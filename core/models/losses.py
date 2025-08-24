from torch import Tensor
import torch.nn as nn
import math
import torch
from core.models.entropy_ import entropy
import torch.nn.functional as F
class NormalizedEntropyLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ['none', 'mean', 'sum'], 'invalid reduction'
        self.reduction = reduction

    def forward(self, logits: Tensor):
        assert logits.dim() in [3, 4]
        dim = logits.shape[1]#19
        p_log_p = nn.functional.softmax(logits, dim=1) * nn.functional.log_softmax(logits, dim=1)
        ent = -1.0 * p_log_p.sum(dim=1)  # b x h x w OR b x n
        loss = ent / math.log(dim)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
class CDCLoss(nn.Module):#全部默认参数
    def __init__(self,
                 feat_dim: int = 128,#128
                 temperature: float = 0.3,#0.3
                 num_grid: int = 7,#7
                 queue_len: int = 65536,
                 warm_up_steps: int = 2500,
                 confidence_threshold: float = 0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.num_grid = num_grid
        self.queue_len = queue_len
        self.warm_up_steps = warm_up_steps
        self.confidence_threshold = confidence_threshold
        #queue queue_ptr  初始化
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))#128 * 65536(均值0，标准差1）
        self.queue = nn.functional.normalize(self.queue, p=2, dim=0).cuda()
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, emb_anc, emb_pos):
        # since the features are normalized afterwards, we don't need to worry about the由于这些特征是在之后规范化的，所以我们不需要担心
        # normalization factor during pooling (for a weighted average)池化期间的归一化因子(用于加权平均值)
        emb_anc = emb_anc
        emb_anc = nn.functional.adaptive_avg_pool2d(emb_anc, self.num_grid).permute(0, 2, 3, 1).contiguous().view(-1, self.feat_dim)
        emb_anc_zero_mask = torch.linalg.norm(emb_anc, dim=1) != 0

        emb_pos = emb_pos
        emb_pos = nn.functional.adaptive_avg_pool2d(emb_pos, self.num_grid).permute(0, 2, 3, 1).contiguous().view(-1, self.feat_dim)
        emb_pos_zero_mask = torch.linalg.norm(emb_pos, dim=1) != 0

        zero_mask = emb_anc_zero_mask & emb_pos_zero_mask

        emb_anc = emb_anc[zero_mask]
        emb_pos = emb_pos[zero_mask]
        emb_anc = nn.functional.normalize(emb_anc, p=2, dim=1)
        emb_pos = nn.functional.normalize(emb_pos, p=2, dim=1)

        l_pos_dense = torch.einsum('nc,nc->n', [emb_anc, emb_pos]).unsqueeze(-1)
        l_neg_dense = torch.einsum('nc,ck->nk', [emb_anc, self.queue.clone().detach()])

        logits = torch.cat((l_pos_dense, l_neg_dense), dim=1) / self.temperature
        labels = torch.zeros((logits.size(0), ), dtype=torch.long, device=logits.device)
        loss = nn.functional.cross_entropy(logits, labels, reduction='mean')

        return loss

    @torch.no_grad()
    def update_queue(self, emb_neg):#1 128 256 256
        emb_neg = nn.functional.adaptive_avg_pool2d(emb_neg, self.num_grid).permute(0, 2, 3, 1).contiguous().view(-1, self.feat_dim)
        emb_neg = nn.functional.normalize(emb_neg, p=2, dim=1)

        batch_size = emb_neg.shape[0]#49
        #ptr 队列当前指针头位置
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_len:
            sec1, sec2 = self.queue_len - ptr, ptr + batch_size - self.queue_len
            emb_neg1, emb_neg2 = torch.split(emb_neg, [sec1, sec2], dim=0)
            self.queue[:, -sec1:] = emb_neg1.transpose(0, 1)
            self.queue[:, :sec2] = emb_neg2.transpose(0, 1)
        else:
            self.queue[:, ptr:ptr + batch_size] = emb_neg.transpose(0, 1)

        ptr = (ptr + batch_size) % self.queue_len  # move pointer
        self.queue_ptr[0] = ptr
class MinimumClassConfusionLoss(nn.Module):
    r"""
    Minimum Class Confusion loss minimizes the class confusion in the target predictions.

    You can see more details in `Minimum Class Confusion for Versatile Domain Adaptation (ECCV 2020) <https://arxiv.org/abs/1912.03699>`_

    Args:
        temperature (float) : The temperature for rescaling, the prediction will shrink to vanilla softmax if
          temperature is 1.0.

    .. note::
        Make sure that temperature is larger than 0.

    Inputs: g_t
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`

    Shape:
        - g_t: :math:`(minibatch, C)` where C means the number of classes.
        - Output: scalar.

    Examples::
        >>> temperature = 2.0
        >>> loss = MinimumClassConfusionLoss(temperature)
        >>> # logits output from target domain
        >>> g_t = torch.randn(batch_size, num_classes)
        >>> output = loss(g_t)

    MCC can also serve as a regularizer for existing methods.
    Examples::
        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> temperature = 2.0
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim, hidden_size=1024)
        >>> cdan_loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> mcc_loss = MinimumClassConfusionLoss(temperature)
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> total_loss = cdan_loss(g_s, f_s, g_t, f_t) + mcc_loss(g_t)
    """

    def __init__(self, temperature: float):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss
