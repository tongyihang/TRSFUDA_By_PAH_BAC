import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter

from typing import Optional, Any, Tuple
import numpy as np
import torch.nn as nn
from torch.autograd import Function

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None
class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float64(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )# 根据当前iter计算λ
        if self.auto_step:
            self.step()# 自动加1
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1
class ASPP_Classifier_V2(nn.Module):
    def __init__(self, in_channels, dilation_series, padding_series, num_classes):
        super(ASPP_Classifier_V2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    in_channels,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )
        # self.conv = nn.Conv2d(num_classes,num_classes,kernel_size=1)
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x, size=None):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        # print("out1",out.shape)
        # out = self.conv(out)#自己加的 看权重文件呢
        # print("out",out.shape)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out
        
# if __name__=='__main__':
#     classifier = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], 19)
#     a = torch.rand([1,2048,96,192])
#     d = classifier(a)
#     print("d",d.shape)#1 19 h w
#     print(classifier.conv.weight.shape)
class ASPP_Classifier_WORSE(nn.Module):
    def __init__(self, in_channels, dilation_series, padding_series, num_classes):
        super(ASPP_Classifier_WORSE, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    in_channels,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=20000,
                                                       auto_step=True)
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x, size=None):
        f_adv = self.grl_layer(x)
        out = self.conv2d_list[0](f_adv)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](f_adv)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out
