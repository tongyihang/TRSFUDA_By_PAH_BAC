import argparse
import os
import copy
import datetime
import logging
import time
import math
import numpy as np
from collections import OrderedDict

import torch
import random
import pickle
import torch.nn.functional as F
from PIL import Image
from core.datasets.transform import Compose
import wandb
from core.configs import cfg
from core.datasets import build_dataset, rand_mixer_v2
from core.models import build_model, build_feature_extractor, build_classifier 
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU 
from core.utils.misc import get_color_pallete, strip_prefix_if_present, WeightEMA, denormalizeimage
from core.utils.losses import BinaryCrossEntropy, pseudo_labels_probs, update_running_conf, full2weak

from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.apis.inference import inference, multi_scale_inference, sample_val_images
from core.apis.inference import evel_stu, run_test, run_candidate
from core.apis.inference import soft_ND_measure, entropy_measure
import pandas as pd
from datasets.generate_city_label_info import gen_lb_info
import higher
from core.models.losses import MinimumClassConfusionLoss
import torch.nn as nn

def dilation(mask, kernel_size=7):
    # max pooling 
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    return F.max_pool2d(mask.float(), kernel_size, stride=1, padding=kernel_size // 2)
    

def erosion(mask, kernel_size=7):
    # min pooling 
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    return -F.max_pool2d(-mask.float(), kernel_size, stride=1, padding=kernel_size // 2)

def entropy_loss(pred_probs):
    pixel_entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-6),dim = 1)
    al = torch.mean(pixel_entropy)
    return al

def shift_log(x, offset=1e-6):
    """
    First shift, then calculate log for numerical stability.
    """

    return torch.log(torch.clamp(x + offset, max=1.))
class WorstCaseEstimationLoss(nn.Module):
   

    def __init__(self):
        super(WorstCaseEstimationLoss, self).__init__()

    def forward(self,y_u, y_u_adv):
        
        prob = F.softmax(y_u_adv, dim=1)
 
        """
        conf,psd_label = prob.max(1)
        mask = conf < 0.40                     
        filtered_psd_label = y_u.clone()  
        filtered_psd_label[~mask] = 255        
        y_u = filtered_psd_label
        loss = F.cross_entropy(y_u_adv, y_u, ignore_index=255)
        return loss
        """
        loss_u = F.nll_loss(shift_log(1. - prob), y_u, ignore_index=255, reduction='none')
        al = entropy_loss(prob)
        return loss_u.mean() -  2  * al

def train(cfg, local_rank, distributed):
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
        entity="SFDA_01",
    # Set the wandb project where this run will be logged.
        project="TR_FUBEN",
        mode="offline",
    # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "100",
            "epochs": 10,
    },
)
    logger = logging.getLogger("DTST.trainer")
    logger.info("Start training")

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)
    
    classifier = build_classifier(cfg)
    classifier.to(device)

    classifier_worse = build_classifier(cfg,1)
    classifier_worse.to(device)

    ## distributed training
    batch_size = cfg.SOLVER.BATCH_SIZE#4
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size())//2
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg2
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()
    
    ## optimizer 
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()
    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    optimizer_worse = torch.optim.SGD(classifier_worse.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_worse.zero_grad()


    
    output_dir = cfg.OUTPUT_DIR
    local_rank = 0
    start_epoch = 0
    iteration = 0
    
    ####  resume the ckpt 
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        classifier_worse.load_state_dict(classifier_weights)
    
    ####  loss define
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    binary_ce = BinaryCrossEntropy(ignore_index=255)
    worse_loss = WorstCaseEstimationLoss() 
    ####  model define
    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    classifier.train()
    classifier_worse.train()
    classifier_his = copy.deepcopy(classifier).cuda(0)
    feature_extractor_his = copy.deepcopy(feature_extractor).cuda(0)
    classifier_his.eval()
    feature_extractor_his.eval()

    minloss = MinimumClassConfusionLoss(5.0)
    
    
    ###### Mixup and  rsc init 
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'CTR')) and \
        os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'CTR_O')):
        start_from_None = False
        print('+++++++++++++++++++++++  mixup from stable')
    else:
        print('+++++++++++++++++++++++  mixup from None')
        start_from_None = True
        run_candidate(cfg, feature_extractor_his, classifier_his, local_rank, distributed, init_candidate=True, update_meta=True)
        gen_lb_info(cfg, 'CTR')  ## for mixup 
        gen_lb_info(cfg, 'CTR_O')  ## for rsc 
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False, rsc='CTR_O.p')
    print('len(tgt_train_data)', len(tgt_train_data))
    if cfg.TCR.OPEN:
        tgt_train_data.mixer.label_to_file, _ = pickle.load(open(os.path.join(cfg.OUTPUT_DIR, 'CTR.p'), "rb"))

    if distributed:
        tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    else:
        tgt_train_sampler = None
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True, sampler=tgt_train_sampler, drop_last=True)

    ###### confident  init 
    #default param in SAC (https://github.com/visinf/da-sac)
    THRESHOLD_BETA = 0.001
    running_conf = torch.zeros(cfg.MODEL.NUM_CLASSES).cuda(0)
    running_conf.fill_(THRESHOLD_BETA)

    ###### Dynamic teacher init
    if cfg.DTU.DYNAMIC:
        stu_eval_list = []
        stu_score_buffer = []
        res_dict = {'stu_ori':[], 'stu_now':[], 'update_iter':[]}
        
        
    cls_his_optimizer = WeightEMA(
        list(classifier_his.parameters()), 
        list(classifier.parameters()),
        list(copy.deepcopy(classifier_his).parameters()),
        alpha= cfg.DTU.EMA_WEIGHT,
        v_r=0.999,

    )  
    feature_extractor_his = copy.deepcopy(feature_extractor).cuda(0)
    fea_his_optimizer = WeightEMA(
        list(feature_extractor_his.parameters()), 
        list(feature_extractor.parameters()),
        list(copy.deepcopy(feature_extractor_his).parameters()),
        alpha= cfg.DTU.EMA_WEIGHT,
        v_r=0.999,
    )      
    
    
    if cfg.METAPL.OPEN:
        cls_meta = build_classifier(cfg).to(device)
        gen_lb_info(cfg, 'meta_val')
        gen_lb_info(cfg, 'meta_val_mixup')
        

    start_training_time = time.time()
    end = time.time()
    
    #duibi-1
    cur_class_dist = np.zeros(cfg.MODEL.NUM_CLASSES)
    w = 1 / np.log(1 + 1e-2 + cur_class_dist)
    w = w / w.sum()

    for rebuild_id in range(255):
        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
        for i, (tgt_input, y, names, tgt_trans_param, tgt_img_full, mix_label) in enumerate(tgt_train_loader):
            data_time = time.time() - end
            current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
            for index in range(len(optimizer_fea.param_groups)):
                optimizer_fea.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_cls.param_groups)):
                optimizer_cls.param_groups[index]['lr'] = current_lr*10
            for index in range(len(optimizer_worse.param_groups)):
                optimizer_worse.param_groups[index]['lr'] = current_lr*5

            optimizer_fea.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_worse.zero_grad()

            tgt_input = tgt_input.to(device)
            tgt_size = tgt_input.shape[-2:]
            tgt_img_full = tgt_img_full.to(device)
            mix_label = mix_label.to(device)
            tgt_full_size = tgt_img_full.shape[-2:]
            
            tgt_img_full_resized = F.interpolate(tgt_img_full, size=tgt_size, mode='bilinear', align_corners=True)
            tgt_concat = torch.cat([tgt_input, tgt_img_full_resized], dim=0)
            tgt_features  = feature_extractor(tgt_concat)[1]
            feat_input, feat_full = torch.chunk(tgt_features, chunks=2, dim=0)

            ### stu forward
            fea_size = feat_input.shape[-2:]
            tgt_pred = classifier(feat_input)
            tgt_pred = F.interpolate(tgt_pred, size=tgt_size, mode='bilinear', align_corners=True)
            ### worse forward
            tgt_pred_worse = classifier_worse(feat_full)
            tgt_pred_worse = F.interpolate(tgt_pred_worse, size=tgt_full_size, mode='bilinear', align_corners=True)


            ######### dy update
            if cfg.DTU.DYNAMIC:#False
                with torch.no_grad():
                    tgt_pred_full = classifier(feature_extractor(tgt_img_full.clone().detach())[1])
                    output = F.softmax(tgt_pred_full.clone().detach(), dim=1).detach()
                    if cfg.DTU.PROXY_METRIC == 'ENT': 
                        entropy_val = entropy_measure(output)
                        stu_score_buffer.append(entropy_val)
                        stu_eval_list.append([tgt_img_full.clone().detach().cpu()])
                    elif cfg.DTU.PROXY_METRIC == 'Soft_ND':
                        soft_ND_val, soft_ND_state = soft_ND_measure(output, select_point=100)
                        stu_score_buffer.append(soft_ND_val)
                        stu_eval_list.append([tgt_img_full.clone().detach().cpu(), soft_ND_state.cpu()])
                    else:
                        print('no support')
                        return
            ###########
                    
            #### history model 
            with torch.no_grad():   
                
                size = tgt_img_full.shape[-2:]
                train_feas = feature_extractor_his(tgt_img_full)[1]
                tgt_pred_his_full = classifier_his(train_feas, tgt_img_full.shape[-2:])
                tgt_prob_his = F.softmax(full2weak(tgt_pred_his_full, tgt_trans_param), dim=1)
                train_feas = full2weak(train_feas, tgt_trans_param, down_ratio=8)
                
                # pos label
                running_conf = update_running_conf(F.softmax(tgt_pred_his_full, dim=1), running_conf, THRESHOLD_BETA)
                #psd_label, _, _ = pseudo_labels_probs(tgt_prob_his, running_conf, THRESHOLD_BETA)
                if cfg.METAPL.OPEN:
                    psd_label = tgt_prob_his.max(1)[1].detach()
                else:
                    #psd_label, _, _ = pseudo_labels_probs(tgt_prob_his, running_conf, THRESHOLD_BETA)
                    #print(type(tgt_prob_his))
                    _,new_label = F.softmax(tgt_pred_his_full,dim=1).max(1)
                    conf,psd_label = tgt_prob_his.max(1)
                    #mask = conf > 0.95                     
                    #filtered_psd_label = psd_label.clone() 
                    #filtered_psd_label[~mask] = 255       
                    #psd_label = filtered_psd_label        

            if cfg.METAPL.OPEN:#false
                # and (iteration+1) > cfg.METAPL.VAL_UPDATE 
                # meta-pseudo labeling
                cls_meta.load_state_dict(classifier_his.state_dict())
                cls_meta.train()
                inner_optimiser = torch.optim.SGD(cls_meta.parameters(), lr= 0.1)
                para_w = torch.ones_like(psd_label).float().cuda()
                para_w = torch.nn.Parameter(para_w, requires_grad=True)
                optimiser_para_w = torch.optim.Adam([para_w], lr=9.0e-1) 

                
                psd_label = psd_label * (mix_label==255) + mix_label * ((mix_label!=255))
                psd_label = psd_label.long()
                
                ## start inner-loop for calculate w
                optimize_cnt = 1
                for optimize_id in range(optimize_cnt):
                    inner_optimiser.zero_grad()
                    optimiser_para_w.zero_grad()
                    with higher.innerloop_ctx(cls_meta, inner_optimiser, copy_initial_weights=False) as (fcls, diffopt):
                        ####### pesudo optimize
                        virtual_pred = fcls(train_feas.detach(), psd_label.shape[-2:])
                        s_ce_loss = criterion(virtual_pred, psd_label.long()) * para_w 
                        s_ce_loss = s_ce_loss.mean()
                        diffopt.step(s_ce_loss)
                        virtual_psd = virtual_pred.max(1)[1]
                        #uni_psd = torch.unique(virtual_psd)
                        
                        ##### updating the w by unbaised data and meta cls
                        val_cnt = 1
                        val_loss = 0

                        #### mixup resampling 
                        for cnt in range(val_cnt):
                            val_input, val_label = sample_val_images(cfg, psd_label, meta_val='meta_val')
                            val_input = val_input.cuda(non_blocking=True)
                            val_label = val_label.cuda(non_blocking=True).long()              
                            
                            v_size = val_label.shape[-2:]
                            with torch.no_grad():
                                val_feas = feature_extractor(val_input)[1]
                            val_pred = fcls(val_feas.detach(), v_size)
                            val_loss += criterion(val_pred, val_label).mean()
                        (val_loss/val_cnt).backward()
                            
                    optimiser_para_w.step()            
                ## end inner-loop for calculate w
                uc_map_eln = para_w.clone().detach()
                if cfg.TCR.OPEN:
                    uc_map_eln[mix_label!=255] = 1
            else:
                uc_map_eln = torch.ones_like(psd_label).float()

            psd_label_1 = psd_label * (mix_label==255) + mix_label * ((mix_label!=255))
            
            output_w = F.softmax(tgt_pred.clone().detach(), dim=1).detach()
            max_values, max_indices = torch.max(output_w, dim=1)
            #w_add = -(5 - 5*torch.exp(-(max_values * max_values) / 2 ))+2
            w_add = -(5 - 5*torch.exp(-(max_values * max_values) / 2 )) + 2
            """
            #tgt_pred_loss = tgt_pred.permute(0, 2, 3, 1).contiguous().view(-1,19)
            #loss3 = minloss(tgt_pred_loss)#
           
            w = -torch.sum(output_w*torch.log2(output_w+1e-5),dim=1)
            max_entropy = torch.log2(torch.tensor(cfg.MODEL.NUM_CLASSES,dtype=w.dtype,device=w.device))
            w = w / max_entropy
            w = torch.exp(-w)
            """
            st_loss = criterion(tgt_pred, psd_label_1.long())# * w_add
            wo_loss = worse_loss(new_label.long(),tgt_pred_worse)
            #pesudo_p_loss = st_loss.mean()
            pesudo_p_loss = (st_loss * (0.5 + 0.5*(uc_map_eln-1).exp()) ).mean()
          
            
            loss_dilation = 0.0
            temperature = 1
            B1, _, _, _ = feat_input.shape
            c = np.random.choice(cfg.MODEL.NUM_CLASSES, p=w)
            
            for b in range(B1):

                init_mask = (psd_label_1[b].unsqueeze(0) == c).int()# 1 512 1024
                dilation_mask  = dilation(init_mask)
                private_mask = init_mask.unsqueeze(0)
                common_mask = torch.logical_and((1-private_mask), dilation_mask) # [B,1,256,256]

                M_p = erosion(init_mask)
                private_mask = (F.interpolate(private_mask.float(), scale_factor=1/8, mode='nearest')>0.5).int()
                dilation_mask = (F.interpolate(dilation_mask.float(), scale_factor=1/8, mode='nearest')>0.5).int()
                M_p = (F.interpolate(M_p.float(), scale_factor=1/8, mode='nearest')>0.5).int()
                common_mask = (F.interpolate(common_mask.float(), scale_factor=1/8, mode='nearest')>0.5).int()

                private_mask = private_mask.reshape(-1,1)#torch.Size([8192, 1])
                dilation_mask = dilation_mask.reshape(-1,1).repeat(1,2048)
                M_p = M_p.reshape(-1,1)
                common_mask = common_mask.reshape(-1,1)
                feat_input1 = feat_input[b].permute(1,2,0).reshape(-1,2048)
                if torch.count_nonzero(private_mask)>0  and torch.count_nonzero(M_p)>0 and torch.count_nonzero(common_mask)>64:
                    private_feat = feat_input1[(M_p!=0).squeeze(1)]
                    private_proto = feat_input1[(M_p!=0).squeeze(1)]
                    private_proto = torch.mean(private_proto, dim=0, keepdim=True)
                    common_feat = feat_input1[(common_mask!=0).squeeze(1)]
                    
                    target_feat = torch.cat([private_proto, common_feat], dim=0) 
                    logits = torch.matmul(F.normalize(private_feat, dim=-1), F.normalize(target_feat, dim=-1).T)
                    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(feat_input.device)
                    loss_dilation += criterion(logits, labels).mean()
                #loss_dilation = loss_dilation.mean()
            #pesudo_p_loss += 0.1 * loss_dilation
            cur_class_dist[c] += 1
            w = 1 / np.log(1 + 1e-2 + cur_class_dist)
            w = w / w.sum()
            
            #pesudo_n_loss = binary_ce(tgt_pred.view(m_batchsize*C, 1, height, width), t_neg_label) * 0.5
            st_loss = pesudo_p_loss +wo_loss  + 0.1 * loss_dilation
            

            st_loss.backward() 
            
            ### update current model
            optimizer_fea.step()
            optimizer_cls.step()
            optimizer_worse.step()

        
            ### update history model
            ### eval student perfromance
            if cfg.DTU.DYNAMIC:
                if len(stu_score_buffer) >= cfg.DTU.Query_START and int(len(stu_score_buffer)-cfg.DTU.Query_START) % cfg.DTU.Query_STEP ==0:   
                    all_score = evel_stu(cfg, feature_extractor, classifier, stu_eval_list)
                    compare_res = np.array(all_score) - np.array(stu_score_buffer)
                    if np.mean(compare_res > 0) > 0.5 or len(stu_score_buffer) > cfg.DTU.META_MAX_UPDATE:
                        update_iter = len(stu_score_buffer)

                        cls_his_optimizer.step()
                        fea_his_optimizer.step()
                        
                        res_dict['stu_ori'].append(np.array(stu_score_buffer).mean())
                        res_dict['stu_now'].append(np.array(all_score).mean())
                        res_dict['update_iter'].append(update_iter)
                        
                        df = pd.DataFrame(res_dict)
                        df.to_csv('dyIter_FN.csv')

                        ## reset
                        stu_eval_list = []
                        stu_score_buffer = []

            else:
                if iteration % cfg.DTU.FIX_ITERATION == 0:
                    
                    #cls_his_optimizer.update_vc_params()
                    #fea_his_optimizer.update_vc_params()
                    cls_his_optimizer.step()
                    fea_his_optimizer.step()
           
            ## update
            if cfg.TCR.OPEN:
                tgt_train_data.mix_p = (1-running_conf[tgt_train_data.mix_classes]) / ((1-running_conf[tgt_train_data.mix_classes]).sum() )
            
            meters.update(loss_p_loss=pesudo_p_loss.item())
            
            iteration = iteration + 1

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iters:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters),
                            lr=optimizer_fea.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )

            if (iteration == cfg.SOLVER.MAX_ITER or (iteration+1) % (cfg.SOLVER.CHECKPOINT_PERIOD)==0):
                filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(int(iteration)))
                torch.save({'iteration': iteration, 'feature_extractor': feature_extractor_his.state_dict(), 'classifier':classifier_his.state_dict()}, filename)
                acc = run_test(cfg, feature_extractor_his, classifier_his, local_rank, distributed)
                run.log({"mIoU": acc})

            run.log({"mIoU": acc if "acc" in locals() else 0,
                     #"hunxiao_loaa": loss3,
                     #"wo_loss": wo_loss,
                     "total_loss": st_loss
                })


            
            ### re-build candidate and dataloader
            if (cfg.TCR.OPEN and (iteration+1) % cfg.TCR.UPDATE_FREQUENCY == 0) or \
                (cfg.METAPL.OPEN and (iteration+1) % cfg.METAPL.VAL_UPDATE == 0):
                run_candidate(cfg, feature_extractor_his, classifier_his, local_rank, distributed, update_meta=start_from_None)
                gen_lb_info(cfg, 'CTR')  ## for mixup 
                gen_lb_info(cfg, 'CTR_O')  ## for rsc 
                tgt_train_data = build_dataset(cfg, mode='train', is_source=False, rsc='CTR_O.p')
                if distributed:
                    tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
                else:
                    tgt_train_sampler = None
                tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True, sampler=tgt_train_sampler, drop_last=True)
                tgt_train_data.mixer.label_to_file, _ = pickle.load(open(os.path.join(cfg.OUTPUT_DIR, 'CTR.p'), "rb"))
                if cfg.METAPL.OPEN:
                    ## meta_val
                    gen_lb_info(cfg, 'meta_val')
                    ## meta_val_mixup
                    gen_lb_info(cfg, 'meta_val_mixup')
        
                break
            
            if iteration == cfg.SOLVER.STOP_ITER:
                break
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (cfg.SOLVER.MAX_ITER)
        )
    )

    run.finish()

    return feature_extractor_his, classifier_his          

            


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        RANK = int(os.environ["RANK"])
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            NGPUS_PER_NODE = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        else:
            NGPUS_PER_NODE = torch.cuda.device_count()
        assert NGPUS_PER_NODE > 0, "CUDA is not supported"
        GPU = RANK % NGPUS_PER_NODE
        torch.cuda.set_device(GPU)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='tcp://{}:{}'.format(
                                                 master_address, master_port),
                                             rank=RANK, world_size=WORLD_SIZE)
        NUM_GPUS = WORLD_SIZE
        print(f"RANK and WORLD_SIZE in environ: {RANK}/{WORLD_SIZE}")
        # torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(
        #     backend="nccl", init_method="env://"
        # )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("DTST", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    fea, cls = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, fea, cls, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()

