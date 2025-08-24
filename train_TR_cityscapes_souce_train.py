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
import torch.nn.functional as F
import wandb
from core.configs import cfg
from core.datasets import build_dataset, rand_mixer_v2
from core.models import build_model, build_feature_extractor, build_classifier 
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU 

from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.apis.inference import evel_stu, run_test, run_candidate

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
            "dataset": "CIFAR-100",
            "epochs": 10,
    },
)
    logger = logging.getLogger("TR.trainer")
    logger.info("Start training")

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)
    
    classifier = build_classifier(cfg)
    classifier.to(device)



    ## distributed training
    batch_size = cfg.SOLVER.BATCH_SIZE#4
    ## optimizer
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()
    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()



    
    output_dir = cfg.OUTPUT_DIR
    local_rank = 0
    start_epoch = 0
    iteration = 0
    
    ####  resume the ckpt 
   
    
    ####  loss define
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    ####  model define
    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    classifier.train()



    ###### Mixup and  rsc init

    tgt_train_data = build_dataset(cfg, mode='train', is_source=True)
    print('len(tgt_train_data)', len(tgt_train_data))

    tgt_train_sampler = None
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True, sampler=tgt_train_sampler, drop_last=True)

    ###### confident  init 
    #default param in SAC (https://github.com/visinf/da-sac)
    THRESHOLD_BETA = 0.001
    running_conf = torch.zeros(cfg.MODEL.NUM_CLASSES).cuda(0)
    running_conf.fill_(THRESHOLD_BETA)


    start_training_time = time.time()
    end = time.time()
    
    #duibi-1


    for rebuild_id in range(255):
        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
        for i, (image, label, name) in enumerate(tgt_train_loader):

            data_time = time.time() - end
            current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
            for index in range(len(optimizer_fea.param_groups)):
                optimizer_fea.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_cls.param_groups)):
                optimizer_cls.param_groups[index]['lr'] = current_lr*10

            optimizer_fea.zero_grad()
            optimizer_cls.zero_grad()

            image = image.to(device)
            label = label.to(device)
            image_size = image.shape[-2:]

            features  = feature_extractor(image)[1]

            ### stu forward
            fea_size = features.shape[-2:]
            _pred = classifier(features)
            _pred = F.interpolate(_pred, size=image_size, mode='bilinear', align_corners=True)



            st_loss = criterion(_pred, label.long())# * w_add
            pesudo_p_loss = st_loss.mean()

            st_loss = pesudo_p_loss
            

            st_loss.backward() 
            
            ### update current model
            optimizer_fea.step()
            optimizer_cls.step()



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
                torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(), 'classifier':classifier.state_dict()}, filename)
                acc = run_test(cfg, feature_extractor, classifier, local_rank, distributed)
                run.log({"mIoU": acc})

            run.log({"mIoU": acc if "acc" in locals() else 0,
                     #"hunxiao_loaa": loss3,
                     "total_loss": st_loss
                })



            
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

    return feature_extractor, classifier

            


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

    logger = setup_logger("TR", output_dir, args.local_rank)
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