import argparse
import logging
import os
import csv
from datetime import datetime
import pickle
import time

import numpy as np

import torch
import torch.distributed as dist
import torch.utils.data.sampler
from torch.utils.data.dataloader import default_collate

from ssd.data.build import *
from ssd.data.datasets import build_dataset
from ssd.data.transforms import build_transforms, build_target_transform

from ssd.config import cfg
from ssd.modeling.detector import build_detection_model
from ssd.solver.build import make_optimizer, make_lr_scheduler

from ssd.utils import *
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger

from ssd.active.strategies import random_sampling, uncertainty_aldod_sampling
from ssd.active.helpers import *


def get_strategy(name):
    if name == 'random_sampling':
        return random_sampling
    elif name == 'uncertainty_aldod_sampling':
        return uncertainty_aldod_sampling
    else:
        raise ValueError(f'Strategy {name} unrecognized')


def active_train(cfg, args):
    logger = logging.getLogger("SSD.trainer")
    raw_model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    raw_model.to(device)

    lr = cfg.SOLVER.LR * args.num_gpus
    optimizer = make_optimizer(cfg, raw_model, lr)

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    
    checkpointer = None
    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(raw_model, optimizer, scheduler, args.model_dir, save_to_disk, logger)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus

    is_train = True
    train_transform = build_transforms(cfg, is_train=is_train)
    target_transform = build_target_transform(cfg) if is_train else None
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(dataset_list, transform=train_transform, target_transform=target_transform, is_train=is_train)
    
    logger.info(f'Creating query loader...')
    query_loader = QueryLoader(datasets[0], args, cfg)

    logger.info(f'Creating al model...')
    strategy = get_strategy(args.strategy)
    model = ALModel(raw_model, strategy, optimizer, device, scheduler, arguments, args, checkpointer, cfg)

    logger.info(f'Training on initial data with size {args.init_size}...')
    n_bbox = query_loader.len_annotations()
    t1 = time.time()
    model.fit(query_loader.get_labeled_loader())
    init_time = time.time() - t1
    logger.info(f'Scoring after initial training...')
    score = model.score()
    logger.info(f'SCORE : {score:.4f}')

    fields = [args.strategy, {}, 0, score, init_time, 0,
        init_time, len(query_loader), n_bbox]
    save_to_csv(args.filename, fields)

    for step in range(args.query_step):
        logger.info(f'STEP NUMBER {step}')
        logger.info('Querying assets to label')
        t1 = time.time()
        query_idx = model.query(unlabeled_loader=query_loader.get_unlabeled_loader(), cfg=cfg,
                    n_instances=args.query_size, length_ds=len(datasets[0]))
        logger.info('Adding labeled samples to train dataset')
        query_loader.add_to_labeled(query_idx, step+1)
        t2 = time.time()
        logger.info('Fitting with new data...')
        model.fit(query_loader.get_labeled_loader())
        total_time = time.time() - t1
        train_time = time.time() - t2
        active_time = total_time - train_time
        logger.info('Scoring model...')
        score = model.score()
        n_bbox = query_loader.len_annotations()
        fields = [args.strategy, {}, step+1, score, train_time, active_time,
            total_time, len(query_loader), n_bbox]
        save_to_csv(args.filename, fields)
        logger.info(f'SCORE : {score:.4f}')

    return model.model

def main():
    """
    python train.py --config-file ../SSD/configs/mobilenet_v2_ssd320_voc0712.yaml \
                    --log_step 10 \
                    --init_size 500 \
                    --query_size 100 \
                    --query_step 2 \
                    --train_step_per_query 50 \
                    --strategy uncertainty_aldod_sampling

    nohup python train.py --config-file ../SSD/configs/mobilenet_v2_ssd320_voc0712.yaml \
                    --log_step 10 \
                    --init_size 1000 \
                    --query_size 300 \
                    --query_step 10 \
                    --train_step_per_query 1000 \
                    --strategy uncertainty_aldod_sampling &     
    """
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--init_size', default=1000, type=int, help='Number of initial labeled samples')
    parser.add_argument('--query_step', default=10, type=int, help='Number of queries')
    parser.add_argument('--query_size', default=300, type=int, help='Number of assets to query each time')
    parser.add_argument('--strategy', default='random_sampling', type=str, help='Strategy to use to sample assets')
    parser.add_argument('--train_step_per_query', default=500, type=int, help='Number of training steps after each query')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
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
    args.save_step = 10000000
    args.eval_step = 10000000

    np.random.seed(42)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_dir = os.path.join(cfg.OUTPUT_DIR, f'results/{args.strategy}/experiment-{time}')
    args.result_dir = experiment_dir
    
    filename = os.path.join(experiment_dir, f'csv.txt')
    argspath = os.path.join(experiment_dir, f'args.pickle')
    querypath = os.path.join(experiment_dir, f'queries.txt')
    model_dir = os.path.join(experiment_dir, 'model')

    mkdir(experiment_dir)
    mkdir(model_dir)

    args.filename = filename
    args.querypath = querypath
    args.model_dir = model_dir
    fields = ['strategy', 'args', 'step', 'mAP', 'train_time', 'active_time',
        'total_time', 'total_samples', 'bboxes']
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    with open(querypath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'indices'])
    with open(argspath, 'wb') as f:
        pickle.dump(args, f)

    logger = setup_logger("SSD", dist_util.get_rank(), experiment_dir)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    active_train(cfg, args)


if __name__ == '__main__':
    main()
