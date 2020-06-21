import argparse
import logging
import os
import csv
from datetime import datetime

import numpy as np

import torch
import torch.distributed as dist
import torch.utils.data.sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from ssd.data import samplers
from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import *
from ssd.engine.trainer import do_train
from ssd.modeling.detector import build_detection_model
from ssd.data.datasets import build_dataset

from ssd.data.transforms import build_transforms, build_target_transform
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool

from ssd.active.strategies import random_sampling, uncertainty_aldod_sampling

def get_strategy(name):
    if name == 'random_sampling':
        return random_sampling
    elif name == 'uncertainty_aldod_sampling':
        return uncertainty_aldod_sampling
    else:
        raise ValueError(f'Strategy {name} unrecognized')


class MaskSampler(BatchSampler):

    def __init__(self, labeled_indices, length_ds, on_labeled=False):
        self.labeled_indices = labeled_indices
        self.length_ds = length_ds
        self.indices = list(range(length_ds))
        self.on_labeled = on_labeled
        self.unlabeled_indices = list(set(self.indices).difference(set(self.labeled_indices)))

    def __iter__(self):
        if self.on_labeled:
            return (self.indices[i] for i in self.labeled_indices)
        else:
            return (self.indices[i] for i in self.unlabeled_indices)

    def __len__(self):
        return len(self.labeled_indices)

    def add_to_labeled(self, labeled):
        self.labeled_indices += list(labeled)
        self.unlabeled_indices = list(set(self.indices).difference(set(self.labeled_indices)))


class QueryLoader():

    def __init__(self, dataset, args, cfg):
        self.dataset = dataset
        self.logger = logging.getLogger("SSD.trainer")
        self.args = args
        self.labeled = list(np.random.choice(range(len(dataset)), size=args.init_size, replace=False))
        self.labeled_sampler = MaskSampler(self.labeled, len(self.dataset), on_labeled=True)
        self.unlabeled_sampler = MaskSampler(self.labeled, len(self.dataset), on_labeled=False)
        self.batch_size = cfg.SOLVER.BATCH_SIZE
        self.num_workers = cfg.DATA_LOADER.NUM_WORKERS
        self.pin_memory = cfg.DATA_LOADER.PIN_MEMORY

    def __len__(self):
        return len(self.labeled)

    def add_to_labeled(self, indices):
        self.labeled_sampler.add_to_labeled(indices)
        self.unlabeled_sampler.add_to_labeled(indices)

    def get_labeled_loader(self):
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=self.labeled_sampler, batch_size=self.batch_size, drop_last=False)
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=self.args.train_step_per_query, start_iter=0)
        data_loader = torch.utils.data.DataLoader(self.dataset, num_workers=self.num_workers, batch_sampler=batch_sampler,
                                 pin_memory=self.pin_memory, collate_fn=BatchCollator(True))
        return data_loader

    def get_unlabeled_loader(self):
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=self.unlabeled_sampler, batch_size=self.batch_size, drop_last=False)
        data_loader = torch.utils.data.DataLoader(self.dataset, num_workers=self.num_workers, batch_sampler=batch_sampler,
                                 pin_memory=self.pin_memory, collate_fn=BatchCollator(True))
        return data_loader


class ALModel():

    def __init__(self, model, strategy, optimizer, device, scheduler, arguments, args, checkpointer, cfg):
        self.model = model
        self.strategy = strategy
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.arguments = arguments
        self.args = args
        self.checkpointer = checkpointer
        self.cfg = cfg

    def fit(self, train_loader):
        self.model = do_train(self.cfg, self.model, train_loader, self.optimizer,
                        self.scheduler, self.checkpointer, self.device,
                        self.arguments, self.args)
        return self.model

    def score(self):
        torch.cuda.empty_cache()
        eval_results = do_evaluation(self.cfg, self.model, distributed=self.args.distributed)
        mAP = eval_results[0]['metrics']['mAP']
        return mAP

    def query(self, *args, **kwargs):
        self.model = self.model.eval()
        return list(self.strategy(self, *args, **kwargs))


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
    checkpointer = CheckPointer(raw_model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    # extra_checkpoint_data = checkpointer.load()
    # arguments.update(extra_checkpoint_data)

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
    model.fit(query_loader.get_labeled_loader())
    # logger.info(f'Scoring after initial training...')
    # score = model.score()
    # logger.info(f'SCORE : {score:.4f}')
    # with open(args.filename, 'a') as f:
    #     writer = csv.writer(f)
    #     fields = [args.strategy, {}, 0, score]
    #     writer.writerow(fields)

    for step in range(args.query_step):
        logger.info(f'STEP NUMBER {step}')
        logger.info('Querying assets to label')
        query_idx = model.query(unlabeled_loader=query_loader.get_unlabeled_loader(), cfg=cfg,
                    n_instances=args.query_size, length_ds=len(datasets[0]))
        logger.info('Adding labeled samples to train dataset')
        query_loader.add_to_labeled(query_idx)
        logger.info('Fitting with new data...')
        model.fit(query_loader.get_labeled_loader())
        logger.info('Scoring model...')
        score = model.score()
        with open(args.filename, 'a') as f:
            writer = csv.writer(f)
            fields = [args.strategy, {}, step+1, score]
            writer.writerow(fields)
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
    # parser.add_argument('--save_step', default=200, type=int, help='Save checkpoint every save_step')
    # parser.add_argument('--eval_step', default=200, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
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
    # args.save_step = args.train_step_per_query
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

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)
        mkdir(cfg.OUTPUT_DIR + '/results')
        mkdir(cfg.OUTPUT_DIR + '/models')

    fields = ['strategy', 'args', 'step', 'mAP']
    filename = os.path.join(cfg.OUTPUT_DIR, f'results/{args.strategy}-{datetime.now().strftime("%Y%m%d%H%M%S")}.txt')
    args.filename = filename
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = active_train(cfg, args)

    # if not args.skip_test:
    #     logger.info('Final evaluation...')
    #     torch.cuda.empty_cache()  # speed up evaluating after training finished
    #     do_evaluation(cfg, model, distributed=args.distributed)


if __name__ == '__main__':
    main()
