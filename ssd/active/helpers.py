import logging

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler

from ssd.utils import save_to_csv
from ssd.engine.inference import do_evaluation
from ssd.engine.trainer import do_train
from ssd.data.build import BatchCollator, samplers


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
        save_to_csv(args.querypath, [0, self.labeled])
        self.labeled_sampler = MaskSampler(self.labeled, len(self.dataset), on_labeled=True)
        self.unlabeled_sampler = MaskSampler(self.labeled, len(self.dataset), on_labeled=False)
        self.batch_size = cfg.SOLVER.BATCH_SIZE
        self.num_workers = cfg.DATA_LOADER.NUM_WORKERS
        self.pin_memory = cfg.DATA_LOADER.PIN_MEMORY

    def __len__(self):
        return len(self.labeled)

    def add_to_labeled(self, indices, step):
        save_to_csv(self.args.querypath, [step, indices])
        self.labeled_sampler.add_to_labeled(indices)
        self.unlabeled_sampler.add_to_labeled(indices)

    def len_annotations(self):
        n_bbox = 0
        for i in self.labeled:
            n_bbox += len(self.dataset.get_annotation(i)[1][0])
        return n_bbox

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


