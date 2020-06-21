import numpy as np
from tqdm import tqdm

import torch
import numpy as np

from ssd.active.aldod import *



def compute_logits(classifier, loader, device):
    model = classifier.model
    detections = []
    loader_ids = []
    with torch.no_grad():
        for _, (images, labels, id_) in tqdm(enumerate(loader), total=len(loader)):
            loader_ids += list(id_.numpy())
            images = images.to(device)
            labels = labels.to(device)
            features = model.backbone(images)
            cls_logits, bbox_pred = model.box_head.predictor(features)
            detection_batch, _ = model.box_head._forward_active(cls_logits, bbox_pred)
            detections += detection_batch
    return detections, loader_ids


def random_sampling(classifier, unlabeled_loader, n_instances=100, length_ds=1000, *args, **kwargs):
    query_idx = np.random.choice(length_ds, size=n_instances, replace=False)
    return query_idx


def uncertainty_aldod_sampling(classifier, unlabeled_loader, cfg, n_instances=100, agregation='sum',
                                weighted=False, *args, **kwargs):
    device = torch.device(cfg.MODEL.DEVICE)
    detections, unlabeled_ids = compute_logits(classifier, unlabeled_loader, device)
    id2uncertaintyId = {i: k for k, i in enumerate(unlabeled_ids)}
    uncertaintyId2id = {v: k for k, v in id2uncertaintyId.items()}
    uncertainties = compute_uncertainties(detections, agregation, weighted)
    query_uncertainty_idx = select_top_indices(uncertainties, permut=False, batch_size=10, n_instances=n_instances)
    query_uncertainty_idx = list(map(int, query_uncertainty_idx))
    print()
    print(f'Maximum id of the query : {max(query_uncertainty_idx)}')
    print(f'Number of selected indices : {len(query_uncertainty_idx)}')
    print(f'Number of uncertainties computed : {len(uncertainties)}')
    print()
    assert max(query_uncertainty_idx) < len(uncertainties)
    keys = set(list(uncertaintyId2id.keys()))
    for id_ in query_uncertainty_idx:
        assert id_ in keys, f"id {id_} not in keys !"
    query_idx = [uncertaintyId2id[x] for x in query_uncertainty_idx]
    return query_idx