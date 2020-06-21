# C. Brust, C. Käding and J. Denzler (2019) Active learning for deep object detection.
# In VISAPP. https://arxiv.org/pdf/1809.09875.pdf
import numpy as np


def compute_uncertainties_asset(detection):
    probas = detection['logits'].softmax(axis=1)
    labels = detection['labels']
    probas = probas.detach().cpu().numpy()
    rev = np.sort(probas, axis=1)[:, ::-1]
    values = (1 - rev[:, 0] - rev[:, 1])**2
    return values, labels

def agregate_detection_uncertainties(agregation_method, weighted):
    def agregation(values, labels):
        if agregation_method == 'mean':
            return values.mean()
        elif agregation_method == 'sum':
            return values.sum()
        elif agregation_method == 'max':
            return values.max()
    return agregation

def compute_uncertainties(detections, agregation, weighted):
    agregate_func = agregate_detection_uncertainties(agregation, weighted)
    uncertainties = np.array(list(map(lambda x: agregate_func(*compute_uncertainties_asset(x)), detections)))
    return uncertainties


def select_top_batches(uncertainties, n_instances=100, batch_size=10):    
    n_samples = len(uncertainties)
    batch_uncertainties = np.array([
        uncertainties[i:i+batch_size].sum()
        for i in range(0, n_samples, batch_size)])
    last_batch_size = n_samples % batch_size
    if last_batch_size > 0:
        batch_uncertainties[-1] *= batch_size / last_batch_size
    ranked = np.argsort(batch_uncertainties)[::-1]
    n_batches = int(np.floor(n_instances // batch_size))
    selected_indices = []
    n_selected = 0
    for batch_id in ranked[:n_batches-1]:
        selected_indices += list(range(batch_size*batch_id, batch_size*(batch_id+1)))
        n_selected += batch_size
    if batch_size < n_instances:
        batch_id = ranked[n_batches-1]
        selected_indices += list(np.random.choice(
            list(range(batch_size*batch_id, batch_size*(batch_id+1))),
            size=n_instances-n_selected))
    return selected_indices

def select_top_indices(uncertainties, permut=True, n_instances=100, batch_size=10):
    if permut:
        permutation = np.random.permutation(len(uncertainties))
        inverse_permutation = np.argsort(permutation)
        permuted2original = dict(zip(inverse_permutation, np.arange(len(permutation))))
        permuted_uncertainties = uncertainties[permutation]
        selected_idx_permuted = select_top_batches(permuted_uncertainties,
                n_instances=n_instances, batch_size=batch_size)
        return np.array([permuted2original[x] for x in selected_idx_permuted])
    else:
        return np.array(select_top_batches(uncertainties, n_instances=n_instances, batch_size=batch_size))

