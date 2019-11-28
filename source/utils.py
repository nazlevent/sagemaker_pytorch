from preproc import *
from sklearn.metrics import matthews_corrcoef


def transer_gpu_dataloader(X, mask, y, batch_size, device):
    X = X.to(device)
    mask = mask.to(device)
    y = y.to(device)
    data = TensorDataset(X, mask, y)
    sampler = RandomSampler(data)
    return DataLoader(data, sampler=sampler, batch_size=batch_size)


def simple_accuracy(preds, labels):
    return np.sum(preds == labels) / len(labels)


def flatten_list(l):
    fl = [item for sl in l for item in sl]
    return np.array(fl)


def matthew_correlation(preds, labels):
    return matthews_corrcoef(labels, preds)
