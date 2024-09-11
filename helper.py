import torch


def pixel_accuracy(output, mask, one_hot=True, use_softmax=False):
    if one_hot:
        if use_softmax:
            output = torch.softmax(output, dim=1)
        preds = torch.argmax(output, dim=1)
        mask = torch.argmax(mask, dim=1)
        correct = (preds == mask).float()
        acc = correct.sum() / correct.numel()
        return acc
    else:
        preds = torch.argmin(output, dim=1)
        masks = torch.argmin(mask, dim=1)
        correct = (preds == masks).float()
        acc = correct.sum() / correct.numel()
        return acc


def counter(x):
    count_total = 0
    count = 0
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            for k in range(x.shape[3]):
                count_total += 1
                if x[0][i][j][k] != 0:
                    count += 1
    return count/count_total
