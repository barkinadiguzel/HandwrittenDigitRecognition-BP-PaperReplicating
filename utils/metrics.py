import torch

def mean_squared_error(output, target):
    return ((output - target) ** 2).mean()

def accuracy(output, target):
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()

def rejection_threshold(output, target, threshold=0.8):
    probs = torch.softmax(output, dim=1)
    max_probs, preds = probs.max(dim=1)
    accepted = max_probs >= threshold
    correct = preds == target
    return (correct & accepted).float().sum().item() / max(1, accepted.sum().item())
