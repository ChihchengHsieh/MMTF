import math
import sys
import time
import numpy as np
import torch
import torchmetrics
import torch.nn.functional as F
import torch.nn as nn

from utils.loggers import MetricLogger, SmoothedValue
from utils.dict import reduce_dict
from utils.tensor import nested_to_device
from utils.train import TrainingInfo
from torch.optim.optimizer import Optimizer
from copy import deepcopy
from sklearn.metrics import mean_squared_error


cpu_device = torch.device("cpu")


class RegressionEvaluator(object):
    def __init__(self) -> None:
        self.logits = []
        self.labels = []

    def update(self, logits, targets):
        for l in logits:
            self.logits.append(l.to(cpu_device).detach().cpu().numpy())  # .numpy())

        for t in targets:
            self.labels.append(t.to(cpu_device).detach().cpu().numpy())  # .numpy())

    def get_performance(
        self,
        clinical_labels,
    ):
        mse = mean_squared_error(self.labels, self.logits, multioutput="raw_values")

        p = {f"MSE-{c}": m for c, m in zip(clinical_labels, mse)}
        p["MSE-mean"] = mse.mean()

        # return {
        #     "MSE": mean_squared_error(self.labels, self.logits),
        # }
        return p


def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    device,
    epoch,
    max_norm=0,
    lr_scheduler=None,
    dropout_adp=None,
):
    model.train()
    criterion.train()

    evaluator = RegressionEvaluator()

    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(dataloader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    if dropout_adp:
        dropout_adp.update(model, epoch)

    losses = []
    iters = len(dataloader)

    for i, (x, targets) in enumerate(dataloader):
        # x = torch.stack(x).to(device)
        x = nested_to_device(x, device)
        targets = torch.stack(targets).to(device)

        optimizer.zero_grad()
        outputs = model(x)
        # model.outputs = outputs
        # model.targets = targets
        loss = criterion(outputs, targets)
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if lr_scheduler is not None:
            if isinstance(
                lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            ):
                lr_scheduler.step(epoch + i / iters)
            else:
                lr_scheduler.step()

        # evaluator.update(torch.argmax(outputs, dim=1), targets) # classification
        evaluator.update(outputs, targets)

        losses.append(loss.item())

    return losses, evaluator


@torch.inference_mode()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()
    evaluator = RegressionEvaluator()
    losses = []

    for x, targets in data_loader:
        x = nested_to_device(x, device)
        targets = torch.stack(targets).to(device)
        outputs = model(x)
        loss = criterion(outputs, targets)
        # evaluator.update(torch.argmax(outputs, dim=1), targets)
        evaluator.update(outputs, targets)
        losses.append(loss.item())

    return losses, evaluator
