import math
import sys
import torch
from model import OursModel
from torch.utils.data import DataLoader
from utils import MetricLogger, SmoothedValue ,InfoBuffer
from eval import evaluate, update_infoBuffer


def train_one_epoch(
    model: OursModel,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    infolist: InfoBuffer,
    epoch: int,
    scheduler=None,
    args=None,
):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    if args.print_this_epoch:
        data_loader = enumerate(metric_logger.log_every(data_loader_train, args.print_freq, header))
    else:
        data_loader = enumerate(data_loader_train)

    tag = {"rectify": epoch < args.start_rectify_epoch}
    param = {"mmt": args.momentum, "rho":args.rho, "dist": None}
    model.train(True)
    optimizer.zero_grad()
    for data_iter_step, (idx, samples, label1, label2, id1, id2, label_align) in data_loader:
        #droplast
        if len(idx) < math.ceil(args.batch_size / 2):
            continue
        smooth_epoch = epoch + (data_iter_step + 1) / len(data_loader_train)
        if scheduler is not None:
            lr = scheduler(optimizer, smooth_epoch)

        for i in range(args.n_views):
            samples[i] = samples[i].to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=False):
            dist = None
            if epoch > args.start_guide_epoch:
                dist = calculate_dist(infolist, samples, idx, args)
            param["dist"] = dist
            loss = model(samples, param, tag)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.print_this_epoch:
            metric_logger.update(lr=lr)
            metric_logger.update(loss=loss_value)

    if epoch == args.start_guide_epoch:
        update_infoBuffer(infolist, model, data_loader_train, device, args, epoch, 'init')
    elif epoch > args.start_guide_epoch:
        update_infoBuffer(infolist, model, data_loader_train, device, args, epoch, 'guide')

    if args.print_this_epoch:
        eval_result = evaluate(model, data_loader_test, device, args)
    else:
        eval_result = None
    return eval_result

def calculate_dist(infolist, samples, idx, args):
    emp_dist = infolist.get_Buffer()
    ret_sample_dist = torch.zeros(args.n_views, len(idx), device=samples[0].device)
    for i in range(args.n_views):
        ret_sample_dist[i] = emp_dist[i][idx]
    return ret_sample_dist

