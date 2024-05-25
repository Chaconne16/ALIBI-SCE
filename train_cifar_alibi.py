#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Runs CIFAR10 and CIFAR100 training with ALIBI for Label Differential Privacy
"""
import argparse
import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed
import torch.utils.tensorboard as tensorboard
import torchvision.transforms as transforms

from lib import models
from lib.alibi import Ohm, RandomizedLabelPrivacy, NoisedCIFAR
from lib.dataset.canary import fill_canaries
from opacus.utils import stats
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm
import matplotlib.pyplot as plt


#######################################################################
# Settings
#######################################################################
@dataclass
class LabelPrivacy:
    sigma: float = 0.1  # noise multiplier -- to be done
    max_grad_norm: float = 1e10
    delta: float = 1e-5  # epsilon-delta DP
    post_process: str = "mapwithprior"
    mechanism: str = "Laplace"
    noise_only_once: bool = True
    alpha: float = 1.0  # TODO 默认为CE


@dataclass
class Learning:
    lr: float = 0.1
    batch_size: int = 128
    epochs: int = 200
    momentum: float = 0.9
    weight_decay: float = 1e-4
    random_aug: bool = False


@dataclass
class Settings:
    dataset: str = "cifar100"
    canary: int = 0  # 错误标记的样本
    arch: str = "wide-resnet"
    # 每次实例化创建新的LabelPrivacy实例作为默认值，避免不同实例的属性共享内存
    privacy: LabelPrivacy = field(default_factory=LabelPrivacy)
    learning: Learning = field(default_factory=Learning)
    gpu: int = 0
    world_size: int = 1
    out_dir_base: str = "/tmp/alibi/"
    data_dir_root: str = "/tmp/"
    seed: int = 0
    savefolder: str = "tr"  # todo


MAX_GRAD_INF = 1e6

#######################################################################
# CIFAR transforms
#######################################################################
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


FIXMATCH_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
FIXMATCH_CIFAR10_STD = (0.2471, 0.2435, 0.2616)
FIXMATCH_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
FIXMATCH_CIFAR100_STD = (0.2675, 0.2565, 0.2761)

#######################################################################
# Stat Collection settings
#######################################################################

# The following few lines, enable stats gathering about the run
_clipping_stats = {}  # will be used to collect stats from different layers
_norm_stats = {}  # will be used to find histograms


def enable_stats(stats_dir):  # to be done

    if stats_dir is None:
        return None
    # 1. where the stats should be logged
    summary_writer = tensorboard.SummaryWriter(stats_dir)
    stats.set_global_summary_writer(summary_writer)
    # 2. enable stats
    stats.add(
        # stats on training accuracy
        stats.Stat(stats.StatType.TRAIN, "accuracy", frequency=1),
        # stats on validation accuracy
        stats.Stat(stats.StatType.TEST, "accuracy"),
        stats.Stat(stats.StatType.TRAIN, "privacy", frequency=1),
    )
    return summary_writer


#######################################################################
# train, test, functions
#######################################################################
def save_checkpoint(state, filename=None):
    torch.save(state, filename)


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, train_loader, optimizer, criterion, device):
    # model.train()将模型设置为训练模式，会自动进行Batch Norm、Dropout等操作，来提升训练效果并防止过拟合
    model.train()
    losses = []
    acc = []

    for i, batch in enumerate(tqdm(train_loader)):  # zero_grad, forward, backward, update
        # 每个batch由images和labels组成，分别对应batch_size个样本的图像和标签
        # 通常等价于images, targets = batch
        images = batch[0].to(device)
        targets = batch[1].to(device)  # 带噪soft label
        labels = targets if len(batch) == 2 else batch[2].to(device)  # 原始数值label，用于计算acc

        # compute output
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        # tensor.detach()返回新的张量，不与计算图相关联
        # output维度是batch_size*class_num，沿class_num方向找最大值索引，作为预测的标签
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)  # gpu上无法进行numpy操作，需要先移至cpu上
        labels = labels.detach().cpu().numpy()

        # measure accuracy and record loss
        acc1 = accuracy(preds, labels)

        losses.append(loss.item())  # 单元素tensor用.item()得到数值
        acc.append(acc1)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

    # --Train acc & loss，该epoch的平均损失&准确率
    # print(
    #     f"Train Loss: {np.mean(losses):.4f} ",
    #     f"Train Acc@1: {np.mean(acc) :.4f} ",
    # )

    return np.mean(acc), np.mean(losses)


def test(model, test_loader, criterion, device, epoch):
    # model.eval()将模型设置为评估模式，会关闭一些训练时的特定操作，确保模型的输出是稳定和可重复的
    # 例如，在评估模式下，Batch Norm层会使用保存的移动平均值和方差来标准化输入，而不是使用当前批次的均值和方差
    model.eval()
    losses = []
    acc = []

    with torch.no_grad():
        for images, target in tqdm(test_loader):  # 等价于for data in tqdm(test_loader): images, target = data
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            acc.append(acc1)

    # print(
    #     f"Test epoch {epoch}:",
    #     f"Loss: {np.mean(losses):.6f} ",
    #     f"Acc@1: {np.mean(acc) :.6f} ",
    # )
    return np.mean(acc), np.mean(losses)


def adjust_learning_rate(optimizer, epoch, lr):
    """
    使用SGD+lr调整or直接使用Adam
    """
    if epoch < 30:  # warm-up
        lr = lr * float(epoch + 1) / 30
    # if epoch < 30 and epoch % 10 != 9:  # --warm-up
    #     return  # --new lr adjusting trial
    # elif epoch < 30 and epoch % 10 == 9:  # --warm-up
    #     lr = lr / 3  # --new lr adjusting trial
    else:
        lr = lr * (0.2 ** (epoch // 60))
        # lr = lr * (0.4 ** (epoch // 60))  # --new lr adjusting trial
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def create_model(arch: str, num_classes: int):
    if "wide" in arch.lower():
        print("Created Wide Resnet Model!")
        return models.wideresnet(
            depth=28,
            widen_factor=8 if num_classes == 100 else 4,
            dropout=0,
            num_classes=num_classes,
        )
    else:
        print("Created simple Resnet Model!")
        return models.resnet18(num_classes=num_classes)


def pic_plot(epc, loss, acc, tloss, tacc, savepath):  # todo
    """plot epoch-loss and epoch-acc curve"""
    plt.plot(epc, loss, label='test loss')
    plt.plot(epc, tloss, label='train loss')
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(savepath, 'epc-loss.png'))  # todo
    plt.clf()  # todo

    plt.plot(epc, acc, label='test acc')
    plt.plot(epc, tacc, label='train acc')
    plt.ylabel("acc")
    plt.xlabel("epochs")
    plt.yticks(ticks=[0.1 * x for x in range(1, 11)])
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(savepath, 'epc-acc.png'))  # todo
    plt.clf()  # todo


#######################################################################
# main worker
#######################################################################


def make_deterministic(seed):
    """固定随机数种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main_worker(settings: Settings):
    print(f"settings are {settings}")
    make_deterministic(settings.seed)
    out_dir_base = settings.out_dir_base
    os.makedirs(out_dir_base, exist_ok=True)  # 指定目录不存在时创建目录

    best_acc = 0
    num_classes = 100 if settings.dataset.lower() == "cifar100" else 10
    model = create_model(settings.arch, num_classes)
    device = torch.device("cuda") if settings.gpu >= 0 else torch.device("cpu")
    model = model.to(device)

    # DEFINE LOSS FUNCTION (CRITERION)
    sigma = settings.privacy.sigma
    noise_only_once = settings.privacy.noise_only_once  # True表示只需要在创建数据集时给label加噪声，一次性完成
    randomized_label_privacy = RandomizedLabelPrivacy(
        sigma=sigma,
        delta=settings.privacy.delta,
        mechanism=settings.privacy.mechanism,
        device=None if noise_only_once else device,
    )
    criterion = Ohm(
        privacy_engine=randomized_label_privacy,
        alpha=settings.privacy.alpha,  # TODO
        post_process=settings.privacy.post_process,
    )
    # DEFINE OPTIMIZER
    optimizer = optim.SGD(
        model.parameters(),
        lr=settings.learning.lr,
        momentum=settings.learning.momentum,
        weight_decay=settings.learning.weight_decay,
        nesterov=True,  # 启用nesterov动量
    )

    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=settings.learning.lr,
    #     betas=(settings.learning.momentum, 0.999),
    #     eps=1e-8,
    #     weight_decay=settings.learning.weight_decay
    # )  # optimizer trial

    # DEFINE DATA
    rand_aug = [
        transforms.RandomHorizontalFlip(),  # 随机水平翻转，默认以0.5的概率进行翻转；只有当DataLoader读取数据时才会进行随机变换
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),  # 裁剪得到32*32的图像，边缘进行镜像的padding操作
    ]
    normalize = []
    # --ALIBI并未使用FIXMATCH，需要使用FIXMATCH对应的mean和std吗?
    if settings.dataset.lower() == "cifar100":
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(FIXMATCH_CIFAR100_MEAN, FIXMATCH_CIFAR100_STD),
        ]
    else:  # CIFAR-10
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(FIXMATCH_CIFAR10_MEAN, FIXMATCH_CIFAR10_STD),
        ]
    # --rand_aug是对原始图像操作还是对ToTensor()之后的tensor进行操作?
    # 先随机变换，再toTensor和Normalize，可以使得tensor的mean和std分别是0和1?
    train_transform = transforms.Compose(
        rand_aug + normalize if settings.learning.random_aug else normalize
    )

    # train data
    CIFAR = CIFAR100 if settings.dataset.lower() == "cifar100" else CIFAR10
    settings.data_dir_root = os.path.join(
        settings.data_dir_root, settings.dataset.lower()
    )
    train_dataset = CIFAR(
        train=True,
        transform=train_transform,
        root=settings.data_dir_root,
        download=True,
    )
    if 0 < settings.canary < len(train_dataset):
        # capture debug info
        original_label_sum = sum(train_dataset.targets)
        original_last10_labels = [train_dataset[-i][1] for i in range(1, 11)]
        # inject canaries
        train_dataset = fill_canaries(
            train_dataset, num_classes, N=settings.canary, seed=settings.seed
        )
        # capture debug info
        canary_label_sum = sum(train_dataset.targets)  # .targets属性是包含所有训练集图像的标签的列表，每个标签是0-9的整数
        canary_last10_labels = [train_dataset[-i][1] for i in range(1, 11)]
        # verify presence，标签求和和最后10个标签都能对应上的话，说明canary injection可能失败
        if original_label_sum == canary_label_sum:
            raise Exception(
                "Canary infiltration has failed."
                f"\nOriginal label sum: {original_label_sum} vs"
                f" Canary label sum: {canary_label_sum}"
                f"\nOriginal last 10 labels: {original_last10_labels} vs"
                f" Canary last 10 labels: {canary_last10_labels}"
            )
    if noise_only_once:
        train_dataset = NoisedCIFAR(
            train_dataset, num_classes, randomized_label_privacy
        )  # 给CIFAR数据集加噪声，__getitem__方法得到图像、带噪soft label、原始数值label
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=settings.learning.batch_size,
        shuffle=True,
        drop_last=True,
    )  # Dataloader会与自定义数据集对应，返回三元tuple的迭代器，分别是图像、带噪soft label、原始数值label

    # test data
    test_dataset = CIFAR(
        train=False,
        transform=transforms.Compose(normalize),
        root=settings.data_dir_root,
        download=True,
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=settings.learning.batch_size, shuffle=False
    )

    cudnn.benchmark = True

    stats_dir = os.path.join(out_dir_base, "stats")
    summary_writer = enable_stats(stats_dir)

    # train and test epochs
    epc_lst, loss_lst, acc_lst, tloss_lst, tacc_lst = [], [], [], [], []

    for epoch in range(settings.learning.epochs):
        adjust_learning_rate(optimizer, epoch, settings.learning.lr)  # --使用Adam，不调整学习率

        randomized_label_privacy.train()  # train状态调用noise()方法会给标签加噪声，eval状态直接返回None
        assert isinstance(criterion, Ohm)  # double check!
        if not noise_only_once:
            randomized_label_privacy.increase_budget()  # 不是一次性给标签加噪声则需要动态计算dp参数

        # train for one epoch
        # model, train_loader, optimizer, criterion, device; 语句似乎不起任何作用
        train_acc, train_loss = train(model, train_loader, optimizer, criterion, device)

        epsilon, alpha = randomized_label_privacy.privacy
        label_change = 0
        label_change = (
            train_dataset.label_change if noise_only_once else criterion.label_change
        )

        stats.update(
            stats.StatType.TRAIN,
            top1Acc=train_acc,
            loss=train_loss,
            epsilon=epsilon,
            alpha=alpha,
            label_change_prob=label_change,
        )  # opacus.utils.stats

        # evaluate on validation set
        if randomized_label_privacy is not None:
            randomized_label_privacy.eval()
        acc, loss = test(model, test_loader, criterion, device, epoch)
        stats.update(stats.StatType.TEST, top1Acc=acc, loss=loss)

        # --train&test reporter
        print(
            f"Train epoch: {epoch}",
            f"Train Loss: {train_loss:.4f} ",
            f"Train Acc: {train_acc:.4f} ",
        )
        print(
            f"Test epoch: {epoch}:",
            f"Loss: {loss:.6f} ",
            f"Acc@1: {acc:.6f} ",
        )

        # add test data to 3 lists
        epc_lst.append(epoch)
        loss_lst.append(loss)
        acc_lst.append(acc)
        tloss_lst.append(train_loss)
        tacc_lst.append(train_acc)

        # remember best acc@1 and save checkpoint
        chkpt_file_name = os.path.join(out_dir_base, f"checkpoint-{epoch}.tar")
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": settings.arch,
                "state_dict": model.state_dict(),
                "acc1": acc,
                "optimizer": optimizer.state_dict(),
            },
            chkpt_file_name,
        )
        if acc > best_acc:
            best_acc = acc
            file_name = os.path.join(out_dir_base, "model.tar")
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": settings.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc,
                    "optimizer": optimizer.state_dict(),
                },
                file_name,
            )
    return acc, best_acc, summary_writer, epc_lst, loss_lst, acc_lst, tloss_lst, tacc_lst


def main():
    parser = argparse.ArgumentParser(description="CIFAR LabelDP Training with ALIBI")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset to run training on (cifar100 or cifar10)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="wide-resnet",
        help="Resnet-18 architecture (wide-resnet vs resnet)",
    )
    # learning
    parser.add_argument(
        "--bs",
        default=128,
        type=int,
        help="mini-batch size",
    )
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="LR momentum")
    parser.add_argument(
        "--weight_decay", default=0.0001, type=float, help="LR weight decay"
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="maximum number of epochs",
    )
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument(
        "--out-dir-base", type=str, default="/tmp/", help="path to save outputs"
    )
    # Privacy
    parser.add_argument(
        "--sigma",
        type=float,
        # default=1.0,
        # default=0.0,  # No noise
        default=2.0 ** 0.5 / 4.0,  # --epsilon = 8
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "--post-process",
        type=str,
        default="mapwithprior",
        help="Post-processing scheme for noised labels "
        "(MinMax, SoftMax, MinProjection, MAP, MAPWithPrior, RandomizedResponse, MAPWithProjPrior)",  # --pp
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        default="Laplace",
        help="Noising mechanism (Laplace or Gaussian)",
    )

    # Attacks
    parser.add_argument(
        "--canary", type=int, default=0, help="Introduce canaries to dataset"
    )

    parser.add_argument("--seed", type=int, default=11337, help="Seed")
    parser.add_argument("--savefolder", type=str, default='tr', help="Folder to save pics and lists")  # todo
    parser.add_argument("--alpha", type=float, help="CE coefficient for SCE")  # TODO

    args = parser.parse_args()

    if args.alpha is None:  # TODO
        args.alpha = 0.7 if args.dataset.lower() == 'cifar100' else 0.2  # TODO

    privacy = LabelPrivacy(
        sigma=args.sigma,
        post_process=args.post_process,
        mechanism=args.mechanism,
        alpha=args.alpha,  # TODO
    )

    learning = Learning(
        lr=args.lr,
        batch_size=args.bs,
        epochs=args.epochs,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        random_aug=True,  # --随机增广
    )

    settings = Settings(
        dataset=args.dataset,
        arch=args.arch,
        privacy=privacy,
        learning=learning,
        canary=args.canary,
        gpu=args.gpu,
        out_dir_base=args.out_dir_base,
        seed=args.seed,
        savefolder=args.savefolder  # todo
    )

    # create the folder to save pics and lists
    save_path = os.path.join(settings.out_dir_base, settings.savefolder)  # todo
    os.makedirs(save_path, exist_ok=True)  # todo

    _, best_acc, _, epc_lst, loss_lst, acc_lst, tloss_lst, tacc_lst = main_worker(settings)
    print(f"Best Test Acc@1: {best_acc:.6f}")
    pic_plot(epc_lst, loss_lst, acc_lst, tloss_lst, tacc_lst, savepath=save_path)  # todo
    save_dict = {
        "best_acc": best_acc,
        "epochs": epc_lst,
        "train_loss": loss_lst,
        "train_acc": acc_lst,
        "test_loss": tloss_lst,
        "test_acc": tacc_lst,
    }  # todo
    json_path = os.path.join(save_path, 'lists.json')  # todo
    with open(json_path, 'w') as json_file:  # todo
        json.dump(save_dict, json_file)  # todo


if __name__ == "__main__":
    main()
