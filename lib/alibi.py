#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Optional, List

import torch
import torch.nn.functional as f
import warnings
from opacus.privacy_analysis import compute_rdp, get_privacy_spent
from scipy.optimize import Bounds, LinearConstraint, minimize


EPS = 1e-10
ROOT2 = 2.0 ** 0.5
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class RandomizedLabelPrivacy:
    def __init__(
        self,
        sigma: float,
        delta: float = 1e-10,
        mechanism: str = "Laplace",
        device: Any = None,
        seed: Optional[int] = None,
    ):
        r"""
        A privacy engine for randomizing labels.

        Arguments
            mechanism: type of the mechansim, for now either normal or laplacian
        """
        self.sigma = sigma
        self.delta = delta
        assert mechanism.lower() in ("gaussian", "laplace")
        self.isNormal = mechanism.lower() == "gaussian"  # else is laplace
        self.seed = (
            seed if seed is not None else (torch.randint(0, 255, (1,)).item())
        )  # this is not secure but ok for experiments
        self.device = device
        self.randomizer = torch.Generator(device) if self.sigma > 0 else None  # 使用设置的随机数种子得到一个随机数生成器
        self.reset_randomizer()  # 重置随机数生成器，使伪随机数从头开始
        self.step: int = 0
        self.eps: float = float("inf")  # 初始化为正无穷大的浮点数
        self.alphas: List[float] = [i / 10.0 for i in range(11, 1000)]
        self.alpha = float("inf")
        self.train()  # 将self._train设置为True

    def train(self):
        # 非__init__中定义的实例属性也可以在别的方法中直接访问，但是要确保该属性已经初始化
        # 如果定义的是局部变量(无self.)而非实例属性，就不能在别的方法中直接访问
        self._train = True

    def eval(self):
        self._train = False

    def reset_randomizer(self):
        if self.randomizer is not None:
            self.randomizer.manual_seed(self.seed)

    def increase_budget(self, step: int = 1):
        if self.sigma <= 0 or step <= 0:
            return  # 实际上会return None

        self.step += step
        if self.isNormal:
            rdps = compute_rdp(1.0, self.sigma / ROOT2, self.step, self.alphas)
            self.eps, self.alpha = get_privacy_spent(self.alphas, rdps, self.delta)
        else:
            if self.step > 1:
                warnings.warn(
                    "It is not optimal to use multiple steps with Laplace mechanism"
                )
                self.eps *= self.step
            else:
                self.eps = 2 * ROOT2 / self.sigma

    def noise(self, shape):
        if not self._train or self.randomizer is None:  # 不在训练状态或没有随机数生成器
            return None
        noise = torch.zeros(shape, device=self.device)
        if self.isNormal:
            noise.normal_(0, self.sigma, generator=self.randomizer)  # 标准差为sigma的正态分布噪声
        else:  # is Laplace
            # Exp(lambda) - Exp(lambda) ~ Lap(1/lambda)
            # 得到的noise服从Lap(sigma/ROOT2)，标准差为sigma，对应的epsilon为2*ROOT2/sigma
            tmp = noise.clone()
            noise.exponential_(ROOT2 / self.sigma, generator=self.randomizer)
            tmp.exponential_(ROOT2 / self.sigma, generator=self.randomizer)
            noise = noise - tmp
        return noise

    @property
    def privacy(self):
        return self.eps, self.alpha


class Ohm:
    def __init__(
        self, privacy_engine: RandomizedLabelPrivacy, alpha: float, post_process: str = "mapwithprior"  # TODO
    ):
        """
        One Hot Mixer

        creates a noised one-hot version of the targets and returns
        the cross entropy loss w.r.t. the noised targets.

        Args:
            sigma: Normal distribution standard deviation to sample noise from,
                if 0 no noising happens and this becomes strictly equivalent
                to the normal cross entropy loss.
            post_process: mode for converting the noised output to proper
                probabilities, current supported modes are:
                MinMax, SoftMax, MinProjection, MAP, RandomizedResponse
                see `post_process` for more details.
        """
        self.mode = post_process.lower()
        assert self.mode in (
            "minmax",
            "softmax",
            "minprojection",
            "map",
            "mapwithprior",
            "randomizedresponse",
            "mapwithprojprior",  # --pp
        ), self.mode
        self.alpha = alpha  # TODO
        self.privacy_engine = privacy_engine
        self.device = privacy_engine.device
        self.label_change = 0.0  # tracks the probability of label changes
        self.beta = 0.99  # is used to update label_change

    def post_process(self, in_vec: torch.Tensor, output: Optional[torch.Tensor] = None):
        """
        convert a given vector to a probability mass vector.

        Has five modes for now:
            MinMax: in -> (in - min(in)) / (max(in) - min(in)) -> in / sum(in)
            SoftMax: in -> e^in / sum(e^in)
            MinpProjection: returns closes point in the surface that defines all
                possible probability mass functions for a given set
                of classes
            MAP: returns the probability mass function that solve the MAP for a
                cross entroy loss.
            RandomizedResponse: sets the largest value to 1 and the rest to zero,
                this way either the original label is kept or some random label is
                assigned so this is equivalent to randomized response.
        """
        if self.mode == "minmax":
            return self._minmax(in_vec)
        elif self.mode == "minprojection":
            return self._duchi_projection(in_vec)
        elif self.mode == "softmax":
            return self._map_normal(in_vec)
        elif "map" in self.mode:
            assert not ("prior" in self.mode and output is None)
            prior = (
                # 默认就是在最后一个维度做softmax，也就是其他维度索引不变，依次遍历最后一个维度索引值做softmax
                # 如果是2维tensor，就是对行做softmax；和torch.max()的dim参数类似
                # 将模型(经过softmax)的output作为先验分布
                # --prior为什么需要detach? 是否是将prior作为常量，避免prior因output在链式法则时产生额外支路?
                # f.softmax(output.detach(), dim=-1)  # --detach之后会对计算图产生什么影响?
                self.get_prior(output, in_vec)  # --pp
                if "prior" in self.mode and output is not None
                else None
            )
            return (
                self._map_normal(in_vec, self.privacy_engine.sigma, prior)
                if self.privacy_engine.isNormal
                else self._map_laplace(in_vec, self.privacy_engine.sigma / ROOT2, prior)
            )
        else:  # self.mode == "randomizedresponse"
            return self._select_max(in_vec)

    def soft_target(self, output: torch.Tensor, target: torch.Tensor):
        """
        convert to one had and noise.

        output is just used to create the vector of same size and on the
        same device.
        """
        if len(target.shape) == 1:  # targets are not soft labels；标签还是数值，target.shape == torch.size([batch_size])
            target, is_noised = self._create_soft_target(output, target)
        else:
            is_noised = True  # TODO we assume that a soft target is always noised
        if is_noised:
            target = self.post_process(target, output)

        # --label smoothing
        # if self.privacy_engine._train and epoch >= 30:
        #     smooth_rate = -0.97
        #     smooth_term = torch.ones_like(target).to(device)
        #     target = (1. - smooth_rate) * target + smooth_rate / 10 * smooth_term

        return target

    # def __call__(self, output: torch.Tensor, target: torch.Tensor):
    #     """
    #     calculates loss.
    #
    #     Args:
    #         orutput: output of the netwok
    #         target: the labels (same dim 0 as the output)
    #     """
    #     pmf = self.soft_target(output, target)  # 得到后处理之后的soft label
    #
    #     # calculate prob. of label change
    #     if len(target.shape) == 1:  # targets were not soft labels; targets是0-9的数值型label
    #         label = target.view(-1)  # 拉成1维tensor
    #         maxidx = torch.argmax(pmf, dim=1).view(-1)  # 将soft label转化为概率最大的类别并拉平
    #         lc = 1 - float((label == maxidx).sum().item()) / label.numel()  # tensor.numel()返回元素总数；lc是label改变比例
    #         # label_change是lc的指数加权平均；指数加权平均是近似求平均的方法，受近期的数据影响较大，且在内存占用上有优势
    #         self.label_change = self.label_change * self.beta + (1 - self.beta) * lc
    #
    #     # 计算CrossEntropyLoss; 模型输出output不需要softmax层
    #     output = f.softmax(output, dim=-1)
    #     output = output + EPS  # 防止对数运行出现负无穷大
    #     output = -torch.log(output)
    #     return (pmf * output).sum(dim=-1).mean()  # mean是对batch求均值

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        """SCE Loss"""
        pmf = self.soft_target(output, target)  # 得到后处理之后的soft label

        # calculate prob. of label change
        if len(target.shape) == 1:  # targets were not soft labels; targets是0-9的数值型label
            label = target.view(-1)  # 拉成1维tensor
            maxidx = torch.argmax(pmf, dim=1).view(-1)  # 将soft label转化为概率最大的类别并拉平
            lc = 1 - float((label == maxidx).sum().item()) / label.numel()  # tensor.numel()返回元素总数；lc是label改变比例
            # label_change是lc的指数加权平均；指数加权平均是近似求平均的方法，受近期的数据影响较大，且在内存占用上有优势
            self.label_change = self.label_change * self.beta + (1 - self.beta) * lc

        # --SCE = alpha * ce + beta * rce
        alpha, beta = self.alpha, 1.0 - self.alpha  # TODO
        # alpha, beta = 0.7, 0.3  # for CIFAR-100
        pred = f.softmax(output, dim=-1)
        output = pred + EPS
        output = -torch.log(output)
        ce = (pmf * output).sum(dim=-1).mean()

        pred = torch.clamp(pred, min=1e-7, max=1.0)  # fixed
        pmf = torch.clamp(pmf, min=1e-4, max=1.0)  # A = 1e-4, you can also try 1e-6
        pmf = -torch.log(pmf)
        rce = (pred * pmf).sum(dim=-1).mean()
        return alpha * ce + beta * rce

    def get_prior(self, output: torch.Tensor, target: torch.Tensor):  # --pp
        """to get the according prior for map"""
        assert self.mode in ('mapwithprior', "mapwithprojprior", "map")
        if self.mode == "mapwithprior":
            prior = f.softmax(output.detach(), dim=-1)
        else:  # self.mode == "map"
            prior = None
        return prior

    def _create_soft_target(self, output: torch.Tensor, target: torch.Tensor):
        """
        Creates a soft target；需要时给one-hot标签加噪声

        onehot representation if not noised and a noisy tensor if noised.
        returns a tuple of the soft target and whether it was noised
        """
        # print('output.shape:', output.shape)  # torch.size([128, 10]), batch_size*class_num
        # print('target.shape:', target.shape)  # torch.size([128]), batch_size
        onehot = torch.zeros_like(output)
        target = target.type(torch.int64)  # 返回一个新的torch.int64数据类型的Tensor并赋值给target
        onehot.scatter_(1, target.view(-1, 1), 1.0)  # inplace操作为对应的one-hot向量
        rand = self.privacy_engine.noise(onehot.shape)
        noised = onehot if rand is None else onehot + rand
        return noised, rand is not None

    def _map_normal(
        self,
        in_vec: torch.Tensor,
        noise_sigma: float = 0,
        prior: Optional[torch.Tensor] = None,
    ):
        """
        The function is calculating the posterior {P(label=i | X) | i in C}
        through P(X | label = i) * P(label = i)

        For Gaussian mechanism boils down to be

        1 / Normalizer * e ^ (x_i / sigma) * P(label = i)
               =
        1 / Normalizer * e ^ (x_i / sigma + ln(P(label= i)))

        posterior P(y = c | output, σ) = SoftMax(output_c / (σ ** 2) + ln(p(y = c))),
        where p(y = c) is the prior.
        """
        in_vec = in_vec / (1.0 if noise_sigma <= 0 else noise_sigma)  # --不应该是noise_sigma ** 2吗?
        in_vec = in_vec + (
            0.0 if prior is None or noise_sigma <= EPS else torch.log(prior)
        )
        return f.softmax(in_vec, dim=-1)

    def _map_laplace(
        self,
        in_vec: torch.Tensor,
        noise_b: float = 0,
        prior: Optional[torch.Tensor] = None,
    ):
        """
        The function is calculating the posterior {P(label=i | X) | i in C}
        through P(X | label = i) * P(label = i)

        For Laplace mechanism boils down to be

        1 / Normalizer * e ^ (- sum_j |x_j - (j == i)| / b) * P(label = i)
               =
        1 / Normalizer * e ^ (- sum_j |x_j - (j == i)| / b + ln(P(label= i)))

        p(y = c | output, λ) = SoftMax(f(output_c) / λ + ln(p(y = c))),
        where f(output_c) = - sum_k |output_k - [c = k]|
        """
        # 这里的张量运算很熟练
        n, c = in_vec.shape  # (batch_size, class_num)
        in_vec = in_vec.repeat(1, c).view(-1, c, c)  # .repeat() -> n * c^2; .view() -> n * c * c
        in_vec = in_vec - torch.eye(c).to(device=in_vec.device).repeat(n, 1, 1)  # 得到output_k - [c = k]
        in_vec = -1 * in_vec.abs().sum(dim=-1)  # sum的维度会"坍缩"; - sum_k |output_k - [c = k]|; n * c
        in_vec = in_vec / (1.0 if noise_b <= 0 else noise_b)  # f(output_c) / λ
        in_vec = in_vec + (0.0 if prior is None or noise_b <= EPS else torch.log(prior))  # f(output_c)/λ + ln(p(y = c))
        return f.softmax(in_vec, dim=-1)

    def _minmax(self, in_vec: torch.Tensor):
        """
        Converts `in_vec` in to a probability mass function.

        does this:
        in = (in - min(in)) / (max(in) - min(in))
        in = in / sum(in)
        """
        m = in_vec.min(-1)[0].reshape(-1, 1)  # .min(-1)返回namedtuple(values, indices)，两者都是tensor，.min(-1)[0]即values
        M = in_vec.max(-1)[0].reshape(-1, 1)
        in_vec = (in_vec - m) / M  # 分母本应是M - m，但下一步有.sum(-1)运算，所以系数无影响
        return in_vec / in_vec.sum(-1).reshape(-1, 1)

    def _select_max(self, in_vec: torch.Tensor):
        """
        Implements randomized response.

        With prob x keeps the class with prob 1 - x assings random other class.

        用最大概率索引作为预测类别
        """
        maxidx = torch.argmax(in_vec, dim=1).view(-1, 1).type(torch.int64)
        onehot = torch.zeros_like(in_vec)
        onehot.scatter_(1, maxidx, 1.0)  # 将预测类别转化为one-hot向量
        return onehot

    def _minprojection_with_optimizer(self, in_vec: torch.Tensor):
        """
        Converts `in_vec` in to a probability mass function.

        minimizes the distance of in_vec to the plain `∑ x = 1`
        with constraints being `0 <= x <= 1`
        """
        n = in_vec.shape[-1]  # num classes
        bounds = Bounds([0] * n, [1] * n)  # all values in [0, 1]
        linear_constraint = LinearConstraint([[1] * n], [1], [1])  # values sum to 1
        x0 = [1 / n] * n  # initial point in the middle of the plain

        results = []

        class optim_wrapper:
            """
            wrapps optimiztion process for a single point
            """

            def __init__(self, p):
                self.p = p

            def func(self, x):
                return ((x - self.p) * (x - self.p)).sum()

            def jac(self, x):
                return 2 * (x - self.p)

            def hess(self, x):
                return 2 * torch.eye(n)

            def __call__(self):
                res = minimize(
                    self.func,
                    x0,
                    method="trust-constr",
                    jac=self.jac,
                    hess=self.hess,
                    constraints=linear_constraint,
                    bounds=bounds,
                )
                return res.x

        results = [optim_wrapper(x)() for x in in_vec.tolist()]
        return torch.Tensor(results).to(in_vec.device)

    def _minprojection_fast(self, in_vec: torch.Tensor):
        """
        Provides a much faster way to calculate  `_minprojection_with_optimizer`.

        This is our proposed method for calculating the min projection on a
        probability surface, i.e. ∑ x = 1 , 0 <= x <= 1

        The method completely bypasses an optimizer and instead uses recursion.
        The number of recursions in the stack is bounded by the number of classes.

        The method works as follows:
        given `p` is the input (it is a `k` element vector).

        1. Update `p` to its projection on `∑ x = 1`.
        2. Set negative elements of `p` to 0.
        3. Tracking the indices, reproject non-negative elemets of `p` to a `∑ x = 1` in the new space.
        4. Repeat from 2 until there are not negative elements.

        This algorithm is > 200 times faster on an average server
        """

        def fast_minimizer(vec):
            n = vec.shape[-1]
            point = vec + (1 - vec.sum().item()) / n
            if (point < 0).sum() > 0:
                idx = point >= 0
                point[point < 0] = 0
                p = point[idx]
                result = fast_minimizer(p)
                point[idx] = result
            return point

        results = [fast_minimizer(torch.Tensor(x)) for x in in_vec.tolist()]
        return torch.stack(results, 0).to(in_vec.device)

    def _minprojection_faster(self, in_vec: torch.Tensor):
        """
        This is a yet faster version of ``_minprojection_fast``.

        The code vectorizes the operation for a batch. Also converts
        recursion to a loop. This ads another >60 times speed up on
        top of ``_minprojection_fast``coming to about 4 orders of magnitude
        speed-up!
        """
        projection = in_vec.clone()
        n = projection.shape[-1]
        idx = torch.ones_like(projection) < 0  # all false
        for _ in range(n):
            projection[idx] = 0
            projection += (
                ((1 - projection.sum(-1)) / (n - idx.sum(-1))).view(-1, 1).repeat(1, n)
            )
            projection[idx] = 0
            idx_new = projection < 0
            if idx_new.sum().item() == 0:
                break
            else:
                idx = torch.logical_or(idx_new, idx)
        return projection

    def _duchi_projection(self, in_vec: torch.Tensor):
        """
        This implementation implements the procedure for projection
        onto a probabilistic simplex following the same notations as
        Algorithm 4 from our paper.
        """
        o = in_vec.clone().cpu()  # --为什么要转移到cpu?
        # o = in_vec.clone()
        B, C = o.shape  # batch_size, class_num
        s, _ = torch.sort(o, dim=1, descending=True)  # 返回namedtuple(values, indices)
        cumsum = torch.cumsum(s, dim=1)  # 累计求和; running total
        indices = torch.arange(1, C + 1)
        k, _ = torch.max(
            (s * indices > (cumsum - 1)) * indices, dim=1
        )  # hack to get the last argmax; 得到满足条件的最大序号值
        # 2维tensor用2个1维tensor索引，得到1维tensor; 实质上相当于用两个索引tensor的对应值依次取出元素组成tensor
        u = (cumsum[torch.arange(B), k - 1] - 1) / k  # 上述k对应的running total - 1再除以k
        proj_o = (o - u.unsqueeze(1)).clamp(min=0)  # .unsqueeze(1)添加dim=1；.clamp(min=0)将最小值限制为0
        return proj_o.to(device)  # --添加.to(device)


class NoisedCIFAR(torch.utils.data.Dataset):
    def __init__(
        self,
        cifar: torch.utils.data.Dataset,
        num_classes: int,
        randomized_label_privacy: RandomizedLabelPrivacy,
    ):
        self.cifar = cifar
        self.rlp = randomized_label_privacy
        targets = cifar.targets
        self.soft_targets = [self._noise(t, num_classes) for t in targets]  # 依次给数值label加噪声转化为soft label
        self.rlp.increase_budget()  # increase budget; to get the dp parameters
        # calculate probability of label change
        num_label_changes = sum(
            label != torch.argmax(soft_target).item()
            for label, soft_target in zip(targets, self.soft_targets)
        )
        self.label_change = num_label_changes / len(targets)
        # print(self.label_change)  # --report the label change rate

    def _noise(self, label, n):
        onehot = torch.zeros(n).to()  # .to()会将tensor放在默认设备即cpu上
        onehot[label] = 1
        rand = self.rlp.noise((n,))
        return onehot if rand is None else onehot + rand

    def __len__(self):
        return self.cifar.__len__()

    def __getitem__(self, index):
        image, label = self.cifar.__getitem__(index)
        return image, self.soft_targets[index], label
