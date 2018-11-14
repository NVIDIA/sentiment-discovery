from torch.optim.lr_scheduler import _LRScheduler
import math

class LinearLR(_LRScheduler):
    """
    A scheduler for linear learning rate decay to 0 over a specified number of steps.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_iters (int): Period of learning rate decay. When last_iter==max_iters lr=max(min_lr,0)
        last_iter (int): The index of last iteration step. Default: -1
        min_lr (float): smallest allowed learning rate (acts as a clamp to prevent too small learning rates). Default: 1e-8
    Example:
        >>> # Assuming optimizer also uses lr = 0.0005 for all groups
        >>> scheduler = LinearLR(optimizer, max_iters=10, last_iter=-1, min_lr=1e-8)
        >>> for iter in range(10):
        >>>     train(...)
        >>>        scheduler.step()
        >>> validate(...)
    """
    def __init__(self, optimizer, max_iters, last_iter=-1, min_lr=1e-8):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.num_iters = last_iter
        self.min_lr = min_lr
        self.done = False
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_iter + 1)

    def get_lr(self):
        return [self.decay_func(base_lr) for base_lr in self.base_lrs]

    def decay_func(self, init_lr):
        new_lr = init_lr*((self.max_iters-self.num_iters)/self.max_iters)
        return max(new_lr, self.min_lr)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.num_iters + 1
        self.num_iters = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        return self.done

class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group['lr'] = lr

    def step(self, step_num=None):
        pass

class SlantedTriangularLR(_LRScheduler):
    """
    Implements the "slanted triangular learning rate schedule used for ULMFiT as a function of
    the number of training iterations" (arxiv.org/pdf/1801.06146.pdf)
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_ratio (float): ratio of minimum to maximum learning rate (32 in paper)
        max_val (float): highest learning rate (attained at peak of slanted triangle - 0.01 in paper)
        cut_frac (float): proportion of iterations during which learning rate is increasing (0.1 in paper)
        num_iters (int): total number of iterations expected (should be one epoch)
    """
    def __init__(self, optimizer, lr_ratio=100, max_val=6.25e-5, cut_frac=0.002, num_iters=1000):
        self.optimizer = optimizer
        self.min_val = max_val / lr_ratio
        self.max_val = max_val
        self.peak_iter = num_iters * cut_frac
        self.end_triangle_iter = num_iters
        self.num_iters = 0
        self.lr_func = self.create_lr_func()


        for group in optimizer.param_groups:
            group['weight_decay'] = 0.01
            if 'name' in group.keys() and group['name'] == 'low':
                group['lr'] = self.min_val / 2.6
            else:
                group['lr'] = self.min_val

    def create_lr_func(self):
        lr_range = self.max_val - self.min_val

        up_slope = lr_range / self.peak_iter
        up_intercept = self.min_val
        down_slope = -lr_range / (self.end_triangle_iter - self.peak_iter)
        down_intercept = -down_slope * self.peak_iter + self.max_val

        def lr_func():
            if self.num_iters <= self.peak_iter:
                return up_slope * self.num_iters + up_intercept
            else:
                return down_slope * self.num_iters + down_intercept

        return lr_func

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.lr_func()
        for group in self.optimizer.param_groups:
            if 'name' in group.keys() and group['name'] == 'low':
                group['lr'] = new_lr / 2.6
            else:
                group['lr'] = new_lr


class CosineAnnealingLR(_LRScheduler):
    """Anneals the learning rate from start to zero along a cosine curve."""

    def __init__(self, optimizer, start_lr, warmup_iter, num_iters):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.num_iters = 0
        self.end_iter = num_iters

    def get_lr(self):
        # https://openreview.net/pdf?id=BJYwwY9ll pg. 4
        if self.num_iters <= self.warmup_iter:
            return float(self.start_lr) * self.num_iters / self.warmup_iter
        else:
            return self.start_lr / 2.0 * (math.cos(math.pi * (self.num_iters - self.warmup_iter) / self.end_iter) + 1)

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

class AnnealingLR(_LRScheduler):
    """Anneals the learning rate from start to zero along a cosine curve."""

    DECAY_STYLES = ['linear', 'cosine', 'exponential', 'constant', 'None']

    def __init__(self, optimizer, start_lr, warmup_iter, num_iters, decay_style=None):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.num_iters = 0
        self.end_iter = num_iters
        self.decay_style = decay_style.lower() if isinstance(decay_style, str) else None
        print('decaying', decay_style)

    def get_lr(self):
        # https://openreview.net/pdf?id=BJYwwY9ll pg. 4
        if self.num_iters <= self.warmup_iter:
            return float(self.start_lr) * self.num_iters / self.warmup_iter
        else:
            if self.decay_style == self.DECAY_STYLES[0]:
                return self.start_lr*((self.end_iter-(self.num_iters-self.warmup_iter))/self.end_iter)
            elif self.decay_style == self.DECAY_STYLES[1]:
                return self.start_lr / 2.0 * (math.cos(math.pi * (self.num_iters - self.warmup_iter) / self.end_iter) + 1)
            elif self.decay_style == self.DECAY_STYLES[2]:
                #TODO: implement exponential decay
                return self.start_lr
            else:
                return self.start_lr

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr


class DiscriminativeFinetuneWrapper(object):
    def __init__(self, optimizer, layer_lambda, lr_ratio=0.3):
        pass


class WarmupLR:
    def __init__(self, optimizer, max_iters, last_iter=-1):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.num_iters = last_iter
        self.step(last_iter + 1)

    def scale_lr(self, lr):
        return (lr * (self.num_iters+1) / self.max_iters)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.num_iters + 1
        self.num_iters = epoch
        if self.num_iters >= self.max_iters:
            return
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            param_group['lr'] = self.scale_lr(lr)
