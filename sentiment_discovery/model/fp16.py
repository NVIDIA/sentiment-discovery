from .dynamic_loss_scaler import DynamicLossScaler
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

def scale_gradient(module, grad_in, grad_out):
    scaler = module.loss_scaler
    return tuple(scaler.loss_scale * g for g in grad_in)
def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val` is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn =  [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn

def fp32_to_fp16(val):
    """Convert fp32 `val` to fp16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (torch.FloatTensor, torch.cuda.FloatTensor)):
            val = val.half()
        return val
    return conversion_helper(val, half_conversion)

def fp16_to_fp32(val):
    """Convert fp16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (torch.HalfTensor, torch.cuda.HalfTensor)):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)

class FP16_Optimizer(object):
    """Wrapper for PyTorch optimizers that enables
       FP16 training with FP32 weights"""
    def __init__(self, optimizer, module, *args, **kwargs):
        if not torch.cuda.is_available:
            raise SystemError('Cannot use fp16 without CUDA')

        self.module = module
        self.optimizer = optimizer
        self.state = optimizer.state
        self.param_groups = optimizer.param_groups

        self.fp16_params = []
        self.fp32_params = []
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                self.fp16_params.append(param)
                fp32_param = param
                if isinstance(fp32_param.data, torch.cuda.HalfTensor):
                    fp32_param = param.clone().type(torch.cuda.FloatTensor).detach()
                fp32_param.requires_grad = param.requires_grad
                self.fp32_params.append(fp32_param)
                group['params'][i] = fp32_param

        self.loss_scaler = DynamicLossScaler()
        self.module.register_backward_hook(scale_gradient)
        # self.optimizer = optim_class(module.parameters(), *args, **kwargs)

    def zero_grad(self):
        # clear fp32 parameter grads
        self.optimizer.zero_grad()
        # clear fp16 parameter grads
        for p in self.fp16_params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def update_grad_(self):
        """Checks for overflow, removes loss scaling factor,
           makes gradients fp32"""
        # Check for overflow
        # ----------------------------------------------
        params = list(self.fp16_params)
        has_overflow = DynamicLossScaler.has_overflow(params)

        # Copy grads to fp32 and scale
        # ----------------------------------------------------------------
        for i, param in enumerate(params):
            fp32_param = self.fp32_params[i]
            fp32_param._grad = param.grad.clone().type_as(fp32_param).detach()
            fp32_param._grad.mul_(1./self.loss_scaler.loss_scale)

        self.loss_scaler.update_scale(has_overflow)
        return has_overflow

    def update_weights_(self):
        for i, param in enumerate(self.fp16_params):
            fp32_param = self.fp32_params[i]
            param.data.copy_(fp32_param.data)

    def step(self):
        """1) Copies the fp16 grad to fp32
           2) If no overflow updates weights in fp32 with normal optimizer
           3) Copies weights back to fp16
        """
        has_overflow = self.update_grad_()  # Copy fp16 grad to fp32
        if has_overflow:
            scale = self.loss_scaler.loss_scale
            print("OVERFLOW! Not taking step. loss scale: {}".format(scale))
            return

        self.optimizer.step()
        self.update_weights_()  # Copy fp32 accumulated weights to fp16
        return
