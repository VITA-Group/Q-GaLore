import pdb
import math
import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def _quantize_tensor_int8(w, q_group_size=-1, n_bit=8):

    org_w_shape = w.shape
    if q_group_size > 0:
        assert w.nelement() % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w = w.reshape(org_w_shape).to(torch.uint8)

    return w, scales, zeros


class W8Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)

        def forward_w_float_weight(weight, x, bias):
            float_weight = weight.to(x.dtype).reshape(-1, weight.group_size)   
            (float_weight.sub_(weight.zeros)).mul_(weight.scales)
            float_weight = float_weight.reshape(weight.shape)

            if bias is not None:
                return x @ float_weight.t() + bias
            else:
                return x @ float_weight.t()

        output = forward_w_float_weight(weight, x, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors

        def backward_w_float_weight(weight, grad_output):
            float_weight = weight.to(x.dtype).reshape(-1, weight.group_size)   
            (float_weight.sub_(weight.zeros)).mul_(weight.scales)
            float_weight = float_weight.reshape(weight.shape)
            grad_input = grad_output @ float_weight
            return grad_input

        grad_input = backward_w_float_weight(weight, grad_output)

        if bias is not None:
            out_features = bias.shape[0]
            grad_bias = grad_output.reshape(-1, out_features).sum(0)
        else:
            grad_bias = None

        out_features, in_features = weight.shape
        # gradient accumulation
        if not hasattr(weight, 'float_grad'):
            weight.__setattr__('float_grad', None)

        if weight.float_grad is not None:
            weight.float_grad += grad_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features) 
        else:
            weight.float_grad = grad_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features)

        if hasattr(weight, 'backward_hook'):
            weight.backward_hook(weight)

        return grad_input, None, grad_bias


class QGaLoreLinear(nn.Module):
    def __init__(self, weight, bias, device=None, dtype=None, num_bits=8, group_size=256, stochastic_round=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        int8_weight, scales, zeros = _quantize_tensor_int8(weight.data, q_group_size=group_size)
        torch.cuda.empty_cache()

        self.weight = Parameter(int8_weight, requires_grad=False).to(device) # Only Tensors of floating point and complex dtype can require gradients, using float_gradient to store the gradient
        
        self.weight.__setattr__('scales', scales.to(device))
        self.weight.__setattr__('zeros', zeros.to(device))
        self.weight.__setattr__('group_size', group_size)
        self.weight.__setattr__('saved_data_dtype', int8_weight.dtype)
        self.weight.__setattr__('stochastic_round', stochastic_round)

        if not num_bits == 8:
            raise NotImplementedError

        self.bias = Parameter(bias, requires_grad=True).to(device) if bias is not None else None

    def forward(self, input: Tensor) -> Tensor:
        output = W8Linear.apply(input, self.weight, self.bias)
        return output


def prepare_model_for_int8_training(model, args, target_module):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prepare_model_for_int8_training(module, args, target_module)

        if isinstance(module, nn.Linear):
            if not name in target_module: continue

            bias_data = module.bias.data if module.bias is not None else None
            new_layers = QGaLoreLinear(module.weight, bias_data, num_bits=args.weight_bits, group_size=args.weight_group_size, stochastic_round=args.stochastic_round)
            model._modules[name] = new_layers

    return model


if __name__ == '__main__':
    GROUP_SIZE=256
    print('*** Memory checking for a single linear layer ***')
    fp16_linear1 = nn.Linear(4096, 4096, bias=False).to('cuda:0').to(torch.bfloat16)
    print('after initial weight for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    mem_weight_float = torch.cuda.memory_allocated('cuda:0')//1024/1024
    x = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device='cuda:0', requires_grad=True)
    print('after initial input for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    start = time.time()
    output = fp16_linear1(x)
    print('after forward for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    output.sum().backward()
    end = time.time()
    print('after backward for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    print('Time for FW+BW = {:.2f} s'.format(end-start))
    print('------------------------------------')

    int8_linear1 = QGaLoreLinear(fp16_linear1.weight, None, device='cuda:1', num_bits=8, group_size=GROUP_SIZE)
    print('after initial weight for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    mem_weight_int = torch.cuda.memory_allocated('cuda:1')//1024/1024
    x1 = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device='cuda:1', requires_grad=True)
    print('after initial input for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    start = time.time()
    output_int8 = int8_linear1(x1)
    print('after forward for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    output_int8.sum().backward()
    end = time.time()
    print('after backward for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    print('Time for FW+BW = {:.2f} s'.format(end-start))
    print('------------------------------------')

    print('Memory saving for weight: {:.2f} MB, ratio: {:.2f}%'.format(mem_weight_float - mem_weight_int, mem_weight_int / mem_weight_float * 100))
    print('------------------------------------')

