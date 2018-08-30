import torch
from .optimizer import Optimizer, required
from .sgd import SGD


def _zeros_like(p):
    if isinstance(p, torch._C.XLATensor):
        return torch._C.XLATensor(torch.zeros(p.size()))
    return torch.zeros_like(p.data)

def _apply_weight_decay(d_p_multi, p_multi, weight_decay):
    assert len(d_p_multi) == len(p_multi)
    torch._C._xla_multi_mul_add(1, d_p_multi, weight_decay, p_multi)

def _apply_momentum(buf_multi, d_p_multi, is_first, momentum, nesterov, dampening):
    assert len(d_p_multi) == len(buf_multi) == len(is_first)
    assert len(set(is_first)) == 1
    if is_first[0]:
        torch._C._xla_multi_mul_add(momentum, buf_multi, 1, d_p_multi)
    else:
        torch._C._xla_multi_mul_add(momentum, buf_multi, 1 - dampening, d_p_multi)

def _apply_lr(p_multi, d_p_multi, lr):
    assert len(d_p_multi) == len(p_multi)
    torch._C._xla_multi_mul_add(1, p_multi, -lr, d_p_multi)

def _apply_nesterov(d_p_multi, buf_multi, momentum):
    assert len(d_p_multi) == len(buf_multi)
    torch._C._xla_multi_mul_add(1, d_p_multi, momentum, buf_multi)

class XlaSGD(SGD):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(XlaSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    grads.append(p.grad)
        torch._C._xla_multi_zero(grads)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            d_p_multi = []
            p_multi = []
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p_multi.append(p.grad.data)
                p_multi.append(p.data)

            if weight_decay != 0:
                _apply_weight_decay(d_p_multi, p_multi, weight_decay)

            if momentum != 0:
                is_first = []
                buf_multi = []
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = _zeros_like(p)
                        buf_multi.append(buf)
                        is_first.append(True)
                    else:
                        buf = param_state['momentum_buffer']
                        buf_multi.append(buf)
                        is_first.append(False)
                _apply_momentum(buf_multi, d_p_multi, is_first, momentum, nesterov, dampening)
                if nesterov:
                    _apply_nesterov(d_p_multi, buf_multi, momentum)
                else:
                    for i in range(0, len(d_p_multi)):
                        d_p_multi[i] = buf_multi[i]

            _apply_lr(p_multi, d_p_multi, group['lr'])

        return loss
