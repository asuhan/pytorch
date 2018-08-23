import torch
from .optimizer import Optimizer, required
from .sgd import SGD


def _zeros_like(p):
    if isinstance(p, torch._C.XLATensor):
        return torch._C.XLATensor(torch.zeros_like(p.to_tensor()))
    return torch.zeros_like(p.data)

def _apply_weight_decay(d_p_multi, p_multi, weight_decay):
    assert len(d_p_multi) == len(p_multi)
    for i in range(0, len(d_p_multi)):
        d_p_multi[i].add_(weight_decay, p_multi[i].data)

def _apply_momentum(buf_multi, d_p_multi, is_first, momentum, nesterov, dampening):
    assert len(d_p_multi) == len(buf_multi) == len(is_first)
    for i in range(0, len(d_p_multi)):
        if is_first[i]:
            buf_multi[i].mul_(momentum).add_(d_p_multi[i])
        else:
            buf_multi[i].mul_(momentum).add_(1 - dampening, d_p_multi[i])

def _apply_lr(p_multi, d_p_multi, lr):
    assert len(d_p_multi) == len(p_multi)
    for i in range(0, len(d_p_multi)):
        p_multi[i].data.add_(-lr, d_p_multi[i])

def _apply_nesterov(d_p_multi, buf_multi, momentum):
    assert len(d_p_multi) == len(buf_multi)
    for i in range(0, len(d_p_multi)):
        d_p_multi[i].add_(momentum, buf_multi[i])

class XlaSGD(SGD):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(XlaSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

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
