import torch
from torch.optim.optimizer import Optimizer

def soft_thresholding(x, threshold):
    """
    prox_{lambda * t}(x) = sign(x) * max(0, |x| - lambda * t)
    """
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

class FISTA(Optimizer):
    """
    FISTA: min_x f(x) + g(x),  f(x) smooth convex, g(x) nonsmooth convex.
    Args:
        params: parameters to optimize.
        lr: The learning rate, = 1/L, where L: L-Lipschitz constant.
        prox_op: proximal operator for g(x).
    """
    def __init__(self, params, lr=1e-3, prox_op=None):
        defaults = dict(lr=lr)
        super(FISTA, self).__init__(params, defaults)
        self.prox_op = prox_op

        for group in self.param_groups:
            group['y'] = [p.clone().detach() for p in group['params']]
            group['k'] = 1

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            k = group['k']
            
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # Get the current parameter 'x_k' and the momentum term 'y_k'
                x_k = p.data
                y_k = group['y'][i].data

                # Gradient at the momentum point 'y_k'.
                grad_y = p.grad.data
                
                # Proximal gradient step: x_{k+1} = prox_tg(y_k - t*grad(f(y_k)))
                x_next = self.prox_op(y_k - lr * grad_y, lr)

                # Momentum-based acceleration step: y_{k+1} = x_{k+1} + (k / (k + 3)) * (x_{k+1} - x_k)
                momentum_coeff = k / (k + 3)
                y_next = x_next + momentum_coeff * (x_next - x_k)

                # Update
                p.data = x_next
                group['y'][i].data = y_next
            
            group['k'] += 1
