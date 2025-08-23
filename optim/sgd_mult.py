import torch
from torch.optim.optimizer import Optimizer

class ManyStepSGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, n_step=2):
        defaults = dict(lr=lr, momentum=momentum, n_step=n_step)
        super(ManyStepSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, n_step = group['lr'], group['momentum'], group['n_step']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                    
                grad = param.grad 

                if momentum != 0:
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = torch.clone(grad).detach()
                        param_state['momentum_buffer'] = buf
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)

                for _ in range(n_step):
                    # Update param = param - lr * grad
                    param.add_(grad, alpha=-lr)
                    # Update the grad
                    grad = param.grad