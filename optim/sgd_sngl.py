import torch
from torch.optim.optimizer import Optimizer

class OneStepSGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0):
        defaults = dict(lr=lr, momentum=momentum)
        super(OneStepSGD, self).__init__(params, defaults)

    @torch.no_grad() 
    def step(self):
        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']

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

                # Update: param = param - lr * grad
                param.add_(grad, alpha=-lr) 