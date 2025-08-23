import torch
from torch.optim.optimizer import Optimizer

class FISTA(Optimizer):
    def __init__(self, params, lr=1e-3, ministeps = 5):
        defaults = dict(lr=lr, ministeps=ministeps)
        super(FISTA, self).__init__(params, defaults)

        for group in self.param_groups:
            group['y'] = [p.clone().detach() for p in group['params']]
#             group['steps'] = 1

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
#             k = group['k']
            ministeps = group['ministeps']
            
            # Loop over ministeps
            for k in range(ministeps):
                
                # Reset, k=0, update p
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue

                    x_k = p.data
                    y_k = group['y'][i].data

                    # y_k grad
                    grad_y = .data

                    # x_{k+1} = y_k - t * grad(f(y_k))
                    x_next = y_k - lr * grad_y
                    # Question: Should we update multi-step here, inside the loop

                    #  y_{k+1} = x_{k+1} + (k / (k + 3)) * (x_{k+1} - x_k)
                    momentum_coeff = k / (k + 3)
                    y_next = x_next + momentum_coeff * (x_next - x_k)

                    # Update
                    p.data = x_next
                    group['y'][i].data = y_next
            
                group['k'] += 1
