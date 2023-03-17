import torch
import torch.nn as nn
#from ..registry import LOSSES

#@LOSSES.register_module
class EntropyLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(EntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, x, **kwargs):
        ''' Colume-wise entropy loss.
        '''
        # colume
        if len(x.shape) == 4:
            n, num_ccpoint, h, w = x.shape
            x = x.view(n, num_ccpoint, -1).permute(0, 2, 1)
        elif len(x.shape) == 3:
            n, h, w = x.shape
            x = x.view(-1, w)
        x += 1e-7
        # calculate entropy
        entropy = -x * torch.log(x)
        loss = self.loss_weight * entropy
        return loss.mean()

if __name__ == '__main__':
    x = torch.zeros(1, 4, 4)
    #x = torch.zeros(1, 7, 4, 4)
    # x = torch.zeros(1, 19, 10, 10) # for ccnet
    x[0, :, 0] = 100
    #x = x.permute(0, 2, 1)
    x = torch.softmax(x, -1)
    print(x)
    entropy_loss = EntropyLoss()
    loss = entropy_loss(x)
    #print(x.sum(-1)) # sum of each row = 1
    print(loss)

