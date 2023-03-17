import numpy as np
import torch
import torch.nn as nn
from functools import reduce
import pdb

def show_statistic(x, info='default'):
    ''' Show statistical data for analyse.
        Args: tensor(torch.Tensor), input tensor.
    '''
    tensor = x.cpu().data
    max_count = (tensor==1.0).sum().float()
    value_count = reduce(lambda x,y: x*y, tensor.shape)
    print(max_count, value_count)
    max_percent = max_count / value_count

    mean = tensor.mean()
    maximum = tensor.max()
    minimum = tensor.min()
    var = tensor.var()
    median = tensor.median()
    value_list = [mean, maximum, minimum, var, median]
    print('{}, mean:{}, max:{}, max_percent:{}, min:{}, var:{}, median:{}'
            .format(info, mean, maximum, max_percent, minimum, var, median))



if __name__ == '__main__':
    x = torch.rand((1,2,4,4))
    x[0][0][0][0] = 1.0
    #print(dir(x))
    show_statistic(x)
