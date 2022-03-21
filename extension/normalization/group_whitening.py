import torch
import torch.nn as nn
from torch.nn import Parameter



class GroupItN(nn.Module):
    def __init__(self, num_features, num_groups=32, T=5, num_channels=0, dim=4, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        super(GroupItN, self).__init__()
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        self.T = T
        if self.num_groups>num_features:
            self.num_groups=num_features
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[1] = num_features

       # print('GroupItN --- num_groups=', self.num_groups, '--T=', self.T)
        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            #nn.init.uniform_(self.weight)
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def matrix_power3(self, Input):
        B = torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.view(size[0], self.num_groups, size[1] // self.num_groups, *size[2:])

        x = x.view(size[0], self.num_groups, -1)
        IG, d, m = x.size()
        mean = x.mean(-1, keepdim=True)
        x_mean = x - mean
        P = [torch.Tensor([]) for _ in range(self.T+1)]
        sigma = x_mean.matmul(x_mean.transpose(1, 2)) / m

        P[0] = torch.eye(d).to(x).expand(sigma.shape)
        M_zero = sigma.clone().fill_(0)
        trace_inv = torch.addcmul(M_zero, sigma, P[0]).sum((1, 2), keepdim=True).reciprocal_()
        sigma_N=torch.addcmul(M_zero, sigma, trace_inv)
        for k in range(self.T):
            P[k+1] = torch.baddbmm(1.5, P[k], -0.5, self.matrix_power3(P[k]), sigma_N)
        wm = torch.addcmul(M_zero, P[self.T], trace_inv.sqrt())
        y = wm.matmul(x_mean)
        output = y.view(size[0], self.num_groups, size[1] // self.num_groups, *size[2:])
        output = output.view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)


if __name__ == '__main__':
    dbn = GroupItN(16, 4, num_channels=0, T=5, affine=False, momentum=1.)
    x = torch.randn(4, 16, 4, 4)
    print(dbn)
    y = dbn(x)
    print('y size:', y.size())
    y = y.view(y.size(0), dbn.num_groups, y.size(1) // dbn.num_groups, *y.size()[2:])
    y = y.view(y.size(0), dbn.num_groups, -1)
    print('y reshaped:', y.size())
    z = y.matmul(y.transpose(1,2))/y.size(2)
    #print('train mode:', z.diag())
    print('z_ins:', z)
    y = y.transpose(0, 1).contiguous().view(dbn.num_groups, -1)
    print('y reshaped:', y.size())
    z = y.matmul(y.transpose(0,1))/y.size(1)
    print('z_batch:', z)
    print(__file__)
