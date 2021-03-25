import torch
import torch.nn as nn
import torch.nn.functional as F

thresh, lens, decay, if_bias = (0.5, 0.5, 0.2, True)

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # print(input.gt(thresh))
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        # temp = input.gt(thresh)
        return grad_input * temp.float()

act_fun = ActFun.apply

def mem_update(ops, x, mem, spike, lateral = None):
    mem = mem * decay * (1. - spike) + ops(x)
    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem)
    return mem, spike

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size, bias = if_bias),
        nn.BatchNorm2d(chann_out)
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Dropout(),
        nn.Linear(size_in, size_out, bias = if_bias),
        nn.BatchNorm1d(size_out)
    )
    return layer

class LISNN(nn.Module):
    def __init__(self, opt):
        super(LISNN, self).__init__()
        self.batch_size = opt.batch_size
        self.dts = opt.dts
        self.time_window = opt.time_window
        # self.fc = (512, 10)
        if self.dts == 'CIFAR10' or self.dts == 'MNIST':
            self.fc = (512, 10)
        elif self.dts == 'CIFAR100':
            self.fc = (512, 100)

        if self.dts == 'CIFAR10' or self.dts == 'CIFAR100':
            self.cnn = ((3, 64, 3, 1, 2), (64, 128, 3, 1, 2),(128, 256, 3, 1, 2),(256, 512, 3, 1, 2),(512, 512, 3, 1, 2))
            self.li = (5, 1, 2)
            self.kernel = (32, 16, 8, 4, 2, 1)
        elif self.dts == 'MNIST':
            self.cnn = ((1, 64, 3, 1, 2), (64, 128, 3, 1, 2),(128, 256, 3, 1, 2),(256, 512, 3, 1, 2),(512, 512, 3, 1, 2))
            self.li = (5, 1, 2)
            self.kernel = (32, 16, 8, 4, 2, 1)

        # Conv blocks
        self.block1 = vgg_conv_block([self.cnn[0][0],self.cnn[0][1]], [self.cnn[0][1],self.cnn[0][1]], [self.cnn[0][2],self.cnn[0][2]], [self.cnn[0][3],self.cnn[0][3]], self.cnn[0][4], self.cnn[0][4])
        self.block2 = vgg_conv_block([self.cnn[1][0],self.cnn[1][1]], [self.cnn[1][1],self.cnn[1][1]], [self.cnn[1][2],self.cnn[1][2]], [self.cnn[1][3],self.cnn[1][3]], self.cnn[1][4], self.cnn[1][4])
        self.block3 = vgg_conv_block([self.cnn[2][0],self.cnn[2][1],self.cnn[2][1]], [self.cnn[2][1],self.cnn[2][1],self.cnn[2][1]], [self.cnn[2][2],self.cnn[2][2],self.cnn[2][2]], [self.cnn[2][3],self.cnn[2][3],self.cnn[2][3]], self.cnn[2][4], self.cnn[2][4])
        self.block4 = vgg_conv_block([self.cnn[3][0],self.cnn[3][1],self.cnn[3][1]], [self.cnn[3][1],self.cnn[3][1],self.cnn[3][1]], [self.cnn[3][2],self.cnn[3][2],self.cnn[3][2]], [self.cnn[3][3],self.cnn[3][3],self.cnn[3][3]], self.cnn[3][4], self.cnn[3][4])
        self.block5 = vgg_conv_block([self.cnn[4][0],self.cnn[4][1],self.cnn[4][1]], [self.cnn[4][1],self.cnn[4][1],self.cnn[4][1]], [self.cnn[4][2],self.cnn[4][2],self.cnn[4][2]], [self.cnn[4][3],self.cnn[4][3],self.cnn[4][3]], self.cnn[4][4], self.cnn[4][4])
        # Fc layers
        self.fc1 = vgg_fc_layer(self.kernel[-1] * self.kernel[-1] * self.cnn[-1][1], self.fc[0])
        self.fc2 = vgg_fc_layer(self.fc[0], self.fc[0])
        self.fc3 = vgg_fc_layer(self.fc[0], self.fc[1])

        if opt.if_lateral:
            self.lateral1 = nn.Conv2d(self.cnn[0][1], self.cnn[0][1], kernel_size = self.li[0], stride = self.li[1], padding = self.li[2], groups = self.cnn[0][1], bias = False)
            self.lateral2 = nn.Conv2d(self.cnn[1][1], self.cnn[1][1], kernel_size = self.li[0], stride = self.li[1], padding = self.li[2], groups = self.cnn[1][1], bias = False)
            self.lateral3 = nn.Conv2d(self.cnn[2][1], self.cnn[2][1], kernel_size = self.li[0], stride = self.li[1], padding = self.li[2], groups = self.cnn[2][1], bias = False)
            self.lateral4 = nn.Conv2d(self.cnn[3][1], self.cnn[3][1], kernel_size = self.li[0], stride = self.li[1], padding = self.li[2], groups = self.cnn[3][1], bias = False)
            self.lateral5 = nn.Conv2d(self.cnn[4][1], self.cnn[4][1], kernel_size = self.li[0], stride = self.li[1], padding = self.li[2], groups = self.cnn[4][1], bias = False)
        else:
            self.lateral1 = None
            self.lateral2 = None
            self.lateral3 = None
            self.lateral4 = None
            self.lateral5 = None

    def forward(self, input):
        c1_mem = c1_spike = torch.zeros(self.batch_size, self.cnn[0][1], self.kernel[0], self.kernel[0]).cuda()
        c2_mem = c2_spike = torch.zeros(self.batch_size, self.cnn[1][1], self.kernel[1], self.kernel[1]).cuda()
        c3_mem = c3_spike = torch.zeros(self.batch_size, self.cnn[2][1], self.kernel[2], self.kernel[2]).cuda()
        c4_mem = c4_spike = torch.zeros(self.batch_size, self.cnn[3][1], self.kernel[3], self.kernel[3]).cuda()
        c5_mem = c5_spike = torch.zeros(self.batch_size, self.cnn[4][1], self.kernel[4], self.kernel[4]).cuda()

        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.fc[0]).cuda()
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.fc[0]).cuda()
        h3_mem = h3_spike = h3_sumspike = torch.zeros(self.batch_size, self.fc[1]).cuda()

        for step in range(self.time_window):
            if self.dts == 'CIFAR10' or self.dts == 'CIFAR100' or self.dts == 'MNIST':
                x = input > torch.rand(input.size()).cuda()
            elif self.dts == 'NMNIST':
                x = input[:, :, :, :, step]

            c1_mem, c1_spike = mem_update(self.block1, x.float(), c1_mem, c1_spike, self.lateral1)
            x = F.avg_pool2d(c1_spike, 2)
            # print(x[1])

            c2_mem, c2_spike = mem_update(self.block2, x, c2_mem, c2_spike, self.lateral2)
            x = F.avg_pool2d(c2_spike, 2)
            # print(x[1])
            
            c3_mem, c3_spike = mem_update(self.block3, x, c3_mem, c3_spike, self.lateral3)
            x = F.avg_pool2d(c3_spike, 2)
            # print('max:%f, min:%f'%(torch.max(x).item(),torch.min(x).item()))

            c4_mem, c4_spike = mem_update(self.block4, x, c4_mem, c4_spike, self.lateral4)
            x = F.avg_pool2d(c4_spike, 2)
            # print('max:%f, min:%f'%(torch.max(x).item(),torch.min(x).item()))

            c5_mem, c5_spike = mem_update(self.block5, x, c5_mem, c5_spike, self.lateral5)
            x = F.avg_pool2d(c5_spike, 2)
            # print('max:%f, min:%f'%(torch.max(x).item(),torch.min(x).item()))
            
            x = x.view(self.batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            # print('max:%f, min:%f'%(torch.max(h1_sumspike).item(),torch.min(h1_sumspike).item()))
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike
            # print('max:%f, min:%f'%(torch.max(h2_sumspike).item(),torch.min(h2_sumspike).item()))
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            h3_sumspike += h3_spike
            # print('max:%f, min:%f'%(torch.max(h3_sumspike).item(),torch.min(h3_sumspike).item()))

        outputs = h3_sumspike / self.time_window
        # print('max:%f, min:%f'%(torch.max(outputs).item(),torch.min(outputs).item()))
        #print('mean:%f, var:%f'%(torch.mean(outputs).item(),torch.var(outputs).item()))
        return outputs