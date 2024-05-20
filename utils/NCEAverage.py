import torch
from torch.autograd import Function
from torch import nn
from .alias_multinomial import AliasMethod
import math
import numpy as np

class NCEFunction(Function):
    @staticmethod
    def forward(self, x, y, memory_da, memory, params):
        K = int(params[0].item()) #ndata
        T = params[1].item()
        Z = params[2].item()
        # K_da = memory_da.shape[0]

        momentum = params[3].item()
        batchSize = x.shape[0]
        outputSize = memory_da.shape[0] #ndata_da
        inputSize = memory_da.shape[1] - 1 #feature_dim

        # # sample positives & negatives
        # # idx.select(1,0).copy_(y.data)

        # # sample correspoinding weights
        # weight = torch.index_select(memory, 0, idx.view(-1))
        # memory.resize_(batchSize, K+1, inputSize)

        # sample correspoinding weights

        weight = torch.index_select(memory, 0, y.view(-1)) 
        # print('weight size: ', weight.shape)
        # weight.resize_(batchSize, K, inputSize)
        # print('weight resize: ', weight.shape)

        memory_da = memory_da[:, 1:].repeat(batchSize, 1, 1).permute(0, 2, 1) #(batchSize, 512, ndata_da)
        x = x.detach().unsqueeze(1)  #(batchSize, 1, 512)

        # inner product
        out = torch.bmm(x, memory_da).squeeze(1)
        out.div_(T).exp_() # batchSize * self.K+1
        # print(out.shape) #batchSize ndata_da
        x.resize_(batchSize, inputSize)
        # print(x.shape)

        if Z < 0:
            Z1 = out.mean(1) * outputSize
            # print('Z: ', Z1)
            # print("normalization constant Z is set to {:.1f}".format(Z))
        out.permute(1, 0).div_(Z1).permute(1, 0).resize_(batchSize, outputSize)
        # out_all = torch.bmm(x, memory_da.repeat(batchSize, 1).permute(0, 2, 1))
        # out_all.div_(T).exp_()
        # out_all = torch.sum(out_all, 1)
        # out_pos = torch.zeros(batchSize)
        # for i in range(batchSize):
        #     memory_po = torch.index_select(memory, 0, idx[i])
        #     out_po = torch.bmm(x[i], memory_po.permute(1, 0))
        #     out_po.div_(T).exp_()
        #     out_pos[i] = torch.sum(out_po, 0)


        self.save_for_backward(x, memory, y, weight, out, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, weight, out, params = self.saved_tensors
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        inputSize = memory.shape[1] - 1
        momentum = params[3].item()
        batchSize = gradOutput.size(0)
        
        # gradients d Pm / d linear = exp(linear) / Z
        gradOutput.detach().mul_(out.detach())
        # add temperature
        gradOutput.detach().div_(T)

        gradOutput.resize_(batchSize, 1, K)
        # print(gradOutput.shape)
        # print(weight.shape)
        # gradient of linear

        # gradInput = torch.bmm(gradOutput.detach(), weight[:, :, 1:])
        # x_resize = x.resize(batchSize, K, inputSize)
        x_resize = x.unsqueeze(1).repeat(1, K, 1)
        gradInput = torch.bmm(gradOutput.detach(), x_resize)
        gradInput.resize_as_(x)

        # update the non-parametric data
        # weight_pos = weight.select(1, 0).resize_as_(x)
        weight = torch.index_select(memory, 0, y)[:, 1:]
        weight.mul_(momentum)
        weight.add_(torch.mul(x.detach(), 1-momentum))
        w_norm = weight.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight.div(w_norm)
        # updated_weight = torch.cat((labels, updated_weight), dim = 1)
        memory[:, 1:].index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None, None

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, dataset_labels, T=0.07, momentum=0.5, Z=None):
        super(NCEAverage, self).__init__()
        # self.nLem = outputSize
        # self.unigrams = torch.ones(self.nLem)
        # self.multinomial = AliasMethod(self.unigrams)
        # self.multinomial.cuda()
        self.K = outputSize

        self.register_buffer('params',torch.tensor([self.K, T, -1, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))
        # self.register_buffer('memory', torch.ones(outputSize, inputSize))
        self.memory[:, 0] = torch.from_numpy(dataset_labels).clone()
 
    def forward(self, x, y, labels, memory_da):
        batchSize = x.size(0)
        # n_da = memory.shape[0]
        # idx_po, idx_ne = self.multinomial.draw(labels, memory_da).view(batchSize, -1)
        idx = []
        # print(labels)
        # labels = labels.detach().cpu().numpy()
        # print(labels)
        label_da = memory_da[:, 0]
        # print(label_da.shape)
        for label in labels:
            # print(label)
            # label = label.detach().cpu().numpy()
            
            idx_po = torch.where(label_da == label)
            # idx_ne = np.where(label_da != label)
            # print(idx_po)
            # idx_po = torch.tensor(idx_po).cuda()
            idx.append(idx_po[0])
        out = NCEFunction.apply(x, y, memory_da, self.memory, self.params)
        return out, idx

