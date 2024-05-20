import torch
from torch import nn
# import math

eps = 1e-7

# class NCECriterion(nn.Module):

#     def __init__(self):
#         super(NCECriterion, self).__init__()
 
#     def forward(self, x, idx):
#         # print(x.shape)
#         # print(len(idx))
#         # print(len(idx[1]))
#         batchSize = x.size(0)
#         out_all = x.sum(1) + eps
#         # print(out_all)
#         # print(x.shape) #batchSize ndata_da
#         out_po = torch.zeros(batchSize).cuda()
#         for i in range(batchSize):
#             out_po[i] = torch.index_select(x[i], 0, idx[i]).sum(0) + eps
#         loss = out_po / out_all
#         loss = loss.log_().sum(0)
#         # print(-loss / batchSize)
#         # print(-loss / batchSize)
#         return -loss / batchSize


class NCECriterion(nn.Module):

    def __init__(self, contrast_feat, T):
        super(NCECriterion, self).__init__()
        self.contrast_feat = contrast_feat
        self.pc_feat = self.contrast_feat['pc_feat'].cuda()
        self.image_feat = self.contrast_feat['image_feat'].cuda()
        self.label_feat = self.contrast_feat['label'].cuda()
        self.T = T
 
    def forward(self, x, labels, mode):
        # x: batchsize 512
        # contrast_feat: ndata_src 512
        batchSize = x.size(0)
        outputSize = self.label_feat.shape[0]
        if mode == 'img':
            memory_da = self.image_feat.repeat(batchSize, 1, 1).permute(0, 2, 1) #(batchSize, 512, ndata_src)
        elif mode == 'pc':
            memory_da = self.pc_feat.repeat(batchSize, 1, 1).permute(0, 2, 1) #(batchSize, 512, ndata_src)

        x = x.unsqueeze(1)

        # inner product
        out = torch.div(torch.bmm(x, memory_da).squeeze(1), outputSize)
        out = torch.exp(torch.div(out, self.T))

        Z1 = out.mean(1)

        # print('loss: ', loss)

        out = out.permute(1, 0)
        out = out / Z1
        out = out.permute(1, 0)

        idx = []
        for label in labels:
            # print(label)
            idx_po = torch.where(self.label_feat == label)
            idx.append(idx_po[0])
        out_po = torch.zeros(batchSize, requires_grad = True).cuda()

        for i in range(batchSize):
            out_po[i] = torch.index_select(out[i], 0, idx[i]).sum(0)
        out_all = out.sum(1)
        loss = out_po / out_all
        loss = torch.log(loss).sum(0)
        loss = - loss / batchSize      
        # print('loss grad: ', loss.requires_grad)

        return loss

class DARE_GRAM_LOSS(nn.Module):

    def __init__(self, treshold, tradeoff_angle, tradeoff_scale):
        super(DARE_GRAM_LOSS, self).__init__()

        self.treshold = treshold
        self.tradeoff_angle = tradeoff_angle
        self.tradeoff_scale = tradeoff_scale

    def forward(self, H1, H2):    
        b,p = H1.shape
        # print(H1.shape)

        A = torch.cat((torch.ones(b,1).cuda(), H1), 1)
        B = torch.cat((torch.ones(b,1).cuda(), H2), 1)

        cov_A = (A.t()@A)
        cov_B = (B.t()@B) 

        _,L_A,_ = torch.linalg.svd(cov_A)
        _,L_B,_ = torch.linalg.svd(cov_B)
        
        eigen_A = torch.cumsum(L_A.detach(), dim=0)/L_A.sum()
        eigen_B = torch.cumsum(L_B.detach(), dim=0)/L_B.sum()

        if(eigen_A[1]>self.treshold):
            T = eigen_A[1].detach()
        else:
            T = self.treshold
            
        index_A = torch.where(eigen_A.detach()<=T)[0][-1]
        # print(index_A)

        if(eigen_B[1]>self.treshold):
            T = eigen_B[1].detach()
        else:
            T = self.treshold

        index_B = torch.where(eigen_B.detach()<=T)[0][-1]
        
        k = max(index_A.item(), index_B.item())
        # print((L_A[k]/L_A[0]).detach())
        # print(cov_A.unsqueeze(0).shape)
        A = torch.linalg.pinv(cov_A, (L_A[k]/L_A[0]).detach(), False)
        B = torch.linalg.pinv(cov_B, (L_B[k]/L_B[0]).detach(), False)
        
        cos_sim = nn.CosineSimilarity(dim=0,eps=1e-6)
        cos = torch.dist(torch.ones((p+1)).cuda(),(cos_sim(A,B)),p=1)/(p+1)
        
        return self.tradeoff_angle*(cos) + self.tradeoff_scale*torch.dist((L_A[:k]),(L_B[:k]))/k