from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from trainers.cross_block import CrossBlock
from utils.pos_embed import get_2d_sincos_pos_embed
from utils import emd
import numpy


def cos_sim(x1, x2):
    scores = torch.acos(torch.cosine_similarity(x1, x2, dim=1)) / numpy.pi
    return scores.mean()


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx.cuda() + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #  batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


# Position Encoding with cos-sin function
class PosE(nn.Module):
    def __init__(self, in_dim=3, out_dim=72, alpha=1000, beta=100):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, x):
        B, C, N = x.shape
        feat_dim = self.out_dim // (self.in_dim * 2)  # 12

        feat_range = torch.arange(feat_dim).float().cuda()  # [0, 1, ..., 11]
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * x.unsqueeze(-1), dim_embed)  # [B, 3, N, 12])

        sin_embed = torch.sin(div_embed)  # [B, 3, N, 12])
        cos_embed = torch.cos(div_embed)  # [B, 3, N, 12])
        # position_embed = torch.cat([sin_embed, cos_embed], -1)  # [B, 3, N, 24])
        # position_embed = position_embed.permute(0, 1, 3, 2).contiguous()  # [B, 3, 24, N]
        # position_embed = position_embed.view(B, self.out_dim, N)  # [B, 72, N]

        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)

        # # Weigh
        # x = torch.cat((x, position_embed), dim=1)  # [B, 72+3, N]

        return position_embed


class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        B, _, N = xyz.shape  
        # print(xyz.shape)  
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim).float().cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        # print(cos_embed.shape)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        # print(position_embed.permute(0, 1, 3, 2).shape)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        
        return position_embed


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        # focal_loss = (1 / pt) ** self.gamma * ce_loss
        return focal_loss


class Pointnet(nn.Module):
    def __init__(self, channel):
        super(Pointnet, self).__init__()
        # self.raw_point_embed_0 = PosE_Initial(3, 36, 100, 1000)
        self.raw_point_embed_1 = PosE_Initial(3, 72, 100, 1000)
        # self.raw_point_embed_2 = PosE_Initial(3, 144, 100, 1000)
        self.conv1 = torch.nn.Conv1d(75, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 768, 1)
        # self.conv3 = torch.nn.Conv1d(768, 768, 1)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(768)
        # self.bn3 = nn.BatchNorm1d(768)

    def forward(self, x):
        B, P, N, C = x.shape
        x = x.reshape(B * P, N, C)
        c = x.mean(1, keepdims=True)
        x = x - c
        xyz = x.transpose(2, 1)
        # x_36 = self.raw_point_embed_0(xyz)
        x_72 = self.raw_point_embed_1(xyz)
        # x_144 = self.raw_point_embed_2(xyz)
        x = torch.cat([xyz, x_72], dim = 1)
        # x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.squeeze(-1)
        x = x.reshape(B, P, -1)
        return x


class DGCNN(nn.Module):
    def __init__(self, k, emb_dims):
        super(DGCNN, self).__init__()
        self.k = k
        self.emb_dims = emb_dims // 2
        # self.posE = PosE()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm1d(self.emb_dims)
        # self.bn4 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 256, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(320, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv4 = nn.Sequential(nn.Conv1d(self.emb_dims * 2, self.emb_dims, kernel_size=1, bias=False),
        #                            self.bn4,
        #                            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):  # [batch_size, patch_num, 32, 3]
        batch_size, patch_num, num_points, dims = x.shape
        x = x.reshape(batch_size * patch_num, num_points, dims)
        x = x.permute(0, 2, 1)  # [batch_size*patch_num, 3, 32], batch_size*patch_num denotes as batch_size following
        # x = self.posE(x)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2), dim=1)  # (batch_size, 64+384, num_points)

        x = self.conv3(x)  # (batch_size, 64+384, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x2 = F.adaptive_avg_pool1d(x, 1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = torch.cat((x1, x2), dim=1)
        # x = self.conv4(x)

        x = x.view(batch_size, patch_num, -1)

        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, activation='relu', bias=True):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                nn.BatchNorm1d(out_ch),
                # nn.LayerNorm(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                self.ac
            )

    def forward(self, x):
        # x = l2_norm(x, 1)
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, num_class):
        super(Classifier, self).__init__()

        self.mlp1 = fc_layer(input_dim, 512, bias=True, activation='leakyrelu', bn=True)
        self.dp1 = nn.Dropout(p=0.5)
        self.mlp2 = fc_layer(512, 256, bias=True, activation='leakyrelu', bn=True)
        self.dp2 = nn.Dropout(p=0.5)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.dp1(self.mlp1(x))
        # print('weight: ', self.mlp1.fc[0].weight)
        x2 = self.dp2(self.mlp2(x))
        logits = self.mlp3(x2)
        return logits


class Projector_img(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projector_img, self).__init__()

        self.mlp1 = fc_layer(input_dim * 10, 512, bias=True, activation='leakyrelu', bn=True)
        self.dp1 = nn.Dropout(p=0.5)
        self.mlp2 = fc_layer(512, output_dim)

    def forward(self, x):
        x = x[:, 0, :] # [B*10, 768]
        # x = x @ self.proj  # [B*10, output_dim_img]

        # x = x / x.norm(dim=-1, keepdim=True)

        x = x.reshape(-1, 10 * 768)

        x = self.dp1(self.mlp1(x))
        x2 = self.mlp2(x)
        return x2


class Projector_pc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projector_pc, self).__init__()

        self.mlp1 = fc_layer(768, 512, bias=True, activation='leakyrelu', bn=True)
        self.dp1 = nn.Dropout(p=0.5)
        self.mlp2 = fc_layer(512, output_dim)

    def forward(self, x):
        # print(x.shape)
        x = x[:, 0, :] # [B, 768]
        # x = x @ self.proj  # [B, output_dim_pc]

        # x = x / x.norm(dim=-1, keepdim=True)

        x = self.dp1(self.mlp1(x))
        x2 = self.mlp2(x)
        return x2


class PCPosEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.conv1 = nn.Sequential(nn.Conv1d(3, 128, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(128 * 2, embed_dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pose_embed_cls_token = nn.Parameter(torch.zeros(1, embed_dim, 1), requires_grad=True)

    def forward(self, x):
        batch_size, patch_num, num_points, dims = x.shape
        x = x.reshape(batch_size * patch_num, num_points, dims)
        c = x.mean(1, keepdims=True).permute(0, 2, 1)  # batch_size*patch_num, 3, 1
        g = self.conv1(c)  # batch_size*patch_num, 128, 1
        g = g.reshape(batch_size, patch_num, -1)  # batch_size, patch_num, 128
        g = g.permute(0, 2, 1)  # batch_size, 128, patch_num
        g_p = F.adaptive_avg_pool1d(g, 1)  # batch_size, 128, 1
        g_p = g_p.repeat(1, 1, patch_num)  # batch_size, 128, patch_num
        x = self.conv2(torch.cat((g, g_p), 1))  # batch_size, embed_dims, patch_num
        pose_embed_cls_token = self.pose_embed_cls_token.repeat(batch_size, 1, 1).cuda()  # batch_size, embed_dims, 1
        x = torch.cat((pose_embed_cls_token, x), 2).permute(0, 2, 1)  # batch_size, patch_num + 1, embed_dims
        return x


class SmoothCrossEntropy(nn.Module):
    """
    loss = SmoothCrossEntropy()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    """
    def __init__(self, alpha=0.3):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()


class TeacherModel(nn.Module):

    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, args=None):
        super().__init__()

        # ---------------------------MAE encoder specifics-------------------------
        # frozen blocks shared between img and pc
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        # for i in range(depth):
        #     self.blocks[i] = self.blocks[i].half()

        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)#.half()
        num_patches = self.patch_embed.num_patches  # 196
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))#.half()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)#.half()
        self.norm = norm_layer(embed_dim)#.half()

        # --------------------------------------------------------------------------

        self.projector_img = Projector_img(embed_dim, 512)
        self.classifier_img = Classifier(512, args.num_cls)
        self.norm_pix_loss = norm_pix_loss
        # self.criterion_ce = SmoothCrossEntropy()#torch.nn.CrossEntropyLoss() # FocalLoss() #
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.train_layer = 3

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

    def forward_encoder_img(self, x):
        with torch.no_grad():
            # embed patches
            # x = x.to(torch.float16)
            x = self.patch_embed(x)

            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            for i in range(12-self.train_layer):
                x = self.blocks[i](x)
        for i in range(self.train_layer):
            x = self.blocks[12-self.train_layer+i](x)
        x = self.norm(x)
        # x = x.to(torch.float32)

        return x  # , mask, ids_restore

    def forward(self, imgs, label=-1, mode='train', mask_ratio=0.75):

        if mode == 'train':
            # for image
            # batch_size = int(imgs.shape[0] / 2)
            latent_all_img = self.forward_encoder_img(imgs)  # [B*10, 197, 768]
            img_feat = self.projector_img(latent_all_img)
            logits_img = self.classifier_img(img_feat)
            loss_ce_img = self.criterion_ce(logits_img, label)
            # loss_re_img = 0.0

            return loss_ce_img, latent_all_img, logits_img
        else:
            latent_all_img = self.forward_encoder_img(imgs)  # [B*10, 197, 768]
            img_feat = self.projector_img(latent_all_img)
            logits_img = self.classifier_img(img_feat)
            loss_ce_img = self.criterion_ce(logits_img, label)

            return logits_img, loss_ce_img


class StudentModel(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, args=None):
        super().__init__()

        # ---------------------------MAE encoder specifics-------------------------
        # frozen blocks shared between img and pc
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.pc_patch_embed = DGCNN(20, embed_dim)
        self.pc_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pc_pos_embed = PCPosEmbed(embed_dim)
        self.pc_norm = norm_layer(embed_dim)

        # -------------------------MAE decoder specifics----------------------------

        self.pc_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.pc_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pc_decoder_pos_embed = PCPosEmbed(decoder_embed_dim)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.pc_decoder_blocks = nn.ModuleList([
            CrossBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.pc_decoder_norm = norm_layer(decoder_embed_dim)
        self.pc_decoder_pred = nn.Linear(decoder_embed_dim, 128 * 3, bias=True)  # decoder to patch
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # --------------------------------------------------------------------------

        self.projector_pc = Projector_pc(embed_dim, 512)  # (1536, 512)
        self.classifier_pc = Classifier(512, args.num_cls)
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.pc_cls_token, std=.02)
        torch.nn.init.normal_(self.pc_mask_token, std=.02)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        B, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder_pc(self, x):
        """
        X: [B,L,N,C]
        """
        # embed patches
        pc_pos_embed = self.pc_pos_embed(x)  # [1, 161, 768]
        x = self.pc_patch_embed(x)

        # add pos embed w/o cls token
        x = x + pc_pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.pc_cls_token + pc_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # can not use torch.no_grad here, as the grad needs to backward to pc_patch_embed.
        # apply Transformer blocks (parameters frozen)
        for blk in self.blocks:
            x = blk(x)
        # for blk in self.pc_blocks:
        #     x = blk(x)
        x = self.pc_norm(x)  # [B*10, 197, 768]
        # x = self.projector_share(x)

        return x  # , mask, ids_restore

    def forward_decoder_pc(self, x, y, ids_restore, pc_decoder_pos_embed):
        batch_size, no_mask_pc_patch_num, pc_fea_dim = x.shape
        _, img_patch_num, img_fea_dim = y.shape

        # embed img tokens
        y = y.reshape(batch_size, -1, img_fea_dim)
        y = self.decoder_embed(y)  # [B, 1970, 512]

        # embed pc tokens
        x = self.pc_decoder_embed(x)  # [B, no_mask_pc_patch_num, 512]

        # append mask tokens to sequence
        mask_tokens = self.pc_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + pc_decoder_pos_embed

        # apply Transformer blocks (including self-attention module)
        for crossblk, selfblk in zip(self.pc_decoder_blocks, self.decoder_blocks):
            x = crossblk(x, y)  # [x:q;  y:k,v] ---> x.shape
            x = selfblk(x)
        x = self.pc_decoder_norm(x)

        # predictor projection
        x = self.pc_decoder_pred(x)  # [B, 28, 768] --> # [B, 28, 128*3]

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss_pc(self, pc_patches, pred, mask, logits_pc, label=None):
        """
        pc_patches: [B, L, N, C]
        pred: [B, L, N*C]
        mask: [B, L], 0 is keep, 1 is remove,
        """
        B, L, N, C = pc_patches.shape
        pc_patches = pc_patches.reshape(-1, N, C)  # [B*L, N, C]
        pred = pred.reshape(-1, N, C)  # [B*L, N, C]
        loss_re = emd.earth_mover_distance(pc_patches.permute(0, 2, 1), pred.permute(0, 2, 1))
        loss_re = loss_re.reshape(B, L)
        loss_re = 1 * (loss_re * mask).sum() / mask.sum()

        return loss_re

    def forward(self, pc_patches, latent_all_img, logits_img, label=-1, mode='train', mask_ratio=0.85):

        if mode == 'train':
            # for point cloud
            latent_all_pc = self.forward_encoder_pc(pc_patches)
            pc_feat = self.projector_pc(latent_all_pc)
            logits_pc = self.classifier_pc(pc_feat)
            loss_ce = 1 * self.criterion_ce(logits_pc[:label.shape[0], :], label)

            latent_pc, mask_pc, ids_restore_pc = self.random_masking(latent_all_pc[:, 1:, :], mask_ratio)
            latent_pc = torch.cat((latent_all_pc[:, 0, :].unsqueeze(1), latent_pc), dim=1)
            latent_img, mask_img, ids_restore_img = self.random_masking(latent_all_img[:, 1:, :], mask_ratio)
            latent_img = torch.cat((latent_all_img[:, 0, :].unsqueeze(1), latent_img), dim=1)
            pc_decoder_pos_embed = self.pc_decoder_pos_embed(pc_patches)
            pred = self.forward_decoder_pc(latent_pc, latent_img, ids_restore_pc, pc_decoder_pos_embed)

            loss_re = self.forward_loss_pc(pc_patches, pred, mask_pc, logits_pc, label)
            # loss_align = torch.tensor(0.0)

            criterion_aign = nn.KLDivLoss(reduction="batchmean", log_target=True)
            loss_align = criterion_aign(F.log_softmax(logits_pc, dim=1), F.log_softmax(logits_img, dim=1)) 

            return loss_re, loss_ce, 2 * loss_align

        else:
            latent_all_pc = self.forward_encoder_pc(pc_patches)
            pc_feat = self.projector_pc(latent_all_pc)
            logits_pc = self.classifier_pc(pc_feat)
            loss_ce = self.criterion_ce(logits_pc, label)
            # loss_align = torch.tensor(0.0)
            criterion_aign = nn.KLDivLoss(reduction="batchmean", log_target=True)
            loss_align = criterion_aign(F.log_softmax(logits_pc, dim=1), F.log_softmax(logits_img, dim=1)) 

            return logits_pc, loss_ce, loss_align


def mae_vit_base_patch16_dec512d8b_img(**kwargs):
    model = TeacherModel(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b_pc(**kwargs):
    model = StudentModel(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16_img = mae_vit_base_patch16_dec512d8b_img  # decoder: 512 dim, 8 blocks
mae_vit_base_patch16_pc = mae_vit_base_patch16_dec512d8b_pc  # decoder: 512 dim, 8 blocks
