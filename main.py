import torch
import numpy
import random
import sklearn.metrics as metrics
import torch.nn as nn

from datasets.dataloader import ModelNet, ScanNet, ShapeNet
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torchsampler import ImbalancedDatasetSampler
import torch.optim as optim
from trainers.mv_utils_zs import Realistic_Projection
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from utils import log, checkpoints
from tqdm import tqdm
from trainers import model
from utils.pos_embed import interpolate_pos_embed


def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    if args.src_dataset == 'shapenet' or args.src_dataset == 'scannet':
        train_sampler = ImbalancedDatasetSampler(dataset, labels=dataset.label[train_indices], indices=train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
    else:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


def load_mae_to_cpu():
    img_model = model.__dict__['mae_vit_base_patch16_img'](norm_pix_loss=False)
    pc_model = model.__dict__['mae_vit_base_patch16_pc'](norm_pix_loss=False)

    checkpoint = torch.load('./pretrained/mae_finetuned_vit_base.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']

    # interpolate position embedding
    interpolate_pos_embed(img_model, checkpoint_model)

    # load pre-trained model
    img_model.load_state_dict(checkpoint_model, strict=False)
    pc_model.load_state_dict(checkpoint_model, strict=False)

    return img_model.eval(), pc_model


class Trainer:
    def __init__(self, img_model, pc_model, optimizer_img, optimizer_pc, device=None):
        self.img_model = img_model
        self.pc_model = pc_model
        self.optimizer_img = optimizer_img
        self.optimizer_pc = optimizer_pc
        self.device = device

        # Realistic projection
        self.num_views = 10
        pc_views = Realistic_Projection()
        self.get_img = pc_views.get_img

        self.channel = 512

    def real_proj(self, pc, imsize=224):
        img = self.get_img(pc).to(self.device)
        img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)
        return img

    def model_forward(self, pc, pc_patches, label, role=None, mode='train'):
        with torch.no_grad():
            # Realistic Projection
            images = self.real_proj(pc)

        # Image features
        if mode == 'train':
            if role == 'teacher':
                loss_ce_img, latent_all_img, logits_img = self.img_model(images, label, mode)
                loss_re, loss_ce_pc, loss_align = self.pc_model(pc_patches, latent_all_img, logits_img.detach(), label, mode)
                loss_ce = loss_ce_img + loss_ce_pc
            else:
                self.img_model.eval()
                with torch.no_grad():
                    loss_ce_img, latent_all_img, logits_img = self.img_model(images, label, mode)
                loss_re, loss_ce, loss_align = self.pc_model(pc_patches, latent_all_img, logits_img, label, mode)
            return loss_re, loss_ce, loss_align
        else:
            logits_img, _ = self.img_model(images, label, mode)
            logits_pc, _, _ = self.pc_model(pc_patches, None, logits_img, label, mode)
            return logits_img, logits_pc

    def model_train_teacher(self, data, domain=None):
        self.img_model.train()
        self.optimizer_img.zero_grad()
        self.pc_model.train()
        self.optimizer_pc.zero_grad()
        pc, pc_patch, label = data[0].cuda(), data[1].cuda(), data[2].cuda()
        loss_re, loss_ce, loss_align = self.model_forward(pc, pc_patch, label, role = 'teacher', mode='train')
        loss = loss_re + loss_ce + loss_align
        if domain == 'target':
            loss = loss_re + loss_align
        loss.backward()
        self.optimizer_img.step()
        self.optimizer_pc.step()
        return loss_re.item(), loss_ce.item(), loss_align.item()

    def model_train_student(self, data, domain=None):
        self.pc_model.train()
        self.optimizer_pc.zero_grad()
        pc, pc_patch, label = data[0].cuda(), data[1].cuda(), data[2].cuda()
        loss_re, loss_ce, loss_align = self.model_forward(pc, pc_patch, label, role = 'student', mode='train')
        loss = loss_re + loss_ce + loss_align
        if domain == 'target':
            loss = loss_re + loss_align
        loss.backward()
        self.optimizer_pc.step()
        return loss_re.item(), loss_ce.item(), loss_align.item()

    def model_eval_img_pc(self, data_loader, io):
        self.pc_model.eval()
        self.img_model.eval()
        with torch.no_grad():
            pred_list = []
            pred_list_img = []
            pred_list_pc = []
            true_list = []
            all_number = 0
            for batch_idx, data in enumerate(data_loader):
                pc, pc_patch, label = data[0].cuda(), data[1].cuda(), data[2].cuda()
                logits_img, logits_pc = self.model_forward(pc, pc_patch, label, mode='eval')
                logits = logits_img + logits_pc
                preds = logits.max(dim=1)[1]
                preds_img = logits_img.max(dim=1)[1]
                preds_pc = logits_pc.max(dim=1)[1]

                true_list.append(label.cpu().numpy())
                pred_list.append(preds.detach().cpu().numpy())
                pred_list_img.append(preds_img.detach().cpu().numpy())
                pred_list_pc.append(preds_pc.detach().cpu().numpy())
            true = numpy.concatenate(true_list)
            pred = numpy.concatenate(pred_list)
            acc = metrics.accuracy_score(true, pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(true, pred)
            io.cprint("Evaluate - acc: %.4f, avg acc: %.4f" % (acc, avg_per_class_acc))

            pred_img = numpy.concatenate(pred_list_img)
            acc_img = metrics.accuracy_score(true, pred_img)
            avg_per_class_acc_img = metrics.balanced_accuracy_score(true, pred_img)
            io.cprint("Evaluate img - acc: %.4f, avg img acc: %.4f" % (acc_img, avg_per_class_acc_img))

            pred_pc = numpy.concatenate(pred_list_pc)
            acc_pc = metrics.accuracy_score(true, pred_pc)
            avg_per_class_acc_pc = metrics.balanced_accuracy_score(true, pred_pc)
            io.cprint("Evaluate pc - acc: %.4f, avg pc acc: %.4f" % (acc_pc, avg_per_class_acc_pc))

            return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DA on Point Clouds')
    parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
    parser.add_argument('--dataroot', type=str, default='..', metavar='N', help='data path')
    parser.add_argument('--src_dataset', type=str, default='modelnet', choices=['modelnet', 'shapenet', 'scannet'])
    parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
    parser.add_argument('--epochs', type=int, default=400, help='number of episode to train')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
    parser.add_argument('--batch_size', type=int, default=24, metavar='batch_size',
                        help='Size of train batch per domain')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of test batch per domain')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')

    args = parser.parse_args()

    if args.src_dataset == 'modelnet' and args.trgt_dataset == 'shapenet':
        args.exp_name = 'm2s'
    elif args.src_dataset == 'modelnet' and args.trgt_dataset == 'scannet':
        args.exp_name = 'm2r'
    elif args.src_dataset == 'shapenet' and args.trgt_dataset == 'modelnet':
        args.exp_name = 's2m'
    elif args.src_dataset == 'shapenet' and args.trgt_dataset == 'scannet':
        args.exp_name = 's2r'
    elif args.src_dataset == 'scannet' and args.trgt_dataset == 'modelnet':
        args.exp_name = 'r2m'
    elif args.src_dataset == 'scannet' and args.trgt_dataset == 'shapenet':
        args.exp_name = 'r2s'
    else:
        args.exp_name = 'other'

    io = log.IOStream(args)
    io.cprint(args)

    random.seed(args.seed)
    np.random.seed(args.seed)  # to get the same point choice in ModelNet and ScanNet leave it fixed
    torch.manual_seed(args.seed)
    args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
    if args.cuda:
        io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
                  str(torch.cuda.device_count()) + ' devices available')
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        io.cprint('Using CPU')

    src_dataset = args.src_dataset
    trgt_dataset = args.trgt_dataset
    data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}

    src_train_dataset = data_func[src_dataset](io, args.dataroot, 'train')
    src_test_dataset = data_func[src_dataset](io, args.dataroot, 'test')
    trgt_train_dataset = data_func[trgt_dataset](io, args.dataroot, 'train')
    trgt_test_dataset = data_func[trgt_dataset](io, args.dataroot, 'test')

    src_train_sampler, src_valid_sampler = split_set(src_train_dataset, src_dataset, "source")

    src_train_loader = torch.utils.data.DataLoader(src_train_dataset, batch_size=args.batch_size, num_workers=4,
                                                   sampler=src_train_sampler, drop_last=True)
    src_val_loader = torch.utils.data.DataLoader(src_train_dataset, batch_size=args.batch_size, num_workers=4,
                                                 sampler=src_valid_sampler)
    src_test_loader = torch.utils.data.DataLoader(src_test_dataset, batch_size=args.batch_size, num_workers=4)

    trgt_train_loader = torch.utils.data.DataLoader(trgt_train_dataset, batch_size=args.batch_size, num_workers=4,
                                                    drop_last=True)
    trgt_test_loader = torch.utils.data.DataLoader(trgt_test_dataset, batch_size=args.batch_size, num_workers=4)

    img_model, pc_model = load_mae_to_cpu()

    need_frozend_layers = ['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']
    for param in img_model.named_parameters():
        if param[0] in need_frozend_layers or (param[0].split('.')[0] == 'blocks'and int(param[0].split('.')[1]) <= 8):
            param[1].requires_grad = False
    for param in pc_model.named_parameters():
        if param[0].split('.')[0] == 'blocks'and int(param[0].split('.')[1]) <= 8:
            param[1].requires_grad = False

    img_model = img_model.to(device)
    pc_model = pc_model.to(device)

    optimizer_img = optim.Adam(filter(lambda p: p.requires_grad, img_model.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler_img = CosineAnnealingLR(optimizer_img, args.epochs)
    optimizer_pc = optim.Adam(filter(lambda p: p.requires_grad, pc_model.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler_pc = CosineAnnealingLR(optimizer_pc, args.epochs)
    trainer = Trainer(img_model, pc_model, optimizer_img, optimizer_pc, device=device)

    checkpoint_dir = args.out_path + '/' + args.exp_name
    checkpoint_io_img = checkpoints.CheckpointIO(checkpoint_dir, model=img_model, optimizer=optimizer_img)
    checkpoint_io_pc = checkpoints.CheckpointIO(checkpoint_dir, model=pc_model, optimizer=optimizer_pc)
    try:
        load_dict_img = checkpoint_io_img.load('model_img_200.pt')
        load_dict_pc = checkpoint_io_pc.load('model_pc_200.pt')
    except FileExistsError:
        load_dict_img = dict()
        load_dict_pc = dict()
    epoch_it = load_dict_img.get('epoch_it', -1)
    batch_it = load_dict_img.get('batch_it', -1)
    src_metric_val_best = 0.0

    while True:
        src_loss_re_sum = 0.0
        src_loss_ce_sum = 0.0
        src_loss_align_sum = 0.0
        trgt_loss_re_sum = 0.0
        trgt_loss_align_sum = 0.0
        it = 0
        epoch_it += 1
        if epoch_it > args.epochs:
            break
        for src_batch, trgt_batch in tqdm(zip(src_train_loader, trgt_train_loader)):
            batch_it += 1
            it += 1

            if epoch_it % 10 < 5:  # update teacher
                src_loss_re, src_loss_ce, src_loss_align = trainer.model_train_teacher(src_batch, domain='source')
                trgt_loss_re, _, _ = trainer.model_train_teacher(trgt_batch, domain='target')
            else:  # update student
                src_loss_re, src_loss_ce, src_loss_align = trainer.model_train_student(src_batch, domain='source')
                trgt_loss_re, _, _ = trainer.model_train_student(trgt_batch, domain='target')

            src_loss_ce_sum += src_loss_ce
            src_loss_re_sum += src_loss_re
            src_loss_align_sum += src_loss_align
            trgt_loss_re_sum += trgt_loss_re
        scheduler_img.step()
        scheduler_pc.step()
        if epoch_it % 10 == 0:
            checkpoint_io_img.save('model_img_new.pt', epoch_it=epoch_it, batch_it=batch_it)
            checkpoint_io_pc.save('model_pc_new.pt', epoch_it=epoch_it, batch_it=batch_it)

        io.cprint('lr_rate=%.9f' % scheduler_pc.get_last_lr()[0])
        io.cprint('Train -SRC - [Epoch %02d], loss_re=%.4f, loss_ce=%.4f, loss_align=%.4f' %
                  (epoch_it, src_loss_re_sum / it, src_loss_ce_sum / it, src_loss_align_sum / it))
        io.cprint('Train -TRT - [Epoch %02d], loss_re=%.4f' % (epoch_it, trgt_loss_re_sum / it))

        # Run validation
        src_metric_val = trainer.model_eval_img_pc(src_val_loader, io)
        trgt_metric_test = trainer.model_eval_img_pc(trgt_test_loader, io)
