import torch
import numpy
import random
import os
import sklearn.metrics as metrics
from datasets.dataloader import ModelNet, ScanNet, ShapeNet, label_to_idx
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchsampler import ImbalancedDatasetSampler
from trainers.mv_utils_zs import Realistic_Projection
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from utils import log, checkpoints
from tqdm import tqdm
from trainers import model
from utils.pos_embed import interpolate_pos_embed


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


def select_target_by_conf(io, trgt_train_loader, threshold,  pc_model=None, img_model=None, trainer=None):
    pc_list = []
    pc_patches_list = []
    pseudo_label_list = []
    gt_list = []
    sfm = torch.nn.Softmax(dim=1)
    pc_model.eval()
    img_model.eval()
    # thd_l = args.thd_low
    # thd_h = args.thd_high
    # threshold = {0: thd_l, 1: thd_h, 2: thd_h, 3: thd_l, 4: thd_h, 5: thd_h, 6: thd_h, 7: thd_h, 8: thd_h, 9: thd_h}
    with torch.no_grad():
        for data in tqdm(trgt_train_loader):
            pc, pc_patches, label = data[0].cuda(), data[1].cuda(), data[2].cuda()
            logits_img, logits_pc = trainer.model_forward(pc, pc_patches, label, mode='eval')
            cls_conf = (sfm(logits_img) + sfm(logits_pc)) / 2
            mask = torch.max(cls_conf, 1)
            index = 0
            for i in mask[0]:
                # thd = threshold.get(mask[1][index].item())
                if i > threshold:
                    pc_list.append(pc[index].cpu().numpy())
                    pc_patches_list.append(pc_patches[index].cpu().numpy())
                    pseudo_label_list.append(mask[1][index].cpu().numpy())
                    gt_list.append(label[index].cpu().numpy())
                index += 1
        io.cprint('pseudo label acc: '
                  + str(round(sum(np.array(pseudo_label_list) == np.array(gt_list)) / len(pc_list), 3)))
        io.cprint('data num: ' + str(len(pc_list)))
    return np.array(pc_list), np.array(pc_patches_list), np.array(pseudo_label_list), np.array(gt_list)


class DataLoadST(Dataset):
    def __init__(self, data):
        self.pc, self.pc_patches, self.label, self.gt_label = data
        self.num_examples = len(self.pc)

        print("Number of examples in trgt_new_dataset: " + str(len(self.pc)))
        unique, counts = np.unique(self.label, return_counts=True)
        print("Occurrences count of classes in trgt_new_dataset set: " + str(dict(zip(unique, counts))))
        unique, counts = np.unique(self.gt_label, return_counts=True)
        print("Occurrences count of classes in trgt_org_dataset set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.copy(self.pc[item])
        pc_patched = np.copy(self.pc_patches[item])
        label = np.copy(self.label[item])
        return pointcloud, pc_patched, label, item

    def __len__(self):
        return len(self.pc)

    def get_labels(self):
        return self.label


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

    def model_forward(self, pc, pc_patches, label, mode='eval'):
        with torch.no_grad():
            # Realistic Projection
            images = self.real_proj(pc)

        # Image features
        if mode == 'train':
            loss_ce_img, latent_all_img, logits_img = self.img_model(images, label, mode)
            loss_re, loss_ce_pc, loss_align = self.pc_model(pc_patches, latent_all_img, logits_img.detach(), label,
                                                            mode)
            loss_ce = loss_ce_img + loss_ce_pc
            return loss_re, loss_ce, loss_align
        else:
            logits_img, _ = self.img_model(images, label, mode)
            logits_pc, _, _ = self.pc_model(pc_patches, None, logits_img, label, mode)
            return logits_img, logits_pc

    def model_train(self, data):
        self.img_model.train()
        self.optimizer_img.zero_grad()
        self.pc_model.train()
        self.optimizer_pc.zero_grad()
        pc, pc_patch, label = data[0].cuda(), data[1].cuda(), data[2].cuda()
        loss_re, loss_ce, loss_align = self.model_forward(pc, pc_patch, label, mode='train')
        loss = loss_re + loss_ce + loss_align
        loss.backward()
        self.optimizer_img.step()
        self.optimizer_pc.step()
        return loss_ce.item(), loss_re.item(), loss_align.item()

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
            conf_mat = metrics.confusion_matrix(true, pred, labels=list(label_to_idx.values())).astype(int)
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
    parser.add_argument('--exp_name', type=str, default='spst', help='Name of the experiment')
    parser.add_argument('--in_path', type=str, default='./experiments', help='log folder path')
    parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
    parser.add_argument('--dataroot', type=str, default='..', metavar='N', help='data path')
    parser.add_argument('--src_dataset', type=str, default='modelnet', choices=['modelnet', 'shapenet', 'scannet'])
    parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
    parser.add_argument('--round', type=int, default=5, help='number of episode to train')
    parser.add_argument('--epochs', type=int, default=1, help='number of epoch to train')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of train batch per domain')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of test batch per domain')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--thd_high', type=float, default=0.725, help='thd high')
    parser.add_argument('--thd_low', type=float, default=0.2, help='thd low')
    parser.add_argument('--thd', type=float, default=0.8, help='thd')
    parser.add_argument('--epsilon', type=float, default=0.02, help='thd increase constant')

    args = parser.parse_args()

    if args.src_dataset == 'modelnet' and args.trgt_dataset == 'shapenet':
        img_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "m2s", "model_img_new.pt")
        pc_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "m2s", "model_pc_new.pt")
        args.out_path = os.path.join(os.path.abspath('.'), args.out_path, "m2s")
    elif args.src_dataset == 'modelnet' and args.trgt_dataset == 'scannet':
        img_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "m2r", "model_img_new.pt")
        pc_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "m2r", "model_pc_new.pt")
        args.out_path = os.path.join(os.path.abspath('.'), args.out_path, "m2r")
    elif args.src_dataset == 'shapenet' and args.trgt_dataset == 'modelnet':
        img_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "s2m", "model_img_new.pt")
        pc_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "s2m", "model_pc_new.pt")
        args.out_path = os.path.join(os.path.abspath('.'), args.out_path, "s2m")
    elif args.src_dataset == 'shapenet' and args.trgt_dataset == 'scannet':
        img_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "s2r", "model_img_new.pt")
        pc_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "s2r", "model_pc_new.pt")
        args.out_path = os.path.join(os.path.abspath('.'), args.out_path, "s2r")
    elif args.src_dataset == 'scannet' and args.trgt_dataset == 'modelnet':
        img_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "r2m", "model_img_new.pt")
        pc_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "r2m", "model_pc_new.pt")
        args.out_path = os.path.join(os.path.abspath('.'), args.out_path, "r2m")
    elif args.src_dataset == 'scannet' and args.trgt_dataset == 'shapenet':
        img_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "r2s", "model_img_new.pt")
        pc_model_dir = os.path.join(os.path.abspath('.'), args.in_path, "r2s", "model_pc_new.pt")
        args.out_path = os.path.join(os.path.abspath('.'), args.out_path, "r2s")
    else:
        img_model_dir = ''
        pc_model_dir = ''
        args.out_path = os.path.join(os.path.abspath('.'), args.out_path, "other")

    io = log.IOStream(args)
    io.cprint(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
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

    trgt_dataset = args.trgt_dataset
    data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}

    trgt_train_dataset = data_func[trgt_dataset](io, args.dataroot, 'train')
    trgt_test_dataset = data_func[trgt_dataset](io, args.dataroot, 'test')

    trgt_train_loader = torch.utils.data.DataLoader(trgt_train_dataset, batch_size=args.batch_size, num_workers=4,
                                                    drop_last=True)
    trgt_test_loader = torch.utils.data.DataLoader(trgt_test_dataset, batch_size=args.batch_size, num_workers=4)

    img_model, pc_model = load_mae_to_cpu()

    need_frozend_layers = ['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']
    for param in img_model.named_parameters():
        if param[0] in need_frozend_layers or (param[0].split('.')[0] == 'blocks' and int(param[0].split('.')[1]) <= 8):
            param[1].requires_grad = False
    for param in pc_model.named_parameters():
        if param[0].split('.')[0] == 'blocks' and int(param[0].split('.')[1]) <= 8:
            param[1].requires_grad = False

    img_model = img_model.to(device)
    pc_model = pc_model.to(device)
    optimizer_img = optim.Adam(filter(lambda p: p.requires_grad, img_model.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler_img = CosineAnnealingLR(optimizer_img, args.epochs)
    optimizer_pc = optim.Adam(filter(lambda p: p.requires_grad, pc_model.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler_pc = CosineAnnealingLR(optimizer_pc, args.epochs)
    trainer = Trainer(img_model, pc_model, optimizer_img, optimizer_pc, device=device)

    checkpoint_dir = args.out_path + '/' + args.exp_name
    checkpoint_io_img = checkpoints.CheckpointIO(checkpoint_dir, model=img_model)
    checkpoint_io_pc = checkpoints.CheckpointIO(checkpoint_dir, model=pc_model)
    try:
        load_dict_img = checkpoint_io_img.load(img_model_dir)
        load_dict_pc = checkpoint_io_pc.load(pc_model_dir)
    except FileExistsError:
        print('FileExistsError')
        load_dict_img = dict()
        load_dict_pc = dict()
    epoch_it = -1
    batch_it = -1
    src_metric_val_best = 0.0
    threshold = args.thd

    # trgt_metric_test = trainer.model_eval_img_pc(trgt_test_loader, io)

    for r in range(args.round):
        trgt_select_data = select_target_by_conf(io, trgt_train_loader, threshold, pc_model, img_model, trainer)
        trgt_new_data = DataLoadST(trgt_select_data)
        train_new_loader = torch.utils.data.DataLoader(trgt_new_data, sampler=ImbalancedDatasetSampler(trgt_new_data),
                                                       num_workers=4, batch_size=args.batch_size, drop_last=True)
        threshold += args.epsilon

        for i in range(args.epochs):
            trgt_loss_ce_sum = 0.0
            trgt_loss_re_sum = 0.0
            trgt_loss_align_sum = 0.0
            it = 0
            epoch_it += 1
            for trgt_batch in tqdm(train_new_loader):
                batch_it += 1
                it += 1
                loss_ce, loss_re, loss_align = trainer.model_train(trgt_batch)
                trgt_loss_ce_sum += loss_ce
                trgt_loss_re_sum += loss_re
                trgt_loss_align_sum += loss_align
            scheduler_img.step()
            scheduler_pc.step()

            io.cprint('lr_rate=%.9f' % scheduler_img.get_last_lr()[0])
            io.cprint('Train -TRT - [Epoch %02d], loss_re=%.4f, loss_ce=%.4f, loss_align=%.4f' %
                      (epoch_it, trgt_loss_re_sum / it, trgt_loss_ce_sum / it, trgt_loss_align_sum / it))

            # # Run validation
            # src_metric_val = trainer.model_eval(src_val_loader, io)
            trgt_metric_test, trgt_conf_mat = trainer.model_eval_img_pc(trgt_test_loader, io)
            io.cprint("Test confusion matrix:")
            io.cprint('\n' + str(trgt_conf_mat))
            checkpoint_io_img.save('model_img_spst.pt', epoch_it=epoch_it, batch_it=batch_it)
            checkpoint_io_pc.save('model_pc_spst.pt', epoch_it=epoch_it, batch_it=batch_it)
