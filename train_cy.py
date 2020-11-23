import configs
import numpy as np
import time
import os
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataloader import efficientdet_dataset_collate, EfficientdetDataset
from nets.efficientdet import EfficientDetBackbone
from nets.efficientdet_training import Generator, FocalLoss
from tqdm import tqdm


class train_model(object):
    def __init__(self):
        cfg = configs.parse_config()
        self.input_shape = (cfg.input_sizes[cfg.phi], cfg.input_sizes[cfg.phi])
        self.lr_first = cfg.learning_rate_first_stage
        self.Batch_size_first = cfg.Batch_size_first_stage
        self.Init_Epoch = cfg.Init_Epoch
        self.Freeze_Epoch = cfg.Freeze_Epoch
        self.opt_weight_decay = cfg.opt_weight_decay
        self.CosineAnnealingLR_T_max = cfg.CosineAnnealingLR_T_max
        self.CosineAnnealingLR_eta_min = cfg.CosineAnnealingLR_eta_min
        self.StepLR_step_size = cfg.StepLR_step_size
        self.StepLR_gamma = cfg.StepLR_gamma
        self.num_workers = cfg.num_workers
        self.Save_num_epoch = cfg.Save_num_epoch
        self.lr_second = cfg.learning_rate_second_stage
        self.Batch_size_second = cfg.Batch_size_second_stage
        self.Unfreeze_Epoch = cfg.Unfreeze_Epoch

        # TODO:tricks的使用设置
        self.Cosine_lr, self.mosaic = cfg.Cosine_lr, cfg.use_mosaic
        self.Cuda = torch.cuda.is_available()
        self.smoooth_label = cfg.smoooth_label
        self.Use_Data_Loader, self.annotation_path = cfg.Use_Data_Loader, cfg.train_annotation_path
        # TODO:获得类
        self.classes_path = cfg.classes_path
        self.class_names = self.get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        # TODO:创建模型
        self.model = EfficientDetBackbone(self.num_classes, cfg.phi)
        pretrain_weight_name = os.listdir(cfg.pretrain_dir)
        index = [item for item in pretrain_weight_name if str(cfg.phi) in item][0]
        # 加快模型训练的效率
        print('Loading pretrain_weights into state dict...')
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(cfg.pretrain_dir + index)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print('Finished!')
        self.net = self.model.train()
        if self.Cuda:
            self.net = torch.nn.DataParallel(self.model)  # 多GPU进行训练，但这个设置有问题
            cudnn.benchmark = True
            self.net = self.net.cuda()

        # TODO:建立loss函数
        self.efficient_loss = FocalLoss()
        # cfg.val_split用于验证，1-cfg.val_split用于训练
        val_split = cfg.val_split
        with open(self.annotation_path) as f:
            self.lines = f.readlines()
        np.random.seed(101)
        np.random.shuffle(self.lines)
        np.random.seed(None)
        self.num_val = int(len(self.lines) * val_split)
        self.num_train = len(self.lines) - self.num_val

        self.train_first_stage()
        self.train_second_stage()

    def get_classes(self, classes_path):
        """
        loads the classes name
        :param classes_path:
        :return:
        """
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def fit_one_epoch(self, net, model, optimizer, focal_loss, epoch, epoch_size, epoch_size_val, gen, genval, Epoch,
                      cuda):
        """
        :param net: 网络
        :param yolo_losses: yolo损失类
        :param epoch: 第几个epoch
        :param epoch_size: train中每个epoch中有多少个数据
        :param epoch_size_val: valid中每个epoch里面的数据
        :param gen: train DataLoader
        :param genval: valid DataLoader
        :param Epoch: 截至epoch
        :param cuda:
        :return:
        """
        total_r_loss = 0
        total_c_loss = 0
        total_loss = 0
        val_loss = 0
        start_time = time.time()
        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_size:
                    break
                images, targets = batch[0], batch[1]
                with torch.no_grad():
                    if cuda:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                    else:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

                optimizer.zero_grad()
                # regression shape is (batch_size, H*W*9+……, 4), classification shape is (batch_size, H*W*9+……, num_classes)
                _, regression, classification, anchors = net(images)  # anchors先验框
                loss, c_loss, r_loss = focal_loss(classification, regression, anchors, targets, cuda=cuda)
                loss.backward()
                optimizer.step()

                total_loss += loss.detach().item()
                total_r_loss += r_loss.detach().item()
                total_c_loss += c_loss.detach().item()
                waste_time = time.time() - start_time

                pbar.set_postfix(**{'Conf Loss': total_c_loss / (iteration + 1),
                                    'Regression Loss': total_r_loss / (iteration + 1),
                                    'lr': self.get_lr(optimizer),
                                    'step/s': waste_time})
                pbar.update(1)
                start_time = time.time()

        print('Start Validation')
        with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(genval):
                if iteration >= epoch_size_val:
                    break
                images_val, targets_val = batch[0], batch[1]

                with torch.no_grad():
                    if cuda:
                        images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                        targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in
                                       targets_val]
                    else:
                        images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                        targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                    optimizer.zero_grad()
                    _, regression, classification, anchors = net(images_val)
                    loss, c_loss, r_loss = focal_loss(classification, regression, anchors, targets_val, cuda=cuda)
                    val_loss += loss.detach().item()

                pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
                pbar.update(1)

        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
        if (epoch + 1) % self.Save_num_epoch == 0:
            print('Saving state, iter:', str(epoch + 1))
            torch.save(model.state_dict(), 'model_weight/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
                (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    def train_first_stage(self):
        """
        主干特征提取网络特征通用，冻结训练可以加快训练速度
        也可以在训练初期防止权值被破坏。
        Init_Epoch为起始世代
        Freeze_Epoch为冻结训练的世代
        Epoch总训练世代
        提示OOM或者显存不足请调小Batch_size
        :return:
        """
        optimizer_stage1 = optim.Adam(self.net.parameters(), self.lr_first, weight_decay=self.opt_weight_decay)
        if self.Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_stage1, T_max=self.CosineAnnealingLR_T_max, eta_min=self.CosineAnnealingLR_eta_min)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer_stage1, step_size=self.StepLR_step_size, gamma=self.StepLR_gamma)

        if self.Use_Data_Loader:
            train_dataset = EfficientdetDataset(self.lines[:self.num_train], (self.input_shape[0], self.input_shape[1]))
            val_dataset = EfficientdetDataset(self.lines[self.num_train:], (self.input_shape[0], self.input_shape[1]))
            gen = DataLoader(train_dataset, batch_size=self.Batch_size_first, num_workers=self.num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=efficientdet_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=self.Batch_size_first, num_workers=self.num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=efficientdet_dataset_collate)
        else:
            gen = Generator(self.Batch_size_first, self.lines[:self.num_train],
                            (self.input_shape[0], self.input_shape[1])).generate()
            gen_val = Generator(self.Batch_size_first, self.lines[self.num_train:],
                                (self.input_shape[0], self.input_shape[1])).generate()

        epoch_size = max(1, self.num_train // self.Batch_size_first)
        epoch_size_val = self.num_val // self.Batch_size_first
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in self.model.backbone_net.parameters():
            param.requires_grad = False

        for epoch in range(self.Init_Epoch, self.Freeze_Epoch):
            self.fit_one_epoch(self.net, self.model, optimizer_stage1, self.efficient_loss, epoch, epoch_size,
                               epoch_size_val, gen, gen_val, self.Freeze_Epoch, self.Cuda)
            lr_scheduler.step()

    def train_second_stage(self):
        """
        整个网络的参数一起更新
        :return:
        """
        optimizer_stage2 = optim.Adam(self.net.parameters(), self.lr_second, weight_decay=self.opt_weight_decay)
        if self.Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_stage2, T_max=self.CosineAnnealingLR_T_max, eta_min=self.CosineAnnealingLR_eta_min)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer_stage2, step_size=self.StepLR_step_size, gamma=self.StepLR_gamma)

        if self.Use_Data_Loader:
            train_dataset = EfficientdetDataset(self.lines[:self.num_train], (self.input_shape[0], self.input_shape[1]))
            val_dataset = EfficientdetDataset(self.lines[self.num_train:], (self.input_shape[0], self.input_shape[1]))
            gen = DataLoader(train_dataset, batch_size=self.Batch_size_first, num_workers=self.num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=efficientdet_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=self.Batch_size_first, num_workers=self.num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=efficientdet_dataset_collate)
        else:
            gen = Generator(self.Batch_size_first, self.lines[:self.num_train],
                            (self.input_shape[0], self.input_shape[1])).generate()
            gen_val = Generator(self.Batch_size_first, self.lines[self.num_train:],
                                (self.input_shape[0], self.input_shape[1])).generate()

        epoch_size = max(1, self.num_train // self.Batch_size_second)
        epoch_size_val = self.num_val // self.Batch_size_second
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in self.model.backbone_net.parameters():
            param.requires_grad = True

        for epoch in range(self.Freeze_Epoch, self.Unfreeze_Epoch):
            self.fit_one_epoch(self.net, self.model, optimizer_stage2, self.efficient_loss, epoch, epoch_size,
                               epoch_size_val, gen, gen_val, self.Unfreeze_Epoch, self.Cuda)
            lr_scheduler.step()


if __name__ == "__main__":
    train_model()
