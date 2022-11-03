import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
import copy
import wandb
wandb.login() # 각자 WandB 로그인 하기
# 9eee70600a60d9d41eecef494a78a696bd12d252

# 성우 0.0001 & 32
# 원국 0.0005 & 32
# 진녕 0.0001 & 64
# 기용 0.0005 & 64

# 🐝 initialise a wandb run
wandb.init(
    project="Effi_v2_l_aug3", # 프로젝트 이름 "모델_버전_성명"
    config = {
    "lr": 0.0005,
    "epochs": 200,
    "batch_size": 64,
    "optimizer" : "Adam",
    "resize" : [384, 384],
    "criterion" : 'weight_cross_entropy'
    }
 )

# Copy your config 
config = wandb.config

'''class weight_cross_entropy(_WeightedLoss):

    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super(weight_cross_entropy, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.num_ins = [
            2745, 2050, 415,
            3660, 4085, 545,
            549, 410, 83,
            732, 817, 109,
            549, 410, 83,
            732, 817, 109]
        self.weights = [1 - (x/(sum(self.num_ins))) for x in self.num_ins]
        self.class_weights = torch.FloatTensor(self.weights).cuda()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, weight=self.class_weights,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

    복사해서 loss 파일 제일 아래에 넣기
    넣은 후에 _criterion_entrypoints 사전 목록에 아래 추가

    'weight_cross_entropy' : weight_cross_entropy
    
    아래 모듈 loss에 import

    from torch import Tensor
    from typing import Callable, Optional

    '''


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18
    
    #dataset_aug3 = copy.deepcopy(dataset)
  
    # -- preprocessing --data_set
    transform_module = getattr(import_module("dataset"), args.preprocessing)  # default: preprocessing
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)
    
    dataset_aug = copy.deepcopy(dataset)
    
    # augmentation 적용
    transform_module_aug = getattr(import_module("dataset"), args.RealAugmentation)  # default: RealAugmentation
    transform_aug = transform_module_aug(
        resize=args.resize,
        mean=dataset_aug.mean,
        std=dataset_aug.std,
    )
    dataset_aug.set_transform(transform_aug)

    # augmentation3 적용
    '''transform_module_aug = getattr(import_module("dataset"), args.RealAugmentation_3)  # default: RealAugmentation
    transform_aug3 = transform_module_aug(
        resize=args.resize,
        mean=dataset_aug.mean,
        std=dataset_aug.std,
    )
    dataset_aug3.set_transform(transform_aug3)
    
    train_set_aug3,val_set3 = dataset_aug3.split_dataset()'''
    
    #torch.manual_seed(42)
    train_set,val_set = dataset.split_dataset() 
    
    # augmentation_set 생성
    torch.manual_seed(42)
    train_set_aug,val_set2 = dataset_aug.split_dataset() 
    

    # train_set + augmentaion_set
    # train_set = ConcatDataset([train_set,train_set_aug])
    train_set = train_set + train_set_aug #+ train_set_aug3
    
    # # -- data_loader
    # train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )
        
    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4 # weight_decay 너무 튀지 않게??
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf

    early_stop = 0
    breaker = False
    early_stop_arg = args.early_stop

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                early_stop = 0
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()
            wandb.log({"val_loss": val_loss,"val_acc": val_acc})
            print(f'{early_stop_arg-early_stop} Epoch left until early stopping..')                
            if val_acc < best_val_acc:                
                if early_stop == early_stop_arg:
                    breaker = True
                    print(f'--------epoch {epoch} early stopping--------')
                    print(f'--------epoch {epoch} early stopping--------')                                       
                    break
                early_stop += 1

        if breaker == True:
            break        

            # Optional
            wandb.watch(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--preprocessing', type=str, default='Basepreprocessing', help='data augmentation type (default: Basepreprocessing)')
    parser.add_argument('--RealAugmentation', type=str, default='RealAugmentation', help='data augmentation type (default: RealAugmentation)')
    parser.add_argument('--RealAugmentation_3', type=str, default='RealAugmentation_3', help='data augmentation type (default: RealAugmentation_3)')
    parser.add_argument("--resize", nargs="+", type=list, default=config.resize, help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=250, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='vit_base_patch16_384', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default=config.optimizer, help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default=config.criterion, help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--early_stop', type=int, default=10, help='number of early_stop (default : 10')

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
    wandb.finish()