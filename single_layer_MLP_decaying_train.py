from __future__ import print_function
import os
import logging
from logging import handlers
import random
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import math

import random
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wsd_decay_steps', default=0.0, type=float)
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--save', default="",help='path to save trained nets')
parser.add_argument('--save_epoch', default=2, type=float, help='save every save_epochs')
parser.add_argument('--seed', default=0, type=int, help='seed for random num generator')
parser.add_argument('--resume', default=None, type=str, help='path to checkpoint to resume from')
parser.add_argument('--continue_after_wsd', action='store_true', help='finish WSD decay then continue training to full epochs')
parser.add_argument('--save_milestones', default="", type=str, help='comma-separated epoch progress values to save ckpts at (e.g., "0.5,1.0,2.0"). If set, overrides periodic save_epoch for pretrain (wsd_decay_steps==0).')
parser.add_argument('--run_tag', default="", type=str, help='optional tag appended to save directory name to differentiate runs')
parser.add_argument('--post_wsd_lr', default=1e-6, type=float, help='learning rate to use after WSD decay if continuing training')
args = parser.parse_args()




random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
    #cudnn.benchmark = True

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
])

# 加载数据集
kwargs = {'num_workers': 8, 'pin_memory': True}
train_dataset = datasets.MNIST(root='/chenyupeng/old_files/yupeng_landscape/landscape_LLM/MNIST_MLP/data2', 
                              train=True, 
                              transform=transform,
                              download=True)
test_dataset = datasets.MNIST(root='/chenyupeng/old_files/yupeng_landscape/landscape_LLM/MNIST_MLP/data2', 
                             train=False, 
                             transform=transform)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, 
                         batch_size=args.batch_size, 
                         shuffle=True,**kwargs)
test_loader = DataLoader(dataset=test_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False,**kwargs)
input_size = 784  # 28x28
hidden_size =  args.hidden_size
output_size = 10  # 0-9的数字

# 定义两层MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size,bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size,bias=False)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

############### 设置WSD schedular #######################################
def wsd_schedule(
    n_iterations,
    final_lr_factor=0.0,
    n_warmup=1000,
    init_div_factor=100,
    fract_decay=0.1,
    decay_type="linear",
    wsd_decay_steps=None,
):
    """Warmup, hold, and decay schedule.
    Args:
        n_iterations: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        warmup_fract: fraction of iterations used for warmup
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
        decay_type: type of decay schedule to use
        wsd_decay_steps: fixed number of steps for decay (only used when fract_decay=0)
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    # If fract_decay is 0 and wsd_decay_steps is provided, use fixed steps for decay
    if fract_decay == 0 and wsd_decay_steps is not None:
        n_anneal_steps = wsd_decay_steps
    else:
        n_anneal_steps = int(fract_decay * n_iterations)
    
    n_hold = n_iterations - n_anneal_steps

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < n_iterations:
            if decay_type == "linear":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
            elif decay_type == "exp":
                return final_lr_factor ** ((step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                return (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "miror_cosine":
                cosine_value = (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
                linear_value = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
                return linear_value * 2 - cosine_value
            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - ((step - n_hold) / n_anneal_steps) ** 2
                )

            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - math.sqrt((step - n_hold) / n_anneal_steps)
                )

            else:
                raise ValueError(
                    f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp']"
                )

        else:
            return final_lr_factor

    return schedule
"""
Yupengh 7.18：
通过wsd_decay_steps 来控制是否使用wsd， 
我觉得默认wsd_decay_steps = 0，则不启用wsd， 
如果非0就在wsd_decay_steps 步数内进行learning rate decaying的训练。

更进一步，我把args.wsd_decay_steps 默认为x.x "epochs"， 比如args.wsd_decay_steps = 0.2 就是默认decaying 0.2 epochs
"""
wsd_decay_steps = int(len(train_loader)*args.wsd_decay_steps)
lambda_schedule = wsd_schedule(
                    n_iterations=len(train_loader)* args.epochs if wsd_decay_steps==0 else wsd_decay_steps,
                    n_warmup=0,
                    fract_decay=0,
                    init_div_factor=1e2,
                    final_lr_factor=0,  # should be 0 here
                    decay_type="linear",
                    wsd_decay_steps=wsd_decay_steps if hasattr(args, 'wsd_decay_steps') else None,
                )
############### 

if args.save_epoch <1:
    save_interval = int(len(train_loader)*args.save_epoch)
else:
    save_interval = int(args.save_epoch)


model = MLP(input_size, hidden_size, output_size)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,momentum=args.momentum)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_schedule)
# 将 wsd 步数体现在保存目录名中，便于区分
save_name = (
    f"bs{args.batch_size}_lr{args.learning_rate}_e{args.epochs}_"
    f"hidden{args.hidden_size}_SGD_m{args.momentum:.1f}_wsd{args.wsd_decay_steps}_seed{args.seed}"
)
if isinstance(args.run_tag, str) and args.run_tag.strip() != "":
    save_name = f"{save_name}_{args.run_tag.strip()}"



save_folder = args.save+save_name
import os

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 统一日志输出到控制台与文件
def setup_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("train")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

logger = setup_logger(save_folder)
logger.info(f"保存目录: {save_folder}")
logger.info(
    f"配置: bs={args.batch_size}, lr={args.learning_rate}, epochs={args.epochs}, "
    f"hidden={args.hidden_size}, momentum={args.momentum}, weight_decay={args.weight_decay}, "
    f"wsd_decay_steps={args.wsd_decay_steps}, post_wsd_lr={args.post_wsd_lr}, save_epoch={args.save_epoch}, seed={args.seed}"
)
logger.info(str(model))

"""
给一个loss 计算的函数
"""
def eval_loss(datalodaer, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        running_loss = 0.0
        for images, labels in datalodaer:
            labels = labels.cuda()
            images = images.reshape(-1, 28 * 28).cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

    return running_loss / len(train_loader), 100 * correct / total


def save_checkpoint(state: dict, base_dir: str, epoch_progress: float, global_step: int, tag: str = "") -> str:
    ckpt_root = os.path.join(base_dir, "ckpts")
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_dir_name = f"epoch_{epoch_progress:.3f}_step_{global_step:06d}"
    if tag:
        ckpt_dir_name += f"_{tag}"
    ckpt_dir = os.path.join(ckpt_root, ckpt_dir_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
    torch.save(state, ckpt_path)
    latest_path = os.path.join(ckpt_root, "latest.pt")
    try:
        torch.save(state, latest_path)
    except Exception:
        pass
    return ckpt_path


# 训练模型
train_losses = []
train_accuracies = []
test_accuracies = []

start_epoch = 0
# 当从检查点恢复时，记录恢复时的精确 epoch 进度（可为小数）
resume_epoch_progress = None
if args.resume is not None:
    if os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] 
        # 记录恢复时的精确进度（若无则退化为整数epoch）
        resume_epoch_progress = checkpoint.get('epoch_progress', float(start_epoch))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    else:
        logger.warning(f"=> no checkpoint found at '{args.resume}'")


model = model.cuda()
num_epochs = args.epochs
stop_training = False

global_step = 0
milestones = []
if isinstance(args.save_milestones, str) and args.save_milestones.strip() != "":
    try:
        milestones = sorted({float(x.strip()) for x in args.save_milestones.split(',') if x.strip() != ''})
    except Exception:
        milestones = []
milestones_done = set()
prev_progress = resume_epoch_progress if 'resume_epoch_progress' in globals() and resume_epoch_progress is not None else float(int(start_epoch))
# 用于在任何保存频率下都能准确切换到后续固定学习率
wsd_base_progress = resume_epoch_progress if resume_epoch_progress is not None else float(int(start_epoch))
post_wsd_activated = False
# 记录已执行的调度步数（与scheduler保持一致）；用于精准判断WSD是否完成
wsd_steps_done = 0
wsd_active = (wsd_decay_steps > 0)
for epoch in range(int(start_epoch),args.epochs):
    if stop_training:  # 检查是否需要终止训练
        break

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_idx = 0
    for images, labels in train_loader:
        # 当前训练进度（可为小数）
        current_progress = epoch + (batch_idx / len(train_loader))
        # 将图像展平
        labels = labels.cuda()
        images = images.reshape(-1, 28 * 28).cuda()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)


        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        if wsd_active:
            scheduler.step()
            wsd_steps_done += 1
            # 当调度步数达到设定的wsd_decay_steps时，立即切换到固定学习率
            if args.continue_after_wsd and not post_wsd_activated and wsd_steps_done >= wsd_decay_steps:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.post_wsd_lr
                logger.info(f"WSD decay完成（步数判定），切换至后续固定学习率 post_wsd_lr={args.post_wsd_lr}")
                args.wsd_decay_steps = 0
                wsd_active = False
                post_wsd_activated = True
            # 保险：若因边界/取整导致 lr 已为 0，也立即切换
            if (
                args.continue_after_wsd
                and not post_wsd_activated
                and optimizer.param_groups[0]['lr'] <= 0.0
            ):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.post_wsd_lr
                logger.info(f"检测到学习率为 0，已立即切换至后续固定学习率 post_wsd_lr={args.post_wsd_lr}")
                args.wsd_decay_steps = 0
                wsd_active = False
                post_wsd_activated = True
        
        # 若选择在decay完成后继续训练，则在达到wsd步数后切换为固定学习率，并停止scheduler
        if (
            wsd_active
            and args.continue_after_wsd
            and not post_wsd_activated
        ):
            total_steps_trained = (epoch + (batch_idx / len(train_loader))) - wsd_base_progress
            if total_steps_trained >= args.wsd_decay_steps:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.post_wsd_lr
                logger.info(f"WSD decay完成（通用检测），切换至后续固定学习率 post_wsd_lr={args.post_wsd_lr}")
                args.wsd_decay_steps = 0
                wsd_active = False
                post_wsd_activated = True

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        batch_idx +=1

        global_step += 1

        # 仅在预训练阶段（wsd_decay_steps==0）按里程碑保存；若未设置里程碑，回退到周期性保存
        if args.wsd_decay_steps == 0:
            if milestones:
                to_save = []
                for m in milestones:
                    if m not in milestones_done and prev_progress < m <= current_progress:
                        to_save.append(m)
                for m in to_save:
                    training_loss = eval_loss(train_loader, model)
                    current_lr = optimizer.param_groups[0]['lr']
                    state = {
                        'epoch_progress': m,
                        'epoch': int(m),
                        'global_step': global_step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': training_loss[0],
                        'accuracy': training_loss[1],
                        'learning_rate': current_lr,
                    }
                    ckpt_path = save_checkpoint(state, save_folder, m, global_step)
                    logger.info(
                        f"按里程碑保存检查点：{os.path.basename(ckpt_path)} | epoch={m:.3f} | "
                        f"loss={training_loss[0]:.6f} | acc={training_loss[1]:.2f}% | lr={current_lr:.8e}"
                    )
                    milestones_done.add(m)
            else:
                if args.save_epoch < 1 and save_interval > 0 and batch_idx % save_interval == 0:
                    fractional = (batch_idx / save_interval) * args.save_epoch
                    save_epoch_progress = epoch + fractional
                    training_loss = eval_loss(train_loader, model)
                    current_lr = optimizer.param_groups[0]['lr']
                    state = {
                        'epoch_progress': save_epoch_progress,
                        'epoch': epoch,
                        'global_step': global_step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': training_loss[0],
                        'accuracy': training_loss[1],
                        'learning_rate': current_lr,
                    }
                    ckpt_path = save_checkpoint(state, save_folder, save_epoch_progress, global_step)
                    logger.info(
                        f"保存检查点：{os.path.basename(ckpt_path)} | epoch={save_epoch_progress:.3f} | "
                        f"loss={training_loss[0]:.6f} | acc={training_loss[1]:.2f}% | lr={current_lr:.8e}"
                    )


        if wsd_active and args.save_epoch < 1 and batch_idx % save_interval == 0:
            fractional = (batch_idx / save_interval) * args.save_epoch
            save_epoch_progress = epoch + fractional
            training_loss = eval_loss(train_loader, model)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"WSD decaying | epoch={save_epoch_progress:.3f} | loss={training_loss[0]:.6f} | "
                f"acc={training_loss[1]:.2f}% | lr={current_lr:.8e}"
            )
            state = {
                'epoch_progress': save_epoch_progress,
                'epoch': epoch,
                'global_step': global_step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': training_loss[0],
                'accuracy': training_loss[1],
                'learning_rate': current_lr,
            }
            ckpt_path = save_checkpoint(state, save_folder, save_epoch_progress, global_step, tag="wsd")
            logger.info(f"保存WSD检查点：{os.path.basename(ckpt_path)}")
            # 如果设置仅进行 WSD 训练，在达到 WSD 步数后停止；否则继续完整 epochs
            # 以恢复点的精确进度为基准来计算相对训练长度
            base_progress = resume_epoch_progress if resume_epoch_progress is not None else float(int(start_epoch))
            total_steps_trained = save_epoch_progress - base_progress
            if wsd_active and total_steps_trained >= args.wsd_decay_steps:
                if not args.continue_after_wsd:
                    logger.info(
                        f"WSD decay完成，已训练{total_steps_trained:.3f}个epoch，达到指定的{args.wsd_decay_steps}个epoch，停止训练"
                    )
                    stop_training = True
                    break
                else:
                    logger.info(
                        f"WSD decay完成，已训练{total_steps_trained:.3f}个epoch，继续训练直至 {args.epochs} 个epoch"
                    )
                    # 达到衰减步后，不再 step scheduler，并将学习率切换为固定 post_wsd_lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.post_wsd_lr
                    logger.info(f"已切换至后续固定学习率 post_wsd_lr={args.post_wsd_lr}")
                    args.wsd_decay_steps = 0
                    wsd_active = False
                    post_wsd_activated = True

        
    # 计算训练集上的准确率和损失
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # 在测试集上评估
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            labels = labels.cuda()
            images = images.reshape(-1, 28 * 28).cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)

    current_lr = optimizer.param_groups[0]['lr']
    logger.info(
        f"Epoch [{epoch+1}/{num_epochs}] | Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | "
        f"LR: {current_lr:.8e}"
    )

    if args.save_epoch >= 1 and ((epoch + 1) % int(args.save_epoch) == 0):
        save_epoch_progress = epoch + 1
        training_loss = eval_loss(train_loader, model)
        state = {
            'epoch_progress': save_epoch_progress,
            'epoch': epoch,
            'global_step': global_step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': training_loss[0],
            'accuracy': training_loss[1],
            'learning_rate': current_lr,
        }
        ckpt_path = save_checkpoint(state, save_folder, save_epoch_progress, global_step, tag="epoch")
        logger.info(f"按整epoch间隔保存检查点：{os.path.basename(ckpt_path)}")


"""
Yupeng 7.19:
这里加入判断收敛的点是否到达稳定线的机制。

训练结束之后继续训练10个step， 输出10个step的loss， 判断loss是否有下降。
"""
# 保存最终的模型
final_training_loss = eval_loss(train_loader, model)
final_state = {
    'epoch': epoch + 1,  # 最终的epoch
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss': final_training_loss[0],
    'accuracy': final_training_loss[1],
    'learning_rate': optimizer.param_groups[0]['lr']
}
torch.save(final_state, save_folder+f"/model_para_final.pt")
logger.info(
    f"保存最终模型：loss: {final_training_loss[0]:.6f}, accuracy: {final_training_loss[1]:.2f}%"
)

# 如果使用了WSD decay，进行额外的稳定性检查
import copy
model_to_save = copy.deepcopy(model)
loss_check_pipe = []
if args.wsd_decay_steps > 0:
    # 保存原始学习率
    original_lr = optimizer.param_groups[0]['lr']
    # 设置一个小的学习率进行稳定性测试
    for param_group in optimizer.param_groups:
        param_group['lr'] = 5e-3

    original_loss = eval_loss(train_loader, model)[0]
    logger.info(f"开始稳定性测试，原始loss: {original_loss}")

    batch_idx = 0
    for images, labels in train_loader:
        # 将图像展平
        labels = labels.cuda()
        images = images.reshape(-1, 28 * 28).cuda()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        batch_idx +=1
        prev_progress = current_progress
        
        loss_eval = eval_loss(train_loader, model)[0]
        loss_check_pipe.append(loss_eval)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"稳定性测试 step {batch_idx} with lr {current_lr}, loss is {loss_eval}")

        if batch_idx>10:
            break

    # 判断模型是否稳定（loss是否上升）
    if loss_eval > original_loss:
        stability = "unstable"  # 不稳定，loss上升
    else:
        stability = "stable"    # 稳定，loss下降或不变
        
    # 保存稳定性测试结果
    stability_state = {
        'epoch': epoch + 1,
        'state_dict': model_to_save.state_dict(),
        'check_loss': loss_check_pipe,
        'original_loss': original_loss,
        'final_check_loss': loss_eval,
        'stability': stability
    }
    torch.save(stability_state, save_folder+f"/model_para_stability_check_{stability}.pt")
    logger.info(
        f"稳定性测试完成，模型{'稳定' if stability == 'stable' else '不稳定'}，原始loss: {original_loss}，最终loss: {loss_eval}"
    )
    
    # 恢复原始学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = original_lr

#
## 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig(save_folder+f"/training_loss.png")
plt.show()

# 保存模型
#torch.save(model.state_dict(), 'mnist_mlp.pth')
