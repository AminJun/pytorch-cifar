'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import time

import os
import argparse

from torchvision.datasets import CIFAR10
from torch.utils.data.dataset import Subset, Dataset

from models import *


# from utils import progress_bar


class IDCifar(CIFAR10):
    def __getitem__(self, item):
        return item, super(IDCifar, self).__getitem__(item)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--fast', default=1, type=int, help='Do it fast or slow')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--change_lr', default=0, type=int)
args = parser.parse_args()

if args.fast:
    print("Fast method")
else:
    print("Boring method")
start_time = time.time()
print("Start time:\t{}".format(start_time))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = IDCifar(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)


class BadAssLoader(Subset):
    def __init__(self):
        super().__init__(trainset, [])

    def reset(self):
        self.indices = []

    def append(self, idx):
        self.indices.append(idx)


bad_ass_set = BadAssLoader()

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)


# Training
def train(epoch, fast=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    my_loader = trainloader
    if fast:
        my_loader = torch.utils.data.DataLoader(bad_ass_set, batch_size=128, shuffle=True, num_workers=2)
    else:
        bad_ass_set.reset()
    for batch_idx, (ids, (inputs, targets)) in enumerate(my_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        eq_ = predicted.eq(targets)
        if not fast:
            bad_ass_set.append(ids[eq_.eq(0)])
        correct += eq_.sum().item()

        # progress_bar(batch_idx, len(my_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) [Train]'
        #             % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print("Train", epoch, 100. * correct / total)
    if not fast:
        bad_ass_set.indices = torch.cat(bad_ass_set.indices).cpu().numpy()


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) [Test]'
            #             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    print("Test", epoch, 100. * correct / total)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt{}_{}.pth'.format(args.fast, args.epochs))
        best_acc = acc


lr = args.lr


def update_lr():
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


nst = args.epochs
old_lr = lr

for epoch in range(start_epoch, start_epoch + nst, 2):
    # if epoch >= (nst / 4) * 3:
    #     lr = 0.001
    # elif epoch >= (nst / 2):
    #     lr = 0.01
    # else:
    #     lr = 0.1
    # update_lr()
    if args.fast and args.change_lr:
        lr = old_lr

    train(epoch)
    test(epoch)
    print('Time:', epoch, time.time() - start_time)

    if args.fast and args.change_lr:
        old_lr = lr
        lr *= float(len(bad_ass_set)) / float(len(trainset))
        update_lr()

    train(epoch + 1, fast=(args.fast == 1))
    test(epoch + 1)
    print('Time:', epoch, time.time() - start_time)

end_time = time.time()
print("End time:\t{}".format(end_time))
print("Total time:\t{}".format(end_time - start_time))
