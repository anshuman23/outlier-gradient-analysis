# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from typing import Optional, Sized, Iterator

from tqdm import tqdm

from torch import autograd

from pyod.models.iforest import IForest

from sklearn.manifold import TSNE
from sklearn.random_projection import SparseRandomProjection

from cleanlab.rank import get_label_quality_scores


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--is_human', action='store_true', default=False)
parser.add_argument('--baseline_method', type = str, default = 'self_confidence', help='baseline method type: self_confidence / normalized_margin / confidence_weighted_entropy')

# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(epoch, train_loader, model, optimizer):
    train_total=0
    train_correct=0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
       
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
       
        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        # prec = 0.0
        train_total+=1
        train_correct+=prec
        loss = F.cross_entropy(logits, labels, reduce = True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))


    train_acc=float(train_correct)/float(train_total)
    return train_acc

# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()    # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc


def get_probs_preds(train_loader, model):
    model.eval()    # Change model to 'eval' mode.
    correct = 0
    total = 0
    probs = None
    preds = []
    for images, labels, _ in train_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)

        if probs is None:
            probs = outputs.cpu().detach().numpy()
        else:
            probs = np.concatenate((probs, outputs.cpu().detach().numpy()), axis=0)

        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()

        preds += pred.cpu().numpy().tolist()

    acc = 100*float(correct)/float(total)

    return probs, np.array(preds)



class RemovalSampler(RandomSampler):
    """Sample elements randomly. 
    Not everything from RandomSampler is implemented.

    Args:
        data_source (Dataset): dataset to sample from
        forbidden  (Optional[list]): list of forbidden numbers
    """
    data_source: Sized
    forbidden: Optional[list]

    def __init__(self, data_source: Sized, forbidden: Optional[list] = []) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.forbidden = forbidden
        self.refill()

    def remove(self, new_forbidden):
        # Remove numbers from the available indices
        for num in new_forbidden:
            if not (num in self.forbidden):
                self.forbidden.append(num)
        self._remove(new_forbidden)

    def _remove(self, to_remove):
        # Remove numbers just for this epoch
        for num in to_remove:
            if num in self.idx:
                self.idx.remove(num)

        self._num_samples = len(self.idx)

    def refill(self):
        # Refill the indices after iterating through the entire DataLoader
        self.idx = list(range(len(self.data_source)))
        self._remove(self.forbidden)

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples // 32):
            batch = random.sample(self.idx, 32)
            self._remove(batch)
            yield from batch
        yield from random.sample(self.idx, self.num_samples % 32)
        self.refill()


def sample_remove_dataloader(ds, idx_list, bs=128):
    sampler = RemovalSampler(ds, forbidden=idx_list)

    dl = DataLoader(dataset = train_dataset,
                                batch_size = bs,
                                num_workers=args.num_workers,
                                shuffle=False, #True originally
                                drop_last = False,
                                sampler=sampler)
    return dl


def plot_grad(grads, od_idxs, figname):
    plt.clf()

    for i,g in enumerate(grads):
        if i in od_idxs:
            plt.scatter(g[0], g[1], color='tab:purple', marker='X', s=75, edgecolor='black')
            continue

        plt.scatter(g[0], g[1], color='tab:purple', marker='o')

    plt.xlabel('Gradient Value (Projection 1)')
    plt.ylabel('Gradient Value (Projection 2)')
    o_point = Line2D([0], [0], label='Predicted Non-outlier', marker='o', markersize=10, markerfacecolor='tab:purple', markeredgecolor='tab:purple', linestyle='')
    x_point = Line2D([0], [0], label='Predicted Outlier', marker='X', markersize=10, markerfacecolor='tab:purple', markeredgecolor='black', linestyle='')

    plt.legend(bbox_to_anchor=(1.5, 0.5), handles=[o_point, x_point])

    plt.savefig(figname, dpi=300, bbox_inches='tight')


#####################################main code ################################################
beginning = time.time()
args = parser.parse_args()
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else: 
        raise NameError(f'Undefined dataset {args.dataset}')


train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.is_human)
noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = 128,
                                   num_workers=args.num_workers,
                                   shuffle=False, #True originally
                                   drop_last = False)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 64,
                                  num_workers=args.num_workers,
                                  shuffle=False)

alpha_plan = [0.1] * 60 + [0.01] * 40

epoch=0
train_acc = 0

org_acc, new_acc = [], []

# load model
print('building model...')
#model = ResNet18(num_classes)
model = ResNet34(num_classes)
print('building model done')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
model.cuda()

# training
noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
# train models
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    model.train()
    train_acc = train(epoch, train_loader, model, optimizer)
    # evaluate models
    test_acc = evaluate(test_loader=test_loader, model=model)
    # save results
    print('train acc on train images is ', train_acc)
    print('test acc on test images is ', test_acc)
    org_acc.append(test_acc)

# original model on full data has been trained

probs, preds = get_probs_preds(train_loader=train_loader, model=model)

# compute label quality scores using baseline approaches
start_time = time.time()
label_quality_scores = get_label_quality_scores(labels=preds, pred_probs=probs, method=args.baseline_method, adjust_pred_probs=False)
end_time = time.time()
print("\nBaseline method completed in {} seconds.\n".format(end_time-start_time))

del_idxs = np.argpartition(label_quality_scores, int(0.05*len(label_quality_scores)))
del_idxs = del_idxs[:int(0.05*len(label_quality_scores))]

print("\nBaseline method identified {} outliers.\n".format(len(del_idxs)))

# remove samples from original dataset
new_train_dl = sample_remove_dataloader(train_dataset, del_idxs)

# retrain model on new dataset
print('building model...')
#model = ResNet18(num_classes)
model = ResNet34(num_classes)
print('building model done')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
model.cuda()

noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    model.train()
    train_acc = train(epoch, new_train_dl, model, optimizer)
    test_acc = evaluate(test_loader=test_loader, model=model)
    print('train acc on train images is ', train_acc)
    print('test acc on test images is ', test_acc)
    new_acc.append(test_acc)

# new model on trimmed data has been trained
org_acc = np.max(org_acc)
new_acc = np.max(new_acc)

ending = time.time()
print("\nEntire experiment concluded in {} seconds.\n".format(ending-beginning))
print("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc, new_acc))
