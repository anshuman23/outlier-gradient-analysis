from src.dataloader import create_dataloaders
from src.lora_model import LORAEngine, LORAEngine2
from src.influence import IFEngine

import numpy as np
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

import pickle

from torch.utils.data.sampler import RandomSampler

from sklearn.ensemble import IsolationForest

from typing import Optional, Sized, Iterator

from torch.utils.data import DataLoader

import random

class RemovalSampler(RandomSampler):
    r"""Sample elements randomly. 
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


def find_idxs(scores, contamination=0.3):
    del_idxs = np.argpartition(scores, int(contamination*len(scores)))
    del_idxs = del_idxs[:int(contamination*len(scores))]
    return del_idxs


def grads_dict_to_arr(d):
    N = len(d)
    arr = np.zeros((N, 2048))
    for i in range(N):
        arr[i] = d[i]['base_model.model.classifier.modules_to_save.default.out_proj.weight'].numpy().flatten()
    return arr


def sample_remove_dataloader(ds, batch_size, idx_list):
    sampler = RemovalSampler(ds, forbidden=idx_list)

    dl = DataLoader(ds, collate_fn=collate_fn, batch_size=batch_size, sampler=sampler)

    return dl


def retrain_model(dl, task, model_name_or_path="roberta-large", noise_ratio=0.2, batch_size=32, target_modules=["value"], device="cuda", num_epochs=10, lr=3e-4):
    lora_engine2 = LORAEngine2(model_name_or_path=model_name_or_path,
                            target_modules=target_modules,
                            train_dataloader=dl,
                            eval_dataloader=eval_dataloader,
                            device=device,
                            num_epochs=num_epochs,
                            lr=lr,
                            low_rank=4, 
                            task=task)

    lora_engine2.build_LORA_model()
    accs = lora_engine2.train_LORA_model()
    return accs






#INITIALIZE PARAMS, TASK AND MODEL
model_name_or_path="roberta-large"
task="mrpc" #choose between qnli, sst2, qqp, mrpc
noise_ratio=0.2
batch_size=32
target_modules=["value"]
device="cuda"
num_epochs=10
lr=3e-4
DATASET=task

#CREATE DATALOADERS
dataloader_outputs = create_dataloaders(model_name_or_path=model_name_or_path,
                                           task=task,
                                           noise_ratio=noise_ratio,
                                           batch_size=batch_size)
train_dataloader, eval_dataloader, noise_index, tokenized_datasets, collate_fn = dataloader_outputs

lora_engine = LORAEngine(model_name_or_path=model_name_or_path,
                            target_modules=target_modules,
                            train_dataloader=train_dataloader,
                            eval_dataloader=eval_dataloader,
                            device=device,
                            num_epochs=num_epochs,
                            lr=lr,
                            low_rank=4, 
                            task=task)


#FINETUNE ROBERTA
lora_engine.build_LORA_model()
lora_engine.train_LORA_model()


#OBTAIN GRADIENTS
tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn) #We will only use training set gradients for fair comparison


#OBTAIN INFLUENCE VALUES FOR BASELINES
influence_engine = IFEngine()
influence_engine.preprocess_gradients(tr_grad_dict, tr_grad_dict, noise_index) #Note no validation set gradients will be used
influence_engine.compute_hvps(compute_accurate=False)
influence_engine.compute_IF()

iden_idxs = find_idxs(influence_engine.IF_dict['identity'])

datainf_idxs = find_idxs(influence_engine.IF_dict['proposed'])

lissa_idxs = find_idxs(influence_engine.IF_dict['LiSSA'])


#DO OUTLIER GRADIENT ANALYSIS
grads = grads_dict_to_arr(tr_grad_dict)
print(grads.shape)

clf = IsolationForest(contamination=0.3)
clf.fit(grads)
scores = clf.score_samples(grads)

od_idxs = find_idxs(scores)

print(len(od_idxs), len(iden_idxs), len(datainf_idxs), len(lissa_idxs))



#RETRAIN THE MODEL FOR EACH BASELINE AND OUTLIER TRIMMING
accuracies = {'outlier': [], 'identity': [], 'lissa': [], 'datainf': []}
od_trainloader = sample_remove_dataloader(tokenized_datasets['train'], batch_size, od_idxs)
iden_trainloader = sample_remove_dataloader(tokenized_datasets['train'], batch_size, iden_idxs)
datainf_trainloader = sample_remove_dataloader(tokenized_datasets['train'], batch_size, datainf_idxs)
lissa_trainloader = sample_remove_dataloader(tokenized_datasets['train'], batch_size, lissa_idxs)

for _ in range(3):
    accuracies['outlier'].append(retrain_model(od_trainloader, task))
    accuracies['identity'].append(retrain_model(iden_trainloader, task))
    accuracies['datainf'].append(retrain_model(datainf_trainloader, task))
    accuracies['lissa'].append(retrain_model(lissa_trainloader, task))

#SAVED THE RESULTS
with open('saved_figs_roberta/{}.pkl'.format(DATASET), 'wb') as f:
    pickle.dump(accuracies, f)
