import os
import time
import argparse
import numpy as np
from typing import Sequence

from dataset import fetch_data, DataTemplate
from eval import Evaluator
#from model import LogisticRegression, NNLastLayerIF, MLPClassifier, NN, MLPClassifier2
from utils import fix_seed, save2csv

import json

import pickle
import random

import copy

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import seaborn as sns
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pyod.models.ecod import ECOD

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from scipy import stats

from deep_model import build_mlp, fit_mlp, pred_mlp, loss_mlp

from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest

import random
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.random_projection import SparseRandomProjection

from sklearn.metrics import classification_report

from datainf_influence_functions import IFEngine

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Influence Baselines')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--seed', type=float, help="random seed")

    args = parser.parse_args()

    return args


def get_full_dataset(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train.csv'
        f.seek(0)        
        json.dump(json_data, f, indent=4)
        f.truncate()

    data: DataTemplate = fetch_data(args.dataset)
    return data


def train_model(args, data):
    if args.seed is not None:
        fix_seed(args.seed)

    model = build_mlp(data.dim)
    model, grads, grads_second = fit_mlp(model, data.x_train, data.y_train)

    return model, grads, grads_second


def train_model_interim(args, x,y):
    if args.seed is not None:
        fix_seed(args.seed)

    model = build_mlp(data.dim)
    model, _, _ = fit_mlp(model, x,y)

    return model



def find_sorted_points(I2, num_points): #Ascending order <-> Descending order

    indices_to_delete = I2.argsort()[::-1][-num_points:][::-1].tolist()
    #indices_to_delete = I2.argsort()[-num_points:][::-1].tolist()

    return indices_to_delete



def find_idxs(scores, contamination=0.1):
    del_idxs = np.argpartition(scores, int(contamination*len(scores)))
    del_idxs = del_idxs[:int(contamination*len(scores))]
    return del_idxs




def delete_pred_outliers_exp(args, data, del_idxs):
    X, y = data.x_train, data.y_train
    X = np.delete(X, del_idxs, axis=0)
    y = np.delete(y, del_idxs, axis=0)

    model = train_model_interim(args, X,y)

    test_evaluator = Evaluator(data.s_test, "test")
    test_res = test_evaluator(data.y_test, pred_mlp(model, data.x_test)[1])



if __name__ == "__main__":
    args = parse_args()

    args.seed = 42
    args.dataset = 'half_moons'

    data = get_full_dataset(args)
    
    model, grads, grads_second = train_model(args, data)

    grads_dict = {i: {'main': None} for i in range(len(grads))}
    for i in range(len(grads)):
        grads_dict[i]['main'] = torch.from_numpy(grads[i])


    inf_eng = IFEngine()
    inf_eng.preprocess_gradients(grads_dict, grads_dict)
    inf_eng.compute_hvps()
    inf_eng.compute_IF()

    print(inf_eng.IF_dict['accurate'].shape)
    print(inf_eng.IF_dict['identity'].shape)
    print(inf_eng.IF_dict['proposed'].shape)
    print(inf_eng.IF_dict['LiSSA'].shape)


    test_evaluator = Evaluator(data.s_test, "test")
    test_res = test_evaluator(data.y_test, pred_mlp(model, data.x_test)[1])

    od_idxs = np.load('data/half_moons/od_idxs.npy')
    gt_od_list = [1 if i in od_idxs else 0 for i in range(data.x_train.shape[0])]

    acc_idxs = find_idxs(inf_eng.IF_dict['accurate'])
    acc_list = [1 if i in acc_idxs else 0 for i in range(data.x_train.shape[0])]

    iden_idxs = find_idxs(inf_eng.IF_dict['identity'])
    iden_list = [1 if i in iden_idxs else 0 for i in range(data.x_train.shape[0])]

    prop_idxs = find_idxs(inf_eng.IF_dict['proposed'])
    prop_list = [1 if i in prop_idxs else 0 for i in range(data.x_train.shape[0])]

    lissa_idxs = find_idxs(inf_eng.IF_dict['LiSSA'])
    lissa_list = [1 if i in lissa_idxs else 0 for i in range(data.x_train.shape[0])]


    # Measure how good each approach is at identifying ground truth outliers
    print("\n\nOUTLIER CLASSIFICATION EXPERIMENTS\n\n")

    print("\nResults for Exact Hessian Trimming\n")
    print(classification_report(gt_od_list, acc_list))

    print("\nResults for Hessian Free (Identity) Baseline\n")
    print(classification_report(gt_od_list, iden_list))

    print("\nResults for LiSSA  Baseline\n")
    print(classification_report(gt_od_list, lissa_list))

    print("\nResults for DataInf Baseline\n")
    print(classification_report(gt_od_list, prop_list))

    print("\n\nRETRAINING EXPERIMENTS\n\n")

    # Measure test set performance as a result of retraining wrt each baseline deletion indices
    print("\nResults for Exact Hessian Trimming\n")
    delete_pred_outliers_exp(args, data, acc_idxs)

    print("\nResults for Hessian Free (Identity) Baseline\n")
    delete_pred_outliers_exp(args, data, iden_idxs)

    print("\nResults for LiSSA  Baseline\n")
    delete_pred_outliers_exp(args, data, lissa_idxs)

    print("\nResults for DataInf Baseline\n")
    delete_pred_outliers_exp(args, data, prop_idxs)
