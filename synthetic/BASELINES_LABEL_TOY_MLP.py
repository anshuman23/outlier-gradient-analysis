import os
import time
import argparse
import numpy as np
from typing import Sequence

from dataset import fetch_data, DataTemplate
from eval import Evaluator
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

from cleanlab.rank import get_label_quality_scores

from sklearn.metrics import classification_report

def parse_args():
    parser = argparse.ArgumentParser(description='Label correction baselines including Outlier gradient analysis')
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




def do_outlier_analysis(X):
    clf = IForest(contamination=0.1)
    clf.fit(X)

    scores = clf.predict(X)
    del_idxs = []
    for i in range(len(scores)):
        if scores[i] == 1:
            del_idxs.append(i)
    return del_idxs 


def do_baseline(labels, pred_probs, method, contamination=0.1):
    label_quality_scores = get_label_quality_scores(labels=labels, pred_probs=pred_probs, method=method, adjust_pred_probs=False)
    del_idxs = np.argpartition(label_quality_scores, int(contamination*len(label_quality_scores)))
    del_idxs = del_idxs[:int(contamination*len(label_quality_scores))]
    return del_idxs


def delete_pred_outliers_exp(args, data, del_idxs):
    X, y = data.x_train, data.y_train
    X = np.delete(X, del_idxs, axis=0)
    y = np.delete(y, del_idxs, axis=0)

    model = train_model_interim(args, X,y)

    test_evaluator = Evaluator(data.s_test, "test")
    test_res = test_evaluator(data.y_test, pred_mlp(model, data.x_test)[1])


def convert_binary_probs(probs):
    arr_probs = np.zeros((probs.shape[0], 2))
    for i, prob in enumerate(probs):
        if prob > 0.5:
            arr_probs[i,1] = prob
            arr_probs[i,0] = 1.0 - prob
        else:
            arr_probs[i,0] = 1.0 - prob
            arr_probs[i,1] = prob

    return arr_probs


if __name__ == "__main__":
    args = parse_args()

    args.seed = 42
    args.dataset = 'half_moons'

    data = get_full_dataset(args)
    
    model, grads, grads_second = train_model(args, data)

    test_evaluator = Evaluator(data.s_test, "test")
    test_res = test_evaluator(data.y_test, pred_mlp(model, data.x_test)[1])

    od_idxs = np.load('data/half_moons/od_idxs.npy')
    gt_od_list = [1 if i in od_idxs else 0 for i in range(data.x_train.shape[0])]

    #grads_combined = np.hstack((grads, grads_second))
    grads_combined = grads

    outlier_trim_idxs = do_outlier_analysis(grads_combined) 
    outlier_trim_list = [1 if i in outlier_trim_idxs else 0 for i in range(data.x_train.shape[0])]

    pred_probs = convert_binary_probs(pred_mlp(model, data.x_train)[0])
    pred_lbls = pred_mlp(model, data.x_train)[1].astype(int)

    self_confidence_idxs = do_baseline(pred_lbls, pred_probs, 'self_confidence')
    sc_list = [1 if i in self_confidence_idxs else 0 for i in range(data.x_train.shape[0])]

    normalized_margin_idxs = do_baseline(pred_lbls, pred_probs, 'normalized_margin')
    nm_list = [1 if i in normalized_margin_idxs else 0 for i in range(data.x_train.shape[0])]

    cw_entropy_idxs = do_baseline(pred_lbls, pred_probs, 'confidence_weighted_entropy')
    cwe_list = [1 if i in cw_entropy_idxs else 0 for i in range(data.x_train.shape[0])]


    # Measure how good each approach is at identifying ground truth outliers
    print("\n\nOUTLIER CLASSIFICATION EXPERIMENTS\n\n")

    print("\nResults for Outlier Gradient Trimming\n")
    print(classification_report(gt_od_list, outlier_trim_list))

    print("\nResults for Self Confidence Baseline\n")
    print(classification_report(gt_od_list, sc_list))

    print("\nResults for Normalized Margin Baseline\n")
    print(classification_report(gt_od_list, nm_list))

    print("\nResults for Cross-weighted Entropy Baseline\n")
    print(classification_report(gt_od_list, cwe_list))

    print("\n\nRETRAINING EXPERIMENTS\n\n")

    # Measure test set performance as a result of retraining wrt each baseline deletion indices
    print("\nResults for Outlier Gradient Trimming\n")
    delete_pred_outliers_exp(args, data, outlier_trim_idxs)

    print("\nResults for Self Confidence Baseline\n")
    delete_pred_outliers_exp(args, data, self_confidence_idxs)

    print("\nResults for Normalized Margin Baseline\n")
    delete_pred_outliers_exp(args, data, normalized_margin_idxs)

    print("\nResults for Cross-weighted Entropy Baseline\n")
    delete_pred_outliers_exp(args, data, cw_entropy_idxs)
