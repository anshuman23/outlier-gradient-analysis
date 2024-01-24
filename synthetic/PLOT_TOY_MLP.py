import os
import time
import argparse
import numpy as np
from typing import Sequence

from dataset import fetch_data, DataTemplate
from eval import Evaluator
from fair_fn import grad_ferm, grad_dp, loss_ferm, loss_dp
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

from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest

from deep_model import build_mlp, fit_mlp, pred_mlp, loss_mlp


def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--metric', type=str, default="eop", help="eop or dp")
    parser.add_argument('--seed', type=float, help="random seed")
    parser.add_argument('--total_points', type=int, default=500, help="points to delete")
    parser.add_argument('--model_type', type=str, default="logreg", help="logreg/nn")
    parser.add_argument('--color', type=str, default="tab:blue", help="color")
    parser.add_argument('--infl_on_train', type=str, default="y", help="infl on train?")

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


def obtain_loo_influence(args, data, model):
    #org_loss = loss_mlp(model, data.x_train, data.y_train)
    org_loss = loss_mlp(model, data.x_test, data.y_test)

    influences = []

    for i in range(len(data.x_train)):
       X,Y = data.x_train, data.y_train
       X = np.delete(X,i,axis=0)
       Y = np.delete(Y,i,axis=0)
       model = train_model_interim(args, X,Y)
       #new_loss = loss_mlp(model, data.x_train, data.y_train)
       new_loss = loss_mlp(model, data.x_test, data.y_test)
       influences.append(org_loss-new_loss)

    return np.array(influences)


def obtain_loo_accuracies(args, data, model):
    #org_acc = accuracy_score(data.y_test, pred_mlp(model, data.x_test)[1])

    influences = []

    for i in range(len(data.x_train)):
       X,Y = data.x_train, data.y_train
       X = np.delete(X,i,axis=0)
       Y = np.delete(Y,i,axis=0)
       model = train_model_interim(args, X,Y)
       new_acc = accuracy_score(data.y_test, pred_mlp(model, data.x_test)[1])
       influences.append(new_acc)

    return np.array(influences)



def find_sorted_points(I2, num_points): #Ascending order <-> Descending order

    indices_to_delete = I2.argsort()[::-1][-num_points:][::-1].tolist()
    #indices_to_delete = I2.argsort()[-num_points:][::-1].tolist()

    return indices_to_delete


def plotter(infls, grads, od_idxs, xname, figname):
    rho = np.round(stats.spearmanr(infls, grads).correlation, 3)

    plt.clf()
    for i,infl in enumerate(infls):
        if i in od_idxs:
            plt.scatter(grads[i], infl, color='tab:green', marker='X', s=75, edgecolor='black')
            continue

        plt.scatter(grads[i], infl, color='tab:green', marker='o')

    #plt.gca().axline((0, 0), slope=1) #Cannot use y=x line since scales are different-- need to use regression line
    coef = np.polyfit(grads,infls,1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(grads, poly1d_fn(grads), color='black', linestyle='dashed', alpha=0.5)

    #plt.text(1,0.5, s='Rho = {}'.format(rho), transform=plt.gca().transAxes)

    plt.xlabel(xname)
    plt.ylabel('Influence Value')

    o_point = Line2D([0], [0], label='Ground-truth Non-outlier', marker='o', markersize=10, markerfacecolor='tab:green', markeredgecolor='black', linestyle='')
    x_point = Line2D([0], [0], label='Ground-truth Outlier', marker='X', markersize=10, markerfacecolor='tab:green', markeredgecolor='black', linestyle='')
    txt = Line2D([0], [0], label='Rho = {}'.format(rho), marker='o', markersize=10, markerfacecolor='white', markeredgecolor='white', linestyle='')

    plt.legend(bbox_to_anchor=(1.1, 1.05), handles=[o_point, x_point, txt])

    plt.savefig(figname, dpi=300, bbox_inches='tight')


def plot_grad(ax, grads, od_idxs, y, y_colormap):
    print(len(y), len(grads), len(od_idxs))

    for i,g in enumerate(grads):
        if i in od_idxs:
            ax.scatter(g[0], g[1], color=y_colormap[y[i]], marker='X', s=175, edgecolor='black')
            continue

        ax.scatter(g[0], g[1], color=y_colormap[y[i]], marker='o', s=65)

    ax.set_xlabel('Gradient (Feature 1)', fontsize=16)
    ax.set_ylabel('Gradient (Feature 2)', fontsize=16)
    #o_point = Line2D([0], [0], label='Ground-truth Non-outlier', marker='o', markersize=10, markerfacecolor='tab:purple', markeredgecolor='tab:purple', linestyle='')
    #x_point = Line2D([0], [0], label='Ground-truth Outlier', marker='X', markersize=10, markerfacecolor='tab:purple', markeredgecolor='black', linestyle='')

    #plt.legend(bbox_to_anchor=(1.5, 0.5), handles=[o_point, x_point])

    #plt.savefig(figname, dpi=300, bbox_inches='tight')



def do_outlier_analysis(X):
    #clf = ECOD()
    clf = IForest()
    clf.fit(X)

    scores = clf.predict(X)
    return scores


def plot_outlier_analysis(ax, infls, od_scores, od_idxs):

    colormap = {1: 'tab:orange', 0: 'tab:green'}
    for i,infl in enumerate(infls):
        ax.scatter(i, infl, color=colormap[od_scores[i]], marker='o', s=65)

    for i,infl in enumerate(infls):
        if i in od_idxs:
            ax.scatter(i, infl, color=colormap[od_scores[i]], marker='X', s=175, edgecolor='black')
            continue


    ax.set_xlabel('Sample Index', fontsize=16)
    ax.set_ylabel('Influence Value', fontsize=16)


    #green_patch = mpatches.Patch(color='tab:green', label='Predicted Non-outlier (IForest on Gradient Space)')
    #orange_patch = mpatches.Patch(color='tab:orange', label='Predicted Outlier (IForest on Gradient Space)')
    #o_point = Line2D([0], [0], label='Ground-truth Non-outlier', marker='o', markersize=10, markerfacecolor='white', markeredgecolor='black', linestyle='')
    #x_point = Line2D([0], [0], label='Ground-truth Outlier', marker='X', markersize=10, markerfacecolor='white', markeredgecolor='black', linestyle='')

    #plt.legend(bbox_to_anchor=(1.1, 1.05), handles=[green_patch, orange_patch, o_point, x_point])

    #plt.savefig('without_hess_figs/mlp/without_hess_od.png', dpi=300, bbox_inches='tight')


def delete_pred_outliers_exp(args, data, od_scores):
    del_idxs = []
    for i in range(len(od_scores)):
        if od_scores[i] == 1:
            del_idxs.append(i)

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
    args.infl_on_train = 'n'

    data = get_full_dataset(args)

    args.total_points = 10
    model, grads, grads_second = train_model(args, data)

    test_evaluator = Evaluator(data.s_test, "test")
    test_res = test_evaluator(data.y_test, pred_mlp(model, data.x_test)[1])

    loo_influences = np.load('data/half_moons/loo_influences.npy')

    loo_accuracies = 0.9 - np.load('data/half_moons/loo_accuracies.npy')

    od_idxs = np.load('data/half_moons/od_idxs.npy')

    grads_combined = np.hstack((grads, grads_second))

    outlier_scores = do_outlier_analysis(grads_combined) #Both first and second order gradient information is used




    fig, axs = plt.subplots(1, 4, figsize=(18,3))

    df = pd.read_csv('data/half_moons/train.csv', header=None)

    f1, f2, y = df[0].to_list(), df[1].to_list(), df[3].to_list()

    indices = np.load('data/half_moons/od_idxs.npy').tolist()
    print(indices)

    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}

    for i, (a,b) in enumerate(zip(f1, f2)):
        axs[0].scatter(a,b, marker='o', color=y_colormap[y[i]], s=65) #, edgecolor='black')

    for i, (a,b) in enumerate(zip(f1, f2)):
        if i in indices:
            axs[0].scatter(a,b, marker='X', color=y_colormap[y[i]], s=175, edgecolor='black')
            continue


    axs[0].set_xlabel('Feature 1', fontsize=16)
    axs[0].set_ylabel('Feature 2', fontsize=16)




    df = pd.read_csv('data/half_moons/test.csv', header=None)

    f1, f2, y_te = df[0].to_list(), df[1].to_list(), df[3].to_list()

    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}

    for i, (a,b) in enumerate(zip(f1, f2)):
        axs[1].scatter(a,b, marker='o', color=y_colormap[y_te[i]], s=65) #, edgecolor='black')

    axs[1].set_xlabel('Feature 1', fontsize=16)
    axs[1].set_ylabel('Feature 2', fontsize=16)




    plot_grad(axs[2], grads, od_idxs, y, y_colormap)
    #plot_grad(grads_second, od_idxs, 'without_hess_figs/mlp/grad_only_second_order.png')



    plot_outlier_analysis(axs[3], loo_accuracies, outlier_scores, od_idxs) #Plot using loo accuracy values





    fig.subplots_adjust(bottom=0.1)

    for a in axs.flatten():
        a.tick_params(axis='both', which='major', labelsize=12)
        a.tick_params(axis='both', which='minor', labelsize=12)

    axs[0].text(1.5, 0.9, 'E', fontsize=16, weight='bold')
    axs[1].text(1.5, 0.9, 'F', fontsize=16, weight='bold')
    axs[2].text(1.9, -0.7, 'G', fontsize=16, weight='bold')
    axs[3].text(220, -0.007, 'H', fontsize=16, weight='bold')


    red_patch = mpatches.Patch(color='tab:red', label='Class 0')
    blue_patch = mpatches.Patch(color='tab:blue', label='Class 1')
    green_patch = mpatches.Patch(color='tab:green', label='Predicted Non-outlier')
    orange_patch = mpatches.Patch(color='tab:orange', label='Predicted Outlier')
    o_point = Line2D([0], [0], label='Regular Sample', marker='o', markersize=17, markerfacecolor='white', markeredgecolor='black', linestyle='')
    x_point = Line2D([0], [0], label='Noisy Sample', marker='X', markersize=17, markerfacecolor='white', markeredgecolor='black', linestyle='')

    plt.figlegend(handles=[red_patch, blue_patch, green_patch, orange_patch, o_point, x_point], loc='lower center', ncol=6, fontsize=15, prop={'size':15}, bbox_to_anchor=(0.5, -0.21), shadow=True)


    plt.tight_layout()
    plt.savefig('without_hess_figs/toy_full_MLP.png', dpi=300, bbox_inches='tight')



    #delete_pred_outliers_exp(args, data, outlier_scores)
