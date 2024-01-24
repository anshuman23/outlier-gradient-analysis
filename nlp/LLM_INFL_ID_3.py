from src.lora_model import LORAEngineGeneration
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ["WANDB_MODE"] = 'disabled'

def create_mat(preds):
    #Create class-wise matrix from full preds
    new_preds = np.zeros((preds.shape[0], 10))
    for i in range(10):
        new_preds[:,i] = preds[:,i*90]
    new_preds = new_preds.T

    print(new_preds.shape)

    return new_preds


def plot_hm(ax, new_preds, title, fig, bar):
    im = ax.imshow(new_preds, interpolation='nearest', cmap='Blues')
    ax.set_aspect(new_preds.shape[1] / new_preds.shape[0])

    if bar:
        #cbar = fig.colorbar(im, orientation='horizontal', ticks=[np.min(new_preds), np.max(new_preds)])
        #cbar.ax.set_xticklabels(['Detrimental', 'Beneficial'])

        cbar = fig.colorbar(im, ticks=[np.min(new_preds), np.max(new_preds)])
        cbar.ax.set_yticklabels(['Detrimental', 'Beneficial'], fontsize=12)

    #ax.set_yticks([i for i in range(10)])
    ax.set_xticks([(i)*10 for i in range(10)])
    ax.set_yticks([1,3,5,7,9])

    ax.set_xlabel('Test Sample Index', fontsize=13) #12
    ax.set_ylabel('Class #', fontsize=13) #12

    ax.set_title(title, fontsize=14, style='italic')


new_preds_grammars = create_mat(np.load('saved_figs_llm/grammars_preds.npy'))
new_preds_math_reason = create_mat(np.load('saved_figs_llm/math_with_reason_preds.npy'))
new_preds_math_no_reason = create_mat(np.load('saved_figs_llm/math_without_reason_preds.npy'))

#fig, axs = plt.subplots(3, 1, figsize=(12,7))
fig, axs = plt.subplots(1, 3, figsize=(18,3))

fig.subplots_adjust(bottom=0.1)

plot_hm(axs[0], new_preds_grammars, 'Sentence Transformations', fig, True)
plot_hm(axs[1], new_preds_math_no_reason, 'Math Without Reasoning', fig, True)
plot_hm(axs[2], new_preds_math_reason, 'Math With Reasoning', fig, True)

#plt.tight_layout()

for a in axs.flatten():
    a.tick_params(axis='both', which='major', labelsize=10)
    a.tick_params(axis='both', which='minor', labelsize=10)

plt.savefig('saved_figs_llm/heatmap.png', dpi=300, bbox_inches='tight')

