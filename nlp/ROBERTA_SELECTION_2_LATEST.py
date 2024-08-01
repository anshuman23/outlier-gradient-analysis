import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.lines import Line2D

#LOAD DATA

with open('saved_figs_roberta/qnli.pkl', 'rb') as f:
    qnli = pickle.load(f)
with open('saved_figs_roberta/sst2.pkl', 'rb') as f:
    sst2 = pickle.load(f)
with open('saved_figs_roberta/qqp.pkl', 'rb') as f:
    qqp = pickle.load(f)
with open('saved_figs_roberta/mrpc.pkl', 'rb') as f:
    mrpc = pickle.load(f)



#MAKE PLOTS

def plot_sub(ax, accs, title):
    ax.margins(x=0)
    #ax.margins(y=0)
    x = [1,2,3,4,5,6,7,8,9,10]
    ax.plot(x, np.mean(accs['outlier'], axis=0), color='tab:red', lw=2.5)
    ax.fill_between(x, np.mean(accs['outlier'], axis=0) - np.std(accs['outlier'], axis=0), np.mean(accs['outlier'], axis=0) + np.std(accs['outlier'], axis=0), alpha=0.2, color='tab:red')

    ax.plot(x, np.mean(accs['identity'], axis=0), color='tab:orange', lw=2)
    ax.fill_between(x, np.mean(accs['identity'], axis=0) - np.std(accs['identity'], axis=0), np.mean(accs['identity'], axis=0) + np.std(accs['identity'], axis=0), alpha=0.2, color='tab:orange')

    ax.plot(x, np.mean(accs['datainf'], axis=0), color='tab:purple', lw=2)
    ax.fill_between(x, np.mean(accs['datainf'], axis=0) - np.std(accs['datainf'], axis=0), np.mean(accs['datainf'], axis=0) + np.std(accs['datainf'], axis=0), alpha=0.2, color='tab:purple')

    ax.plot(x, np.mean(accs['self_datainf'], axis=0), color='tab:pink', lw=2)
    ax.fill_between(x, np.mean(accs['self_datainf'], axis=0) - np.std(accs['self_datainf'], axis=0), np.mean(accs['self_datainf'], axis=0) + np.std(accs['self_datainf'], axis=0), alpha=0.2, color='tab:pink')

    ax.plot(x, np.mean(accs['lissa'], axis=0), color='tab:blue', lw=2)
    ax.fill_between(x, np.mean(accs['lissa'], axis=0) - np.std(accs['lissa'], axis=0), np.mean(accs['lissa'], axis=0) + np.std(accs['lissa'], axis=0), alpha=0.2, color='tab:blue')

    ax.plot(x, np.mean(accs['self_lissa'], axis=0), color='tab:olive', lw=2)
    ax.fill_between(x, np.mean(accs['self_lissa'], axis=0) - np.std(accs['self_lissa'], axis=0), np.mean(accs['self_lissa'], axis=0) + np.std(accs['self_lissa'], axis=0), alpha=0.2, color='tab:olive')

    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)

    #ax.set_title(title, fontsize=16, weight='bold', style='italic')
    ax.set_title(title, fontsize=16, style='italic')


fig, axs = plt.subplots(1, 4, figsize=(18,3))

plot_sub(axs[0], qnli, 'QNLI')
plot_sub(axs[1], sst2, 'SST2')
plot_sub(axs[2], qqp, 'QQP')
plot_sub(axs[3], mrpc, 'MRPC')


fig.subplots_adjust(bottom=0.1)

for a in axs.flatten():
    a.tick_params(axis='both', which='major', labelsize=12)
    a.tick_params(axis='both', which='minor', labelsize=12)

#axs[0].text(5, 5.2, 'A', fontsize=16, weight='bold')
#axs[1].text(6, 3, 'B', fontsize=16, weight='bold')
#axs[2].text(-2.0, 0.0, 'C', fontsize=16, weight='bold')
#axs[3].text(130, -0.5, 'D', fontsize=16, weight='bold')

od_leg = Line2D([0], [0], label='Outlier Gradient Trimming', color='tab:red', lw=3.5)
iden_leg = Line2D([0], [0], label='Gradient Tracing', color='tab:orange', lw=3.5)
datainf_leg = Line2D([0], [0], label='DataInf', color='tab:purple', lw=3.5)
self_datainf_leg = Line2D([0], [0], label='Self-DataInf', color='tab:pink', lw=3.5)
lissa_leg = Line2D([0], [0], label='LiSSA', color='tab:blue', lw=3.5)
self_lissa_leg = Line2D([0], [0], label='Self-LiSSA', color='tab:olive', lw=3.5)

plt.figlegend(handles=[iden_leg, lissa_leg, datainf_leg, self_lissa_leg, self_datainf_leg, od_leg], loc='lower center', ncol=6, fontsize=15, prop={'size':15}, bbox_to_anchor=(0.5, -0.155), shadow=True)


plt.tight_layout()
plt.savefig('saved_figs_roberta/combined_roberta.png', dpi=300, bbox_inches='tight')
