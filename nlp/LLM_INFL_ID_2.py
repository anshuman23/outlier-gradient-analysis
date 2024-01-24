import numpy as np
import pickle
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

#initialize basic info
DATASET='grammars' #Choose from math_with_reason, math_without_reason, grammars
N_CLASSES = 10
N_PER_CLASS = 90
N_TRAIN = 900
N_VAL = 100

#load gradient pickles saved in previous code run
with open('saved_grads/{}/train.pkl'.format(DATASET), 'rb') as f:
    tr_grads_dict = pickle.load(f)
with open('saved_grads/{}/val.pkl'.format(DATASET), 'rb') as f:
    val_grads_dict = pickle.load(f)

#convert tr_grads and val_grads in numpy array form only from last few layers
def grads_dict_to_arr(d, N):
    arr = np.zeros((N, 163840))
    for i in range(N):
        qA = d[i]['base_model.model.model.layers.39.self_attn.q_proj.lora_A.default.weight'].numpy().flatten()
        qB = d[i]['base_model.model.model.layers.39.self_attn.q_proj.lora_B.default.weight'].numpy().flatten()
        vA = d[i]['base_model.model.model.layers.39.self_attn.v_proj.lora_A.default.weight'].numpy().flatten()
        vB = d[i]['base_model.model.model.layers.39.self_attn.v_proj.lora_B.default.weight'].numpy().flatten()
        s = np.vstack((qA, qB))
        s = np.vstack((s, vA))
        s = np.vstack((s, vB))
        arr[i] = s.flatten()

    return arr

#then combine them into a single grads variable
tr_grads = grads_dict_to_arr(tr_grads_dict, N_TRAIN)
val_grads = grads_dict_to_arr(val_grads_dict, N_VAL)
print(tr_grads.shape, val_grads.shape)
grads = np.vstack((tr_grads, val_grads))
print(grads.shape)

#reduce dimensionality via sparse random projection
grads_embed = SparseRandomProjection(n_components='auto').fit_transform(grads)
print(grads_embed.shape)
tr_grads = grads_embed[:N_TRAIN,:]
val_grads = grads_embed[N_TRAIN:,:]

#train individual outlier detectors for each class
clf_list, idxs_list, scores_list = [], [], []
preds = np.zeros((N_VAL, N_TRAIN))
for cl in range(N_CLASSES):
    clf = IsolationForest(contamination='auto', random_state=42)
    clf.fit(tr_grads[cl*N_PER_CLASS:(cl+1)*N_PER_CLASS])
    scores = clf.score_samples(val_grads)

    clf_list.append(clf)
    scores_list.append(scores)
    #idxs_list.append(np.argpartition(scores, -10)[-10:])
    idxs_list.append(np.argmax(scores))
    sorted_scores_idxs = np.argsort(scores)[::-1] #descending -> least outlier to most
    for i in sorted_scores_idxs:
        preds[i, cl*N_PER_CLASS:(cl+1)*N_PER_CLASS] = scores[i]

#print(idxs_list)
np.save('saved_figs_llm/{}_preds.npy'.format(DATASET), preds)

#get AUC results for outlier gradient trimming
auc_list = []
for i in range(N_VAL):
    gt_array=np.zeros(N_TRAIN)
    gt_array[(i//N_CLASSES)*N_PER_CLASS:((i//N_CLASSES)+1)*N_PER_CLASS]=1
    
    auc_list.append(roc_auc_score(gt_array, preds[i,:]))
    
print("OUTLIER GRADIENT ANALYSIS AUC: {} +- {}".format(np.mean(auc_list), np.std(auc_list)))

#get Recall results for outlier gradient trimming
recall_list = []
for i in range(N_VAL):
    correct_label = i // 10
    sorted_labels = np.argsort(preds[i])[::-1] // 90
    recall = np.count_nonzero(sorted_labels[0:90] == correct_label) / 90.0
    recall_list.append(recall)
    
print("OUTLIER GRADIENT ANALYSIS Recall: {} +- {}".format(np.mean(recall_list), np.std(recall_list)))
