import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch import autograd


############################## <- MLP -> ###############################

def build_mlp(dim, factor=1):
    model = nn.Sequential(OrderedDict([
    ('linear1', nn.Linear(dim, dim // factor, bias=True)),
    ('relu1', nn.ReLU()),
    ('linear2', nn.Linear(dim // factor, dim // factor, bias=True)),
    ('relu2', nn.ReLU()),
    ('linear3', nn.Linear(dim // factor, 1, bias=False)),
    ('sigmoid', nn.Sigmoid())])
    )

    return model


def fit_mlp(model, X,y, l2_reg=1e-4, num_epochs=5000):
    model = model.to('cuda:0')

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_reg)

    X = torch.from_numpy(X).to(torch.float).to('cuda:0')
    y = torch.from_numpy(y).to(torch.float).to('cuda:0')

    grads = np.zeros((X.shape[0], 2))
    grads_second = np.zeros((X.shape[0], 2))

    for n in tqdm(range(num_epochs)):
        y_pred = model(X).reshape((y.shape[0]))
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    for idx, (X_single, y_single) in enumerate(zip(X,y)):
        y_pred = model(X_single).reshape((y_single.shape))
        loss = loss_fn(y_pred, y_single)
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #grads[idx] = model.linear3.weight.grad.cpu().numpy()

        first_drv = autograd.grad(loss, model.linear3.weight, create_graph=True)[0]
        second_drv_0 = autograd.grad(first_drv[0][0], model.linear3.weight, create_graph=True)[0]
        second_drv_1 = autograd.grad(first_drv[0][1], model.linear3.weight, create_graph=True)[0]

        grads[idx] = first_drv.cpu().detach().numpy()
        grads_second[idx] = np.array([second_drv_0.cpu().detach().numpy()[0][0], second_drv_1.cpu().detach().numpy()[0][1]])


    return model, grads, grads_second



def fit_mlp_batched(model, X,y, batch_size = 10, l2_reg=1e-4, num_epochs=1000):
    model = model.to('cuda:0')

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_reg)

    X = torch.from_numpy(X).to(torch.float).to('cuda:0')
    y = torch.from_numpy(y).to(torch.float).to('cuda:0')

    num_batches = X.shape[0] // batch_size

    grads = np.zeros((num_batches, 2))
    grads_second = np.zeros((num_batches, 2))

    for n in tqdm(range(num_epochs)):
        for batch_i in range(num_batches):
            X_batch = X[batch_i*batch_size:(batch_i+1)*batch_size]
            y_batch = y[batch_i*batch_size:(batch_i+1)*batch_size]

            y_pred_batch = model(X_batch).reshape((y_batch.shape[0]))
            loss = loss_fn(y_pred_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    for idx, batch_i in enumerate(range(num_batches)):
        X_batch_i = X[batch_i*batch_size:(batch_i+1)*batch_size]
        y_batch_i = y[batch_i*batch_size:(batch_i+1)*batch_size]

        y_pred = model(X_batch_i).reshape((y_batch_i.shape))
        loss_i = loss_fn(y_pred, y_batch_i)

        first_drv = autograd.grad(loss_i, model.linear3.weight, create_graph=True)[0]
        second_drv_0 = autograd.grad(first_drv[0][0], model.linear3.weight, create_graph=True)[0]
        second_drv_1 = autograd.grad(first_drv[0][1], model.linear3.weight, create_graph=True)[0]

        grads[idx] = first_drv.cpu().detach().numpy()
        grads_second[idx] = np.array([second_drv_0.cpu().detach().numpy()[0][0], second_drv_1.cpu().detach().numpy()[0][1]])


    return model, grads, grads_second




def pred_mlp(model, X):
    X = torch.from_numpy(X).to(torch.float).to('cuda:0')
    with torch.no_grad():
        pred = model(X).reshape((X.shape[0]))
    pred_label = (pred > 0.5).to(torch.float)

    return pred.cpu().numpy(), pred_label.cpu().numpy()


def loss_mlp(model, X, y):
    loss_fn = nn.BCELoss()
    X = torch.from_numpy(X).to(torch.float).to('cuda:0')
    y = torch.from_numpy(y).to(torch.float).to('cuda:0')
    with torch.no_grad():
        pred = model(X).reshape((y.shape[0]))

    return loss_fn(pred, y).cpu()


def get_embeddings_helper(name='hidden'):
    global features
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def get_embeddings_mlp(model, X):
    global features
    features = {}
    model.linear2.register_forward_hook(get_embeddings_helper('hidden'))
    X = torch.from_numpy(X).to(torch.float).to('cuda:0')
    with torch.no_grad():
        pred = model(X).reshape((X.shape[0]))
    return features['hidden'].cpu().numpy()


###############################################################################
