import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import time
import networkx as nx
import random
import pickle
from scipy.special import softmax
from scipy.sparse import csr_matrix
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import pandas as pd
import argparse
import scipy.sparse as sp
import matplotlib.pyplot as plt

# from main.utils import load_dataset, InverseProblemDataset, adj_process, diffusion_evaluation
# from main.model.gat import GAT, SpGAT
# from main.model.model import GNNModel, VAEModel, DiffusionPropagate, Encoder, Decoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
# from main.model.inference import StudentInference

from configuration import args
import utilities as utils
import models
from gat import GAT, SpGAT
import loss_functions
print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1
print(args)
## Loding Dataset
with open('../data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG', 'rb') as f:
    graph = pickle.load(f)

adj, inverse_pairs = graph['adj'], graph['inverse_pairs']
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = utils.normalize_adj(adj + sp.eye(adj.shape[0]))
adj = torch.Tensor(adj.toarray()).to_sparse()

if args.dataset == 'random5':
    batch_size = 2
    hidden_dim = 4096
    latent_dim = 1024
else:
    batch_size = 16
    hidden_dim = 1024 # feature
    latent_dim = 512 # Z output

train_set, test_set = torch.utils.data.random_split(inverse_pairs,
                                                    [len(inverse_pairs)-batch_size,
                                                     batch_size])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader  = DataLoader(dataset=test_set,  batch_size=1, shuffle=False)

## Defining Models

## Regarding distillation model
encoder = models.Encoder(input_dim= inverse_pairs.shape[1],
                  hidden_dim=hidden_dim,
                  latent_dim=latent_dim)

decoder = models.Decoder(input_dim=latent_dim,
                  latent_dim=latent_dim,
                  hidden_dim=hidden_dim,
                  output_dim=inverse_pairs.shape[1])

vae_model = models.VAEModel(Encoder=encoder, Decoder=decoder).to(device)

forward_model = SpGAT(nfeat=1,
                nhid=64,
                nclass=1,
                dropout=0.2,
                nheads=4,
                alpha=0.2)




optimizer = Adam([{'params': vae_model.parameters()}, {'params': forward_model.parameters()}],
                 lr=1e-4)

adj = adj.to(device)

if args.run_mode == 'test':
    diffusion_tic = time.perf_counter()
    checkpoint_distillation = torch.load('../saved_models/Distillation_Model_M_' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.ckpt')
    forward_model.load_state_dict(checkpoint_distillation)
    checkpoint_vae = torch.load('../saved_models/VAE_train_' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.ckpt')
    vae_model.load_state_dict(checkpoint_vae)
    forward_model.to(device)
    forward_model.eval()
    for batch_idx, data_pair in enumerate(test_loader): # train_loader
        # input_pair = torch.cat((data_pair[:, :, 0], data_pair[:, :, 1]), 1).to(device)

        x = data_pair[:, :, 0].float().to(device)
        y = data_pair[:, :, 1].float().to(device)
        # optimizer.zero_grad()

        y_true = y.cpu().detach().numpy()
        x_true = x.cpu().detach().numpy()

        for i, x_i in enumerate(x):
            y_i = y[i]
            x_hat = vae_model(x_i.unsqueeze(0))
    diffusion_toc = time.perf_counter()
    diffusion_train_time = diffusion_toc - diffusion_tic
    print("Diffusion model testing time {}".format(diffusion_train_time))
else:
    diffusion_tic = time.perf_counter()
    forward_model = forward_model.to(device)
    forward_model.train()
    for epoch in range(args.numEpoch):  # 600
        begin = time.time()
        total_overall = 0
        forward_loss = 0
        reproduction_loss = 0
        precision_for = 0
        recall_for = 0
        precision_re = 0
        recall_re = 0

        for batch_idx, data_pair in enumerate(train_loader):
            # input_pair = torch.cat((data_pair[:, :, 0], data_pair[:, :, 1]), 1).to(device)

            x = data_pair[:, :, 0].float().to(device)
            y = data_pair[:, :, 1].float().to(device)
            optimizer.zero_grad()

            y_true = y.cpu().detach().numpy()
            x_true = x.cpu().detach().numpy()

            loss = 0
            for i, x_i in enumerate(x):
                y_i = y[i]

                x_hat = vae_model(x_i.unsqueeze(0))
                y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), adj)
                total, re, forw = loss_functions.loss_all(x_i.unsqueeze(0), x_hat, y_i, y_hat.squeeze(-1))

                loss += total

                x_pred = x_hat.cpu().detach().numpy()
                x_pred[x_pred > 0.01] = 1
                x_pred[x_pred != 1] = 0

                precision_re += precision_score(x_true[i], x_pred[0], zero_division=0)
                recall_re += recall_score(x_true[i], x_pred[0], zero_division=0)

            total_overall += loss.item()
            loss = loss / x.size(0)

            loss.backward()
            optimizer.step()
            for p in forward_model.parameters():
                p.data.clamp_(min=0)

        end = time.time()
        print("Epoch: {}".format(epoch + 1),
              "\tTotal Loss: {:.4f}".format(total_overall / len(train_set)),
              "\tReconstruction Precision: {:.4f}".format(precision_re / len(train_set)),
              "\tReconstruction Recall: {:.4f}".format(recall_re / len(train_set)),
              "\tTime: {:.4f}".format(end - begin)
              )
    diffusion_toc = time.perf_counter()
    diffusion_train_time = diffusion_toc-diffusion_tic
    print("Diffusion model training time {}".format(diffusion_train_time))
    # saving models
    torch.save(vae_model.state_dict(), '../saved_models/VAE_train_' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.ckpt')
    torch.save(forward_model.state_dict(), '../saved_models/Distillation_Model_M_' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.ckpt')
    print(' Model saved')

## Seed set inference Strategy 0
for param in vae_model.parameters():
    param.requires_grad = False

for param in forward_model.parameters():
    param.requires_grad = False

encoder = vae_model.Encoder
decoder = vae_model.Decoder
topk_seed = utils.sampling(inverse_pairs)

z_hat = 0
for i in topk_seed:
    z_hat += encoder(inverse_pairs[i, :, 0].unsqueeze(0).to(device))

z_hat = z_hat / len(topk_seed)
seed_num = int(x_hat.sum().item())
y_true = torch.ones(x_hat.shape).to(device)

z_hat = z_hat.detach()
z_hat_init = z_hat.clone().detach().to(device)
z_hat.requires_grad = True
z_optimizer = Adam([z_hat], lr=1e-4)

iterative_inf_tic = time.perf_counter()
for i in range(args.inferRange):
    x_hat = decoder(z_hat)

    y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), adj)

    y = torch.where(y_hat > 0.05, 1, 0)

    loss, L0 = loss_functions.loss_inverse(y_true, y_hat, x_hat)

    loss.backward()
    z_optimizer.step()

    print('Iteration: {}'.format(i + 1),
          '\t Total Loss:{:.5f}'.format(loss.item())
          )
iterative_inf_toc = time.perf_counter()
iterative_inf_time = iterative_inf_toc-iterative_inf_tic
print("Iterative inference time {}".format(iterative_inf_time))
top_k = x_hat.topk(seed_num)
seed = top_k.indices[0].cpu().detach().numpy()
with open('../data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG', 'rb') as f:
    graph = pickle.load(f)

adj, inverse_pairs = graph['adj'], graph['inverse_pairs']
influence = utils.diffusion_evaluation(adj, seed, diffusion = args.diffusion_model)
print('Diffusion count: {}'.format(influence))

## Student inference
# train student NN
studentNet = models.StudentInference(latent_dim, 1024, latent_dim).to(device)
if args.run_mode == 'test':
    student_inf_tic = time.perf_counter()
    checkpoint_Student_Inference = torch.load('../saved_models/Student_Inference_' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.ckpt')
    studentNet.load_state_dict(checkpoint_Student_Inference)
    studentNet.to(device)
    studentNet.eval()
    z_hat_dprime = studentNet(z_hat_init)  # outputs from Student
    student_inf_toc = time.perf_counter()
    student_inf_time = student_inf_toc - student_inf_tic
    print("Student inference testing time {}".format(student_inf_time))
else:
    student_inf_tic = time.perf_counter()
    studentNet.train()
    max_epochs = 20
    ep_log_interval = 5
    lrn_rate = 0.005
    loss_func = nn.MSELoss()  # no hidden activation
    optimizer_studentInfer = torch.optim.SGD(studentNet.parameters(), lr=lrn_rate)
    # print("\nbat_size = %3d " % bat_size)
    print("loss = " + str(loss_func))
    print("optimizer = SGD")
    print("max_epochs = %3d " % max_epochs)
    print("lrn_rate = %0.3f " % lrn_rate)
    print("\nStarting training the student NN ")
    for epoch in range(0, max_epochs):
        epoch_loss = 0  # for one full epoch
    # for (batch_idx, batch) in enumerate(train_ldr):
    # X = batch[0]
    # Y = batch[1]
    #     Y = z_hat
        optimizer_studentInfer.zero_grad()
        z_hat_dprime = studentNet(z_hat_init)  # outputs from Student
        loss_val = loss_func(z_hat_dprime, z_hat)  # a tensor
        epoch_loss += loss_val.item()  # accumulate
        loss_val.backward()
        optimizer_studentInfer.step()
        if epoch % ep_log_interval == 0:
            print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))
            print("Done ")

    student_inf_toc = time.perf_counter()
    student_inf_time = student_inf_toc - student_inf_tic
    print("Student inference training time [wo/w seed inference] {}".format([student_inf_time,student_inf_time+iterative_inf_time]))
    torch.save(studentNet.state_dict(), '../saved_models/Student_Inference_' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.ckpt')
    print(' Model saved')
# Model Evaluation for student inference
# z_hat_dprime = studentNet(z_hat_init)
x_hat = decoder(z_hat_dprime)
top_k = x_hat.topk(seed_num)
seed2 = top_k.indices[0].cpu().detach().numpy()

with open('../data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG', 'rb') as f:
    graph = pickle.load(f)

adj, inverse_pairs = graph['adj'], graph['inverse_pairs']

influence = utils.diffusion_evaluation(adj, seed2, diffusion = args.diffusion_model)
print('Diffusion count: {}'.format(influence))

## Unsupervised Inference
with open('../data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG', 'rb') as f:
    graph = pickle.load(f)

adj, inverse_pairs = graph['adj'], graph['inverse_pairs']


adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = utils.normalize_adj(adj + sp.eye(adj.shape[0]))
adj = torch.Tensor(adj.toarray()).to_sparse()
adj = adj.to(device)
# encoder2 = models.GCNEncoder1(input_dim= inverse_pairs.shape[1],
#                   hidden_dim=hidden_dim,
#                   latent_dim=latent_dim)
encoder2 = models.Encoder(input_dim= inverse_pairs.shape[1],
                  hidden_dim=hidden_dim,
                  latent_dim=latent_dim)
decoder2 = models.Decoder(input_dim=latent_dim,
                  latent_dim=latent_dim,
                  hidden_dim=hidden_dim,
                  output_dim=inverse_pairs.shape[1])

VAE_Inference = models.VAEModel(Encoder=encoder2, Decoder=decoder2).to(device)
optimizer_VAE_Inference = Adam([{'params': VAE_Inference.parameters()}],
                 lr=1e-4)
if args.run_mode == 'test':
    unsupervised_inf_tic = time.perf_counter()
    checkpoint_VAE_Inference = torch.load('../saved_models/vae_model2_inference_' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.ckpt')
    VAE_Inference.load_state_dict(checkpoint_VAE_Inference)
    VAE_Inference.to(device)
    VAE_Inference.eval()
    for batch_idx, data_pair in enumerate(test_loader):  # train_loader
        # input_pair = torch.cat((data_pair[:, :, 0], data_pair[:, :, 1]), 1).to(device)

        x = data_pair[:, :, 0].float().to(device)
        y = data_pair[:, :, 1].float().to(device)

        y_true = y.cpu().detach().numpy()
        x_true = x.cpu().detach().numpy()

        for i, x_i in enumerate(x):
            y_i = y[i]
            x_hat = VAE_Inference(x_i.unsqueeze(0))
    unsupervised_inf_toc = time.perf_counter()
    unsupervised_inf_time = unsupervised_inf_toc - unsupervised_inf_tic
    print("Unsupervised inference testing time {}".format(unsupervised_inf_time))
else:
    unsupervised_inf_tic = time.perf_counter()
    VAE_Inference.train()
    for epoch in range(args.numEpoch):  # 600
        begin = time.time()
        total_overall = 0
        forward_loss = 0
        reproduction_loss = 0
        precision_for = 0
        recall_for = 0
        precision_re = 0
        recall_re = 0

        for batch_idx, data_pair in enumerate(train_loader):
            # input_pair = torch.cat((data_pair[:, :, 0], data_pair[:, :, 1]), 1).to(device)

            x = data_pair[:, :, 0].float().to(device)
            y = data_pair[:, :, 1].float().to(device)
            optimizer_VAE_Inference.zero_grad()

            y_true = y.cpu().detach().numpy()
            x_true = x.cpu().detach().numpy()

            loss_VAE_Inference = 0
            for i, x_i in enumerate(x):
                y_i = y[i]
                # x_hat = VAE_Inference(x_i.unsqueeze(0), adj.to_sparse())
                # x_hat = VAE_Inference(x_i.squeeze(0).unsqueeze(-1), utils.adj_process(adj).to(device))
                x_hat = VAE_Inference(x_i.unsqueeze(0))
                y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), adj)
                # y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), utils.adj_process(adj).coalesce().to(device))
                # total, re, forw = loss_functions.loss_all(x_i, x_hat, y_i, y_hat.squeeze(-1))
                total, re, forw = loss_functions.loss_all(x_i.unsqueeze(0), x_hat, y_i, y_hat.squeeze(-1))

                # loss_VAE_Inference += -torch.sum(y_hat)
                loss_VAE_Inference += total

                x_pred = x_hat.cpu().detach().numpy()
                x_pred[x_pred > 0.01] = 1
                x_pred[x_pred != 1] = 0

                precision_re += precision_score(x_true[i], x_pred[0], zero_division=0)
                recall_re += recall_score(x_true[i], x_pred[0], zero_division=0)

            total_overall += loss_VAE_Inference.item()
            loss_VAE_Inference = loss_VAE_Inference / x.size(0)

            loss_VAE_Inference.backward()
            optimizer_VAE_Inference.step()
            for p in forward_model.parameters():
                p.data.clamp_(min=0)

        end = time.time()
        print("Epoch: {}".format(epoch + 1),
              "\tTotal Loss: {:.4f}".format(total_overall / len(train_set)),
              "\tReconstruction Precision: {:.4f}".format(precision_re / len(train_set)),
              "\tReconstruction Recall: {:.4f}".format(recall_re / len(train_set)),
              "\tTime: {:.4f}".format(end - begin)
              )
    unsupervised_inf_toc = time.perf_counter()
    unsupervised_inf_time = unsupervised_inf_toc - unsupervised_inf_tic
    print("Unsupervised inference training time {}".format(unsupervised_inf_time))
    # saving models
    torch.save(VAE_Inference.state_dict(), '../saved_models/vae_model2_inference_' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.ckpt')
    print(' Model saved')


# Model Evaluation for unsupervised inference
seed_num = int(x_hat.sum().item())
y_true = torch.ones(x_hat.shape).to(device)
top_k = x_hat.topk(seed_num)
seed3 = top_k.indices[0].cpu().detach().numpy()

with open('../data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG', 'rb') as f:
    graph = pickle.load(f)

adj, inverse_pairs = graph['adj'], graph['inverse_pairs']

influence = utils.diffusion_evaluation(adj, seed3, diffusion = args.diffusion_model)
print('Diffusion count: {}'.format(influence))

G = nx.from_scipy_sparse_matrix(adj)
node_type = 0
nx.set_node_attributes(G, node_type, "node_type")
for s1 in seed:
    G.nodes[s1]['node_type'] = 1
for s2 in seed2:
    if G.nodes[s2]['node_type'] == 0:
        G.nodes[s2]['node_type'] = 2
    else:
        G.nodes[s2]['node_type'] = G.nodes[s2]['node_type']+2+1
for s3 in seed3:
    if G.nodes[s3]['node_type'] == 0:
        G.nodes[s3]['node_type'] = 3
    else:
        G.nodes[s3]['node_type'] = G.nodes[s3]['node_type']+3+1
# 1->1
# 2->2
# 3->3
# 1+2->4
# 1+3->5
# 2+3->6
# 1+2+3->8
df_edges = nx.to_pandas_edgelist(G)
df_edges = df_edges.drop('weight', axis=1)
df_edges.to_csv('../saved_models/edge_list_' + args.dataset + '_' + args.diffusion_model + str(10*args.seed_rate) + '.csv',index=False)
df_nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
df_nodes.columns.name = 'Id'
df_nodes.to_csv('../saved_models/node_list_' + args.dataset + '_' + args.diffusion_model + str(10*args.seed_rate) + '.csv')