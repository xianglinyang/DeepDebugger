import torch
import sys
import os
import json
import time
import numpy as np
import argparse

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params

from singleVis.custom_weighted_random_sampler import CustomWeightedRandomSampler
from singleVis.SingleVisualizationModel import SingleVisualizationModel
from singleVis.losses import SingleVisLoss, UmapLoss, ReconstructionLoss
from singleVis.edge_dataset import DataHandler
from singleVis.trainer import SingleVisTrainer
from singleVis.data import DataProvider
import singleVis.config as config
from singleVis.eval.evaluator import Evaluator
from singleVis.spatial_edge_constructor import kcSpatialEdgeConstructor
from singleVis.temporal_edge_constructor import GlobalTemporalEdgeConstructor

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################


parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('-d','--dataset', type=str)
parser.add_argument('-p',"--preprocess", choices=[0,1], type=int, default=0)
parser.add_argument('-g',"--gpu_id", type=int, choices=[0,1,2,3], default=0)
args = parser.parse_args()

CONTENT_PATH = args.content_path
DATASET = args.dataset
PREPROCESS = args.preprocess
GPU_ID = args.gpu_id

LEN = config.dataset_config[DATASET]["TRAINING_LEN"]
LAMBDA = config.dataset_config[DATASET]["LAMBDA"]
L_BOUND = config.dataset_config[DATASET]["L_BOUND"]
MAX_HAUSDORFF = config.dataset_config[DATASET]["MAX_HAUSDORFF"]
ALPHA = config.dataset_config[DATASET]["ALPHA"]
BETA = config.dataset_config[DATASET]["BETA"]
INIT_NUM = config.dataset_config[DATASET]["INIT_NUM"]
EPOCH_START = config.dataset_config[DATASET]["EPOCH_START"]
EPOCH_END = config.dataset_config[DATASET]["EPOCH_END"]
EPOCH_PERIOD = config.dataset_config[DATASET]["EPOCH_PERIOD"]
HIDDEN_LAYER = config.dataset_config[DATASET]["HIDDEN_LAYER"]

# define hyperparameters
DEVICE = torch.device("cuda:{:d}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
S_N_EPOCHS = config.dataset_config[DATASET]["training_config"]["S_N_EPOCHS"]
B_N_EPOCHS = config.dataset_config[DATASET]["training_config"]["B_N_EPOCHS"]
T_N_EPOCHS = config.dataset_config[DATASET]["training_config"]["T_N_EPOCHS"]
N_NEIGHBORS = config.dataset_config[DATASET]["training_config"]["N_NEIGHBORS"]
PATIENT = config.dataset_config[DATASET]["training_config"]["PATIENT"]
MAX_EPOCH = config.dataset_config[DATASET]["training_config"]["MAX_EPOCH"]

content_path = CONTENT_PATH
sys.path.append(content_path)

from Model.model import *
net = resnet18()
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


########################################################################################################################
#                                                    TRAINING SETTING                                                  #
########################################################################################################################
data_provider = DataProvider(content_path, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, verbose=1)
if PREPROCESS:
    data_provider.initialize(LEN//10, l_bound=L_BOUND)

model = SingleVisualizationModel(input_dims=512, output_dims=2, units=256, hidden_layer=HIDDEN_LAYER)
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)
umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA)

optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)


t0 = time.time()
spatial_cons = kcSpatialEdgeConstructor(data_provider=data_provider, init_num=INIT_NUM, s_n_epochs=S_N_EPOCHS, b_n_epochs=B_N_EPOCHS, n_neighbors=N_NEIGHBORS, MAX_HAUSDORFF=MAX_HAUSDORFF, ALPHA=ALPHA, BETA=BETA)
s_edge_to, s_edge_from, s_probs, feature_vectors, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention = spatial_cons.construct()
temporal_cons = GlobalTemporalEdgeConstructor(X=feature_vectors, time_step_nums=time_step_nums, sigmas=sigmas, rhos=rhos, n_neighbors=N_NEIGHBORS, n_epochs=T_N_EPOCHS)
t_edge_to, t_edge_from, t_probs = temporal_cons.construct()
t1 = time.time()

edge_to = np.concatenate((s_edge_to, t_edge_to),axis=0)
edge_from = np.concatenate((s_edge_from, t_edge_from), axis=0)
probs = np.concatenate((s_probs, t_probs), axis=0)
probs = probs / (probs.max()+1e-3)
eliminate_zeros = probs>1e-3
edge_to = edge_to[eliminate_zeros]
edge_from = edge_from[eliminate_zeros]
probs = probs[eliminate_zeros]

# save result
save_dir = os.path.join(data_provider.model_path, "SV_time_tnn.json")
if not os.path.exists(save_dir):
    evaluation = dict()
else:
    f = open(save_dir, "r")
    evaluation = json.load(f)
    f.close()
evaluation["complex_construction"] = round(t1-t0, 3)
with open(save_dir, 'w') as f:
    json.dump(evaluation, f)
print("constructing timeVis complex in {:.1f} seconds.".format(t1-t0))


dataset = DataHandler(edge_to, edge_from, feature_vectors, attention)
n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
# chosse sampler based on the number of dataset
if len(edge_to) > 2^24:
    sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
else:
    sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)

########################################################################################################################
#                                                       TRAIN                                                          #
########################################################################################################################

trainer = SingleVisTrainer(model, criterion, optimizer, lr_scheduler,edge_loader=edge_loader, DEVICE=DEVICE)

t2=time.time()
trainer.train(PATIENT, MAX_EPOCH)
t3 = time.time()
# save result
save_dir = os.path.join(data_provider.model_path, "SV_time_tnn.json")
if not os.path.exists(save_dir):
    evaluation = dict()
else:
    f = open(save_dir, "r")
    evaluation = json.load(f)
    f.close()
evaluation["training"] = round(t3-t2, 3)
with open(save_dir, 'w') as f:
    json.dump(evaluation, f)
trainer.save(save_dir=data_provider.model_path, file_name="tnn")
# trainer.load(file_path=os.path.join(data_provider.model_path,"SV.pth"))

########################################################################################################################
#                                                      VISUALIZATION                                                   #
########################################################################################################################

from singleVis.visualizer import visualizer

vis = visualizer(data_provider, trainer.model, 200, 10, classes)
save_dir = os.path.join(data_provider.content_path, "img")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
    vis.savefig(i, path=os.path.join(save_dir, "{}_{}_tnn.png".format(DATASET, i)))

    
########################################################################################################################
#                                                       EVALUATION                                                     #
########################################################################################################################
EVAL_EPOCH_DICT = {
    "mnist_full":[4, 12, 20],
    "fmnist_full":[10, 30, 50],
    "cifar10_full":[40, 120, 200]
}
eval_epochs = EVAL_EPOCH_DICT[DATASET]

evaluator = Evaluator(data_provider, trainer)
# evaluator.save_epoch_eval(eval_epochs[0], 10, temporal_k=3, save_corrs=True, file_name="test_evaluation_tnn")
evaluator.save_epoch_eval(eval_epochs[0], 15, temporal_k=5, save_corrs=False, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[0], 20, temporal_k=7, save_corrs=False, file_name="test_evaluation_tnn")

# evaluator.save_epoch_eval(eval_epochs[1], 10, temporal_k=3, save_corrs=True, file_name="test_evaluation_tnn")
evaluator.save_epoch_eval(eval_epochs[1], 15, temporal_k=5, save_corrs=False, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[1], 20, temporal_k=7, save_corrs=False, file_name="test_evaluation_tnn")

# evaluator.save_epoch_eval(eval_epochs[2], 10, temporal_k=3, save_corrs=True, file_name="test_evaluation_tnn")
evaluator.save_epoch_eval(eval_epochs[2], 15, temporal_k=5, save_corrs=False, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[2], 20, temporal_k=7, save_corrs=False, file_name="test_evaluation_tnn")