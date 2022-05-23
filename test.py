import numpy as np
from sklearn import datasets
import torch
import sys
import os

import argparse
import numpy as np
import json

from umap.umap_ import find_ab_params

from singleVis.SingleVisualizationModel import SingleVisualizationModel
from singleVis.losses import SingleVisLoss, UmapLoss, ReconstructionLoss
from singleVis.trainer import SingleVisTrainer
from singleVis.data import DataProvider
import singleVis.config as config
from singleVis.eval.evaluator import Evaluator

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################


parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('-d','--dataset', type=str)
parser.add_argument('-p',"--preprocess", choices=[0,1], default=0)
parser.add_argument('-g',"--gpu_id", type=int, choices=[0,1,2,3], default=0)
args = parser.parse_args()

CONTENT_PATH = args.content_path
DATASET = args.dataset
PREPROCESS = args.preprocess
GPU_ID = args.gpu_id

NET = config.dataset_config[DATASET]["NET"]
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

import Model.model as subject_model
# net = resnet18()
net = eval("subject_model.{}()".format(NET))
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


trainer = SingleVisTrainer(model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, edge_loader=None, DEVICE=DEVICE)
trainer.load(file_path=os.path.join(data_provider.model_path,"tnn.pth"))

########################################################################################################################
#                                                      VISUALIZATION                                                   #
########################################################################################################################

from singleVis.visualizer import visualizer

vis = visualizer(data_provider, trainer.model, 200, 10, classes)
save_dir = os.path.join(data_provider.content_path, "img")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

noise_label = os.path.join(data_provider.content_path, "noisy_label.json")
with open(noise_label, "r") as f:
    noise_labels = json.load(f)

for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
    # vis.savefig(i, path=os.path.join(save_dir, "{}_{}_tnn.png".format(DATASET, i)))
    data = data_provider.train_representation(i)
    labels = data_provider.train_labels(i)
    selected = labels != np.array(noise_labels)
    data = data[selected]
    labels = np.array(noise_labels)[selected]
    vis.savefig_cus(i, data, labels, labels, path=os.path.join(save_dir, "{}_{}_tnn.png".format(DATASET, i)))
########################################################################################################################
#                                                       EVALUATION                                                     #
########################################################################################################################
# EVAL_EPOCH_DICT = {
#     "mnist_full":[4, 12, 20],
#     "fmnist_full":[10, 30, 50],
#     "cifar10_full":[40, 120, 200]
# }
# eval_epochs = EVAL_EPOCH_DICT[DATASET]

# evaluator = Evaluator(data_provider, trainer)
# evaluator.save_epoch_eval(eval_epochs[0], 10, temporal_k=3, save_corrs=True, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[0], 15, temporal_k=5, save_corrs=False, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[0], 20, temporal_k=7, save_corrs=False, file_name="test_evaluation_tnn")

# evaluator.save_epoch_eval(eval_epochs[1], 10, temporal_k=3, save_corrs=True, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[1], 15, temporal_k=5, save_corrs=False, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[1], 20, temporal_k=7, save_corrs=False, file_name="test_evaluation_tnn")

# evaluator.save_epoch_eval(eval_epochs[2], 10, temporal_k=3, save_corrs=True, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[2], 15, temporal_k=5, save_corrs=False, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[2], 20, temporal_k=7, save_corrs=False, file_name="test_evaluation_tnn")
