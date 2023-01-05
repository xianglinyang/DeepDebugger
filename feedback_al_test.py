import numpy as np
import os
import json
import torch
import pickle
import argparse

from singleVis.data import ActiveLearningDataProvider

def add_noise(rate, acc_idxs, rej_idxs):
    if rate == 0:
        return acc_idxs, rej_idxs
    acc_noise = np.random.choice(len(acc_idxs), size=int(len(acc_idxs)*rate))
    acc_noise = acc_idxs[acc_noise]
    new_acc = np.setdiff1d(acc_idxs, acc_noise)

    rej_noise = np.random.choice(len(rej_idxs), size=int(len(rej_idxs)*rate))
    rej_noise = rej_idxs[rej_noise]
    new_rej = np.setdiff1d(rej_idxs, rej_noise)

    new_acc = np.concatenate((new_acc, rej_noise), axis=0)
    new_rej = np.concatenate((new_rej, acc_noise), axis=0)
    return new_acc, new_rej


def init_sampling(tm, method, round, budget, ulb_wrong):
    print("Feedback sampling initialization ({}):".format(method))
    rate = list()
    for _ in range(round):
        correct = np.array([]).astype(np.int32)
        wrong = np.array([]).astype(np.int32)
        
        suggest_idxs, _ = tm.sample_batch_init(correct, wrong, budget)
        suggest_idxs = ulb_idxs[suggest_idxs]
        correct = np.intersect1d(suggest_idxs, ulb_wrong)
        rate.append(len(correct)/budget)
    print("Init success Rate:\t{:.4f}".format(sum(rate)/len(rate)))
    return sum(rate)/len(rate)


def feedback_sampling(tm, method, round, budget, ulb_wrong, noise_rate=0):
    print("Feedback sampling ({}) with noise rate {}:".format(method, noise_rate))
    rate = np.zeros(round)
    correct = np.array([]).astype(np.int32)
    wrong = np.array([]).astype(np.int32)
    map_ulb =ulb_idxs.tolist()

    map_acc_idxs = np.array([map_ulb.index(i) for i in correct]).astype(np.int32)
    map_rej_idxs = np.array([map_ulb.index(i) for i in wrong]).astype(np.int32)
    suggest_idxs, _ = tm.sample_batch_init(map_acc_idxs, map_rej_idxs, budget)
    suggest_idxs = ulb_idxs[suggest_idxs]
    correct = np.intersect1d(suggest_idxs, ulb_wrong)
    wrong = np.setdiff1d(suggest_idxs, correct)
    rate[0] = len(correct)/budget
    # inject noise
    correct, wrong = add_noise(noise_rate, correct, wrong)
    for r in range(1, round):
        map_acc_idxs = np.array([map_ulb.index(i) for i in correct]).astype(np.int32)
        map_rej_idxs = np.array([map_ulb.index(i) for i in wrong]).astype(np.int32)
        suggest_idxs,_ = tm.sample_batch(map_acc_idxs, map_rej_idxs, budget)
        suggest_idxs = ulb_idxs[suggest_idxs]

        c = np.intersect1d(np.intersect1d(suggest_idxs, ulb_idxs), ulb_wrong)
        w = np.setdiff1d(suggest_idxs, c)
        rate[r] = len(c) / budget

        # inject noise
        c, w = add_noise(noise_rate, c, w)
        correct = np.concatenate((correct, c), axis=0)
        wrong = np.concatenate((wrong, w), axis=0)
    print("Success Rate:\t{:.4f}".format(sum(rate)/len(rate)))
    ac_rate = np.array([rate[:i].mean() for i in range(1, len(rate)+1)])
    return ac_rate


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["cifar10","mnist","fmnist"])
parser.add_argument('--rate', type=int, choices=[30,10,20])
parser.add_argument("--tolerance", type=float, help="Feedback noise")
parser.add_argument("--budget", type=int)
# parser.add_argument("--ignore", type=float, default=0.0)
parser.add_argument("--init_round", type=int, default=10000)
parser.add_argument("--round", type=int, default=10, help="Feedback round")
parser.add_argument("-g", default="0")
args = parser.parse_args()

# tensorflow
visible_device = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = visible_device

# get hyperparameters
DATASET = args.dataset
RATE = args.rate
BUDGET = args.budget
TOLERANCE = args.tolerance
# IGNORE_RATE = args.ignore
ROUND = args.round
INIT_ROUND = args.init_round
GPU_ID = args.g


# load meta data
CONTENT_PATH = "/home/xianglin/projects/DVI_data/active_learning/random/resnet18/CIFAR10/{}".format(RATE)
with open(os.path.join(CONTENT_PATH, "config.json"), "r") as f:
    config = json.load(f)
config = config["DVI"]

CLASSES = config["CLASSES"]
if GPU_ID is None:
    GPU_ID = config["GPU"]
EPOCH_START = config["EPOCH_START"]
EPOCH_END = config["EPOCH_END"]
EPOCH_PERIOD = config["EPOCH_PERIOD"]

# Training parameter (subject model)
TRAINING_PARAMETER = config["TRAINING"]
NET = TRAINING_PARAMETER["NET"]
LEN = TRAINING_PARAMETER["train_num"]   # all

# define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))
data_provider = ActiveLearningDataProvider(CONTENT_PATH, net, EPOCH_START,device=DEVICE, classes=CLASSES, iteration_name="Epoch")

# meta info
lb_idxs = data_provider.get_labeled_idx(EPOCH_END)
ulb_idxs = data_provider.get_unlabeled_idx(LEN, lb_idxs)

data = data_provider.train_representation_all(EPOCH_END)
labels = data_provider.train_labels_all(EPOCH_END)
pred = data_provider.get_pred(EPOCH_END, data).argmax(1)
wrong_pred_idx = np.argwhere(pred!=labels).squeeze()
ulb_wrong = np.intersect1d(wrong_pred_idx, ulb_idxs)

# evaluate
with open(os.path.join(CONTENT_PATH, "Model", 'DVI_sample_recommender.pkl'), 'rb') as f:
    dvi_tm = pickle.load(f)
with open(os.path.join(CONTENT_PATH, "Model", 'TimeVis_sample_recommender.pkl'), 'rb') as f:
    timevis_tm = pickle.load(f)

#############################################
#                   init                    #
#############################################
# random init
print("Random sampling init")
random_rate = list()
pool = np.array(ulb_idxs)
for _ in range(INIT_ROUND):
    s_idxs = np.random.choice(pool,size=BUDGET,replace=False)
    random_rate.append(len(np.intersect1d(s_idxs, ulb_wrong))/BUDGET)
print("Success Rate:\t{:.4f}".format(sum(random_rate)/len(random_rate)))

# dvi init
init_sampling(tm=dvi_tm, method="DVI", round=INIT_ROUND, budget=BUDGET, ulb_wrong=ulb_wrong)

# timevis init
init_sampling(tm=timevis_tm, method="TimeVis", round=INIT_ROUND, budget=BUDGET, ulb_wrong=ulb_wrong)

#############################################
#                 Feedback                  #
#############################################
# random sampling
print("Random sampling feedback:")
random_rate = list()
pool = np.array(ulb_idxs)
for _ in range(ROUND):
    s_idxs = np.random.choice(pool,size=BUDGET,replace=False)
    random_rate.append(len(np.intersect1d(s_idxs, ulb_wrong))/BUDGET)
    pool = np.setdiff1d(pool, s_idxs)
print("Success Rate:\t{:.4f}".format(sum(random_rate)/len(random_rate)))

# dvi sampling
feedback_sampling(tm=dvi_tm, method="DVI", round=ROUND, budget=BUDGET, ulb_wrong=ulb_wrong, noise_rate=0.0)

# timevis sampling
feedback_sampling(tm=timevis_tm, method="TimeVis", round=ROUND, budget=BUDGET, ulb_wrong=ulb_wrong, noise_rate=0.0)

#############################################
#              Noise Feedback               #
#############################################
# dvi tolerance
feedback_sampling(tm=dvi_tm, method="DVI", round=ROUND, budget=BUDGET, ulb_wrong=ulb_wrong, noise_rate=TOLERANCE)

# timevis tolerance
feedback_sampling(tm=timevis_tm, method="TimeVis", round=ROUND, budget=BUDGET, ulb_wrong=ulb_wrong, noise_rate=TOLERANCE)

