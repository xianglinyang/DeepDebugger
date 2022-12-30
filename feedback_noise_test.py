import numpy as np
import os
import json
import pickle
import argparse

def add_noise(rate, acc_idxs, rej_idxs):
    if rate == 0:
        return acc_idxs, rej_idxs
    acc_noise = np.random.choice(len(acc_idxs), size=len(acc_idxs)*rate//1)
    acc_noise = acc_idxs[acc_noise]
    new_acc = np.setdiff1d(acc_idxs, acc_noise)

    rej_noise = np.random.choice(len(rej_idxs), size=len(rej_idxs)*rate//1)
    rej_noise = rej_idxs[rej_noise]
    new_rej = np.setdiff1d(rej_idxs, rej_noise)

    new_acc = np.concatenate((new_acc, rej_noise), axis=0)
    new_rej = np.concatenate((new_rej, acc_noise), axis=0)
    return new_acc, new_rej

def init_sampling(tm, method, round, budget):
    print("Feedback sampling initialization ({}):".format(method))
    init_rate = list()
    for _ in range(round):
        correct = np.array([]).astype(np.int32)
        wrong = np.array([]).astype(np.int32)
        selected,_ = tm.sample_batch_init(correct, wrong, budget)
        c = np.intersect1d(selected, noise_idxs)
        init_rate.append(len(c)/budget)
    print("Success Rate:\t{:.4f}".format(sum(init_rate)/len(init_rate)))
    return sum(init_rate)/len(init_rate)

def feedback_sampling(tm, method, round, budget, noise_rate=0.0):
    print("Feedback sampling ({}) with noise {}:".format(method, noise_rate))
    rate = list()
    correct = np.array([]).astype(np.int32)
    wrong = np.array([]).astype(np.int32)
    selected,_ = tm.sample_batch_init(correct, wrong, BUDGET)
    c = np.intersect1d(selected, noise_idxs)
    w = np.setdiff1d(selected, c)
    correct = np.concatenate((correct, c), axis=0)
    wrong = np.concatenate((wrong, w), axis=0)
    rate.append(len(correct)/budget)

    # inject noise
    correct, wrong = add_noise(noise_rate, correct, wrong)

    for _ in range(round-1):
        selected,_ = tm.sample_batch(correct, wrong, BUDGET)
        c = np.intersect1d(selected, noise_idxs)
        w = np.setdiff1d(selected, c)
        rate.append(len(c)/BUDGET)

        # inject noise
        c, w = add_noise(TOLERANCE, c, w)

        correct = np.concatenate((correct, c), axis=0)
        wrong = np.concatenate((wrong, w), axis=0)
    print("Success Rate:\t{:.4f}".format(rate))
    print(rate)
    return rate


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["cifar10","mnist","fmnist"])
parser.add_argument('--noise_rate', type=int, choices=[5,10,20])
parser.add_argument("--tolerance", type=float)
parser.add_argument("--budget", type=int)
parser.add_argument("--init_round", type=int, default=10000)
parser.add_argument("--round", type=int, default=10, help="Feedback round")
args = parser.parse_args()

DATASET = args.dataset
NOISE_RATE = args.noise_rate
BUDGET = args.budget
TOLERANCE = args.tolerance
ROUND = args.round
INIT_ROUND = args.init_round

CONTENT_PATH = "/home/xianglin/projects/DVI_data/noisy/symmetric/{}/{}/".format(DATASET, NOISE_RATE)
with open(os.path.join(CONTENT_PATH, "config.json"), "r") as f:
    config = json.load(f)
config = config["DVI"]
path = "{}/clean_label.json".format(CONTENT_PATH)
with open(path, "r") as f:
    clean_label = np.array(json.load(f))
path = "{}/noisy_label.json".format(CONTENT_PATH)
with open(path, "r") as f:
    noisy_label = np.array(json.load(f))

TRAINING_PARAMETER = config["TRAINING"]
LEN = TRAINING_PARAMETER["train_num"]


# Evaluate
noise_idxs = np.argwhere(clean_label!=noisy_label).squeeze()
with open(os.path.join(CONTENT_PATH,  'DVI_sample_recommender.pkl'), 'rb') as f:
    dvi_tm = pickle.load(f)
with open(os.path.join(CONTENT_PATH,  'TimeVis_sample_recommender.pkl'), 'rb') as f:
    timevis_tm = pickle.load(f)

#############################################
#                   init                    #
#############################################
# random init
print("Random sampling init")
s_rate = list()
pool = np.arange(LEN)
for _ in range(INIT_ROUND):
    s_idxs = np.random.choice(pool,size=BUDGET,replace=False)
    s_rate.append(len(np.intersect1d(s_idxs, noise_idxs))/BUDGET)
print("Success Rate:\t{:.4f}".format(sum(s_rate)/len(s_rate)))

# dvi init
init_sampling(dvi_tm, method="DVI", round=INIT_ROUND, budget=BUDGET)

# timevis init
init_sampling(timevis_tm, method="TimeVis", round=INIT_ROUND, budget=BUDGET)

#############################################
#                 Feedback                  #
#############################################
# random Feedback
print("Random sampling feedback")
random_rate = list()
pool = np.arange(LEN)
for _ in range(11):
    s_idxs = np.random.choice(pool,size=BUDGET,replace=False)
    random_rate.append(len(np.intersect1d(s_idxs, noise_idxs))/BUDGET)
    pool = np.setdiff1d(pool, s_idxs)
print("Success Rate:\t{:.4f}".format(sum(random_rate)/len(random_rate)))
print(random_rate)

# dvi Feedback
feedback_sampling(tm=dvi_tm, method="DVI", round=ROUND, budget=BUDGET)

# timevis Feedback
feedback_sampling(tm=timevis_tm, method="TimeVis", round=ROUND, budget=BUDGET)


#############################################
#              Noise Feedback               #
#############################################

# dvi Feedback with noise
feedback_sampling(tm=dvi_tm, method="DVI", round=ROUND, budget=BUDGET, noise_rate=TOLERANCE)

# timevis Feedback with noise
feedback_sampling(tm=timevis_tm, method="TimeVis", round=ROUND, budget=BUDGET, noise_rate=TOLERANCE)

