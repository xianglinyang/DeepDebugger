from abc import ABC, abstractmethod

import torch
import sys
import os
import time
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params

from singleVis.custom_weighted_random_sampler import CustomWeightedRandomSampler
from singleVis.SingleVisualizationModel import VisModel
from singleVis.losses import HybridLoss, SmoothnessLoss, UmapLoss, ReconstructionLoss
from singleVis.edge_dataset import HybridDataHandler
from singleVis.trainer import HybridVisTrainer
from singleVis.data import NormalDataProvider
from singleVis.spatial_edge_constructor import kcHybridSpatialEdgeConstructor
from singleVis.temporal_edge_constructor import GlobalTemporalEdgeConstructor
from singleVis.projector import Projector
from singleVis.segmenter import Segmenter
from singleVis.eval.evaluator import Evaluator

class StrategyAbstractClass(ABC):
    def __init__(self, config):
        self.config = config
        self._init()

    @abstractmethod
    def visualize_embedding(self):
        pass

class DeepDebugger(StrategyAbstractClass):
    
    def _init(self):
        CONTENT_PATH = self.config.CONTENT_PATH
        sys.path.append(CONTENT_PATH)
        # record output information
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
        sys.stdout = open(os.path.join(CONTENT_PATH, now+".txt"), "w")

        CLASSES = self.config["CLASSES"]
        GPU_ID = self.config["GPU"]
        EPOCH_START = self.config["EPOCH_START"]
        EPOCH_END = self.config["EPOCH_END"]
        EPOCH_PERIOD = self.config["EPOCH_PERIOD"]

        # Training parameter (subject model)
        TRAINING_PARAMETER = self.config["TRAINING"]
        NET = TRAINING_PARAMETER["NET"]

        # Training parameter (visualization model)
        VISUALIZATION_PARAMETER = self.config["VISUALIZATION"]
        LAMBDA = VISUALIZATION_PARAMETER["LAMBDA"]
        S_LAMBDA = VISUALIZATION_PARAMETER["S_LAMBDA"]
        ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
        DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]

        # define hyperparameters
        self.DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

        import Model.model as subject_model
        net = eval("subject_model.{}()".format(NET))

        self.data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=self.DEVICE, classes=CLASSES,verbose=1)        
        self.model = VisModel(ENCODER_DIMS, DECODER_DIMS)
        negative_sample_rate = 5
        min_dist = .1
        _a, _b = find_ab_params(1.0, min_dist)
        umap_loss_fn = UmapLoss(negative_sample_rate, self.DEVICE, _a, _b, repulsion_strength=1.0)
        recon_loss_fn = ReconstructionLoss(beta=1.0)
        smooth_loss_fn = SmoothnessLoss(margin=0.5)
        self.criterion = HybridLoss(umap_loss_fn, recon_loss_fn, smooth_loss_fn, lambd1=LAMBDA, lambd2=S_LAMBDA)
        self.segmenter = Segmenter(data_provider=self.data_provider, threshold=78.5, range_s=EPOCH_START, range_e=EPOCH_END, range_p=EPOCH_PERIOD)
        self.projector = Projector(vis_model=self.model, content_path=CONTENT_PATH, segments=None, device=self.DEVICE)

    def _preprocess(self):
        PREPROCESS = self.config["VISUALIZATION"]["PREPROCESS"]
        # Training parameter (subject model)
        TRAINING_PARAMETER = self.config["TRAINING"]
        LEN = TRAINING_PARAMETER["train_num"]
        # Training parameter (visualization model)
        VISUALIZATION_PARAMETER = self.config["VISUALIZATION"]
        B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
        L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
        if PREPROCESS:
            self.data_provider._meta_data()
            if B_N_EPOCHS >0:
                self.data_provider._estimate_boundary(LEN//10, l_bound=L_BOUND)
    
    def _segment(self):
        SEGMENTS = self.segmenter.segment()
        self.projector.segments = SEGMENTS
    
    def _train(self):
        TRAINING_PARAMETER = self.config["TRAINING"]
        LEN = TRAINING_PARAMETER["train_num"]
        VISUALIZATION_PARAMETER = self.config["VISUALIZATION"]
        SEGMENTS = self.segmenter.segments
        B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
        INIT_NUM = VISUALIZATION_PARAMETER["INIT_NUM"]
        ALPHA = VISUALIZATION_PARAMETER["ALPHA"]
        BETA = VISUALIZATION_PARAMETER["BETA"]
        MAX_HAUSDORFF = VISUALIZATION_PARAMETER["MAX_HAUSDORFF"]
        S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
        T_N_EPOCHS = VISUALIZATION_PARAMETER["T_N_EPOCHS"]
        N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
        PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
        MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]
        VIS_MODEL_NAME = VISUALIZATION_PARAMETER["VIS_MODEL_NAME"]
        
        prev_selected = np.random.choice(np.arange(LEN), size=INIT_NUM, replace=False)
        prev_embedding = None
        start_point = len(SEGMENTS)-1
        c0=None
        d0=None


        for seg in range(start_point,-1,-1):
            epoch_start, epoch_end = SEGMENTS[seg]
            self.data_provider.update_interval(epoch_s=epoch_start, epoch_e=epoch_end)

            optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)

            t0 = time.time()
            spatial_cons = kcHybridSpatialEdgeConstructor(data_provider=self.data_provider, init_num=INIT_NUM, s_n_epochs=S_N_EPOCHS, b_n_epochs=B_N_EPOCHS, n_neighbors=N_NEIGHBORS, MAX_HAUSDORFF=MAX_HAUSDORFF, ALPHA=ALPHA, BETA=BETA, init_idxs=prev_selected, init_embeddings=prev_embedding, c0=c0, d0=d0)
            s_edge_to, s_edge_from, s_probs, feature_vectors, embedded, coefficient, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention, (c0,d0) = spatial_cons.construct()

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

            dataset = HybridDataHandler(edge_to, edge_from, feature_vectors, attention, embedded, coefficient)
            n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
            # chose sampler based on the number of dataset
            if len(edge_to) > 2^24:
                sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
            else:
                sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
            edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)

            ########################################################################################################################
            #                                                       TRAIN                                                          #
            ########################################################################################################################

            trainer = HybridVisTrainer(model, self.criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=self.DEVICE)

            t2=time.time()
            trainer.train(PATIENT, MAX_EPOCH)
            t3 = time.time()

            file_name = "DeepDebugger_time"
            trainer.record_time(file_name, "complex_construction", seg, t1-t0)
            trainer.record_time(file_name, "training", seg, t3-t2)

            trainer.save(save_dir=self.data_provider.model_path, file_name="{}_{}".format(VIS_MODEL_NAME, seg))
            model = trainer.model

            # update prev_idxs and prev_embedding
            prev_selected = time_step_idxs_list[0]
            prev_data = torch.from_numpy(feature_vectors[:len(prev_selected)]).to(dtype=torch.float32, device=self.DEVICE)
            model.to(device=self.DEVICE)
            prev_embedding = model.encoder(prev_data).cpu().detach().numpy()
    

    def _evaluate(self):
        EPOCH_START = self.config["EPOCH_START"]
        EPOCH_END = self.config["EPOCH_END"]
        EPOCH_PERIOD = self.config["EPOCH_PERIOD"]
        VISUALIZATION_PARAMETER = self.config["VISUALIZATION"]
        EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]
        N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
        eval_epochs = list(range(EPOCH_START, EPOCH_END, EPOCH_PERIOD))
        self.evaluator = Evaluator(self.data_provider, self.projector)
        for eval_epoch in eval_epochs:
            self.evaluator.save_epoch_eval(eval_epoch, N_NEIGHBORS, temporal_k=5, file_name="{}".format(EVALUATION_NAME))


    def visualize_embedding(self):
        self._preprocess()
        self._segment()
        self._train()
        self._evaluate()
