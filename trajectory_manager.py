import numpy as np
from sklearn.cluster import Birch
from pynndescent import NNDescent

class TrajectoryManager:
    def __init__(self, samples, embeddings_2d, cls_num, period=100, metric="v"):
        """ trajectory manager with no feedback
        Parameters
        ----------
        samples: ndarray, shape(train_num, repr_dim)
        embeddings_2d : ndarray, shape (epoch_num, train_num, 2)
            all 2d embeddings of representations by timevis
        cls_num: int 
            the number of classes to cluster
        period: int
            We only look at the last *period* epochs of trajectory
        """
        self.samples = samples
        self.embeddings_2d = embeddings_2d
        train_num,time_steps, _ = embeddings_2d.shape
        self.train_num = train_num
        self.time_steps = time_steps
        self.period = period
        self.metric = metric
        self.cls_num = cls_num

        self.v = self.embeddings_2d[:, -period:,:][:,1:,:] - self.embeddings_2d[:, -period:,:][:,:-1,:]
        self.a = self.v[:,1:,:]-self.v[:,:-1,:]
    
    def clustered(self):
        brc = Birch(n_clusters=self.cls_num)
        if self.metric == "v":
            brc.fit(self.v.reshape(self.train_num, -1))
        elif self.metric == "a":
            brc.fit(self.a.reshape(self.train_num, -1))
        else:
            print("Not a valid metric")
        
        self.predict_sub_labels = brc.labels_
        self.suspect_clean = np.argsort(np.bincount(self.predict_sub_labels))[-3:]

        # to be updated each time
        self.sample_rate = np.ones(self.cls_num)
        self.sample_rate[self.suspect_clean] = 0.
        self.selected = np.zeros(self.train_num)
    
    def sample_one(self):
        # sample class
        rate = self.sample_rate/np.sum(self.sample_rate)
        cls = np.random.choice(self.cls_num, size=1, p=rate)[0]
        cls_idxs = np.argwhere(self.predict_sub_labels==cls)

        # check how many left
        selected_idxs = np.argwhere(self.selected==1)
        already_selected = np.intersect1d(cls_idxs, selected_idxs)
        not_selected = np.setdiff1d(cls_idxs, already_selected)

        # select one
        s_idx = np.random.choice(not_selected, size=1)[0]

        # update parameters
        self.selected[s_idx] = 1
        if len(not_selected) ==1:
            self.sample_rate[cls] = 0.
        return s_idx

    def sample_batch(self, budget):
        selected_idxs = list()
        for _ in range(budget):
            selected_idxs.append(self.sample_one())
        return np.array(selected_idxs)


class FeedbackTrajectoryManager(TrajectoryManager):
    def __init__(self, samples, embeddings_2d, cls_num, period=100, metric="v"):
        super().__init__(samples, embeddings_2d, cls_num, period, metric)
    
    def clustered(self):
        super().clustered()
        # to be updated
        # self.selected
        # self.sample_rate
        self.user_interested = np.zeros(self.train_num)
        self.success_rate = np.ones(self.cls_num)
    
    # def sample_one(self):
    #     interested_idxs = np.argwhere(self.user_interested==1).squeeze()
    #     # scores of success rate
    #     cls_rate = self.sample_rate*self.success_rate
    #     success_rate = cls_rate[self.predict_sub_labels]

    #     # similarity scores
    #     sim_rate = np.zeros(self.train_num)+0.5
    #     if len(interested_idxs)>0:
    #         interested_embedding = self.samples[interested_idxs,:]
    #         nd = NNDescent(self.samples)
    #         indices, _ = nd.query(interested_embedding, k=3)
    #         indices = np.unique(indices.reshape(-1))
    #         sim_rate[indices] = 1

    #     rate = sim_rate*success_rate
    #     not_selected = np.argwhere(self.selected==0).squeeze()
    #     norm_rate = rate[not_selected]/np.sum(rate[not_selected])
    #     s_idx = np.random.choice(not_selected, p=norm_rate, size=1)[0]
    #     self.selected[s_idx]=1
    #     return s_idx
    
    def sample_batch(self, budget):
        interested_idxs = np.argwhere(self.user_interested==1).squeeze()
        # scores of success rate
        cls_rate = self.sample_rate*self.success_rate
        success_rate = cls_rate[self.predict_sub_labels]

        # similarity scores
        sim_rate = np.zeros(self.train_num)+0.5
        if len(interested_idxs)>0:
            interested_embedding = self.samples[interested_idxs,:]
            nd = NNDescent(self.samples)
            indices, _ = nd.query(interested_embedding, k=3)
            indices = np.unique(indices.reshape(-1))
            sim_rate[indices] = 1

        rate = sim_rate*success_rate
        not_selected = np.argwhere(self.selected==0).squeeze()
        norm_rate = rate[not_selected]/np.sum(rate[not_selected])
        s_idxs = np.random.choice(not_selected, p=norm_rate, size=budget, replace=False)
        self.selected[s_idxs]=1
        return s_idxs
    
    def update_belief(self, interested_idxs):
        self.user_interested[interested_idxs]=1
        for cls in range(self.cls_num):
            cls_idxs = np.argwhere(self.predict_sub_labels==cls)
            interested_num = np.sum(self.user_interested[cls_idxs])
            query_sum = np.sum(self.selected[cls_idxs])
            if query_sum == 0:
                self.success_rate[cls] = 1
            else:
                self.success_rate[cls] = interested_num/query_sum
    
