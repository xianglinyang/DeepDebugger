import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import Birch
# TODO random ignore

class TrajectoryManager:
    def __init__(self, embeddings_2d, cls_num, period=100, metric="a"):
        """ trajectory manager with no feedback
        Parameters
        ----------
        samples: ndarray, shape(train_num, repr_dim)
        embeddings_2d : ndarray, shape (train_num, epoch_num, 2)
            all 2d embeddings of representations by timevis
        cls_num: int 
            the number of classes to cluster
        period: int
            We only look at the last *period* epochs of trajectory
        """
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
    
    def sample_batch(self, budget):
        not_selected = np.argwhere(self.selected==0).squeeze()
        s_idxs = np.random.choice(not_selected, size=budget)
        return s_idxs
        
    def sample_normal(self, budget):
        # sample class
        normal_rate = 1 - self.sample_rate
        sample_rate = normal_rate[self.predict_sub_labels]
        # check how many left
        not_selected = np.argwhere(self.selected==0).squeeze()
        # select one
        normed_rate = sample_rate[not_selected]/np.sum(sample_rate[not_selected])
        s_idxs = np.random.choice(not_selected, p=normed_rate,size=budget)
        return s_idxs
    
    def update_belief(self, idxs):
        # update parameters
        if len(idxs)>0:
            self.selected[idxs] = 1
    

class FeedbackTrajectoryManager(TrajectoryManager):
    def __init__(self, embeddings_2d, cls_num, period=100, metric="a"):
        super().__init__(embeddings_2d, cls_num, period, metric)
    
    def clustered(self):
        super().clustered()
        # to be updated
        # self.selected
        # self.sample_rate
        self.user_acc = np.zeros(self.train_num)
        self.user_rej = np.zeros(self.train_num)
    
    def sample_batch(self, budget, return_scores=False):
        acc_idxs = np.argwhere(self.user_acc==1).squeeze()
        rej_idxs = np.argwhere(self.user_rej==1).squeeze() 

        acc_rate = np.zeros(self.train_num)
        rej_rate = np.zeros(self.train_num)
        if len(acc_idxs)>0:
            acc_rate[acc_idxs]=1.
        if len(rej_idxs)>0:
            rej_rate[rej_idxs]=1.

        if len(np.intersect1d(acc_idxs, rej_idxs))>0:
            raise Exception("Intersection between acc idxs and rej idxs!")

        exploit_rate = np.zeros(self.cls_num)
        explore_rate = np.zeros(self.cls_num)
        for cls in range(self.cls_num):
            cls_idxs = np.argwhere(self.predict_sub_labels==cls).squeeze()
            acc_num = np.sum(acc_rate[cls_idxs])
            rej_num = np.sum(rej_rate[cls_idxs])
            query_sum = acc_num + rej_num
            if query_sum > 0:
                exploit_rate[cls] = acc_num/query_sum
            explore_rate[cls] = 1 - query_sum/len(cls_idxs)
        
        # remove clean cls
        rate = (explore_rate + exploit_rate)* self.sample_rate
        sample_rate = rate[self.predict_sub_labels]
        not_selected = np.argwhere(self.selected==0).squeeze()
        norm_rate = sample_rate[not_selected]/np.sum(sample_rate[not_selected])
        s_idxs = np.random.choice(not_selected, p=norm_rate, size=budget, replace=False)

        if return_scores:
            scores = sample_rate[s_idxs]
            return s_idxs, scores
        return s_idxs
    
    def update_belief(self, acc_idxs, rej_idxs):
        if len(acc_idxs)>0:
            self.user_acc[acc_idxs]=1
            self.selected[acc_idxs] = 1
        if len(rej_idxs)>0:
            self.user_rej[rej_idxs]=1
            self.selected[rej_idxs] = 1
    

class TBSampling(TrajectoryManager):
    """with no memory, for user study"""
    def __init__(self, embeddings_2d, cls_num, period=100, metric="a"):
        super().__init__(embeddings_2d, cls_num, period, metric)

    def sample_batch(self, acc_idxs, rej_idxs, budget, return_scores=True):
        selected = np.zeros(self.train_num)
        if len(acc_idxs)>0:
            selected[acc_idxs] = 1.
        if len(rej_idxs)>0:
            selected[rej_idxs] = 1.
        if len(np.intersect1d(acc_idxs, rej_idxs))>0:
            raise Exception("Intersection between acc idxs and rej idxs!")
        
        sample_rate = self.sample_rate[self.predict_sub_labels]
        not_selected = np.argwhere(selected==0).squeeze()
        norm_rate = sample_rate[not_selected]/np.sum(sample_rate[not_selected])
        s_idxs = np.random.choice(not_selected, p=norm_rate, size=budget, replace=False)
        if return_scores:
            scores = sample_rate[s_idxs]
            return s_idxs, scores
        return s_idxs


class FeedbackSampling(TrajectoryManager):
    """with no memory, for user study"""
    def __init__(self, embeddings_2d, cls_num, period=100, metric="a"):
        super().__init__(embeddings_2d, cls_num, period, metric)

    def sample_batch(self, acc_idxs, rej_idxs, budget, return_scores=True):
        acc_rate = np.zeros(self.train_num)
        rej_rate = np.zeros(self.train_num)
        selected = np.zeros(self.train_num)
        if len(acc_idxs)>0:
            acc_rate[acc_idxs]=1.
            selected[acc_idxs] = 1.
        if len(rej_idxs)>0:
            rej_rate[rej_idxs]=1.
            selected[rej_idxs] = 1.
        if len(np.intersect1d(acc_idxs, rej_idxs))>0:
            raise Exception("Intersection between acc idxs and rej idxs!")

        exploit_rate = np.zeros(self.cls_num)
        explore_rate = np.zeros(self.cls_num)
        for cls in range(self.cls_num):
            cls_idxs = np.argwhere(self.predict_sub_labels==cls).squeeze()
            acc_num = np.sum(acc_rate[cls_idxs])
            rej_num = np.sum(rej_rate[cls_idxs])
            query_sum = acc_num + rej_num
            if query_sum > 0:
                exploit_rate[cls] = acc_num/query_sum
            explore_rate[cls] = 1 - query_sum/len(cls_idxs)
        
        # remove clean cls
        rate = (explore_rate + exploit_rate)* self.sample_rate
        sample_rate = rate[self.predict_sub_labels]
        not_selected = np.argwhere(selected==0).squeeze()
        norm_rate = sample_rate[not_selected]/np.sum(sample_rate[not_selected])
        s_idxs = np.random.choice(not_selected, p=norm_rate, size=budget, replace=False)
        if return_scores:
            scores = sample_rate[s_idxs]
            return s_idxs, scores
        return s_idxs

class AnormalyDetector:
    def __init__(self, uncertainty, embeddings_2d, cls_num, period=100, metric="a"):
        """ trajectory manager with no feedback
        Parameters
        ----------
        samples: ndarray, shape(train_num, repr_dim)
        embeddings_2d : ndarray, shape (train_num, epoch_num, 2)
            all 2d embeddings of representations by timevis
        cls_num: int 
            the number of classes to cluster
        period: int
            We only look at the last *period* epochs of trajectory
        """
        self.uncertainty = uncertainty
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
        self.cls_score = np.zeros(self.cls_num)
        for cls in range(self.cls_num):
            self.cls_score[cls] = 1 - np.sum(self.predict_sub_labels==cls)/self.train_num
    
    def sample_batch_init(self, budget):
        scores = (self.uncertainty + self.cls_score[self.predict_sub_labels])/2
        norm_rate = scores/np.sum(scores)
        s_idxs = np.random.choice(self.train_num, p=norm_rate, size=budget, replace=False)
        return s_idxs, scores
    
    def sample_batch(self, acc_idxs, rej_idxs, budget):
        s1 = self.uncertainty
        s2 = self.cls_scores[self.predict_sub_labels]
        # X = np.concatenate((s1, s2), axis=1)
        X = np.hstack((s1,s2)).transpose([1,0])


        exp_idxs = np.concatenate((acc_idxs, rej_idxs), axis=0)
        target_X = X[exp_idxs]
        target_Y = np.zeros(len(exp_idxs))
        target_Y[:len(acc_idxs)] = 1
        krr = KernelRidge(alpha=1.0)
        krr.fit(target_X, target_Y)
        scores = krr.predict(X)

        not_selected = np.setdiff1d(np.arange(self.train_num), exp_idxs)
        remain_scores = scores[not_selected]
        args = np.argsort(remain_scores)[-budget:]
        selected_idxs = not_selected[args]
        return selected_idxs, scores[selected_idxs]
