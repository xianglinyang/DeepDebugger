from fileinput import filename
import os
import json

import numpy as np
from scipy import stats
from sklearn import neighbors

from singleVis.eval.evaluate import *
from singleVis.backend import *

class Evaluator:
    def __init__(self, data_provider, projector, verbose=1):
        self.data_provider = data_provider
        self.projector = projector
        self.verbose = verbose

    ####################################### ATOM #############################################

    def eval_nn_train(self, epoch, n_neighbors):
        train_data = self.data_provider.train_representation(epoch)
        embedding = self.projector.batch_project(epoch, train_data)
        val = evaluate_proj_nn_perseverance_knn(train_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#train# nn preserving: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, epoch))
        return val

    def eval_nn_test(self, epoch, n_neighbors):
        train_data = self.data_provider.train_representation(epoch)
        test_data = self.data_provider.test_representation(epoch)
        fitting_data = np.concatenate((train_data, test_data), axis=0)
        embedding = self.projector.batch_project(epoch, fitting_data)
        val = evaluate_proj_nn_perseverance_knn(fitting_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#test# nn preserving : {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, epoch))
        return val

    def eval_b_train(self, epoch, n_neighbors):
        train_data = self.data_provider.train_representation(epoch)
        border_centers = self.data_provider.border_representation(epoch)

        low_center = self.projector.batch_project(epoch, border_centers)
        low_train = self.projector.batch_project(epoch, train_data)

        val = evaluate_proj_boundary_perseverance_knn(train_data,
                                                      low_train,
                                                      border_centers,
                                                      low_center,
                                                      n_neighbors=n_neighbors)
        if self.verbose:
            print("#train# boundary preserving: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, epoch))
        return val

    def eval_b_test(self, epoch, n_neighbors):
        test_data = self.data_provider.test_representation(epoch)
        border_centers = self.data_provider.test_border_representation(epoch)

        low_center = self.projector.batch_project(epoch, border_centers)
        low_test = self.projector.batch_project(epoch, test_data)

        val = evaluate_proj_boundary_perseverance_knn(test_data,
                                                      low_test,
                                                      border_centers,
                                                      low_center,
                                                      n_neighbors=n_neighbors)
        if self.verbose:
            print("#test# boundary preserving: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, epoch))
        return val

    def eval_inv_train(self, epoch):
        train_data = self.data_provider.train_representation(epoch)
        embedding = self.projector.batch_project(epoch, train_data)
        inv_data = self.projector.batch_inverse(epoch, embedding)

        pred = self.data_provider.get_pred(epoch, train_data).argmax(axis=1)
        new_pred = self.data_provider.get_pred(epoch, inv_data).argmax(axis=1)

        val = evaluate_inv_accu(pred, new_pred)
        if self.verbose:
            print("#train# PPR: {:.2f} in epoch {:d}".format(val, epoch))
        return val

    def eval_inv_test(self, epoch):
        test_data = self.data_provider.test_representation(epoch)
        embedding = self.projector.batch_project(epoch, test_data)
        inv_data = self.projector.batch_inverse(epoch, embedding)

        pred = self.data_provider.get_pred(epoch, test_data).argmax(axis=1)
        new_pred = self.data_provider.get_pred(epoch, inv_data).argmax(axis=1)

        val = evaluate_inv_accu(pred, new_pred)
        if self.verbose:
            print("#test# PPR: {:.2f} in epoch {:d}".format(val, epoch))
        return val

    def eval_temporal_train(self, n_neighbors):
        eval_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p
        l = self.data_provider.train_num

        alpha = np.zeros((eval_num, l))
        delta_x = np.zeros((eval_num, l))

        for t in range(eval_num):
            prev_data = self.data_provider.train_representation(t * self.data_provider.p + self.data_provider.s)
            prev_embedding = self.projector.batch_project(t * self.data_provider.p + self.data_provider.s, prev_data)

            curr_data = self.data_provider.train_representation((t+1) * self.data_provider.p + self.data_provider.s)
            curr_embedding = self.projector.batch_project((t+1) * self.data_provider.p + self.data_provider.s, curr_data)

            alpha_ = find_neighbor_preserving_rate(prev_data, curr_data, n_neighbors=n_neighbors)
            delta_x_ = np.linalg.norm(prev_embedding - curr_embedding, axis=1)

            alpha[t] = alpha_
            delta_x[t] = delta_x_

        val_corr, corr_std = evaluate_proj_temporal_perseverance_corr(alpha, delta_x)
        if self.verbose:
            print("Temporal preserving (train): {:.3f}\t std :{:.3f}".format(val_corr, corr_std))
        return val_corr, corr_std

    def eval_temporal_test(self, n_neighbors):
        eval_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p
        l = self.data_provider.train_num + self.data_provider.test_num

        alpha = np.zeros((eval_num, l))
        delta_x = np.zeros((eval_num, l))
        for t in range(eval_num):
            prev_data_test = self.data_provider.test_representation(t * self.data_provider.p + self.data_provider.s)
            prev_data_train = self.data_provider.train_representation(t * self.data_provider.p + self.data_provider.s)
            prev_data = np.concatenate((prev_data_train, prev_data_test), axis=0)
            prev_embedding = self.projector.batch_project(t * self.data_provider.p + self.data_provider.s, prev_data)

            curr_data_test = self.data_provider.test_representation((t+1) * self.data_provider.p + self.data_provider.s)
            curr_data_train = self.data_provider.train_representation((t+1) * self.data_provider.p + self.data_provider.s)
            curr_data = np.concatenate((curr_data_train, curr_data_test), axis=0)
            curr_embedding = self.projector.batch_project((t+1) * self.data_provider.p + self.data_provider.s, curr_data)

            alpha_ = find_neighbor_preserving_rate(prev_data, curr_data, n_neighbors=n_neighbors)
            delta_x_ = np.linalg.norm(prev_embedding - curr_embedding, axis=1)

            alpha[t] = alpha_
            delta_x[t] = delta_x_

        val_corr, corr_std = evaluate_proj_temporal_perseverance_corr(alpha, delta_x)
        if self.verbose:
            print("Temporal preserving (test): {:.3f}\t std:{:.3f}".format(val_corr, corr_std))
        return val_corr, corr_std

    def eval_temporal_nn_train(self, epoch, n_neighbors):
        epoch_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p + 1
        l = self.data_provider.train_num
        high_dists = np.zeros((l, epoch_num))
        low_dists = np.zeros((l, epoch_num))

        curr_data = self.data_provider.train_representation(epoch)
        curr_embedding = self.projector.batch_project(epoch, curr_data)
        
        for t in range(epoch_num):
            data = self.data_provider.train_representation(t * self.data_provider.p + self.data_provider.s)
            embedding = self.projector.batch_project(t * self.data_provider.p + self.data_provider.s, data)

            high_dist = np.linalg.norm(curr_data - data, axis=1)
            low_dist = np.linalg.norm(curr_embedding - embedding, axis=1)
            high_dists[:, t] = high_dist
            low_dists[:, t] = low_dist
        
        # find the index of top k dists
        # argsort descent order
        high_orders = np.argsort(high_dists, axis=1)
        low_orders = np.argsort(low_dists, axis=1)

        high_rankings = high_orders[:, 1:n_neighbors+1]
        low_rankings = low_orders[:, 1:n_neighbors+1]
        
        corr = np.zeros(len(high_dists))
        for i in range(len(data)):
            corr[i] = len(np.intersect1d(high_rankings[i], low_rankings[i]))

        if self.verbose:
            print("Temporal temporal neighbor preserving (train) for {}-th epoch {}: {:.3f}\t std :{:.3f}".format(epoch, n_neighbors, corr.mean(), corr.std()))
        return float(corr.mean())

    def eval_temporal_nn_test(self, epoch, n_neighbors):
        epoch_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p + 1
        l = self.data_provider.test_num
        high_dists = np.zeros((l, epoch_num))
        low_dists = np.zeros((l, epoch_num))

        curr_data = self.data_provider.test_representation(epoch)
        curr_embedding = self.projector.batch_project(epoch, curr_data)

        for t in range(epoch_num):
            data = self.data_provider.test_representation(t * self.data_provider.p + self.data_provider.s)
            embedding = self.projector.batch_project(t * self.data_provider.p + self.data_provider.s, data)

            high_dist = np.linalg.norm(curr_data - data, axis=1)
            low_dist = np.linalg.norm(curr_embedding - embedding, axis=1)
            high_dists[:, t] = high_dist
            low_dists[:,t] = low_dist
        
        # find the index of top k dists
        high_orders = np.argsort(high_dists, axis=1)
        low_orders = np.argsort(low_dists, axis=1)
        
        high_rankings = high_orders[:, 1:n_neighbors+1]
        low_rankings = low_orders[:, 1:n_neighbors+1]
        corr = np.zeros(len(high_dists))
        for i in range(len(data)):
            corr[i] = len(np.intersect1d(high_rankings[i], low_rankings[i]))

        if self.verbose:
            print("Temporal ranking preserving (test) for {}-th epoch {}: {:.3f}\t std:{:.3f}".format(epoch, n_neighbors, corr.mean(), corr.std()))
        return float(corr.mean())

    def eval_spatial_temporal_nn_train(self, n_neighbors, feature_dim):
        """
            evaluate whether vis model can preserve the ranking of close spatial and temporal neighbors
        """
        #TODO: scale up to 100 epochs, need to speed up the process...
        epoch_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p + 1
        train_num = self.data_provider.train_num

        high_features = np.zeros((epoch_num*train_num, feature_dim))
        low_features = np.zeros((epoch_num*train_num, 2))

        for t in range(epoch_num):
            data = self.data_provider.train_representation(t * self.data_provider.p + self.data_provider.s)
            high_features[t*train_num:(t+1)*train_num] = np.copy(data)
            low_features[t*train_num:(t+1)*train_num] = self.projector.batch_project(t * self.data_provider.p + self.data_provider.s, data)
        
        val = evaluate_proj_nn_perseverance_knn(high_features, low_features, n_neighbors)

        if self.verbose:
            print("Spatial/Temporal nn preserving (train):\t{:.3f}/{:d}".format(val, n_neighbors))
        return val


    def eval_spatial_temporal_nn_test(self, n_neighbors, feature_dim):
        # find n temporal neighbors
        epoch_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p + 1
        train_num = self.data_provider.train_num
        test_num = self.data_provider.test_num
        num = train_num + test_num

        high_features = np.zeros((epoch_num*num, feature_dim))
        low_features = np.zeros((epoch_num*num, 2))

        for t in range(epoch_num):
            train_data = self.data_provider.train_representation(t * self.data_provider.p + self.data_provider.s)
            test_data = self.data_provider.test_representation(t * self.data_provider.p + self.data_provider.s)
            data = np.concatenate((train_data, test_data), axis=0)
            low_features[t*num:(t+1)*num] = self.projector.batch_project(t * self.data_provider.p + self.data_provider.s, data)
            high_features[t*num:(t+1)*num] = np.copy(data)

        val =evaluate_proj_nn_perseverance_knn(high_features, low_features, n_neighbors)
    
        if self.verbose:
            print("Spatial/Temporal nn preserving (test):\t{:.3f}/{:d}".format(val, n_neighbors))
        return val

    
    def eval_temporal_global_corr_train(self, start=None, end=None, period=None):
        # check if we use the default value
        if start is None:
            start = self.data_provider.s
            end = self.data_provider.e
            period = self.data_provider.p
        # set parameters
        LEN = self.data_provider.train_num()
        EPOCH = (end - start) // period + 1
        repr_dim = self.data_provider.representation_dim()
        all_train_repr = np.zeros((EPOCH,LEN,repr_dim))
        low_repr = np.zeros((EPOCH,LEN,2))

        # save all representation vectors
        for i in range(start,end + 1, period):
            index = (i - start) //  period
            all_train_repr[index] = self.data_provider.train_representation(i)
            low_repr[index] = self.projector.batch_project(i, all_train_repr[index])

        epochs = [i for i in range(start, end+1, period)]
        corrs = np.zeros((EPOCH,LEN))
        ps = np.zeros((EPOCH,LEN))
        for i in range(LEN):
            high_embeddings = all_train_repr[:,i,:].squeeze()
            low_embeddings = low_repr[:,i,:].squeeze()

            for epoch in epochs:
                e = (epoch - start) // period
                high_dists = np.linalg.norm(high_embeddings - high_embeddings[e], axis=1)
                low_dists = np.linalg.norm(low_embeddings - low_embeddings[e], axis=1)
                corr, p = stats.spearmanr(high_dists, low_dists)
                corrs[e][i] = corr
                ps[e][i] = p
        return corrs, ps
    
    def eval_temporal_global_corr_test(self, start=None, end=None, period=None):
        # check if we use the default value
        if start is None:
            start = self.data_provider.s
            end = self.data_provider.e
            period = self.data_provider.p
        TEST_LEN = self.data_provider.test_num()
        EPOCH = (end - start) // period + 1
        repr_dim = self.data_provider.representation_dim()

        all_test_repr = np.zeros((EPOCH,TEST_LEN,repr_dim))
        low_repr = np.zeros((EPOCH,TEST_LEN,2))
        for i in range(start,end + 1, period):
            all_test_repr[i-1] = self.data_provider.test_representation(i)
            low_repr[i-1] = self.projector.batch_project(i, all_test_repr[i-1])

        epochs = [i for i in range(start, end+1, period)]
        corrs = np.zeros((EPOCH,TEST_LEN))
        ps = np.zeros((EPOCH,TEST_LEN))

        for i in range(TEST_LEN):
            high_embeddings = all_test_repr[:,i,:].squeeze()
            low_embeddings = low_repr[:,i,:].squeeze()

            for epoch in epochs:
                e = (epoch - start) // period
                high_dists = np.linalg.norm(high_embeddings - high_embeddings[e], axis=1)
                low_dists = np.linalg.norm(low_embeddings - low_embeddings[e], axis=1)
                corr, p = stats.spearmanr(high_dists, low_dists)
                corrs[e][i] = corr
                ps[e][i] = p
        return corrs, ps

    #################################### helper functions #############################################

    def save_epoch_eval(self, n_epoch, n_neighbors, temporal_k=5, save_corrs=False, file_name="evaluation"):
        # save result
        save_dir = os.path.join(self.data_provider.model_path)
        save_file = os.path.join(save_dir, file_name + ".json")
        if not os.path.exists(save_file):
            evaluation = dict()
        else:
            f = open(save_file, "r")
            evaluation = json.load(f)
            f.close()
        n_key = str(n_neighbors)

        if n_key not in evaluation.keys():
            evaluation[n_key] = dict()
        if "nn_train" not in evaluation[n_key].keys():
            evaluation[n_key]["nn_train"] = dict()
        if "nn_test" not in evaluation[n_key].keys():
            evaluation[n_key]["nn_test"] = dict()
        if "b_train" not in evaluation[n_key].keys():
            evaluation[n_key]["b_train"] = dict()
        if "b_test" not in evaluation[n_key].keys():
            evaluation[n_key]["b_test"] = dict()
        if "ppr_train" not in evaluation.keys():
            evaluation["ppr_train"] = dict()
        if "ppr_test" not in evaluation.keys():
            evaluation["ppr_test"] = dict()
        # if "tnn_train" not in evaluation.keys():
        #     evaluation["tnn_train"] = dict()
        # if "tnn_test" not in evaluation.keys():
        #     evaluation["tnn_test"] = dict()

        epoch_key = str(n_epoch)
        evaluation[n_key]["nn_train"][epoch_key] = self.eval_nn_train(n_epoch, n_neighbors)
        evaluation[n_key]["nn_test"][epoch_key] = self.eval_nn_test(n_epoch, n_neighbors)

        # evaluation[n_key]["b_train"][epoch_key] = self.eval_b_train(n_epoch, n_neighbors)
        # evaluation[n_key]["b_test"][epoch_key] = self.eval_b_test(n_epoch, n_neighbors)
        evaluation[n_key]["b_train"][epoch_key] = 1
        evaluation[n_key]["b_test"][epoch_key] = 1

        evaluation["ppr_train"][epoch_key] = self.eval_inv_train(n_epoch)
        evaluation["ppr_test"][epoch_key] = self.eval_inv_test(n_epoch)

        # if epoch_key not in evaluation["tnn_train"].keys():
        #     evaluation["tnn_train"][epoch_key] = dict()
        # if epoch_key not in evaluation["tnn_test"].keys():
        #     evaluation["tnn_test"][epoch_key] = dict()

        # evaluation["tnn_train"][epoch_key][str(temporal_k)] = self.eval_temporal_nn_train(n_epoch, temporal_k)
        # evaluation["tnn_test"][epoch_key][str(temporal_k)] = self.eval_temporal_nn_test(n_epoch, temporal_k)

        # t_train_val, t_train_std = self.eval_temporal_train(n_neighbors)
        # evaluation[n_key]["temporal_train_mean"] = t_train_val
        # evaluation[n_key]["temporal_train_std"] = t_train_std
        # t_test_val, t_test_std = self.eval_temporal_test(n_neighbors)
        # evaluation[n_key]["temporal_test_mean"] = t_test_val
        # evaluation[n_key]["temporal_test_std"] = t_test_std

        # if save_corrs:
        #     corrs_train, ps_train = self.eval_temporal_global_corr_train()
        #     np.save(os.path.join(save_dir, filename + "_corrs.npy"), corrs_train)
        #     np.save(os.path.join(save_dir, filename + "_ps.npy"), ps_train)
        #     corrs_test, ps_test = self.eval_temporal_global_corr_test()
        #     np.save(os.path.join(save_dir, filename + "_test_corrs.npy"), corrs_test)
        #     np.save(os.path.join(save_dir, filename + "_test_ps.npy"), ps_test)

        with open(save_file, "w") as f:
            json.dump(evaluation, f)
        if self.verbose:
            print("Successfully save evaluation with {:d} neighbors...".format(n_neighbors))
    
    def get_eval(self, file_name="evaluation"):
        save_dir = os.path.join(self.data_provider.model_path, file_name + ".json")
        f = open(save_dir, "r")
        evaluation = json.load(f)
        f.close()
        return evaluation



