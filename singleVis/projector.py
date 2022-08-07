"""The Projector class for visualization, serve as a helper module for evaluator and visualizer"""
import os

import numpy as np
import torch
from scipy.special import softmax

class Projector:
    def __init__(self, vis_model, content_path, segments, device) -> None:
        self.content_path = content_path
        self.vis_model = vis_model
        self.segments = segments    #[(1,6),(6, 15),(15,42),(42,200)]
        self.DEVICE = device

    def load(self, iteration):
        for i in range(len(self.segments)):
            s = self.segments[i][0]
            e = self.segments[i][1]
            # range (s, e]
            if iteration > s and iteration <= e:
                idx = i
                break
        file_path = os.path.join(self.content_path, "Model", "tnn_hybrid_{}.pth".format(idx))
        save_model = torch.load(file_path, map_location=self.DEVICE)
        self.vis_model.load_state_dict(save_model["state_dict"])
        self.vis_model.to(self.DEVICE)
        self.vis_model.eval()
        print("Successfully load the visualization model for range ({},{}]...".format(s,e))

    def batch_project(self, iteration, data):
        self.load(iteration)
        embedding = self.vis_model.encoder(torch.from_numpy(data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        return embedding
    
    def individual_project(self, iteration, data):
        self.load(iteration)
        embedding = self.vis_model.encoder(torch.from_numpy(np.expand_dims(data, axis=0)).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        return embedding.squeeze(axis=0)
    
    def batch_inverse(self, iteration, embedding):
        self.load(iteration)
        data = self.vis_model.decoder(torch.from_numpy(embedding).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        return data
    
    def individual_inverse(self, iteration, embedding):
        self.load(iteration)
        data = self.vis_model.decoder(torch.from_numpy(np.expand_dims(embedding, axis=0)).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        return data.squeeze(axis=0)