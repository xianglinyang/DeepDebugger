"""The Projector class for visualization, serve as a helper module for evaluator and visualizer"""
import os
import json
import numpy as np
import torch

class Projector:
    def __init__(self, vis_model, content_path, segments, device) -> None:
        self.content_path = content_path
        self.vis_model = vis_model
        self.segments = segments    #[(1,6),(6, 15),(15,42),(42,200)]
        self.DEVICE = device
        self.current_range = (-1,-1)

    def load(self, iteration):
        if iteration < self.current_range[0] or iteration > self.current_range[1]:
            for i in range(len(self.segments)):
                s = self.segments[i][0]
                e = self.segments[i][1]
                # range [s, e)
                if iteration >= s and iteration <= e:
                    idx = i
                    break
            file_path = os.path.join(self.content_path, "Model", "tnn_hybrid_{}.pth".format(idx))
            save_model = torch.load(file_path, map_location=self.DEVICE)
            self.vis_model.load_state_dict(save_model["state_dict"])
            self.vis_model.to(self.DEVICE)
            self.vis_model.eval()
            self.current_range = (s, e)
            print("Successfully load the visualization model for range ({},{}]...".format(s,e))
        else:
            print("Same range as current visualization model...")

    def batch_project(self, iteration, data):
        self.load(iteration)
        embedding = self.vis_model.encoder(torch.from_numpy(data).to(dtype=torch.float32, device=self.DEVICE)).cpu().detach().numpy()
        return embedding
    
    def individual_project(self, iteration, data):
        self.load(iteration)
        embedding = self.vis_model.encoder(torch.from_numpy(np.expand_dims(data, axis=0)).to(dtype=torch.float32, device=self.DEVICE)).cpu().detach().numpy()
        return embedding.squeeze(axis=0)
    
    def batch_inverse(self, iteration, embedding):
        self.load(iteration)
        data = self.vis_model.decoder(torch.from_numpy(embedding).to(dtype=torch.float32, device=self.DEVICE)).cpu().detach().numpy()
        return data
    
    def individual_inverse(self, iteration, embedding):
        self.load(iteration)
        data = self.vis_model.decoder(torch.from_numpy(np.expand_dims(embedding, axis=0)).to(dtype=torch.float32, device=self.DEVICE)).cpu().detach().numpy()
        return data.squeeze(axis=0)


class ALProjector(Projector):
    def __init__(self, vis_model, content_path, vis_model_name, device) -> None:
        super().__init__(vis_model, content_path, None, device)
        self.current_range = None
        self.vis_model_name = vis_model_name

    def load(self, iteration):
        file_path=os.path.join(self.content_path, "Model", "Iteration_{}".format(iteration), self.vis_model_name)

        save_model = torch.load(file_path, map_location=self.DEVICE)
        self.vis_model.load_state_dict(save_model["state_dict"])
        self.vis_model.to(self.DEVICE)
        self.vis_model.eval()
        print("Successfully load the visualization model for Iteration {}...".format(iteration))

class EvalProjector(Projector):
    def __init__(self, vis_model, content_path, device, exp) -> None:
        super().__init__(vis_model, content_path, None, device)
        self.exp = exp
        file_path = os.path.join(content_path, "Model", "{}".format(exp), "segments.json")
        with open(file_path, "r") as f:
            self.segments = json.load(f)
    
    def load(self, iteration):
        if iteration < self.current_range[0] or iteration > self.current_range[1]:
            for i in range(len(self.segments)):
                s = self.segments[i][0]
                e = self.segments[i][1]
                # range [s, e)
                if iteration >= s and iteration <= e:
                    idx = i
                    break
            file_path = os.path.join(self.content_path, "Model", "{}".format(self.exp), "tnn_hybrid_{}.pth".format(idx))
            save_model = torch.load(file_path, map_location=self.DEVICE)
            self.vis_model.load_state_dict(save_model["state_dict"])
            self.vis_model.to(self.DEVICE)
            self.vis_model.eval()
            self.current_range = (s, e)
            print("Successfully load the visualization model for range ({},{}]...".format(s,e))
        else:
            print("Same range as current visualization model...")
        