from abc import ABC, abstractmethod

import os
import torch
from torch.utils.data import DataLoader
import json



class SummaryWriterAbstractClass(ABC):
    """Writes entries directly to event files in the log_dir to be
    consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """
    def __init__(self, log_dir):
        """Creates a `SummaryWriter` that will write out events and summaries
        to the event file.

        Args:
            log_dir (string): Save directory location.
        """
        log_dir = str(log_dir)
        self.log_dir = log_dir
    
    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self.log_dir
    
    @abstractmethod
    def add_training_data(self, dataset):
        pass

    @abstractmethod
    def add_testing_data(self, dataset):
        pass

    @abstractmethod
    def add_checkpoint_data(self, relative_path, state_dict, idxs):
        pass

    @abstractmethod
    def add_config(self, *args, **kwargs):
        pass

    @abstractmethod
    def add_iteration_structure(self, *args, **kwargs):
        pass

class SummaryWriter(SummaryWriterAbstractClass):

    def __init__(self, log_dir, batch_size=1000, num_worker=2):
        super().__init__(log_dir)
        self.batch_size = batch_size
        self.num_worker = num_worker

    def add_training_data(self, dataset):
        #! Noted, dataset transform need to be test transform
        dataloader = DataLoader(
            dataset,
            batch_size=self.bat,
            num_workers=self.num_worker,
            shuffle=False   # need to keep order, otherwise the index saved would be wrong
        )
        trainset_data = None
        trainset_label = None
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if trainset_data != None:
                # print(input_list.shape, inputs.shape)
                trainset_data = torch.cat((trainset_data, inputs), 0)
                trainset_label = torch.cat((trainset_label, targets), 0)
            else:
                trainset_data = inputs
                trainset_label = targets

        training_path = os.path.join(self.log_dir, "Training_data")
        os.makedirs(training_path)
        torch.save(trainset_data, os.path.join(training_path, "training_dataset_data.pth"))
        torch.save(trainset_label, os.path.join(training_path, "training_dataset_label.pth"))

    def add_testing_data(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            drop_last=False,
            shuffle=False
        )
        testset_data = None
        testset_label = None
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if testset_data != None:
                testset_data = torch.cat((testset_data, inputs), 0)
                testset_label = torch.cat((testset_label, targets), 0)
            else:
                testset_data = inputs
                testset_label = targets
        testing_path = os.path.join(self.log_dir, "Testing_data")
        os.makedirs(testing_path)
        torch.save(testset_data, os.path.join(testing_path, "testing_dataset_data.pth"))
        torch.save(testset_label, os.path.join(testing_path, "testing_dataset_label.pth"))
    
    def add_checkpoint_data(self, state_dict, idxs):
        checkpoints_path = os.path.join(self.log_dir, "Model")
        dirs = os.listdir(checkpoint_path)
        C = 0
        for dir in dirs:
            if "C_" in dir:
                C = max(C, int(dir.replace("C_", "")))
        C += 1

        checkpoint_path = os.path.join(checkpoints_path, "C_".format(C))
        os.makedirs(checkpoint_path)
        torch.save(state_dict, os.path.join(checkpoint_path, "subject_model.pth"))
        
        with open(os.path.join(checkpoint_path, "index.json"), "w") as f:
            json.dump(idxs, f)
    



