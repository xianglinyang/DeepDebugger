from abc import ABC, abstractmethod
class SummaryWriterAbstractClass(ABC):
    """Writes entries directly to event files in the log_dir to be
    consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    @abstractmethod
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
    def add_training_data(self, dataset, transform):
        pass

    @abstractmethod
    def add_testing_data(self, dataset, transform):
        pass

    @abstractmethod
    def add_checkpoint_data(self, relative_path, state_dict, idxs):
        pass

    @abstractmethod
    def add_config(self, *args, **kwargs):
        pass

    @abstractmethod
    def add_subject_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def add_iteration_structure(self, *args, **kwargs):
        pass