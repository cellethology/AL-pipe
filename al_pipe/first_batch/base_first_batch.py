"""Abstract base class for first-batch strategies."""

from abc import ABC, abstractmethod

import torch

from al_pipe.data.base_dataset import BaseDataset


class FirstBatchStrategy(ABC):
    """
    An abstract base class for different first batch strategies.
    TODO: whether we need a constructor?
    """

    def __init__(self, dataset: BaseDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    @abstractmethod
    def select_initial_samples(self) -> list[torch.Tensor]:
        """
        Given a pd.Series of initial set of sequence input, return an subset
        of sequences as the first batch of training input.

        Returns:
            list[torch.Tensor]: A list of torch tensors selected as first batch.
        """
        raise NotImplementedError()
