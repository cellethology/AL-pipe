"""General functions to help with the project making code neater."""

import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.embedding_models.static.base_static_embedder import BaseStaticEmbedder
from al_pipe.embedding_models.static.onehot_embedding import OneHotEmbedder
from al_pipe.first_batch.base_first_batch import FirstBatchStrategy
from al_pipe.first_batch.random import RandomFirstBatch
from al_pipe.queries.base_strategy import BaseQueryStrategy
from al_pipe.queries.random_sampling import RandomQueryStrategy

SEQUENCE_CODE = {"A": 0, "C": 1, "T": 2, "G": 3, "N": 0}


def seed_all(seed: int) -> np.random.RandomState:
    """
    Set random seeds for reproducibility across PyTorch, NumPy and Python's random.

    Args:
        seed (int): The random seed value to use for all random number generators.

    Returns:
        np.random.RandomState(): random state from np lib could be used in scikit-learn
    """  # noqa: E501
    # https://scikit-learn.org/stable/common_pitfalls.html
    # TODO: account for the random state instance

    if not seed:
        seed = 0

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return np.random.RandomState(seed)


def avail_device(device: str) -> str:
    """
    Check device availability and return appropriate device.

    Args:
        device (str): Requested device ('cuda' or 'cpu')

    Returns:
        torch.device: Available device to use
    """
    torch.backends.cudnn.enabled = False
    use_cuda = torch.cuda.is_available()
    device = torch.device(device if use_cuda else "cpu")
    print("device: ", device)
    return device


def validate_dna(seq: str) -> None:
    """
    Validates that the DNA sequence contains only the characters A, C, G, T and N.

    Args:
        seq (str): The DNA sequence.

    Raises:
        ValueError: If the sequence contains invalid characters.
    """
    allowed = set("ACGTN")
    invalid = set(seq.upper()) - allowed
    if invalid:
        raise ValueError(f"Invalid characters in DNA sequence: {invalid}")


def onehot_encode_dna(seq: str) -> torch.Tensor:
    """Given DNA sequence input covert it to onehot encoded form."""
    if validate_dna(seq):
        # Convert the string into a NumPy array
        arr = np.array(list(seq))

        # masked_N scores indices where N occurs
        mask_N = arr == "N"

        # Use vectorized indexing to map each character to its corresponding index.
        indices = SEQUENCE_CODE[arr]

        # Create the one-hot encoding matrix by indexing into an identity matrix.
        one_hot = F.one_hot(
            torch.tensor(indices, dtype=torch.long), num_classes=len(SEQUENCE_CODE) - 1
        ).float()  # num_classses - 1 to account for N in the sequence

        # Zero out rows corresponding to 'N'.
        mask_N_tensor = torch.tensor(mask_N, dtype=torch.bool)
        one_hot[mask_N_tensor] = 0.0

        return one_hot


def pad_collate_fn(batch):
    """
    Collate function to pad variable-length sequences in a batch.
    `batch` is a list of tensors of shape (seq_len, num_classes).
    Returns a tensor of shape (batch_size, max_seq_len, num_classes).
    """
    # Use PyTorch's pad_sequence to pad the batch along dimension 0 (time dimension)
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch


def evaluate(loader, model, device) -> dict:
    """Run model in inference mode using a given data loader."""
    model.eval()
    model.to(device)
    pred = []
    truth = []
    results = {}

    for itr, batch in enumerate(loader):
        batch.to(device)

        with torch.no_grad():
            predicted = model(batch)
            target = batch.y
            pred.extend(predicted.cpu())
            truth.extend(target.cpu())

    # all genes
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy()
    results["truth"] = truth.detach().cpu().numpy()

    return results


def print_sys_stderr(s: str) -> None:
    """System print.

    Args:
        s (str): the string to print
    """
    print(s, flush=True, file=sys.stderr)


def initialize_model(model_config: dict, dataset: BaseDataset) -> BaseStaticEmbedder:
    """Initialize a model based on the model configuration."""
    if model_config["type"] == "OneHotEmbedder":
        return OneHotEmbedder(dataset)
    else:
        raise ValueError(f"Model class {model_config['type']} must inherit from BaseStaticEmbedder")


def initialize_first_batch_strategy(first_batch_config: dict, dataset: BaseDataset) -> FirstBatchStrategy:
    """Initialize a first batch strategy based on the first batch configuration."""
    if first_batch_config["type"] == "RandomFirstBatch":
        return RandomFirstBatch(dataset, first_batch_config["batch_size"])
    else:
        raise ValueError(f"First batch strategy class {first_batch_config['type']} is not supported.")


def initialize_query_strategy(query_config: dict, dataset: BaseDataset) -> BaseQueryStrategy:
    """Initialize a query strategy based on the query configuration."""
    if query_config["type"] == "RandomQueryStrategy":
        return RandomQueryStrategy(dataset, query_config["batch_size"])
    else:
        raise ValueError(f"Query strategy class {query_config['type']} is not supported.")
