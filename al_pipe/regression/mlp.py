"""MLP for prediction task using PyTorch Lightning."""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MLP(pl.LightningModule):
    """MLP layer for regression tasks implemented with PyTorch Lightning."""

    def __init__(
        self,
        sizes: list[int],
        learning_rate: float = 1e-3,
        batch_norm: bool = True,
        last_layer_act: str = "linear",
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize MLP model.

        Args:
            sizes: List of integers defining the network architecture
            learning_rate: Learning rate for optimization
            batch_norm: Whether to use batch normalization
            last_layer_act: Activation function for last layer
            weight_decay: L2 regularization factor
        """
        super().__init__()
        self.save_hyperparameters()

        layers = []
        for s in range(len(sizes) - 1):
            layers.extend(
                [
                    torch.nn.Linear(sizes[s], sizes[s + 1]),
                    torch.nn.BatchNorm1d(sizes[s + 1]) if batch_norm and s < len(sizes) - 1 else None,
                    torch.nn.ReLU(),
                ]
            )

        layers = [l for l in layers if l is not None][:-1]  # Remove last activation  # noqa: E741
        self.network = torch.nn.Sequential(*layers)
        self.activation = last_layer_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Model predictions
        """
        return self.network(x.view(x.size(0), -1))

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Tuple of (x, y)
            batch_idx: Index of current batch

        Returns:
            Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = F.smooth_l1_loss(y_hat, y, beta=1.0)
        # Logging the training loss this will be synced to wandb
        self.log(
            "train_loss",
            loss,
            on_step=True,  # log every batch
            on_epoch=True,  # also compute/record the epoch average
            prog_bar=True,  # show in progress bar
            logger=True,  # send to W&B (default)
            sync_dist=True,  # if you’re doing DDP
        )
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Tuple of (x, y)
            batch_idx: Index of current batch
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.smooth_l1_loss(y_hat, y, beta=1.0)
        self.log(
            "val_loss",
            val_loss,
            on_step=True,  # you usually only care per‐epoch for val
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Tuple of (x, y)
            batch_idx: Index of current batch
        """
        x, y = batch
        y_hat = self(x)
        test_loss = F.smooth_l1_loss(y_hat, y, beta=1.0)
        self.log(
            "test_loss",
            test_loss,
            on_step=True,  # you usually only care per‐epoch for val
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self) -> dict:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Optimizer configuration
        """
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.1)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
