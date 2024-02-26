import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import AutoModel, AutoTokenizer, logging
import torchmetrics

from typing import List

# set the verbosity warning
logging.set_verbosity_error()


# =====================================
# Helper Functions
# =====================================


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    """Mean pooling - considers attention mask for correct averaging

    Args:
        model_output (Tensor): The model's output containing the token vectors.
        attention_mask (Tensor): The attention mask showing which tokens to use.

    Returns:
        Tensor: The tensors containing the mean pooled token embeddings.
    """

    # First element of model_output contains all token embeddings
    token_embeds = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    )
    return torch.sum(token_embeds * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# =====================================
# Main Pairing Model
# =====================================


class PairBERT(pl.LightningModule):
    def __init__(
        self,
        model_id: str,
        eps: float = 1e-5,
        lr: float = 1e-5,
        wd: float = 1e-2,
        eval_th: float = 0.5,
    ):
        super().__init__()

        # save the model hyperparameters
        self.save_hyperparameters("model_id", "eps", "lr", "wd", "eval_th")

        # get language model sprecific values
        self.lm = AutoModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

        # add evaluation metrics
        self.eval_accuracy = torchmetrics.Accuracy(task="binary", threshold=eval_th)
        self.eval_precision = torchmetrics.Precision(task="binary", threshold=eval_th)
        self.eval_recall = torchmetrics.Recall(task="binary", threshold=eval_th)
        self.eval_f1 = torchmetrics.F1Score(task="binary", threshold=eval_th)

    @torch.no_grad()
    def forward(self, input1: List[str], input2: List[str], device = None):
        """Classifies if the two text should be paired or not"""

        assert len(input1) == len(
            input2
        ), "The length of input1 is not the same as the length of input2"

        # get the inputs' encodings
        encoded_input1 = self.tokenizer(
            input1, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input2 = self.tokenizer(
            input2, padding=True, truncation=True, return_tensors="pt"
        )
        if device:
            encoded_input1 = encoded_input1.to(device)
            encoded_input2 = encoded_input2.to(device)

        # get the inputs' embeddings
        embedding_input1 = self._get_embeddings(encoded_input1)
        embedding_input2 = self._get_embeddings(encoded_input2)

        # scale the cosine values from [-1, 1] to [0, 1]
        logits = (1 + self.cos(embedding_input1, embedding_input2)) / 2
        return logits

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            eps=self.hparams.eps,
        )
        return optimizer

    def on_training_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, train_batch, batch_idx):
        loss = self._shared_eval_step(train_batch, batch_idx, stage="train")
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._shared_eval_step(val_batch, batch_idx, stage="val")
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        self._shared_log("val_metrics")

    def test_step(self, test_batch, batch_idx):
        loss = self._shared_eval_step(test_batch, batch_idx, stage="test")
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        self._shared_log("test_metrics")

    def _shared_eval_step(self, batch, batch_idx, stage):
        # get the input encodings
        encoded_input1 = batch["encoded_input1"]
        encoded_input2 = batch["encoded_input2"]
        target = batch["labels"]

        # get the inputs' embeddings
        embedding_input1 = self._get_embeddings(encoded_input1)
        embedding_input2 = self._get_embeddings(encoded_input2)

        # scale the values from [-1, 1] to [0, 1]
        logits = (1 + self.cos(embedding_input1, embedding_input2)) / 2

        # get the loss of the model
        loss = self._get_loss(logits, target)

        if stage in ["val", "test"]:
            self.eval_accuracy(logits, target)
            self.eval_precision(logits, target)
            self.eval_recall(logits, target)
            self.eval_f1(logits, target)

        return loss

    def _get_embeddings(self, encoded_input):
        """Calculates the sentence embeddings of the texts

        Args:
            encoded_input: The dictionary containing the encoding tensors.

        Returns:
            Tensor: The tensor containing the token embeddings.

        """
        model_output = self.lm(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _shared_log(self, state):
        self.log_dict(
            {
                f"{state}_accuracy": self.eval_accuracy,
                f"{state}_precision": self.eval_precision,
                f"{state}_recall": self.eval_recall,
                f"{state}_f1score": self.eval_f1,
            }
        )

    def _get_loss(self, logits, target):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(logits, target.to(torch.float32))
