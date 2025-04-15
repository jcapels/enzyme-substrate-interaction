from typing import Sequence, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import esm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(input_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights.data)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # attention_weights: (input_dim, 1)

        attention_scores = torch.matmul(x, self.attention_weights)  # (batch_size, seq_len)
        attention_scores = F.softmax(attention_scores, dim=1)  # normalize over sequence

        # Apply attention scores: weight each vector in the sequence
        attended_output = torch.sum(x * attention_scores.unsqueeze(-1), dim=1)  # (batch_size, input_dim)
        return attended_output, attention_scores

class ESM_NPClassifierFP(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super(ESM_NPClassifierFP, self).__init__()

        # Load ESM-2 model
        self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

        # Define additional layers if needed
        self.classifier = self.np_classifier_network = nn.Sequential(
            nn.Linear(self.esm_model.embed_dim*2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1),
        )

        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()  # Example loss function

        self.np_classifier_attention = AttentionLayer(6144)
        self.np_classifier_network = nn.Sequential(
            nn.Linear(6144, 2048),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 3072),
            nn.BatchNorm1d(3072),
            nn.Linear(3072, 1536),
            nn.BatchNorm1d(1536),
            nn.Linear(1536, 1536),
            nn.Dropout(0.2),
            nn.Linear(1536, self.esm_model.embed_dim)
        )

        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.precision = torchmetrics.Precision(task="binary")

    def forward(self, tokens, compound_fp):
        # Convert input to embeddings using ESM model
        results = self.esm_model(tokens, repr_layers=[6], return_contacts=False)
        token_representations = results['representations'][6][:, 0]

        # Pooling or further processing can be done here
        # For simplicity, take the mean of token representations

        attended_output, attention_scores = self.np_classifier_attention(compound_fp)
        x_compounds = self.np_classifier_network(attended_output)

        pooled_representation = torch.concat([token_representations, x_compounds], dim=1)

        # Pass through classifier
        output = self.classifier(pooled_representation)
        output = nn.Sigmoid()(output)
        return output

    def training_step(self, batch, batch_idx):
        _, _, tokens, compound_fp, y = batch
        outputs = self(tokens, compound_fp)
        loss = self.criterion(outputs.squeeze(), y.float())
        y_pred = outputs.squeeze()
        self.log('train_loss', loss, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=outputs.shape[0])
        
        accuracy = self.accuracy(y_pred, y)

        self.log("train_accuracy", accuracy, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, tokens, compound_fp, y = batch
        outputs = self(tokens, compound_fp)
        loss = self.criterion(outputs.squeeze(), y.float())

        y_pred = outputs.squeeze()
        self.log('val_loss', loss, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=outputs.shape[0])
        
        accuracy = self.accuracy(y_pred, y)

        self.log("val_accuracy", accuracy, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

from torch.utils.data import Dataset

class ProteinCompoundDataset(Dataset):
    def __init__(self, compound_fps, protein_seqs, interactions, labels):
        """
        Args:
            protein_seqs: list of strings, protein sequences
            compound_fps: list or tensor of shape (N, L, 6144), compound representations
            labels: list or tensor of labels (N,)
        """
        self.protein_seqs = protein_seqs
        self.compound_fps = compound_fps
        self.labels = labels
        self.interactions = interactions

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        protein_id, compound_id = self.interactions[idx]
        protein_seq = self.protein_seqs[protein_id]
        compound_fp = self.compound_fps[compound_id]
        label = self.labels[idx]

        return protein_id, protein_seq, compound_fp, label
    

class BatchConverterESMNPClassifierFP(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.

        batch_labels, seq_str_list, compound_fp, y = zip(*raw_batch)

        compound_fp = torch.tensor(compound_fp, dtype=torch.float32)
        y  = torch.tensor(y , dtype=torch.float32)

        batch_size = len(raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [seq_str[:self.truncation_seq_length] for seq_str in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens, compound_fp, y

