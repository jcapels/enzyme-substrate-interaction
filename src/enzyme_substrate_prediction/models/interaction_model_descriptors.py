import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from torch.nn import BCELoss

from torch.utils.data import Dataset
import torchmetrics

from enzyme_substrate_prediction.models.np_classifier import NPClassifierDNN
from enzyme_substrate_prediction.models.protein_model import ESM2_3B




class AttentionBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])) #.cuda()

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)
    
        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2, 3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        self.scale = self.scale.to(V.device)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix

class InteractionModelCustomized(pl.LightningModule):
    def __init__(self, protein_head_layers, compound_head_layers, protein_descriptors_layers, compound_descriptors_layers, final_head_layers, 
                 batch_norm_final, batch_norm_modules, learning_rate, compounds_model_path="np_classifier.ckpt", protein_model_path="esm2_3b.pt"):
        super(InteractionModelCustomized, self).__init__()
        self.save_hyperparameters()

        self.compounds_model = NPClassifierDNN()
        self.proteins_model = ESM2_3B()
        # if dropout > 0:
        #     self.dropout = nn.Dropout(dropout)

        np_classifier_model_pretrained_dict = torch.load(compounds_model_path, map_location="cpu")
        np_classifier_model_pretrained_dict = {k.replace("np_classifier_model.", ""): v 
                                               for k, v in np_classifier_model_pretrained_dict["state_dict"].items() 
                                               if k.replace("np_classifier_model.", "") 
                                               in self.compounds_model.state_dict()}
        protein_model = torch.load(protein_model_path, map_location="cpu")
        
        self.compounds_model.load_state_dict(np_classifier_model_pretrained_dict)
        self.proteins_model.load_state_dict(protein_model)

        self.compounds_model.eval()
        self.proteins_model.eval()

        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.precision = torchmetrics.Precision(task="binary")

        new_module = nn.Sequential()
        new_module.add_module("proteins_head_0", nn.Linear(2560, protein_head_layers[0]))
        if batch_norm_modules:
            new_module.add_module("proteins_head_0_batch_norm", nn.BatchNorm1d(protein_head_layers[0]))
        new_module.add_module("proteins_head_0_relu", nn.ReLU())
        for i, layer in enumerate(protein_head_layers[1:]):
            new_module.add_module(f"proteins_head_{i+1}", nn.Linear(protein_head_layers[i], layer))
            if batch_norm_modules:
                new_module.add_module(f"proteins_head_{i+1}_batch_norm", nn.BatchNorm1d(layer))
            new_module.add_module(f"proteins_head_{i+1}_relu", nn.ReLU())

        self.proteins_head = new_module

        new_module_2 = nn.Sequential()
        new_module_2.add_module("compounds_head_0", nn.Linear(1536, compound_head_layers[0]))
        if batch_norm_modules:
            new_module_2.add_module("compounds_head_0_batch_norm", nn.BatchNorm1d(compound_head_layers[0]))
        new_module_2.add_module("compounds_head_0_relu", nn.ReLU())
        for i, layer in enumerate(compound_head_layers[1:]):
            new_module_2.add_module(f"compounds_head_{i+1}", nn.Linear(compound_head_layers[i], layer))
            if batch_norm_modules:
                new_module_2.add_module(f"compounds_head_{i+1}_batch_norm", nn.BatchNorm1d(layer))
            new_module_2.add_module(f"compounds_head_{i+1}_relu", nn.ReLU())

        self.compounds_head = new_module_2

        new_module_3 = nn.Sequential()
        new_module_3.add_module("compounds_descriptors_head_0", nn.Linear(849, compound_descriptors_layers[0]))
        if batch_norm_modules:
            new_module_3.add_module("compounds_descriptors_head_0_batch_norm", nn.BatchNorm1d(compound_descriptors_layers[0]))
        new_module_3.add_module("compounds_descriptors_head_0_relu", nn.ReLU())
        for i, layer in enumerate(compound_descriptors_layers[1:]):
            new_module_3.add_module(f"compounds_descriptors_head_{i+1}", nn.Linear(compound_descriptors_layers[i], layer))
            if batch_norm_modules:
                new_module_3.add_module(f"compounds_descriptors_head_{i+1}_batch_norm", nn.BatchNorm1d(layer))
            new_module_3.add_module(f"compounds_descriptors_head_{i+1}_relu", nn.ReLU())

        self.compounds_descriptors_head = new_module_3

        new_module_4 = nn.Sequential()
        new_module_4.add_module("proteins_descriptors_head_0", nn.Linear(1105, protein_descriptors_layers[0]))
        if batch_norm_modules:
            new_module_4.add_module("proteins_descriptors_head_0_batch_norm", nn.BatchNorm1d(protein_descriptors_layers[0]))
        new_module_4.add_module("proteins_descriptors_head_0_relu", nn.ReLU())
        for i, layer in enumerate(protein_descriptors_layers[1:]):
            new_module_4.add_module(f"proteins_descriptors_head_{i+1}", nn.Linear(protein_descriptors_layers[i], layer))
            if batch_norm_modules:
                new_module_4.add_module(f"proteins_descriptors_head_{i+1}_batch_norm", nn.BatchNorm1d(layer))

            new_module_4.add_module(f"proteins_descriptors_head_{i+1}_relu", nn.ReLU())

        self.proteins_descriptors_head = new_module_4

        input_dim = compound_head_layers[-1] + protein_head_layers[-1] + compound_descriptors_layers[-1] + protein_descriptors_layers[-1]
        # input_dim = compound_head_layers[-1] + protein_head_layers[-1]

        new_module = nn.Sequential()
        new_module.add_module("prediction_layers_0", nn.Linear(input_dim, final_head_layers[0]))
        if batch_norm_final:
            new_module.add_module("prediction_layers_0_batch_norm", nn.BatchNorm1d(final_head_layers[0]))
        new_module.add_module("prediction_layers_0_relu", nn.ReLU())

        if len(final_head_layers)>1:
            for i, layer in enumerate(final_head_layers[1:]):
                new_module.add_module(f"prediction_layers_{i+1}", nn.Linear(final_head_layers[i], layer))
                if batch_norm_final:
                    new_module.add_module(f"prediction_layers_{i+1}_batch_norm", nn.BatchNorm1d(layer))
                new_module.add_module(f"prediction_layers_{i+1}_relu", nn.ReLU())

        new_module.add_module(f"prediction", nn.Linear(final_head_layers[-1], 1))
        new_module.add_module(f"prediction_sigmoid", nn.Sigmoid())

        self.prediction_head = new_module
        self.learning_rate = learning_rate
        
    def forward(self, data):

        fp, compound_descriptors, protein_descriptors, protein_embedding, y = data
        with torch.no_grad():
            np_prediction, compounds_embedding = self.compounds_model([fp], return_embedding=True)
            protein_ec_number_prediction, protein_embedding = self.proteins_model([protein_embedding], return_embedding=True)

        compounds_embedding = self.compounds_head(compounds_embedding)
        protein_embedding = self.proteins_head(protein_embedding)

        compound_descriptors = self.compounds_descriptors_head(compound_descriptors)
        protein_descriptors = self.proteins_descriptors_head(protein_descriptors)

        final_embedding = torch.concat([compounds_embedding, protein_embedding, compound_descriptors, protein_descriptors], axis=1)

        x = self.prediction_head(final_embedding)
        
        return x

    def training_step(self, batch, batch_idx):
        fp, compound_descriptors, protein_descriptors, protein_embedding, y = batch
        
        y_pred = self(batch)
        # y_pred = torch.nan_to_num(y_pred, nan=0.5)

        loss = BCELoss()(y_pred, y)

        self.log("train_loss", loss.item(), on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        
        accuracy = self.accuracy(y_pred, y)
        f1 = self.f1(y_pred, y)
        recall = self.recall(y_pred, y)
        precision = self.precision(y_pred, y)

        self.log("train_accuracy", accuracy, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        self.log("train_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        self.log("train_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        
        self.log("train_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])

        return loss
    
    def predict_step(self, batch, batch_idx):
        y_pred = self(batch)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        fp, compound_descriptors, protein_descriptors, protein_embedding, y = batch
        y_pred = self(batch)
        loss = BCELoss()(y_pred,  y)
        self.log('test_loss', loss, batch_size=y_pred.shape[0])
        self.accuracy(y_pred,  y)
        self.log('test_acc', self.accuracy, batch_size=y_pred.shape[0])
        return loss
    
    
    
    def validation_step(self, batch, batch_idx):
        y_pred = self(batch)
        fp, compound_descriptors, protein_descriptors, protein_embedding, y = batch
        loss = BCELoss()(y_pred, y)

        self.log("val_loss", loss.item(), on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        
        accuracy = self.accuracy(y_pred, y)
        f1 = self.f1(y_pred, y)
        recall = self.recall(y_pred, y)
        precision = self.precision(y_pred, y)

        self.log("val_accuracy", accuracy, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        self.log("val_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        self.log("val_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        
        self.log("val_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        
        return loss

from sklearn.preprocessing import StandardScaler

# Example dataset class
class FPESMDataset(Dataset):
    def __init__(self, mol_fp, mol_descriptors, protein_descriptors, proteins_embeddings_dict, interactions, labels,
                 mol_scaler=None, protein_scaler=None):
        self.graphs = []
        self.mol_fp = mol_fp
        self.proteins_embeddings_dict = proteins_embeddings_dict
        self.protein_descriptors = protein_descriptors
        self.interactions = interactions
        self.labels = labels
        self.mol_descriptors = mol_descriptors

        self.unique_protein_ids = set()
        self.unique_compound_ids = set()
        # Collect unique protein and compound IDs
        for protein_id, compound_id in self.interactions:
            self.unique_protein_ids.add(protein_id)
            self.unique_compound_ids.add(compound_id)

        if mol_scaler and protein_scaler:
            self.mol_scaler = mol_scaler
            self.protein_scaler = protein_scaler
        else:
            self.scale_fit()

        self.scale_transform()

        self.device="cpu"

    def scale_fit(self):

        # Initialize the scalers
        self.mol_scaler = StandardScaler()
        self.protein_scaler = StandardScaler()

        # Extract unique descriptors
        unique_protein_descriptors = [self.protein_descriptors[protein_id] for protein_id in self.unique_protein_ids]
        unique_compound_descriptors = [self.mol_descriptors[compound_id] for compound_id in self.unique_compound_ids]

        # Fit the scalers on the unique descriptors
        self.protein_scaler.fit(unique_protein_descriptors)
        self.mol_scaler.fit(unique_compound_descriptors)

    def scale_transform(self):

        # Transform the original descriptors using the fitted scalers
        self.protein_descriptors = {protein_id: self.protein_scaler.transform([self.protein_descriptors[protein_id]])[0]
                                    for protein_id in self.unique_protein_ids}
        self.mol_descriptors = {compound_id: self.mol_scaler.transform([self.mol_descriptors[compound_id]])[0]
                                for compound_id in self.unique_compound_ids}

        


    def to(self, device):
        self.device=device

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        protein_id, compound_id = self.interactions[idx]
        label = self.labels[idx]
        fp = torch.tensor(self.mol_fp[compound_id], dtype=torch.float32, device=self.device)
        y = torch.tensor([label], dtype=torch.float32)
        protein_embedding = torch.tensor(self.proteins_embeddings_dict[protein_id], dtype=torch.float32, device=self.device)
        protein_descriptors = torch.tensor(self.protein_descriptors[protein_id], dtype=torch.float32, device=self.device)
        compound_descriptors = torch.tensor(self.mol_descriptors[compound_id], dtype=torch.float32, device=self.device)

        return fp, compound_descriptors, protein_descriptors, protein_embedding, y


