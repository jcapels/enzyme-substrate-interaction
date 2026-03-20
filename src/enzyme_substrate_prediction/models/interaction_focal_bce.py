import torch
from torch.nn import BCELoss
from plants_sm.models.lightning_model import InternalLightningModule
from plants_sm.models.fc.fc import DNN
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
import lightning as L

from enzyme_substrate_prediction.focal_bce import MixedFocalWeightedBCELoss
from enzyme_substrate_prediction.models.np_classifier import NPClassifierDNN
from enzyme_substrate_prediction.models.protein_model import ESM2_3B



class InteractionModelTransferLearning(nn.Module):

    def __init__(self, protein_head_layers, compound_head_layers, final_head_layers, batch_norm, dropout, last_sigmoid=False) -> None:
        super(InteractionModelTransferLearning, self).__init__()
        self.compounds_model = NPClassifierDNN()
        self.proteins_model = ESM2_3B()

        self.compounds_model = self.compounds_model.eval()
        self.proteins_model = self.proteins_model.eval()

        # if self.transfer_learning:
        new_module = nn.Sequential()
        new_module.add_module("proteins_head_0", nn.Linear(2560, protein_head_layers[0]))
        new_module.add_module("proteins_head_0_relu", nn.ReLU())
        new_module.add_module("proteins_head_0_batch_norm", nn.BatchNorm1d(protein_head_layers[0]))
        for i, layer in enumerate(protein_head_layers[1:]):
            new_module.add_module(f"proteins_head_{i+1}", nn.Linear(protein_head_layers[i], layer))
            new_module.add_module(f"proteins_head_{i+1}_relu", nn.ReLU())
            new_module.add_module(f"proteins_head_{i+1}_batch_norm", nn.BatchNorm1d(layer))

        self.proteins_head = new_module

        new_module_2 = nn.Sequential()
        new_module_2.add_module("compounds_head_0", nn.Linear(1536, compound_head_layers[0]))
        new_module_2.add_module("compounds_head_0_relu", nn.ReLU())
        new_module_2.add_module("compounds_head_0_batch_norm", nn.BatchNorm1d(compound_head_layers[0]))
        for i, layer in enumerate(compound_head_layers[1:]):
            new_module_2.add_module(f"compounds_head_{i+1}", nn.Linear(compound_head_layers[i], layer))
            new_module_2.add_module(f"compounds_head_{i+1}_relu", nn.ReLU())
            new_module_2.add_module(f"compounds_head_{i+1}_batch_norm", nn.BatchNorm1d(layer))

        self.compounds_head = new_module_2

        input_dim = compound_head_layers[-1] + protein_head_layers[-1]

        self.interaction_model = DNN(input_dim, final_head_layers, 1, batch_norm=batch_norm, last_sigmoid=last_sigmoid, dropout=dropout)

        self.last_sigmoid = last_sigmoid

        # else:
        #     self.compounds_head = nn.Sequential(
        #         nn.Linear(6144, 3072),
        #         nn.ReLU(),
        #         nn.Linear(3072, 1536),
        #         nn.ReLU(),
        #     )

        #     self.proteins_head = nn.Sequential(
        #         nn.Linear(1280, 640),
        #         nn.ReLU(),
        #         nn.Linear(640, 320),
        #         nn.ReLU(),
        #     )
        #     self.interaction_model = DNN(1856, [1024, 1024, 512, 256], 1, batch_norm=True, last_sigmoid=True, dropout=None)



    def forward(self, x_compounds, x_proteins, return_predictions=False):
        # if self.transfer_learning:
        with torch.no_grad():
            np_prediction, compounds_embedding = self.compounds_model([x_compounds], return_embedding=True)
            protein_ec_number_prediction, protein_embedding = self.proteins_model([x_proteins], return_embedding=True)
        
        compounds_embedding = self.compounds_head(compounds_embedding)
        protein_embedding = self.proteins_head(protein_embedding)
        # else:
        #     compounds_embedding = self.compounds_head(x_compounds)
        #     protein_embedding = self.proteins_head(x_proteins)

        # compound_process = self.compounds_model_fp([x_compounds])
        interaction = torch.cat([compounds_embedding, protein_embedding], dim=1)
        interaction = self.interaction_model([interaction])

        if self.last_sigmoid:
            interaction = torch.sigmoid(interaction)

        if return_predictions:
            return interaction, np_prediction, protein_ec_number_prediction
        else:
            return interaction
    
class InteractionModel(L.LightningModule):

    def __init__(self, protein_head_layers, compound_head_layers,
                       final_head_layers, scheduler=None, 
                       protein_model_path="", compounds_model_path="", transfer_learning=True,
                       protein_model=None, compound_model=None, learning_rate = 1e-3,
                       batch_norm=True, dropout=0.1, alpha=0.25, gamma=2.0, pos_weight=None, lambda_focal=0.5) -> None:
        
        self.scheduler = scheduler
        self.protein_model_path = protein_model_path
        self.compounds_model_path = compounds_model_path
        self.transfer_learning = transfer_learning
        self.learning_rate = learning_rate
        self.protein_head_layers = protein_head_layers
        self.compound_head_layers = compound_head_layers
        self.final_head_layers = final_head_layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_focal = lambda_focal
        self._constructor_parameters = {}

        super().__init__()

        self._create_model()

        self.save_hyperparameters()
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.problem_type = "binary"
        

    def _update_constructor_parameters(self):
        self._constructor_parameters.update({
            'metric': self.metric,
        })

    def load_models(self):
        np_classifier_model_pretrained_dict = torch.load(self.compounds_model_path, map_location="cpu")
        np_classifier_model_pretrained_dict = {k.replace("np_classifier_model.", ""): v 
                                               for k, v in np_classifier_model_pretrained_dict["state_dict"].items() 
                                               if k.replace("np_classifier_model.", "") 
                                               in self.interaction_model.compounds_model.state_dict()}
        protein_model = torch.load(self.protein_model_path, map_location="cpu")
        
        self.interaction_model.compounds_model.load_state_dict(np_classifier_model_pretrained_dict)
        self.interaction_model.proteins_model.load_state_dict(protein_model)

    def _create_model(self):

        self.interaction_model = InteractionModelTransferLearning(
                                                                  protein_head_layers=self.protein_head_layers,
                                                                  compound_head_layers=self.compound_head_layers,
                                                                  final_head_layers=self.final_head_layers,
                                                                  batch_norm=self.batch_norm,
                                                                  dropout=self.dropout)

        self.load_models()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=self.learning_rate)

        # Define a custom learning rate scheduler using LambdaLR
        if self.scheduler:
            scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x_compounds, x_proteins):
        return self.interaction_model(x_compounds, x_proteins)

    def compute_loss(self, logits, y):
        return MixedFocalWeightedBCELoss(pos_weight=self.pos_weight, alpha=self.alpha,
                                         gamma=self.gamma, lambda_focal=self.lambda_focal)(logits, y)
    
    def training_step(self, batch, batch_idx):
        x_proteins, x_compounds, y = batch
        logits = self(x_compounds, x_proteins)
        loss = self.compute_loss(logits, y)
        
        self.log("train_loss", loss.item(), on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        
        accuracy = self.accuracy(logits, y)
        f1 = self.f1(logits, y)
        recall = self.recall(logits, y)
        precision = self.precision(logits, y)

        self.log("train_accuracy", accuracy, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("train_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("train_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        
        self.log("train_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x_proteins, x_compounds, target = batch
        output = self(x_compounds, x_proteins)

        loss = self.compute_loss(output, target)
        accuracy = self.accuracy(output, target)
        f1 = self.f1(output, target)
        recall = self.recall(output, target)
        precision = self.precision(output, target)
        
        self.log("val_loss", loss.item(), on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_accuracy", accuracy, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        
        return loss
        

    def predict_step(self, batch):
        self.interaction_model.last_sigmoid = True
        if len(batch) == 3:
            x_proteins, x_compounds, target = batch
        else:
            x_proteins, x_compounds = batch
        return self(x_compounds, x_proteins)