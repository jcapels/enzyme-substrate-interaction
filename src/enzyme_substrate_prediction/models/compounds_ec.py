import torch
from torch.nn import BCELoss
from plants_sm.models.lightning_model import InternalLightningModule
from plants_sm.models.fc.fc import DNN
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ModelECNumber(InternalLightningModule):

    def __init__(self, input_dim, layers, classification_neurons, metric=None, learning_rate = 1e-3, layers_to_freeze=0, 
                 scheduler = False, dropout=None, weight_decay=0.0) -> None:

        self.layers = layers
        self.classification_neurons = classification_neurons
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.layers_to_freeze = layers_to_freeze
        self.scheduler = scheduler
        self.dropout = dropout
        self.weight_decay = weight_decay
        super().__init__(metric=metric)
        self._create_model()

    def _update_constructor_parameters(self):
        self._contructor_parameters.update({
            'layers': self.layers,
            'classification_neurons': self.classification_neurons,
            'metric': self.metric,
            'learning_rate': self.learning_rate,
            'layers_to_freeze': self.layers_to_freeze,
            'scheduler': self.scheduler,
            'dropout': self.dropout,
            'weight_decay': self.weight_decay,
            "input_dim": self.input_dim
        })


    def _create_model(self):
        self.fc_model = DNN(self.input_dim, self.layers, self.classification_neurons, batch_norm=True, last_sigmoid=True, 
                            dropout=None, layers_to_freeze=self.layers_to_freeze)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.fc_model.parameters()}], lr=self.learning_rate, weight_decay=self.weight_decay)

        # Define a custom learning rate scheduler using LambdaLR
        if self.scheduler:
            scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x):
        return self.fc_model(x)

    def compute_loss(self, logits, y):
        return BCELoss()(logits, y)