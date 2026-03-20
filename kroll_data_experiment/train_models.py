

from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors

from plants_sm.models.lightning_model import InternalLightningModel
from sklearn.metrics import *
from enzyme_substrate_prediction.models.interaction import InteractionModel
from lightning.pytorch.callbacks import EarlyStopping

import pandas as pd

from plants_sm.models.fc.fc import DNN

import torch.nn as nn


from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
    # load datasets

train_dataset = MultiInputDataset.from_csv("train_dataset_w_representation.csv", representation_field={"enzymes": "sequence",
                                                                                                       "ligands": "smiles"},
                                                                                        instances_ids_field={"enzymes": "Uniprot ID",
                                                                                                             "ligands": "molecule ID", 
                                                                                        },
                                                                                        labels_field="Binding")

test_dataset = MultiInputDataset.from_csv("test_dataset_w_representation.csv", representation_field={"enzymes": "sequence",
                                                                                                     "ligands": "smiles", 
                                                                                        },
                                                                                        instances_ids_field={"enzymes": "Uniprot ID",
                                                                                                             "ligands": "molecule ID", 
                                                                                        },
                                                                                        labels_field="Binding")

train_dataset.load_features("features_proteins_esm2_3b", "enzymes")
train_dataset.load_features("features_compounds_np_classifier_fp", "ligands")

test_dataset.load_features("features_proteins_esm2_3b", "enzymes")
test_dataset.load_features("features_compounds_np_classifier_fp", "ligands")

module = InteractionModel(protein_model_path="esm2_3b.pt", compounds_model_path="np_classifier.ckpt", transfer_learning=True, 
                          protein_model = DNN(2560, [2560], 5743, batch_norm=True, last_sigmoid=True, dropout=None))

module.interaction_model.proteins_head = nn.Sequential(
                nn.Linear(2560, 1280),
                nn.ReLU(),
                nn.BatchNorm1d(1280),
                nn.Linear(1280, 640),
                nn.ReLU(),
                nn.BatchNorm1d(640)
            )


module.interaction_model.interaction_model = DNN(1024, [256], 1, batch_norm=True, last_sigmoid=True, dropout=0.5)

# Create the early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # metric to monitor
    patience=5,          # number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode='min'           # stop when the metric is minimized
)

model = InternalLightningModel(module=module, max_epochs=50,
        batch_size=56,
        devices=[0, 1, 2],
        accelerator="gpu",
        callbacks=[early_stopping_callback], 
        strategy="ddp_find_unused_parameters_true"
        )

model.fit(train_dataset, test_dataset)
predictions_proba = model.predict_proba(test_dataset)
predictions = model.predict(test_dataset)
print(roc_auc_score(test_dataset.y, predictions_proba))
print(accuracy_score(test_dataset.y, predictions))
print(f1_score(test_dataset.y, predictions))
print(precision_score(test_dataset.y, predictions))
print(recall_score(test_dataset.y, predictions))