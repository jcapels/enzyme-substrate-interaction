
from plants_sm.hyperparameter_optimization.experiment import Experiment

import optuna

import os
import pandas as pd

from lightning.pytorch.callbacks import EarlyStopping

from sklearn.metrics import *
from enzyme_substrate_prediction.models.interaction import InteractionModel
from lightning.pytorch.callbacks import EarlyStopping

import pandas as pd

from plants_sm.models.fc.fc import DNN

import torch.nn as nn

def get_model(module, batch_size,epochs, callbacks, train_dataset, validation_dataset):
    from plants_sm.models.lightning_model import InternalLightningModel
    model = InternalLightningModel(module=module, max_epochs=epochs,
                batch_size=batch_size,
                devices=[0],
                accelerator="gpu",
                callbacks=[callbacks]
                )
    model.fit(train_dataset, validation_dataset)
    return model

class KrollDataExperiment(Experiment):
    def __init__(self, datasets, baseline = False, results_output_file="results.csv", input_dim=600, 
                 classification_neurons=730,
                 folder_path="trials", **kwargs):
        super().__init__(**kwargs)

        self.folder_path = folder_path
        self.datasets = datasets
        self.baseline = baseline
        self.results_output_file = results_output_file
        self.input_dim = input_dim
        self.classification_neurons = classification_neurons
        self.best_result = float("-inf")

    def _steps(self, trial):

        protein_head_layers = trial.suggest_categorical("protein_head_layers", ["[2560]","[1280]", "[640]", "[2560, 1280]", "[2560, 640]", "[1280, 640]",
                                                                            "[2560, 1280]", "[2560, 1280, 640]",
                                                                            "[2560, 1280, 1280]",
                                                                            "[2560, 1280, 1280, 640]",
                                                                            "[2560, 1280, 1280, 1280]",
                                                                            "[2560, 1280, 1280, 1280, 640]",
                                                                            "[2560, 1280, 1280, 1280, 1280]",
                                                                            "[2560, 2560, 2560, 1280]",
                                                                            "[2560, 2560, 2560, 1280, 640]"],)
        
        compound_head_layers = trial.suggest_categorical("compound_head_layers", ["[2560]","[1280]", "[640]", "[2560, 1280]", "[2560, 640]", "[1280, 640]",
                                                                            "[2560, 1280]", "[2560, 1280, 640]",
                                                                            "[2560, 1280, 1280]",
                                                                            "[2560, 1280, 1280, 640]",
                                                                            "[2560, 1280, 1280, 1280]",
                                                                            "[2560, 1280, 1280, 1280, 640]",
                                                                            "[2560, 1280, 1280, 1280, 1280]",
                                                                            "[2560, 2560, 2560, 1280]",
                                                                            "[2560, 2560, 2560, 1280, 640]"],)
        
        final_head_layers = trial.suggest_categorical("final_head_layers", ["[2560]","[1280]", "[640]", "[2560, 1280]", "[2560, 640]", "[1280, 640]",
                                                                            "[2560, 1280]", "[2560, 1280, 640]",
                                                                            "[2560, 1280, 1280]",
                                                                            "[2560, 1280, 1280, 640]",
                                                                            "[2560, 1280, 1280, 1280]",
                                                                            "[2560, 1280, 1280, 1280, 640]",
                                                                            "[2560, 1280, 1280, 1280, 1280]",
                                                                            "[2560, 2560, 2560, 1280]",
                                                                            "[2560, 2560, 2560, 1280, 640]"],)
        

        
        # evaluate literal
        protein_head_layers = eval(protein_head_layers)
        compound_head_layers = eval(compound_head_layers)
        final_head_layers = eval(final_head_layers)

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)

        batch_norm = trial.suggest_categorical("batch_norm", [True, False])
        dropout = trial.suggest_float("dropout", 0, 1)

        return protein_head_layers, compound_head_layers, final_head_layers, batch_norm, dropout, batch_size, learning_rate

    def objective(self, trial: optuna.Trial) -> float:
        protein_head_layers, compound_head_layers, final_head_layers, batch_norm, dropout, batch_size, learning_rate = self._steps(trial)
        epochs = trial.suggest_int("epochs", 50, 200)
        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame() 
        train_dataset, validation_dataset, test_dataset = self.datasets
        callbacks = EarlyStopping("val_loss", patience=5, mode="min")
        
        module = InteractionModel(protein_model_path="esm2_3b.pt", compounds_model_path="np_classifier.ckpt", transfer_learning=True, 
                          protein_model = DNN(2560, [2560], 5743, batch_norm=True, last_sigmoid=True, dropout=None),
                          learning_rate=learning_rate)
        
        new_module = nn.Sequential()
        new_module.add_module("proteins_head_0", nn.Linear(2560, protein_head_layers[0]))
        new_module.add_module("proteins_head_0_relu", nn.ReLU())
        new_module.add_module("proteins_head_0_batch_norm", nn.BatchNorm1d(protein_head_layers[0]))
        for i, layer in enumerate(protein_head_layers[1:]):
            new_module.add_module(f"proteins_head_{i+1}", nn.Linear(protein_head_layers[i], layer))
            new_module.add_module(f"proteins_head_{i+1}_relu", nn.ReLU())
            new_module.add_module(f"proteins_head_{i+1}_batch_norm", nn.BatchNorm1d(layer))

        module.interaction_model.proteins_head = new_module

        new_module_2 = nn.Sequential()
        new_module_2.add_module("compounds_head_0", nn.Linear(1536, compound_head_layers[0]))
        new_module_2.add_module("compounds_head_0_relu", nn.ReLU())
        new_module_2.add_module("compounds_head_0_batch_norm", nn.BatchNorm1d(compound_head_layers[0]))
        for i, layer in enumerate(compound_head_layers[1:]):
            new_module_2.add_module(f"compounds_head_{i+1}", nn.Linear(compound_head_layers[i], layer))
            new_module_2.add_module(f"compounds_head_{i+1}_relu", nn.ReLU())
            new_module_2.add_module(f"compounds_head_{i+1}_batch_norm", nn.BatchNorm1d(layer))

        module.interaction_model.compounds_head = new_module_2
            
        input_dim = compound_head_layers[-1] + protein_head_layers[-1]

        module.interaction_model.interaction_model = DNN(input_dim, final_head_layers, 1, batch_norm=batch_norm, last_sigmoid=True, dropout=dropout)

        import traceback

        try:
            model = get_model(module, batch_size, epochs, callbacks, train_dataset, validation_dataset)
        except Exception as e:
            print("An error occurred:")
            traceback.print_exc()  # This will print the full traceback
            return float('-inf')
        
        predictions_probability = model.predict_proba(validation_dataset)
        predictions = model.predict(validation_dataset)

        f1_macro_validation_set = f1_score(validation_dataset.y, predictions)
        precision_validation_set = precision_score(validation_dataset.y, predictions)
        roc_auc_validation_set = roc_auc_score(validation_dataset.y, predictions_probability)
        accuracy_score_validation_set = accuracy_score(validation_dataset.y, predictions)
        mcc_score_validation_set = matthews_corrcoef(validation_dataset.y, predictions)

        predictions_test_dataset = model.predict(test_dataset)
        predictions_probability_test_dataset = model.predict_proba(test_dataset)

        f1_macro_test_set = f1_score(test_dataset.y, predictions_test_dataset)
        precision_test_set = precision_score(test_dataset.y, predictions_test_dataset)
        roc_auc_test_set = roc_auc_score(test_dataset.y, predictions_probability_test_dataset)
        accuracy_score_test_set = accuracy_score(test_dataset.y, predictions_test_dataset)
        mcc_score_test_set = matthews_corrcoef(test_dataset.y, predictions_test_dataset)

        # instead of append use concatenate
        results = pd.concat([results, pd.DataFrame({"Model type": ["esm2_3b_np_classifier"], 
                                                    "Trial": [trial.number], 
                                                    "F1_macro_validation_set": [f1_macro_validation_set], 
                                                    "Precision validation set": [precision_validation_set],
                                                    "roc_auc validation set": [roc_auc_validation_set],
                                                    "accuracy validation set": [accuracy_score_validation_set],
                                                    "mcc validation set": [mcc_score_validation_set],
                                                    "F1_macro_test_set": [f1_macro_test_set],
                                                    "Precision test set": [precision_test_set],
                                                    "roc_auc test set": [roc_auc_test_set],
                                                    "accuracy test set": [accuracy_score_test_set],
                                                    "mcc test set": [mcc_score_test_set],
                                                    "Protein layers": [protein_head_layers],
                                                    "Compound layers": [compound_head_layers],
                                                    "Final head layers": [final_head_layers],
                                                    "Batch size": [batch_size], 
                                                    "Learning rate": [learning_rate], 
                                                    "Epochs": [epochs],
                                                    "batch_norm": [batch_norm],
                                                    "dropout": [dropout]})], ignore_index=True)
        
        results.to_csv(self.results_output_file, index=False)

        return mcc_score_validation_set
    
def experiment_optimize():

    from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
    # load datasets

    train_dataset = MultiInputDataset.from_csv("train_dataset_w_representation_split.csv", representation_field={"enzymes": "sequence",
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
    
    validation_dataset = MultiInputDataset.from_csv("validation_dataset_w_representation.csv", representation_field={"enzymes": "sequence",
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

    validation_dataset.load_features("features_proteins_esm2_3b", "enzymes")
    validation_dataset.load_features("features_compounds_np_classifier_fp", "ligands")
    datasets = (train_dataset, validation_dataset, test_dataset)
    
    experiment = KrollDataExperiment(datasets=datasets, study_name="kroll_experiment_models", storage="sqlite:///kroll_experiment_models.db", sampler= optuna.samplers.TPESampler(),
                                    direction="maximize", load_if_exists=True, results_output_file="kroll_experiment_models_results.csv",
                                        folder_path="kroll_experiment_models/trials")
    
    experiment.run(n_trials=100, n_jobs=1)
    

if __name__ == "__main__":
   experiment_optimize()