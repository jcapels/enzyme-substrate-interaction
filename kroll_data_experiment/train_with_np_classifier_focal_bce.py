
from copy import copy
import numpy as np
from plants_sm.hyperparameter_optimization.experiment import Experiment

import optuna

import os
import pandas as pd

from sklearn.metrics import *
from enzyme_substrate_prediction.models.interaction_focal_bce import InteractionModel
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import pytorch_lightning as pl



def get_model(module, batch_size,epochs, train_dataset, validation_dataset):
    from plants_sm.models.lightning_model import InternalLightningModel

    if validation_dataset != None:
        model_checkpoint = ModelCheckpoint(
                monitor='val_loss',  # metric to monitor
                dirpath='checkpoints_np_classifier/',  # directory to save the model checkpoints
                filename='best-checkpoint',  # filename for the best checkpoint
                save_top_k=1,  # save the top k models
                mode='min',  # save the model when the metric is minimized
                verbose=True)
        
        early_stopping = EarlyStopping("val_loss", patience=5, mode="min")
        
        callbacks = [model_checkpoint, early_stopping]

        model = InternalLightningModel(module=module, max_epochs=epochs,
                    batch_size=batch_size,
                    devices=[2],
                    accelerator="gpu",
                    callbacks=callbacks
                    )
        model.reset_weights()
        model.fit(train_dataset, validation_dataset)
    
        best_model_path = model_checkpoint.best_model_path
        model.module = InteractionModel.load_from_checkpoint(best_model_path)
        best_epoch = (model.trainer.current_epoch - 1) - early_stopping.wait_count
        os.remove(best_model_path)
        return model, best_epoch

    else:
        model = InternalLightningModel(module=module, max_epochs=epochs,
                    batch_size=batch_size,
                    devices=[2],
                    accelerator="gpu",
                    )
        model.reset_weights()
        model.fit(train_dataset)
        return model, None

import optuna
from plants_sm.hyperparameter_optimization.experiment import Experiment
from sklearn.metrics import *
import pandas as pd

class BindingExperiment(Experiment):
    def __init__(self, train_dataset, validation_dataset, test_dataset, baseline = False, results_output_file="results.csv", input_dim=600, 
                 classification_neurons=1,
                 folder_path="trials", **kwargs):
        
        self.train_dataset = copy(train_dataset)
        self.test_dataset = copy(test_dataset)
        self.validation_dataset = copy(validation_dataset)
        self.folder_path = folder_path
        self.baseline = baseline
        self.results_output_file = results_output_file
        self.input_dim = input_dim
        self.classification_neurons = classification_neurons
        self.best_result = float("-inf")
        self.best_result = 0
        self.best_hyperparameters = {}
        self.best_epoch = 0
        super().__init__(**kwargs)

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

        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame() 

        protein_head_layers, compound_head_layers, final_head_layers, batch_norm, dropout, batch_size, learning_rate = self._steps(trial)
        epochs = trial.suggest_int("epochs", 50, 200)
        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame() 
        
        module = InteractionModel(protein_model_path="esm2_3b.pt", compounds_model_path="np_classifier.ckpt", 
                          learning_rate=learning_rate, protein_head_layers=protein_head_layers, compound_head_layers=compound_head_layers,
                          final_head_layers=final_head_layers, batch_norm=batch_norm, dropout=dropout)
        
        datasets, _ = load_datasets_v2(self.train_dataset, self.validation_dataset, self.test_dataset)

        f1_macro_validation = []
        precision_validation = []
        roc_auc_validation = []
        accuracy_validation = []
        mcc_validation = []
        recall_validation = []

        # Cross-validation loop
        for train_dataset, validation_dataset in datasets:
            import traceback

            try:
                model, best_epoch = get_model(module, batch_size, epochs, train_dataset, validation_dataset)
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()  # This will print the full traceback
                return float('-inf')
            
            predictions_probability = model.predict_proba(validation_dataset, trainer = model.trainer)
            predictions = model.predict(validation_dataset, trainer = model.trainer)
            y_true = validation_dataset.y

            # Evaluate on validation set
            f1_macro_validation.append(f1_score(y_true, predictions))
            precision_validation.append(precision_score(y_true, predictions))
            roc_auc_validation.append(roc_auc_score(y_true, predictions_probability))
            accuracy_validation.append(accuracy_score(y_true, predictions))
            mcc_validation.append(matthews_corrcoef(y_true, predictions))
            recall_validation.append(recall_score(y_true, predictions))


        avg_metrics = {
            "Model type": ["gat"],
            "Trial": [trial.number],
            "F1_macro_validation_set": [np.mean(f1_macro_validation)],
            "Precision validation set": [np.mean(precision_validation)],
            "recall validation set": [np.mean(recall_validation)],
            "roc_auc validation set": [np.mean(roc_auc_validation)],
            "accuracy validation set": [np.mean(accuracy_validation)],
            "mcc validation set": [np.mean(mcc_validation)],
            "Batch size": [batch_size],
            "Learning rate": [learning_rate],
            "Protein layers": [protein_head_layers],
            "Compound layers": [compound_head_layers],
            "Final head layers": [final_head_layers],
            "Batch size": [batch_size], 
            "Learning rate": [learning_rate], 
            "Epochs": [epochs],
            "batch_norm": [batch_norm],
            "dropout": [dropout],
            "best epoch": [best_epoch],
        }

        # Create a DataFrame and save to CSV
        results_df = pd.DataFrame(avg_metrics)
        results = pd.concat([results, results_df])
        
        results.to_csv(self.results_output_file, index=False)

        if np.mean(mcc_validation) > self.best_result:
            self.best_result = np.mean(f1_macro_validation)
            self.best_hyperparameters = {
                "protein_head_layers": protein_head_layers,
                "compound_head_layers": compound_head_layers,
                "final_head_layers": final_head_layers,
                "batch_norm": batch_norm,
                "dropout": dropout,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs
            }
            self.best_epoch = best_epoch
        

        return np.mean(mcc_validation)
    
def load_datasets_v2(train_dataset, validation_dataset, test_dataset, random_state=42, merge_validation_set=False):
    import pandas as pd

    from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset

    if not merge_validation_set:
        datasets = []

        # Set the seed for all random number generators
        train_dataset_ = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        import pandas as pd

        validation_dataset__ = copy(validation_dataset)

        validation_dataset_ = MultiInputDataset(dataframe=validation_dataset__, representation_field={"proteins": "sequence",
                                                                                                        "ligands": "smiles", 
                                                                                    },
                                                                                    instances_ids_field={ "proteins": "Uniprot ID",
                                                                                        "ligands": "molecule ID", 
                                                                                    },
                                                            labels_field="Binding")
        
        validation_dataset_.load_features("features_proteins_esm2_3b", "proteins")
        validation_dataset_.load_features("features_compounds_np_classifier_fp", "ligands")

        train_dataset_ = MultiInputDataset(dataframe=train_dataset_, representation_field={"proteins": "sequence",
                                                                                            "ligands": "smiles", 
                                                                                    },
                                                                                    instances_ids_field={"proteins": "Uniprot ID",
                                                                                        "ligands": "molecule ID", 
                                                                                    },
                                                                labels_field="Binding")

        train_dataset_.load_features("features_proteins_esm2_3b", "proteins")
        train_dataset_.load_features("features_compounds_np_classifier_fp", "ligands")

        
        datasets.append((train_dataset_, validation_dataset_))

        test_dataset_copy = copy(test_dataset)

        test_dataset_ = MultiInputDataset(dataframe=test_dataset_copy, representation_field={"proteins": "sequence",
                                                                                       "ligands": "smiles", 
                                                                                        },
                                                                                        instances_ids_field={
                                                                                            "proteins": "Uniprot ID", 
                                                                                            "ligands": "molecule ID", 
                                                                                        },
                                                                    labels_field="Binding")
        test_dataset_.load_features("features_proteins_esm2_3b", "proteins")
        test_dataset_.load_features("features_compounds_np_classifier_fp", "ligands")
        return datasets, test_dataset_
    else:
        train_dataset = pd.concat([train_dataset, validation_dataset], ignore_index=True)
        train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        train_dataset_ = MultiInputDataset(dataframe=train_dataset, representation_field={"proteins": "sequence",
                                                                                         "ligands": "smiles", 
                                                                                        },
                                                                                        instances_ids_field={
                                                                                            "proteins": "Uniprot ID",
                                                                                            "ligands": "molecule ID", 
                                                                                        },
                                                                    labels_field="Binding")

        train_dataset_.load_features("features_proteins_esm2_3b", "proteins")
        train_dataset_.load_features("features_compounds_np_classifier_fp", "ligands")

        test_dataset_copy = copy(test_dataset)
        test_dataset_ = MultiInputDataset(dataframe=test_dataset_copy, representation_field={
                                                                                        "proteins": "sequence", 
                                                                                        "ligands": "smiles", 
                                                                                        },
                                                                                        instances_ids_field={
                                                                                            "proteins": "Uniprot ID",
                                                                                            "ligands": "molecule ID", 
                                                                                        },
                                                                    labels_field="Binding")
        test_dataset_.load_features("features_proteins_esm2_3b", "proteins")
        test_dataset_.load_features("features_compounds_np_classifier_fp", "ligands")

        return train_dataset_, test_dataset_
    
def experiment_optimize_v2(train_dataset, validation_dataset, test_dataset):
    
    experiment = BindingExperiment(train_dataset, validation_dataset, test_dataset, study_name="binding_np_classifier_weighted_bce", storage="sqlite:///binding_np_classifier_weighted_bce.db", sampler= optuna.samplers.TPESampler(seed=123),
                                    direction="maximize", load_if_exists=True, results_output_file="binding_np_classifier_weighted_bce.csv",
                                        folder_path="kroll_experiment_models")
    experiment.run(n_trials=50, n_jobs=1)

    test_train_model(train_dataset, validation_dataset, test_dataset, experiment.best_hyperparameters["protein_head_layers"], experiment.best_hyperparameters["compound_head_layers"],
                    experiment.best_hyperparameters["final_head_layers"], experiment.best_hyperparameters["batch_norm"], 
                    experiment.best_hyperparameters["dropout"], experiment.best_hyperparameters["batch_size"], 
                    experiment.best_hyperparameters["learning_rate"], experiment.best_epoch+1)

def test_train_model(train_dataset, validation_dataset, test_dataset, protein_head_layers, compound_head_layers, final_head_layers, batch_norm, dropout, batch_size, learning_rate, epochs):

    file_exists = False

    for i in range(5):
        # Set the seed for all random number generators
        pl.seed_everything(i, workers=True)

        train_dataset_ = copy(train_dataset)
        test_dataset_ = copy(test_dataset)
            
        module = InteractionModel(protein_model_path="esm2_3b.pt", compounds_model_path="np_classifier.ckpt", 
                            learning_rate=learning_rate, protein_head_layers=protein_head_layers, compound_head_layers=compound_head_layers, 
                            final_head_layers=final_head_layers, batch_norm=batch_norm, dropout=dropout)
        
        train_dataset_, test_dataset_ = load_datasets_v2(train_dataset_, validation_dataset, test_dataset_, random_state=i, merge_validation_set=True)

        # Cross-validation loop
        model, _ = get_model(module, batch_size, epochs, train_dataset_, None)
        
        # Evaluate on test set
        predictions_probability = model.predict_proba(test_dataset_, trainer = model.trainer)
        predictions = model.predict(test_dataset_, trainer = model.trainer)
        y_true = test_dataset_.y

        result = {
            'seed': i,
            'f1_macro': f1_score(y_true, predictions),
            'precision': precision_score(y_true, predictions),
            'roc_auc': roc_auc_score(y_true, predictions_probability),
            'accuracy': accuracy_score(y_true, predictions),
            'mcc': matthews_corrcoef(y_true, predictions),
            'recall': recall_score(y_true, predictions)
        }

        df = pd.DataFrame([result])
        df.to_csv("results_np_classifier_weighted_bce.csv", mode='a', index=False, header=not file_exists)
        file_exists = True  # Ensure header is not written again


if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    from plants_sm.io.pickle import read_pickle

    train_dataset = pd.read_csv("train_dataset_w_split.csv")
    test_dataset = pd.read_csv("test_dataset_w_representation_filtered.csv")
    validation_dataset = pd.read_csv("validation_dataset.csv")

    # test_train_model(train_dataset, validation_dataset, test_dataset, [2560, 1280, 1280], [2560, 2560, 2560, 1280, 640],[2560, 2560, 2560, 1280, 640], True, 0.596622, 32, 0.000063, 1, 0.1, 2, 0)
    experiment_optimize_v2(train_dataset, validation_dataset, test_dataset)
