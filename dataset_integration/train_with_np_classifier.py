
import numpy as np
from plants_sm.hyperparameter_optimization.experiment import Experiment

import optuna

import os
import pandas as pd

from sklearn.metrics import *
from enzyme_substrate_prediction.models.interaction_individual_module import InteractionModel
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd

from plants_sm.models.fc.fc import DNN

import torch.nn as nn

def get_model(module, batch_size,epochs, train_dataset, validation_dataset):
    from plants_sm.models.lightning_model import InternalLightningModel

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
                devices=[1],
                accelerator="gpu",
                callbacks=callbacks
                )
    model.reset_weights()
    model.fit(train_dataset, validation_dataset)
    
    best_model_path = model_checkpoint.best_model_path
    model.module = InteractionModel.load_from_checkpoint(best_model_path)
    return model

import traceback
import optuna
from plants_sm.hyperparameter_optimization.experiment import Experiment
from torch import nn
from sklearn.metrics import *
import pandas as pd

class BindingExperiment(Experiment):
    def __init__(self, baseline = False, results_output_file="results.csv", input_dim=600, 
                 classification_neurons=1,
                 folder_path="trials", **kwargs):

        self.ids_for_datasets = kwargs.pop("ids_for_datasets")
        self.folder_path = folder_path
        self.baseline = baseline
        self.results_output_file = results_output_file
        self.input_dim = input_dim
        self.classification_neurons = classification_neurons
        self.best_result = float("-inf")
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
        
        datasets = load_datasets(self.ids_for_datasets)

        f1_macro_validation = []
        precision_validation = []
        roc_auc_validation = []
        accuracy_validation = []
        mcc_validation = []
        recall_validation = []

        f1_macro_test = []
        precision_test = []
        roc_auc_test = []
        accuracy_test = []
        mcc_test = []
        recall_test = []

        # Cross-validation loop
        for train_dataset, validation_dataset, test_dataset in datasets:
            import traceback

            try:
                model = get_model(module, batch_size, epochs, train_dataset, validation_dataset)
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()  # This will print the full traceback
                return float('-inf')
            
            predictions_probability = model.predict_proba(validation_dataset)
            predictions = model.predict(validation_dataset)
            y_true = validation_dataset.y

            # Evaluate on validation set
            f1_macro_validation.append(f1_score(y_true, predictions))
            precision_validation.append(precision_score(y_true, predictions))
            roc_auc_validation.append(roc_auc_score(y_true, predictions_probability))
            accuracy_validation.append(accuracy_score(y_true, predictions))
            mcc_validation.append(matthews_corrcoef(y_true, predictions))
            recall_validation.append(recall_score(y_true, predictions))


            # Evaluate on test set
            predictions_probability = model.predict_proba(test_dataset)
            predictions = model.predict(test_dataset)
            y_true = test_dataset.y
            f1_macro_test.append(f1_score(y_true, predictions))
            precision_test.append(precision_score(y_true, predictions))
            roc_auc_test.append(roc_auc_score(y_true, predictions_probability))
            accuracy_test.append(accuracy_score(y_true, predictions))
            mcc_test.append(matthews_corrcoef(y_true, predictions))
            recall_test.append(recall_score(y_true, predictions))

        avg_metrics = {
            "Model type": ["gat"],
            "Trial": [trial.number],
            "F1_macro_validation_set": [np.mean(f1_macro_validation)],
            "Precision validation set": [np.mean(precision_validation)],
            "recall validation set": [np.mean(recall_validation)],
            "roc_auc validation set": [np.mean(roc_auc_validation)],
            "accuracy validation set": [np.mean(accuracy_validation)],
            "mcc validation set": [np.mean(mcc_validation)],
            "F1_macro_test_set": [np.mean(f1_macro_test)],
            "Precision test set": [np.mean(precision_test)],
            "recall test set": [np.mean(recall_test)],
            "roc_auc test set": [np.mean(roc_auc_test)],
            "accuracy test set": [np.mean(accuracy_test)],
            "mcc test set": [np.mean(mcc_test)],
            "Batch size": [batch_size],
            "Learning rate": [learning_rate],
            "Protein layers": [protein_head_layers],
            "Compound layers": [compound_head_layers],
            "Final head layers": [final_head_layers],
            "Batch size": [batch_size], 
            "Learning rate": [learning_rate], 
            "Epochs": [epochs],
            "batch_norm": [batch_norm],
            "dropout": [dropout]
        }

        # Create a DataFrame and save to CSV
        results_df = pd.DataFrame(avg_metrics)
        results = pd.concat([results, results_df])
        
        results.to_csv(self.results_output_file, index=False)

        return np.mean(mcc_validation)
    
def load_datasets(ids_for_datasets):
    import pandas as pd
    dataset = pd.read_csv("integrated_dataset_descriptors_available.csv")

    datasets = []

    from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset

    for train_ids, val_ids, test_ids in ids_for_datasets:

        train_dataset = dataset[dataset["Enzyme ID"].isin(train_ids)]
        train_dataset = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

        train_dataset = MultiInputDataset(dataframe=train_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")
        
        train_dataset.load_features("features_proteins_esm2_3b", "proteins")
        train_dataset.load_features("features_compounds_np_classifier_fp", "ligands")

        import pandas as pd

        validation_dataset = dataset[dataset["Enzyme ID"].isin(val_ids)]

        validation_dataset = MultiInputDataset(dataframe=validation_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")
        validation_dataset.load_features("features_proteins_esm2_3b", "proteins")
        validation_dataset.load_features("features_compounds_np_classifier_fp", "ligands")


        test_dataset = dataset[dataset["Enzyme ID"].isin(test_ids)]
        
        test_dataset = MultiInputDataset(dataframe=test_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")
        test_dataset.load_features("features_proteins_esm2_3b", "proteins")
        test_dataset.load_features("features_compounds_np_classifier_fp", "ligands")
        
        datasets.append((train_dataset, validation_dataset, test_dataset))
    return datasets
    
def experiment_optimize(ids_for_datasets):
    
    experiment = BindingExperiment(ids_for_datasets=ids_for_datasets, study_name="binding_np_classifier", storage="sqlite:///binding_np_classifier.db", sampler= optuna.samplers.TPESampler(seed=123),
                                    direction="maximize", load_if_exists=True, results_output_file="binding_np_classifier.csv",
                                        folder_path="kroll_experiment_models")
    experiment.run(n_trials=100, n_jobs=1)

def test_train_model(ids_for_datasets, protein_head_layers, compound_head_layers, final_head_layers, batch_norm, dropout, batch_size, learning_rate, epochs):

        
    module = InteractionModel(protein_model_path="esm2_3b.pt", compounds_model_path="np_classifier.ckpt", 
                        learning_rate=learning_rate, protein_head_layers=protein_head_layers, compound_head_layers=compound_head_layers, 
                        final_head_layers=final_head_layers, batch_norm=batch_norm, dropout=dropout)
    
    datasets = load_datasets(ids_for_datasets)

    f1_macro_validation = []
    precision_validation = []
    roc_auc_validation = []
    accuracy_validation = []
    mcc_validation = []
    recall_validation = []

    f1_macro_test = []
    precision_test = []
    roc_auc_test = []
    accuracy_test = []
    mcc_test = []
    recall_test = []

    # Cross-validation loop
    for train_dataset, validation_dataset, test_dataset in datasets:
        import traceback

        try:
            model = get_model(module, batch_size, epochs, train_dataset, validation_dataset)
        except Exception as e:
            print("An error occurred:")
            traceback.print_exc()  # This will print the full traceback
            return float('-inf')
        
        predictions_probability = model.predict_proba(validation_dataset)
        predictions = model.predict(validation_dataset)
        y_true = validation_dataset.y

        # Evaluate on validation set
        f1_macro_validation.append(f1_score(y_true, predictions))
        precision_validation.append(precision_score(y_true, predictions))
        roc_auc_validation.append(roc_auc_score(y_true, predictions_probability))
        accuracy_validation.append(accuracy_score(y_true, predictions))
        mcc_validation.append(matthews_corrcoef(y_true, predictions))
        recall_validation.append(recall_score(y_true, predictions))


        # Evaluate on test set
        predictions_probability = model.predict_proba(test_dataset)
        predictions = model.predict(test_dataset)
        y_true = test_dataset.y
        f1_macro_test.append(f1_score(y_true, predictions))
        precision_test.append(precision_score(y_true, predictions))
        roc_auc_test.append(roc_auc_score(y_true, predictions_probability))
        accuracy_test.append(accuracy_score(y_true, predictions))
        mcc_test.append(matthews_corrcoef(y_true, predictions))
        recall_test.append(recall_score(y_true, predictions))

        # Open a file in write mode
        with open('validation_metrics.txt', 'a+') as file:
            # Write the individual metrics to the file
            file.write(f"Individual Metrics:\n")
            for f1, acc, mcc in zip(f1_macro_validation, accuracy_validation, mcc_validation):
                file.write(f"F1 Macro: {f1}, Accuracy: {acc}, MCC: {mcc}\n")
                file.write(f"\n")

    with open('validation_metrics.txt', 'a+') as file:
        # Write the mean metrics to the file
        file.write(f"\nMean Metrics:\n")
        file.write(f"Mean F1 Macro: {np.mean(f1_macro_validation)}\n")
        file.write(f"Mean Accuracy: {np.mean(accuracy_validation)}\n")
        file.write(f"Mean MCC: {np.mean(mcc_validation)}\n")
    

if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    from plants_sm.io.pickle import read_pickle
    splits = read_pickle("splits/splits_0_6_proteins.pkl")
    # test_train_model(splits, [640],[2560, 1280],[2560, 1280],False,0.0990217559638052, 64, 5.238402256370824e-05, 153)
    # experiment_optimize(splits)
    experiment = BindingExperiment(ids_for_datasets=splits, study_name="binding_np_classifier", storage="sqlite:///binding_np_classifier.db", sampler= optuna.samplers.TPESampler(seed=123),
                                    direction="maximize", load_if_exists=True, results_output_file="binding_np_classifier.csv",
                                        folder_path="kroll_experiment_models")
    experiment.run(n_trials=100, n_jobs=1)
