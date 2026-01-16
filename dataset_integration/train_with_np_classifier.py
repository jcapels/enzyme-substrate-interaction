
import numpy as np
from plants_sm.hyperparameter_optimization.experiment import Experiment

import optuna

import os
import pandas as pd

from sklearn.metrics import *
from enzyme_substrate_prediction.models.interaction_individual_module import InteractionModel
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
                    devices=[0],
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
                    devices=[0],
                    accelerator="gpu",
                    )
        model.reset_weights()
        model.fit(train_dataset)
        return model, None

import optuna
from plants_sm.hyperparameter_optimization.experiment import Experiment
from torch import nn
from sklearn.metrics import *
import pandas as pd

class BindingExperiment(Experiment):
    def __init__(self, baseline = False, results_output_file="results.csv", input_dim=600, 
                 classification_neurons=1,
                 folder_path="trials", proteins_split=True, **kwargs):

        self.ids_for_datasets = kwargs.pop("ids_for_datasets")
        self.folder_path = folder_path
        self.baseline = baseline
        self.results_output_file = results_output_file
        self.input_dim = input_dim
        self.classification_neurons = classification_neurons
        self.best_result = float("-inf")
        self.best_result = 0
        self.best_hyperparameters = {}
        self.best_epoch = 0
        self.proteins_split = proteins_split
        super().__init__(**kwargs)

    def _steps(self, trial):

        protein_head_layers = trial.suggest_categorical("protein_head_layers", ["[2560]","[1280]", "[640]", "[2560, 1280]", "[2560, 640]", "[1280, 640]",
                                                                            "[2560, 1280]", "[2560, 1280, 640]"],)
                                                                            # "[2560, 1280, 1280]",
                                                                            # "[2560, 1280, 1280, 640]",
                                                                            # "[2560, 1280, 1280, 1280]",
                                                                            # "[2560, 1280, 1280, 1280, 640]",
                                                                            # "[2560, 1280, 1280, 1280, 1280]",
                                                                            # "[2560, 2560, 2560, 1280]",
                                                                            # "[2560, 2560, 2560, 1280, 640]"],)
        
        compound_head_layers = trial.suggest_categorical("compound_head_layers", ["[2560]","[1280]", "[640]", "[2560, 1280]", "[2560, 640]", "[1280, 640]",
                                                                            "[2560, 1280]", "[2560, 1280, 640]"],)
                                                                            # "[2560, 1280, 1280]",
                                                                            # "[2560, 1280, 1280, 640]",
                                                                            # "[2560, 1280, 1280, 1280]",
                                                                            # "[2560, 1280, 1280, 1280, 640]",
                                                                            # "[2560, 1280, 1280, 1280, 1280]",
                                                                            # "[2560, 2560, 2560, 1280]",
                                                                            # "[2560, 2560, 2560, 1280, 640]"],)
        
        final_head_layers = trial.suggest_categorical("final_head_layers", ["[2560]","[1280]", "[640]", "[2560, 1280]", "[2560, 640]", "[1280, 640]",
                                                                            "[2560, 1280]", "[2560, 1280, 640]"],)
                                                                            # "[2560, 1280, 1280]",
                                                                            # "[2560, 1280, 1280, 640]",
                                                                            # "[2560, 1280, 1280, 1280]",
                                                                            # "[2560, 1280, 1280, 1280, 640]",
                                                                            # "[2560, 1280, 1280, 1280, 1280]",
                                                                            # "[2560, 2560, 2560, 1280]",
                                                                            # "[2560, 2560, 2560, 1280, 640]"],)
        

        
        # evaluate literal
        protein_head_layers = eval(protein_head_layers)
        compound_head_layers = eval(compound_head_layers)
        final_head_layers = eval(final_head_layers)

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)

        batch_norm = trial.suggest_categorical("batch_norm", [True, False])
        # dropout = trial.suggest_float("dropout", 0, 1)
        dropout = trial.suggest_float("dropout", 0, 0.5)

        return protein_head_layers, compound_head_layers, final_head_layers, batch_norm, dropout, batch_size, learning_rate

    def objective(self, trial: optuna.Trial) -> float:

        results = []
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame() 

        protein_head_layers, compound_head_layers, final_head_layers, batch_norm, dropout, batch_size, learning_rate = self._steps(trial)
        epochs = trial.suggest_int("epochs", 50, 200)
        results = []
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame() 
        
        module = InteractionModel(protein_model_path="esm2_3b.pt", compounds_model_path="np_classifier.ckpt", 
                          learning_rate=learning_rate, protein_head_layers=protein_head_layers, compound_head_layers=compound_head_layers,
                          final_head_layers=final_head_layers, batch_norm=batch_norm, dropout=dropout)
        
        if self.proteins_split:
            datasets = load_datasets(self.ids_for_datasets)
        else:
            datasets = load_datasets_compounds(self.ids_for_datasets)

        f1_macro_validation = []
        precision_validation = []
        roc_auc_validation = []
        accuracy_validation = []
        mcc_validation = []
        recall_validation = []

        # Cross-validation loop
        for train_dataset, validation_dataset, _ in datasets:
            import traceback

            try:
                model, best_epoch = get_model(module, batch_size, epochs, train_dataset, validation_dataset)
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
            self.best_result = np.mean(mcc_validation)
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
    
def load_datasets(ids_for_datasets, random_state=42, merge_validation_set=False):
    import pandas as pd
    dataset = pd.read_csv("curated_dataset_no_stereochemistry_duplicates.csv")

    datasets = []

    from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset

    for train_ids, val_ids, test_ids in ids_for_datasets:

        train_dataset = dataset[dataset["Enzyme ID"].isin(train_ids)]
        
        
        import pandas as pd

        validation_dataset = dataset[dataset["Enzyme ID"].isin(val_ids)]

        if merge_validation_set == True:
            train_dataset = pd.concat((train_dataset, validation_dataset), axis=0)

        else:

            validation_dataset = MultiInputDataset(dataframe=validation_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")
            
            validation_dataset.load_features("features_proteins_esm2_3b", "proteins")
            validation_dataset.load_features("features_compounds_np_classifier_fp", "ligands")

        train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_dataset = MultiInputDataset(dataframe=train_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")

        train_dataset.load_features("features_proteins_esm2_3b", "proteins")
        train_dataset.load_features("features_compounds_np_classifier_fp", "ligands")

        test_dataset = dataset[dataset["Enzyme ID"].isin(test_ids)]
        
        test_dataset = MultiInputDataset(dataframe=test_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")
        test_dataset.load_features("features_proteins_esm2_3b", "proteins")
        test_dataset.load_features("features_compounds_np_classifier_fp", "ligands")
        
        if merge_validation_set:
            datasets.append((train_dataset, test_dataset))
        else:
            datasets.append((train_dataset, validation_dataset, test_dataset))
    return datasets

def load_datasets_compounds(ids_for_datasets, random_state=42, merge_validation_set=False):
    import pandas as pd
    dataset = pd.read_csv("curated_dataset_no_stereochemistry_duplicates.csv")

    datasets = []

    from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset

    for train_ids, val_ids, test_ids in ids_for_datasets:

        train_dataset = dataset[dataset["Substrate ID"].isin(train_ids)]
        
        
        import pandas as pd

        validation_dataset = dataset[dataset["Substrate ID"].isin(val_ids)]

        if merge_validation_set == True:
            train_dataset = pd.concat((train_dataset, validation_dataset), axis=0)

        else:

            validation_dataset = MultiInputDataset(dataframe=validation_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")
            
            validation_dataset.load_features("features_proteins_esm2_3b", "proteins")
            validation_dataset.load_features("features_compounds_np_classifier_fp", "ligands")

        train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_dataset = MultiInputDataset(dataframe=train_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")

        train_dataset.load_features("features_proteins_esm2_3b", "proteins")
        train_dataset.load_features("features_compounds_np_classifier_fp", "ligands")

        test_dataset = dataset[dataset["Substrate ID"].isin(test_ids)]
        
        test_dataset = MultiInputDataset(dataframe=test_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")
        test_dataset.load_features("features_proteins_esm2_3b", "proteins")
        test_dataset.load_features("features_compounds_np_classifier_fp", "ligands")
        
        if merge_validation_set:
            datasets.append((train_dataset, test_dataset))
        else:
            datasets.append((train_dataset, validation_dataset, test_dataset))
    return datasets

def test_train_model(ids_for_datasets, protein_head_layers, compound_head_layers, final_head_layers, 
                     batch_norm, dropout, batch_size, learning_rate, epochs, proteins_split=True, 
                     output_file = "results_np_classifier.csv"):

    file_exists = False

    for i in range(1,6):
        # Set the seed for all random number generators
        pl.seed_everything(i, workers=True)
            
        module = InteractionModel(protein_model_path="esm2_3b.pt", compounds_model_path="np_classifier.ckpt", 
                            learning_rate=learning_rate, protein_head_layers=protein_head_layers, compound_head_layers=compound_head_layers, 
                            final_head_layers=final_head_layers, batch_norm=batch_norm, dropout=dropout)
        
        if proteins_split:
            datasets = load_datasets(ids_for_datasets, random_state=i, merge_validation_set=True)
        else:
            datasets = load_datasets_compounds(ids_for_datasets, random_state=i, merge_validation_set=True)

        # Cross-validation loop
        fold_idx = 0
        for train_dataset, test_dataset in datasets:

            model, _ = get_model(module, batch_size, epochs, train_dataset, None)
            
            # Evaluate on test set
            predictions_probability = model.predict_proba(test_dataset)
            predictions = model.predict(test_dataset)
            y_true = test_dataset.y

            result = {
                'seed': i,
                'fold': fold_idx,
                'f1_macro': f1_score(y_true, predictions),
                'precision': precision_score(y_true, predictions),
                'roc_auc': roc_auc_score(y_true, predictions_probability),
                'accuracy': accuracy_score(y_true, predictions),
                'mcc': matthews_corrcoef(y_true, predictions),
                'recall': recall_score(y_true, predictions),
            }
            fold_idx+=1

            df = pd.DataFrame([result])
            df.to_csv(output_file, mode='a', index=False, header=not file_exists)
            file_exists = True  # Ensure header is not written again
    
def experiment_optimize(ids_for_datasets, name="binding_np_classifier", proteins_split=True):
    
    experiment = BindingExperiment(ids_for_datasets=ids_for_datasets, study_name=name, storage="sqlite:///binding_np_classifier_no_stereo.db", sampler= optuna.samplers.TPESampler(seed=123),
                                    direction="maximize", load_if_exists=True, results_output_file=f"{name}.csv",
                                        folder_path="kroll_experiment_models_no_stereo", proteins_split=proteins_split)
    experiment.run(n_trials=50, n_jobs=1)

    test_train_model(ids_for_datasets, experiment.best_hyperparameters["protein_head_layers"], experiment.best_hyperparameters["compound_head_layers"],
                    experiment.best_hyperparameters["final_head_layers"], experiment.best_hyperparameters["batch_norm"], 
                    experiment.best_hyperparameters["dropout"], experiment.best_hyperparameters["batch_size"], 
                    experiment.best_hyperparameters["learning_rate"], experiment.best_epoch+1, proteins_split=proteins_split,
                    output_file=f"results_{name}.csv")
        
    

if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    from plants_sm.io.pickle import read_pickle

    splits = read_pickle("compounds_split/splits_compounds_08.pkl")

    experiment_optimize(splits, name="binding_np_classifier_compounds_08_no_stereo", proteins_split=False)

    splits = read_pickle("compounds_split/splits_compounds_06.pkl")

    experiment_optimize(splits, name="binding_np_classifier_compounds_06_no_stereo", proteins_split=False)

    splits = read_pickle("compounds_split/splits_compounds_04.pkl")

    experiment_optimize(splits, name="binding_np_classifier_compounds_04_no_stereo", proteins_split=False)

    splits = read_pickle("compounds_split/splits_compounds_02.pkl")

    experiment_optimize(splits, name="binding_np_classifier_compounds_02_no_stereo", proteins_split=False)

    splits = read_pickle("splits/splits_0_6_proteins_train_val_test.pkl")

    experiment_optimize(splits, name="binding_np_classifier_06_proteins_no_stereo", proteins_split=True)

    splits = read_pickle("splits/splits_0_8_proteins_train_val_test.pkl")

    experiment_optimize(splits, name="binding_np_classifier_08_proteins_no_stereo", proteins_split=True)

    splits = read_pickle("splits/splits_0_4_proteins_train_val_test.pkl")

    experiment_optimize(splits, name="binding_np_classifier_04_proteins_no_stereo", proteins_split=True)

