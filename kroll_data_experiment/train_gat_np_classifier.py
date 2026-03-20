import os

import numpy as np
from enzyme_substrate_prediction.models.structural_gat_np_classifier import MolecularGAT3D, MolecularGraphDataset, collate_fn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch.utils.data import Dataset, DataLoader




def train_model(dataset, validation_dataset, num_heads_descriptors_cross_attention, num_heads_gat, num_heads_attention, activation_attention, dropout_gat, \
                                prediction_layers, dropout_attention, batch_size, learning_rate, patience, max_epochs = 100):
    

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    callbacks = []
    if validation_dataset is not None:
        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        # Create the early stopping callback
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',  # metric to monitor
            patience=patience,          # number of epochs with no improvement after which training will be stopped
            verbose=True,
            mode='min'           # stop when the metric is minimized
        )

        # Create the model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',  # metric to monitor
            dirpath='checkpoints/',  # directory to save the model checkpoints
            filename='best-checkpoint',  # filename for the best checkpoint
            save_top_k=1,  # save the top k models
            mode='min',  # save the model when the metric is minimized
            verbose=True
        )
        callbacks = [early_stopping_callback, checkpoint_callback]

    model = MolecularGAT3D(in_dim=10,
                           num_heads_gat=num_heads_gat, num_heads_attention=num_heads_attention, 
                           activation_attention=activation_attention, dropout_gat=dropout_gat, 
                           num_heads_descriptors_cross_attention=num_heads_descriptors_cross_attention, prediction_layers=prediction_layers, 
                           dropout_attention=dropout_attention,
                           learning_rate=learning_rate)


    trainer = Trainer(max_epochs=max_epochs, devices=[3], callbacks=callbacks)
    # trainer = Trainer(max_epochs=100, accelerator="cpu", callbacks=[early_stopping_callback, checkpoint_callback])
    
    if validation_dataset is not None:
        trainer.fit(model, dataloader, val_dataloaders=val_dataloader)
    else:
        trainer.fit(model, dataloader)

    if validation_dataset is not None:
        best_model_path = checkpoint_callback.best_model_path
        best_model = MolecularGAT3D.load_from_checkpoint(best_model_path)
        best_epoch = (trainer.current_epoch - 1) - early_stopping_callback.wait_count

        os.remove(best_model_path)
        return best_model, best_epoch
    else:
        best_model = model

        return best_model, None

def test_model(model, test_dataset):

    test_dataloader = DataLoader(test_dataset, batch_size=56, shuffle=False, collate_fn=collate_fn)

    indices = []
    predictions_list = []
    model.eval()
    y_true = []
    model = model.to("cuda:3")

    for batched_data in test_dataloader:
        # Perform your model's forward pass
        indices.extend(list(batched_data.idx.numpy()))
        y_true.extend(list(batched_data.y.numpy()))
        batched_data = batched_data.to("cuda:3")
        predictions = model(batched_data)
        predictions_list.extend(list(predictions.reshape(predictions.shape[0], ).cpu().detach().numpy()))

    predictions_list = np.nan_to_num(predictions_list, nan=0)
    
    # # Zip the predictions with their corresponding indicessportsurge
    # paired = list(zip(indices, predictions_list))

    # # Sort the pairs based on the indices
    # paired_sorted = sorted(paired, key=lambda x: x[0])

    # # Extract the sorted predictions
    # sorted_predictions = np.array([pred for _, pred in paired_sorted])
    binary_array = (np.array(predictions_list) >= 0.5).astype(int)
    binary_array = np.nan_to_num(binary_array, nan=0)

    return predictions_list, binary_array, y_true

import traceback
import optuna
from plants_sm.hyperparameter_optimization.experiment import Experiment
from torch import nn
from sklearn.metrics import *
import pandas as pd

class BindingExperiment(Experiment):
    def __init__(self, train_dataset, validation_dataset, test_dataset, **kwargs):

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.results_output_file = kwargs.pop("results_output_file")
        self.folder_path = kwargs.pop("folder_path")
        self.best_result = 0
        self.best_hyperparameters = {}
        self.best_epoch = 0
        super().__init__(**kwargs)

        self.best_result = float("-inf")

    def _steps(self, trial):

        prediction_layers = trial.suggest_categorical("prediction_layers", ["[2048]","[1024]", "[512]", "[2048, 1024]", "[2048, 512]", "[1024, 512]",
                                                                            "[256]", "[512]", "[512, 256]", "[128]", "[256, 128]",
                                                                            "[2048, 1024]", "[2048, 1024, 512]",
                                                                            "[2048, 1024, 1024]",
                                                                            "[2048, 2048, 2048, 1024]",
                                                                            "[2048, 2048, 2048, 1024, 512]"],)
        

        
        # evaluate literal
        prediction_layers = eval(prediction_layers)

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        # hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024, 2048])
        num_heads_gat = trial.suggest_categorical("num_heads_gat", [1, 2, 4, 8])
        num_heads_descriptors_cross_attention = trial.suggest_categorical("num_heads_descriptors_cross_attention", [1, 5])
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
        num_heads_attention = trial.suggest_categorical("num_heads_attention", [1, 2, 4, 8])
        activation_attention = trial.suggest_categorical("activation_attention", ["relu", "tanh", "elu", "leaky_relu"])
        dropout_gat = trial.suggest_categorical("dropout_gat", [0.1, 0.2, 0.3, 0.4, 0.5])
        dropout_attention = trial.suggest_categorical("dropout_attention", [0.1, 0.2, 0.3, 0.4, 0.5])
        # use_descriptors = trial.suggest_categorical("use_descriptors", ["True", "False"])
        # patience = trial.suggest_int("patience", 5, 20)
        # use_descriptors = eval(use_descriptors)
        # assert type(use_descriptors) == bool

        activation_mapping = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),  # Corrected from "tahn" to "tanh"
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU()
        }

        activation_attention = activation_mapping[activation_attention]

        return num_heads_descriptors_cross_attention, num_heads_gat, num_heads_attention, activation_attention, dropout_gat, \
                prediction_layers, dropout_attention, batch_size, learning_rate, 5
                

    def objective(self, trial: optuna.Trial) -> float:

        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame() 

        num_heads_descriptors_cross_attention, num_heads_gat, num_heads_attention, activation_attention, dropout_gat, \
                prediction_layers, dropout_attention, batch_size, learning_rate, patience = self._steps(trial=trial)
        
        # Initialize lists to store metrics for each fold
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


        datasets = load_datasets(self.train_dataset, self.validation_dataset, self.test_dataset)

        # Cross-validation loop
        for train_dataset, validation_dataset, test_dataset in datasets:
            try:
                model, best_epoch = train_model(train_dataset, validation_dataset, num_heads_descriptors_cross_attention, num_heads_gat, num_heads_attention, activation_attention, dropout_gat, \
                                                                        prediction_layers, dropout_attention, batch_size, learning_rate, patience)
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()
                continue  # Skip this fold if an error occurs

            # Evaluate on validation set
            predictions_probability, predictions, y_true = test_model(model, validation_dataset)
            f1_macro_validation.append(f1_score(y_true, predictions))
            precision_validation.append(precision_score(y_true, predictions))
            roc_auc_validation.append(roc_auc_score(y_true, predictions_probability))
            accuracy_validation.append(accuracy_score(y_true, predictions))
            mcc_validation.append(matthews_corrcoef(y_true, predictions))
            recall_validation.append(recall_score(y_true, predictions))


            # Evaluate on test set
            predictions_probability, predictions, y_true = test_model(model, test_dataset)
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
            "num_heads_gat": [num_heads_gat],
            "num_heads_attention": [num_heads_attention],
            "activation_attention": [activation_attention],
            "dropout_gat": [dropout_gat],
            "num_heads_descriptors_cross_attention": [num_heads_descriptors_cross_attention],
            "prediction_layers": [prediction_layers],
            "dropout_attention": [dropout_attention],
            "patience": [patience],
            "best_epoch": [best_epoch],
        }

        # Create a DataFrame and save to CSV
        results_df = pd.DataFrame(avg_metrics)
        results = pd.concat([results, results_df])
        
        results.to_csv(self.results_output_file, index=False)

        # Save the best model if the current trial is better than the previous best
        if np.mean(mcc_validation) > self.best_result:
            self.best_result = np.mean(mcc_validation)
            self.best_hyperparameters = {
                "num_heads_descriptors_cross_attention": num_heads_descriptors_cross_attention,
                "num_heads_gat": num_heads_gat,
                "num_heads_attention": num_heads_attention,
                "activation_attention": activation_attention,
                "dropout_gat": dropout_gat,
                "prediction_layers": prediction_layers,
                "dropout_attention": dropout_attention,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "patience": patience
            }
            self.best_epoch = best_epoch

        return np.mean(mcc_validation)
    
def load_datasets(train_dataset, validation_dataset, test_dataset):
    from deepmol.loaders import SDFLoader

    dataset = SDFLoader("unique_compounds_with_features.sdf", id_field="_ID").create_dataset()

    molecules_dict = dict(zip(list(dataset.ids), list(dataset.mols)))

    from plants_sm.io.pickle import read_pickle

    enzymes_dict = read_pickle("features_proteins_esm2_3b/features.pkl")["proteins"]
    protein_descriptors = read_pickle("propythia_descriptors/features.pkl")["proteins"]
    compounds_features = read_pickle("compounds_features.pkl")

    np_classifier_features = read_pickle("features_compounds_np_classifier_fp/features.pkl")["ligands"]

    datasets = []

    interaction = list(zip(list(train_dataset["Uniprot ID"]), list(train_dataset["molecule ID"])))
    labels = list(train_dataset["Binding"])

    from copy import copy

    train_dataset = MolecularGraphDataset(copy(molecules_dict), mol_descriptors=compounds_features, protein_descriptors = protein_descriptors,
                                    proteins_embeddings_dict=enzymes_dict, interactions=interaction, labels=labels, np_classifier_fp_dict=np_classifier_features)

    interaction = list(zip(list(test_dataset["Uniprot ID"]), list(test_dataset["molecule ID"])))
    labels = list(test_dataset["Binding"])

    test_dataset = MolecularGraphDataset(copy(molecules_dict), mol_descriptors=compounds_features, protein_descriptors = protein_descriptors,
                                proteins_embeddings_dict=enzymes_dict, interactions=interaction, labels=labels, np_classifier_fp_dict=np_classifier_features,
                                mol_scaler=train_dataset.mol_scaler, protein_scaler=train_dataset.protein_scaler)

    if validation_dataset is not None:
        interaction = list(zip(list(validation_dataset["Uniprot ID"]), list(validation_dataset["molecule ID"])))
        labels = list(validation_dataset["Binding"])

        
        validation_dataset = MolecularGraphDataset(copy(molecules_dict), mol_descriptors=compounds_features, protein_descriptors = protein_descriptors,
                                        proteins_embeddings_dict=enzymes_dict, interactions=interaction, labels=labels, np_classifier_fp_dict=np_classifier_features, 
                                        mol_scaler=train_dataset.mol_scaler, protein_scaler=train_dataset.protein_scaler)
        datasets.append((train_dataset, validation_dataset, test_dataset))
    else:
        datasets.append((train_dataset, test_dataset))

    return datasets
    
def experiment_optimize(train_dataset, validation_dataset, test_dataset):
    
    experiment = BindingExperiment(train_dataset, validation_dataset, test_dataset, study_name="binding_3d_gat_np_classifier", storage="sqlite:///binding_3d_gat_np_classifier.db", sampler= optuna.samplers.TPESampler(seed=123),
                                    direction="maximize", load_if_exists=True, results_output_file="binding_3d_gat_np_classifier.csv",
                                        folder_path="kroll_experiment_models/trials")
    for trial in experiment.study.trials:
        if trial.state == optuna.trial.TrialState.FAIL: 
            experiment.study.enqueue_trial(trial.params)
    experiment.run(n_trials=50, n_jobs=1)
    test_train_model(train_dataset, validation_dataset, test_dataset, num_heads_descriptors_cross_attention=experiment.best_hyperparameters["num_heads_descriptors_cross_attention"],
                        num_heads_gat=experiment.best_hyperparameters["num_heads_gat"],
                        num_heads_attention=experiment.best_hyperparameters["num_heads_attention"],
                        activation_attention=experiment.best_hyperparameters["activation_attention"],
                        dropout_gat=experiment.best_hyperparameters["dropout_gat"],
                        prediction_layers=experiment.best_hyperparameters["prediction_layers"],
                        dropout_attention=experiment.best_hyperparameters["dropout_attention"],
                        batch_size=experiment.best_hyperparameters["batch_size"],
                        learning_rate=experiment.best_hyperparameters["learning_rate"],
                        patience=experiment.best_hyperparameters["patience"],
                        max_epochs=experiment.best_epoch+1)


def load_datasets_cv(ids_for_datasets):
    from deepmol.loaders import SDFLoader

    dataset = SDFLoader("unique_compounds_with_features.sdf", id_field="_ID").create_dataset()

    molecules_dict = dict(zip(list(dataset.ids), list(dataset.mols)))

    from plants_sm.io.pickle import read_pickle

    enzymes_dict = read_pickle("features_proteins_esm2_3b/features.pkl")["proteins"]
    protein_descriptors = read_pickle("propythia_descriptors/features.pkl")["proteins"]
    compounds_features = read_pickle("compounds_features.pkl")

    np_classifier_features = read_pickle("features_compounds_np_classifier_fp/features.pkl")["ligands"]

    import pandas as pd
    dataset = pd.read_csv("curated_dataset.csv")

    datasets = []

    for train_ids, val_ids, test_ids in ids_for_datasets:

        train_dataset = dataset[dataset["Enzyme ID"].isin(train_ids)]

        interaction = list(zip(list(train_dataset["Enzyme ID"]), list(train_dataset["Substrate ID"])))
        labels = list(train_dataset["Binding"])

        import pandas as pd

        validation_dataset = dataset[dataset["Enzyme ID"].isin(val_ids)]
        validation_interaction = list(zip(list(validation_dataset["Enzyme ID"]), list(validation_dataset["Substrate ID"])))

        train_dataset = pd.concat([train_dataset, validation_dataset], axis=0)
        interaction = interaction + validation_interaction
        labels = labels + list(validation_dataset["Binding"])

        from copy import copy

        train_dataset = MolecularGraphDataset(copy(molecules_dict), mol_descriptors=compounds_features, protein_descriptors = protein_descriptors,
                                        proteins_embeddings_dict=enzymes_dict, interactions=interaction, labels=labels, np_classifier_fp_dict=np_classifier_features)

        test_dataset = dataset[dataset["Enzyme ID"].isin(test_ids)]
        interaction = list(zip(list(test_dataset["Enzyme ID"]), list(test_dataset["Substrate ID"])))
        labels = list(test_dataset["Binding"])

        test_dataset = MolecularGraphDataset(copy(molecules_dict), mol_descriptors=compounds_features, protein_descriptors = protein_descriptors,
                                    proteins_embeddings_dict=enzymes_dict, interactions=interaction, labels=labels, np_classifier_fp_dict=np_classifier_features,
                                    mol_scaler=train_dataset.mol_scaler, protein_scaler=train_dataset.protein_scaler)
        
        datasets.append((train_dataset, test_dataset))
    return datasets


def test_train_model_cv():

    import pytorch_lightning as pl

    from plants_sm.io.pickle import read_pickle
    ids_for_datasets = read_pickle("splits/splits_0_6_proteins.pkl")


    datasets = []

    datasets = load_datasets_cv(ids_for_datasets)

    file_exists = False

    for i in range(5):
        # Set the seed for all random number generators
        pl.seed_everything(i, workers=True)
        fold_idx = 0
        for train_dataset, test_dataset in datasets:
            try:
                model = train_model(train_dataset, None, 5, 4,
                                    8, nn.LeakyReLU(negative_slope=0.01), 0.3,
                                    [512], 0.5,
                                    64, 0.000351, 5, max_epochs=10)
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()
                continue  # Skip this fold if an error occurs

            # Evaluate on test set
            predictions_probability, predictions, y_true = test_model(model, test_dataset)
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
            df.to_csv("results_gat_np_classifier_10_epochs.csv", mode='a', index=False, header=not file_exists)
            file_exists = True  # Ensure header is not written again

def test_train_model(train_dataset, validation_dataset, test_dataset, num_heads_descriptors_cross_attention=5, num_heads_gat=4,
                     num_heads_attention=8, activation_attention=nn.LeakyReLU(negative_slope=0.01), dropout_gat=0.3,
                     prediction_layers=[512], dropout_attention=0.5,
                     batch_size=64, learning_rate=0.000351, patience=5, max_epochs=10):

    import pytorch_lightning as pl

    datasets = []

    file_exists = False
    

    for i in range(5):
        # Set the seed for all random number generators
        pl.seed_everything(i, workers=True)
        train_dataset = pd.concat([train_dataset, validation_dataset], axis=0)
        train_dataset = train_dataset.sample(frac=1, random_state=i).reset_index(drop=True)

        datasets = load_datasets(train_dataset, None, test_dataset)
        for train_dataset, test_dataset in datasets:
            try:
                model, _ = train_model(train_dataset, None, num_heads_descriptors_cross_attention, num_heads_gat,
                                   num_heads_attention, activation_attention, dropout_gat,
                                    prediction_layers, dropout_attention,
                                    batch_size, learning_rate, patience, max_epochs)
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()
                continue  # Skip this fold if an error occurs

            # Evaluate on test set
            predictions_probability, predictions, y_true = test_model(model, test_dataset)
            result = {
                'seed': i,
                'f1_macro': f1_score(y_true, predictions),
                'precision': precision_score(y_true, predictions),
                'roc_auc': roc_auc_score(y_true, predictions_probability),
                'accuracy': accuracy_score(y_true, predictions),
                'mcc': matthews_corrcoef(y_true, predictions),
                'recall': recall_score(y_true, predictions),
            }

            df = pd.DataFrame([result])
            df.to_csv("results_gat_np_classifier.csv", mode='a', index=False, header=not file_exists)
            file_exists = True  # Ensure header is not written again




if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    from plants_sm.io.pickle import read_pickle

    train_dataset = pd.read_csv("train_dataset_w_split.csv")
    test_dataset = pd.read_csv("test_dataset_w_representation_filtered.csv")
    validation_dataset = pd.read_csv("validation_dataset.csv")

    experiment_optimize(train_dataset, validation_dataset, test_dataset)
