import os
from enzyme_substrate_prediction.models.structural_gat import MolecularGAT3D
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from enzyme_substrate_prediction.models.structural_gat import MolecularGraphDataset, collate_fn
from torch.utils.data import Dataset, DataLoader

def train_model(dataset, validation_dataset, hidden_dim, num_heads_gat, num_heads_attention, activation_attention, dropout_gat,
                use_descriptors, prediction_layers, dropout_attention, batch_size, learning_rate, patience):
    

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

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

    from torch import nn

    model = MolecularGAT3D(in_dim=10, hidden_dim=hidden_dim, 
                           num_heads_gat=num_heads_gat, num_heads_attention=num_heads_attention, 
                           activation_attention=activation_attention, dropout_gat=dropout_gat, 
                           use_descriptors=use_descriptors, prediction_layers=prediction_layers, dropout_attention=dropout_attention,
                           learning_rate=learning_rate)


    trainer = Trainer(max_epochs=100, devices=[0], callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(model, dataloader, val_dataloaders=val_dataloader)
    # Load the best model weights
    best_model_path = checkpoint_callback.best_model_path
    best_model = MolecularGAT3D.load_from_checkpoint(best_model_path)

    return best_model

def test_model(model, test_dataset):

    test_dataloader = DataLoader(test_dataset, batch_size=56, shuffle=False, collate_fn=collate_fn)

    import torch

    indices = []
    predictions_list = []
    model.eval()
    y_true = []
    model = model.to("cuda:0")

    for batched_data in test_dataloader:
        # Perform your model's forward pass
        indices.extend(list(batched_data.idx.numpy()))
        y_true.extend(list(batched_data.y.numpy()))
        batched_data = batched_data.to("cuda:0")
        predictions = model(batched_data)
        predictions_list.extend(list(predictions.reshape(predictions.shape[0], ).cpu().detach().numpy()))

    import numpy as np

    # # Zip the predictions with their corresponding indices
    # paired = list(zip(indices, predictions_list))

    # # Sort the pairs based on the indices
    # paired_sorted = sorted(paired, key=lambda x: x[0])

    # # Extract the sorted predictions
    # sorted_predictions = np.array([pred for _, pred in paired_sorted])
    binary_array = (np.array(predictions_list) >= 0.5).astype(int)

    return predictions_list, binary_array, y_true

import traceback
import optuna
from plants_sm.hyperparameter_optimization.experiment import Experiment
from torch import nn
from sklearn.metrics import *
import pandas as pd

class KrollDataExperiment(Experiment):
    def __init__(self, **kwargs):

        self.train_dataset, self.validation_dataset, self.test_dataset = kwargs.pop("datasets")
        self.results_output_file = kwargs.pop("results_output_file")
        self.folder_path = kwargs.pop("folder_path")
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

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024, 2048])
        num_heads_gat = trial.suggest_categorical("num_heads_gat", [1, 2, 4, 8])
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
        num_heads_attention = trial.suggest_categorical("num_heads_attention", [1, 2, 4, 8])
        activation_attention = trial.suggest_categorical("activation_attention", ["relu", "tanh", "elu", "leaky_relu"])
        dropout_gat = trial.suggest_categorical("dropout_gat", [0.1, 0.2, 0.3, 0.4, 0.5])
        dropout_attention = trial.suggest_categorical("dropout_attention", [0.1, 0.2, 0.3, 0.4, 0.5])
        use_descriptors = trial.suggest_categorical("use_descriptors", ["True", "False"])
        patience = trial.suggest_int("patience", 5, 20)
        use_descriptors = eval(use_descriptors)
        assert type(use_descriptors) == bool

        activation_mapping = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),  # Corrected from "tahn" to "tanh"
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU()
        }

        activation_attention = activation_mapping[activation_attention]

        return hidden_dim, num_heads_gat, num_heads_attention, activation_attention, dropout_gat, \
                use_descriptors, prediction_layers, dropout_attention, batch_size, learning_rate, patience

    def objective(self, trial: optuna.Trial) -> float:

        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame() 

        hidden_dim, num_heads_gat, num_heads_attention, activation_attention, dropout_gat, \
                use_descriptors, prediction_layers, dropout_attention, batch_size, learning_rate, patience = self._steps(trial=trial)

        try:
            model = train_model(self.train_dataset, self.validation_dataset, hidden_dim, num_heads_gat, num_heads_attention, activation_attention, dropout_gat, \
                use_descriptors, prediction_layers, dropout_attention, batch_size, learning_rate, patience)
        except Exception as e:
            print("An error occurred:")
            traceback.print_exc()  # This will print the full traceback
            return float('-inf')
        
        predictions_probability, predictions, y_true = test_model(model, self.validation_dataset)

        f1_macro_validation_set = f1_score(y_true, predictions)
        precision_validation_set = precision_score(y_true, predictions)
        roc_auc_validation_set = roc_auc_score(y_true, predictions_probability)
        accuracy_score_validation_set = accuracy_score(y_true, predictions)
        mcc_score_validation_set = matthews_corrcoef(y_true, predictions)

        predictions_probability, predictions, y_true = test_model(model, self.test_dataset)

        f1_macro_test_set = f1_score(y_true, predictions)
        precision_test_set = precision_score(y_true, predictions)
        roc_auc_test_set = roc_auc_score(y_true, predictions_probability)
        accuracy_score_test_set = accuracy_score(y_true, predictions)
        mcc_score_test_set = matthews_corrcoef(y_true, predictions)

        # instead of append use concatenate
        # Log results with parameters
        results = pd.concat([results, pd.DataFrame({
            "Model type": ["gat"],
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
            "Batch size": [batch_size],
            "Learning rate": [learning_rate],
            "hidden_dim": [hidden_dim],
            "num_heads_gat": [num_heads_gat],
            "num_heads_attention": [num_heads_attention],
            "activation_attention": [activation_attention],
            "dropout_gat": [dropout_gat],
            "use_descriptors": [use_descriptors],
            "prediction_layers": [prediction_layers],
            "dropout_attention": [dropout_attention],
            "patience": [patience]
        })])
        
        results.to_csv(self.results_output_file, index=False)

        return accuracy_score_validation_set
    
def experiment_optimize():


    from deepmol.loaders import SDFLoader

    dataset = SDFLoader("unique_compounds_conformers.sdf", id_field="_ID").create_dataset()

    molecules_dict = dict(zip(list(dataset.ids), list(dataset.mols)))

    from plants_sm.io.pickle import read_pickle

    enzymes_dict = read_pickle("features_proteins_esm2_3b/features.pkl")["enzymes"]

    import pandas as pd

    train_dataset = pd.read_csv("train_dataset_w_representation_split.csv")

    interaction = list(zip(list(train_dataset["Uniprot ID"]), list(train_dataset["molecule ID"])))
    labels = list(train_dataset["Binding"])

    from copy import copy

    dataset = MolecularGraphDataset(copy(molecules_dict), enzymes_dict, interaction, labels)

    import pandas as pd

    validation_dataset = pd.read_csv("validation_dataset_w_representation.csv")
    interaction = list(zip(list(validation_dataset["Uniprot ID"]), list(validation_dataset["molecule ID"])))
    labels = list(validation_dataset["Binding"])

    validation_dataset = MolecularGraphDataset(copy(molecules_dict), enzymes_dict, interaction, labels)

    test_dataset = pd.read_csv("test_dataset_w_representation.csv")
    interaction = list(zip(list(test_dataset["Uniprot ID"]), list(test_dataset["molecule ID"])))
    labels = list(test_dataset["Binding"])

    test_dataset = MolecularGraphDataset(copy(molecules_dict), enzymes_dict, interaction, labels)

    datasets = (dataset, validation_dataset, test_dataset)
    
    experiment = KrollDataExperiment(datasets=datasets, study_name="kroll_experiment_models_gat", storage="sqlite:///kroll_experiment_gat.db", sampler= optuna.samplers.TPESampler(seed=123),
                                    direction="maximize", load_if_exists=True, results_output_file="kroll_experiment_models_results_gat.csv",
                                        folder_path="kroll_experiment_models/trials")
    
    experiment.run(n_trials=100, n_jobs=1)
    

if __name__ == "__main__":
#    experiment_optimize()
    train_dataset = pd.read_csv("train_dataset_w_representation_filtered.csv")
    test_dataset = pd.read_csv("test_dataset_w_representation_filtered.csv")
    validation_dataset = pd.read_csv("validation_dataset.csv")

    experiment_optimize()