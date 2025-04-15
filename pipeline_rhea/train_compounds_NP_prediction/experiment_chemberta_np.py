
from plants_sm.hyperparameter_optimization.experiment import Experiment

from sklearn.metrics import average_precision_score

import optuna

import os
import pandas as pd

from enzyme_substrate_prediction.models import ModelECNumber
from plants_sm.models.lightning_model import InternalLightningModel
from lightning.pytorch.callbacks import EarlyStopping

class FineTuneExperiment(Experiment):
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

    def _steps(self, trial):

        additional_layers = trial.suggest_categorical("additional_layers", ["[2560]","[1280]", "[640]", "[2560, 1280]", "[2560, 640]", "[1280, 640]",
                                                                            "[2560, 1280]", "[2560, 1280, 640]",
                                                                            "[2560, 1280, 1280]",
                                                                            "[2560, 1280, 1280, 640]",
                                                                            "[2560, 1280, 1280, 1280]",
                                                                            "[2560, 1280, 1280, 1280, 640]",
                                                                            "[2560, 1280, 1280, 1280, 1280]",
                                                                            "[2560, 2560, 2560, 1280]",
                                                                            "[2560, 2560, 2560, 1280, 640]"],)
        
        # evaluate literal
        additional_layers = eval(additional_layers)

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)

        return additional_layers, batch_size, learning_rate

    def objective(self, trial: optuna.Trial) -> float:
        additional_layers, batch_size, learning_rate = self._steps(trial)
        epochs = trial.suggest_int("epochs", 50, 200)
        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame(columns=["Model type", "Trial", "F1_macro_validation_set", "Average precision validation set", "F1_macro_test_set", "Average precision test set", "Additional layers", "Batch size", "Learning rate", "Epochs"]) 
        train_dataset, validation_dataset, test_dataset = self.datasets
        callbacks = EarlyStopping("val_metric", patience=5, mode="max")

        module = ModelECNumber(input_dim=600, layers=additional_layers, classification_neurons=730, metric=average_precision_score,
                               learning_rate=learning_rate)
        model = InternalLightningModel(module=module, max_epochs=100,
                batch_size=batch_size,
                devices=[2],
                accelerator="gpu",
                callbacks=[callbacks]
                )
        model.fit(train_dataset, validation_dataset)
        predictions = model.predict(validation_dataset)

        from sklearn.metrics import f1_score

        f1_macro_validation_set = f1_score(validation_dataset.y, predictions, average="macro")
        average_precision_validation_set = average_precision_score(validation_dataset.y, predictions)

        predictions_test_dataset = model.predict(test_dataset)
        f1_macro_test_set = f1_score(test_dataset.y, predictions_test_dataset, average="macro")
        average_precision_test_set = average_precision_score(test_dataset.y, predictions_test_dataset)

        # instead of append use concatenate
        results = pd.concat([results, pd.DataFrame({"Model type": ["chemberta"], 
                                                    "Trial": [trial.number], 
                                                    "F1_macro_validation_set": [f1_macro_validation_set], 
                                                    "Average precision validation set": [average_precision_validation_set], 
                                                    "F1_macro_test_set": [f1_macro_test_set], 
                                                    "Average precision test set": [average_precision_test_set], 
                                                    "Additional layers": [additional_layers], 
                                                    "Batch size": [batch_size], 
                                                    "Learning rate": [learning_rate], 
                                                    "Epochs": [epochs]})], ignore_index=True)
        
        results.to_csv(self.results_output_file, index=False)
        model.save(f"{self.folder_path}/model_{trial.number}")

        return average_precision_validation_set
    
def experiment_optimize():

    from plants_sm.data_structures.dataset import SingleInputDataset
    # load datasets

    train_dataset = SingleInputDataset.from_csv("train_dataset.csv", representation_field="SMILES", instances_ids_field="key",
                                                    labels_field=slice(2,-1))
    test_dataset = SingleInputDataset.from_csv("test_dataset.csv", representation_field="SMILES", instances_ids_field="key",
                                                    labels_field=slice(2,-1))
    validation_set = SingleInputDataset.from_csv("validation_dataset.csv", representation_field="SMILES", instances_ids_field="key",
                                                    labels_field=slice(2,-1))

    train_dataset.load_features("train_dataset_features")
    test_dataset.load_features("test_dataset_features")
    validation_set.load_features("validation_dataset_features")
        
    datasets = (train_dataset, validation_set, test_dataset)
    
    experiment = FineTuneExperiment(datasets=datasets, study_name="chemberta_np_experiment_with_optimization", storage="sqlite:///transfer_learning_experiment.db", sampler= optuna.samplers.TPESampler(),
                                    direction="maximize", load_if_exists=True, results_output_file="chemberta_np_experiment_with_optimization_results.csv",
                                        folder_path="chembert2a/trials")
    
    experiment.run(n_trials=25, n_jobs=1)
    

if __name__ == "__main__":
   experiment_optimize()