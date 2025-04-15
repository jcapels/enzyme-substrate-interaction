import os

import numpy as np
from enzyme_substrate_prediction.models.interaction_model_descriptors import FPESMDataset, InteractionModelCustomized
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch.utils.data import Dataset, DataLoader




def train_model(dataset, validation_dataset, protein_head_layers, compound_head_layers, protein_descriptors_layers, compound_descriptors_layers, final_head_layers, 
                 batch_norm_final, batch_norm_modules, batch_size, learning_rate, patience):
    

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

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

    model = InteractionModelCustomized(protein_head_layers, compound_head_layers, protein_descriptors_layers, compound_descriptors_layers, final_head_layers, 
                 batch_norm_final, batch_norm_modules, learning_rate)


    trainer = Trainer(max_epochs=100, devices=[3], callbacks=[early_stopping_callback, checkpoint_callback])
    # trainer = Trainer(max_epochs=100, accelerator="cpu", callbacks=[early_stopping_callback, checkpoint_callback])
    
    trainer.fit(model, dataloader, val_dataloaders=val_dataloader)
    # Load the best model weights
    best_model_path = checkpoint_callback.best_model_path
    best_model = InteractionModelCustomized.load_from_checkpoint(best_model_path)

    return best_model

def test_model(model, test_dataset):
    test_dataset.to("cuda:3")
    test_dataloader = DataLoader(test_dataset, batch_size=56, shuffle=False)

    indices = []
    predictions_list = []
    model.eval()
    y_true = []
    model = model.to("cuda:3")
    

    for batched_data in test_dataloader:
        fp, compound_descriptors, protein_descriptors, protein_embedding, y = batched_data
        # Perform your model's forward pass
        y_true.extend(list(y.numpy()))
        # batched_data = batched_data.to("cuda:3")
        predictions = model(batched_data)
        predictions_list.extend(list(predictions.reshape(predictions.shape[0], ).cpu().detach().numpy()))

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
    def __init__(self, **kwargs):

        self.ids_for_datasets = kwargs.pop("ids_for_datasets")
        self.results_output_file = kwargs.pop("results_output_file")
        self.folder_path = kwargs.pop("folder_path")
        super().__init__(**kwargs)

        self.best_result = float("-inf")

    def _steps(self, trial):

        final_head_layers = trial.suggest_categorical("final_head_layers", ["[2048]","[1024]", "[512]", "[2048, 1024]", "[2048, 512]", "[1024, 512]",
                                                                            "[256]", "[512]", "[512, 256]", "[128]", "[256, 128]",
                                                                            "[2048, 1024]", "[2048, 1024, 512]",
                                                                            "[2048, 1024, 1024]",
                                                                            "[2048, 2048, 2048, 1024]",
                                                                            "[2048, 2048, 2048, 1024, 512]"],)
        

        
        # evaluate literal
        final_head_layers = eval(final_head_layers)

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
        batch_norm_final = trial.suggest_categorical("batch_norm_final", ["True", "False"])
        batch_norm_final = eval(batch_norm_final)
        batch_norm_modules = trial.suggest_categorical("batch_norm_modules", ["True", "False"])
        batch_norm_modules = eval(batch_norm_modules)
        
        protein_head_layers = trial.suggest_categorical("protein_head_layers", ["[2048]","[1024]", "[512]", "[2048, 1024]", "[2048, 512]", "[1024, 512]",
                                                                            "[256]", "[512]", "[512, 256]", "[128]", "[256, 128]",
                                                                            "[2048, 1024]", "[2048, 1024, 512]",
                                                                            "[2048, 1024, 1024]",
                                                                            "[2048, 2048, 2048, 1024]",
                                                                            "[2048, 2048, 2048, 1024, 512]"],)
        protein_head_layers = eval(protein_head_layers)
        
        compound_head_layers = trial.suggest_categorical("compound_head_layers", ["[2048]","[1024]", "[512]", "[2048, 1024]", "[2048, 512]", "[1024, 512]",
                                                                            "[256]", "[512]", "[512, 256]", "[128]", "[256, 128]",
                                                                            "[2048, 1024]", "[2048, 1024, 512]",
                                                                            "[2048, 1024, 1024]",
                                                                            "[2048, 2048, 2048, 1024]",
                                                                            "[2048, 2048, 2048, 1024, 512]"],)
        compound_head_layers = eval(compound_head_layers)
        
        protein_descriptors_layers = trial.suggest_categorical("protein_descriptors_layers", ["[2048]","[1024]", "[512]", "[2048, 1024]", "[2048, 512]", "[1024, 512]",
                                                                            "[256]", "[512]", "[512, 256]", "[128]", "[256, 128]",
                                                                            "[2048, 1024]", "[2048, 1024, 512]",
                                                                            "[2048, 1024, 1024]",
                                                                            "[2048, 2048, 2048, 1024]",
                                                                            "[2048, 2048, 2048, 1024, 512]"],)
        protein_descriptors_layers = eval(protein_descriptors_layers)
        
        compound_descriptors_layers = trial.suggest_categorical("compound_descriptors_layers", ["[2048]","[1024]", "[512]", "[2048, 1024]", "[2048, 512]", "[1024, 512]",
                                                                            "[256]", "[512]", "[512, 256]", "[128]", "[256, 128]",
                                                                            "[2048, 1024]", "[2048, 1024, 512]",
                                                                            "[2048, 1024, 1024]",
                                                                            "[2048, 2048, 2048, 1024]",
                                                                            "[2048, 2048, 2048, 1024, 512]"],)
        compound_descriptors_layers = eval(compound_descriptors_layers)
        # patience = trial.suggest_int("patience", 5, 20)



        return protein_head_layers, compound_head_layers, protein_descriptors_layers, compound_descriptors_layers, final_head_layers, \
                 batch_norm_final, batch_norm_modules, batch_size, learning_rate, 5
                

    def objective(self, trial: optuna.Trial) -> float:

        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame() 

        protein_head_layers, compound_head_layers, protein_descriptors_layers, compound_descriptors_layers, final_head_layers, \
                 batch_norm_final, batch_norm_modules, batch_size, learning_rate, patience = self._steps(trial=trial)
        
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


        datasets = load_datasets(self.ids_for_datasets)

        # Cross-validation loop
        for train_dataset, validation_dataset, test_dataset in datasets:
            try:
                model = train_model(train_dataset, validation_dataset, protein_head_layers, compound_head_layers, protein_descriptors_layers, compound_descriptors_layers, final_head_layers, \
                 batch_norm_final, batch_norm_modules, batch_size, learning_rate, patience )
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
            "Protein layers": [protein_head_layers],
            "Protein descriptors layers": [protein_descriptors_layers],
            "Compound layers": [compound_head_layers],
            "Compound descriptors layers": [compound_descriptors_layers],
            "Final head layers": [final_head_layers],
            "Batch size": [batch_size], 
            "Learning rate": [learning_rate], 
            "batch_norm_final": [batch_norm_final],
            "batch_norm_modules": [batch_norm_modules],
        }

        # Create a DataFrame and save to CSV
        results_df = pd.DataFrame(avg_metrics)
        results = pd.concat([results, results_df])
        
        results.to_csv(self.results_output_file, index=False)

        return np.mean(mcc_validation)
    
def load_datasets(ids_for_datasets):
    from deepmol.loaders import SDFLoader

    dataset = SDFLoader("unique_compounds_with_features.sdf", id_field="_ID").create_dataset()

    molecules_dict = dict(zip(list(dataset.ids), list(dataset.mols)))

    from plants_sm.io.pickle import read_pickle

    enzymes_dict = read_pickle("features_proteins_esm2_3b/features.pkl")["proteins"]
    protein_descriptors = read_pickle("propythia_descriptors/features.pkl")["proteins"]
    compounds_features = read_pickle("compounds_features.pkl")

    np_classifier_features = read_pickle("features_compounds_np_classifier_fp/features.pkl")["ligands"]

    import pandas as pd
    dataset = pd.read_csv("integrated_dataset_descriptors_available.csv")

    datasets = []

    for train_ids, val_ids, test_ids in ids_for_datasets:

        train_dataset = dataset[dataset["Enzyme ID"].isin(train_ids)]
        train_dataset = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

        interaction = list(zip(list(train_dataset["Enzyme ID"]), list(train_dataset["Substrate ID"])))
        labels = list(train_dataset["Binding"])

        from copy import copy

        train_dataset = FPESMDataset(copy(np_classifier_features), mol_descriptors=compounds_features, protein_descriptors = protein_descriptors,
                                        proteins_embeddings_dict=enzymes_dict, interactions=interaction, labels=labels)

        import pandas as pd

        validation_dataset = dataset[dataset["Enzyme ID"].isin(val_ids)]
        interaction = list(zip(list(validation_dataset["Enzyme ID"]), list(validation_dataset["Substrate ID"])))
        labels = list(validation_dataset["Binding"])

        validation_dataset = FPESMDataset(copy(np_classifier_features), mol_descriptors=compounds_features, protein_descriptors = protein_descriptors,
                                        proteins_embeddings_dict=enzymes_dict, interactions=interaction, labels=labels, mol_scaler=train_dataset.mol_scaler, protein_scaler=train_dataset.protein_scaler)

        test_dataset = dataset[dataset["Enzyme ID"].isin(test_ids)]
        interaction = list(zip(list(test_dataset["Enzyme ID"]), list(test_dataset["Substrate ID"])))
        labels = list(test_dataset["Binding"])

        test_dataset = FPESMDataset(copy(np_classifier_features), mol_descriptors=compounds_features, protein_descriptors = protein_descriptors,
                                    proteins_embeddings_dict=enzymes_dict, interactions=interaction, labels=labels, mol_scaler=train_dataset.mol_scaler, protein_scaler=train_dataset.protein_scaler)
        
        datasets.append((train_dataset, validation_dataset, test_dataset))
    return datasets
    
def experiment_optimize(ids_for_datasets):
    
    experiment = BindingExperiment(ids_for_datasets=ids_for_datasets, study_name="descriptors_np_classifier", storage="sqlite:///descriptors_np_classifier.db", sampler= optuna.samplers.TPESampler(seed=123),
                                    direction="maximize", load_if_exists=True, results_output_file="descriptors_np_classifier.csv",
                                        folder_path="kroll_experiment_models/trials")
    for trial in experiment.study.trials:
        if trial.state == optuna.trial.TrialState.FAIL: 
            experiment.study.enqueue_trial(trial.params)
    experiment.run(n_trials=100, n_jobs=1)

def test_train_model():

    from plants_sm.io.pickle import read_pickle
    ids_for_datasets = read_pickle("splits/splits_0_6_proteins.pkl")

    from deepmol.loaders import SDFLoader

    dataset = SDFLoader("unique_compounds_with_features.sdf", id_field="_ID").create_dataset()

    molecules_dict = dict(zip(list(dataset.ids), list(dataset.mols)))

    from plants_sm.io.pickle import read_pickle

    enzymes_dict = read_pickle("features_proteins_esm2_3b/features.pkl")["proteins"]
    protein_descriptors = read_pickle("propythia_descriptors/features.pkl")["proteins"]
    compounds_features = read_pickle("compounds_features.pkl")

    np_classifier_features = read_pickle("features_compounds_np_classifier_fp/features.pkl")["ligands"]

    import pandas as pd
    dataset = pd.read_csv("test_integrated_dataset.csv")

    datasets = []

    for train_ids, val_ids, test_ids in ids_for_datasets:

        train_dataset = dataset[dataset["Enzyme ID"].isin(train_ids)]

        interaction = list(zip(list(train_dataset["Enzyme ID"]), list(train_dataset["Substrate ID"])))
        labels = list(train_dataset["Binding"])

        from copy import copy

        train_dataset = FPESMDataset(copy(np_classifier_features), mol_descriptors=compounds_features, protein_descriptors = protein_descriptors,
                                        proteins_embeddings_dict=enzymes_dict, interactions=interaction, labels=labels)

        import pandas as pd

        validation_dataset = dataset[dataset["Enzyme ID"].isin(val_ids)]
        interaction = list(zip(list(validation_dataset["Enzyme ID"]), list(validation_dataset["Substrate ID"])))
        labels = list(validation_dataset["Binding"])

        validation_dataset = FPESMDataset(copy(np_classifier_features), mol_descriptors=compounds_features, protein_descriptors = protein_descriptors,
                                        proteins_embeddings_dict=enzymes_dict, interactions=interaction, labels=labels, mol_scaler=train_dataset.mol_scaler, protein_scaler=train_dataset.protein_scaler)

        test_dataset = dataset[dataset["Enzyme ID"].isin(test_ids)]
        interaction = list(zip(list(test_dataset["Enzyme ID"]), list(test_dataset["Substrate ID"])))
        labels = list(test_dataset["Binding"])

        test_dataset = FPESMDataset(copy(np_classifier_features), mol_descriptors=compounds_features, protein_descriptors = protein_descriptors,
                                    proteins_embeddings_dict=enzymes_dict, interactions=interaction, labels=labels, mol_scaler=train_dataset.mol_scaler, protein_scaler=train_dataset.protein_scaler)
        
        datasets.append((train_dataset, validation_dataset, test_dataset))

    f1_macro_validation = []
    precision_validation = []
    roc_auc_validation = []
    accuracy_validation = []
    mcc_validation = []

    f1_macro_test = []
    precision_test = []
    roc_auc_test = []
    accuracy_test = []
    mcc_test = []

    for train_dataset, validation_dataset, test_dataset in datasets:
        try:
            model = train_model(train_dataset, validation_dataset, [2048, 512], [2048, 512],
                                [2048, 512],[2048, 512], [2048, 512],
                                False, False,
                                128, 0.0010502815562126, 5)
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

        # Evaluate on test set
        predictions_probability, predictions, y_true = test_model(model, test_dataset)
        f1_macro_test.append(f1_score(y_true, predictions))
        precision_test.append(precision_score(y_true, predictions))
        roc_auc_test.append(roc_auc_score(y_true, predictions_probability))
        accuracy_test.append(accuracy_score(y_true, predictions))
        mcc_test.append(matthews_corrcoef(y_true, predictions))
    

if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    from plants_sm.io.pickle import read_pickle
    splits = read_pickle("splits/splits_0_6_proteins.pkl")
    experiment_optimize(splits)
    # test_train_model()
