
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors

from plants_sm.models.lightning_model import InternalLightningModel
from enzyme_substrate_prediction.models.interaction import InteractionModel
from lightning.pytorch.callbacks import EarlyStopping

import pandas as pd

def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    return accuracy, precision, recall, f1, roc_auc

def load_datasets(fold, split):

    multi_input_dataset_csv = f"/home/jcapela/enzyme-substrate-interaction/pipeline_rhea/splits/{split}/train_dataset_{fold}.csv"

    train_dataset = MultiInputDataset.from_csv(file_path=multi_input_dataset_csv,
                                                    representation_field={"proteins": "sequence",
                                                                        "ligands": "SMILES"},
                                                    instances_ids_field={"proteins": "uniprot_id",
                                                                        "ligands": "CHEBI_ID"},
                                                    labels_field="interaction")
    
    validation_dataset_csv = f"/home/jcapela/enzyme-substrate-interaction/pipeline_rhea/splits/{split}/val_dataset_{fold}.csv"
    validation_dataset = MultiInputDataset.from_csv(file_path=validation_dataset_csv,
                                                    representation_field={"proteins": "sequence",
                                                                        "ligands": "SMILES"},
                                                    instances_ids_field={"proteins": "uniprot_id",
                                                                        "ligands": "CHEBI_ID"},
                                                    labels_field="interaction")

    train_dataset.load_features(folder_path="interactions_esm1b_t33_650M_UR50S/")
    validation_dataset.load_features(folder_path="interactions_esm1b_t33_650M_UR50S/")

    featurizer = DeepMolDescriptors(preset="np_classifier_fp", n_jobs=10)
    featurizer.fit_transform(train_dataset, "ligands")
    featurizer.transform(validation_dataset, "ligands")

    test_dataset_csv = f"/home/jcapela/enzyme-substrate-interaction/pipeline_rhea/splits/{split}/test_dataset.csv"
    test_dataset = MultiInputDataset.from_csv(file_path=test_dataset_csv,
                                            representation_field={"proteins": "sequence",
                                                                "ligands": "SMILES"},
                                            instances_ids_field={"proteins": "uniprot_id",
                                                                "ligands": "CHEBI_ID"},
                                            labels_field="interaction")
    
    test_dataset.load_features(folder_path="interactions_esm1b_t33_650M_UR50S/")
    featurizer.transform(test_dataset, "ligands")

    return train_dataset, validation_dataset, test_dataset

def train_model(train_dataset, validation_dataset, transfer_learning=True):
    module = InteractionModel(protein_model_path="esm1b.pt", compounds_model_path="np_classifier.ckpt", transfer_learning=transfer_learning)

    # Create the early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # metric to monitor
        patience=5,          # number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # stop when the metric is minimized
    )

    model = InternalLightningModel(module=module, max_epochs=50,
            batch_size=128,
            devices=[0],
            accelerator="gpu",
            callbacks=[early_stopping_callback]
            )

    model.fit(train_dataset, validation_dataset)
    return model

def make_predictions_and_evaluation(model, train_dataset, validation_dataset, test_dataset, i, results, split, mode="TL"):
    train_predictions = model.predict(train_dataset)
    accuracy, precision, recall, f1, roc_auc = compute_metrics(train_dataset.y, train_predictions)

    results_fold = pd.DataFrame({"model": [split], "mode": [mode], "fold": [i], "split": ["train"], "accuracy": [accuracy], 
                                    "precision": [precision], "recall": [recall], "f1": [f1], "roc_auc": [roc_auc]})


    val_predictions = model.predict(validation_dataset)
    accuracy, precision, recall, f1, roc_auc = compute_metrics(validation_dataset.y, val_predictions)

    results_fold_val = pd.DataFrame({"model": [split], "mode": [mode], "fold": [i], "split": ["validation"], "accuracy": [accuracy],
                                        "precision": [precision], "recall": [recall], "f1": [f1], "roc_auc": [roc_auc]})
    
    results = pd.concat([results, results_fold, results_fold_val])

    test_predictions = model.predict(test_dataset)
    accuracy, precision, recall, f1, roc_auc = compute_metrics(test_dataset.y, test_predictions)

    results_fold_test = pd.DataFrame({"model": [split], "mode": [mode], "fold": [i], "split": ["test"], "accuracy": [accuracy],
                                        "precision": [precision], "recall": [recall], "f1": [f1], "roc_auc": [roc_auc]})
    
    results = pd.concat([results, results_fold_test])
    return results


def train_random():

    results = pd.DataFrame(columns=["model", "mode", "fold", "split", "accuracy", "precision", "recall", "f1", "roc_auc"])

    for i in range(5):

        multi_input_dataset, validation_dataset, test_dataset = load_datasets(i, "random")

        model = train_model(multi_input_dataset, validation_dataset, transfer_learning=True)

        results = make_predictions_and_evaluation(model, multi_input_dataset, validation_dataset, test_dataset, i, results)
        
        results.to_csv("results_random_split.csv", index=False)

    for i in range(5):

        multi_input_dataset, validation_dataset, test_dataset = load_datasets(i, "random")

        model = train_model(multi_input_dataset, validation_dataset, transfer_learning=False)

        results = make_predictions_and_evaluation(model, multi_input_dataset, validation_dataset, test_dataset, i, results, mode="No TL")
        
        results.to_csv("results_random_split.csv", index=False)

def train_0640():
    results = pd.DataFrame(columns=["model", "mode", "fold", "split", "accuracy", "precision", "recall", "f1", "roc_auc"])

    for i in range(5):

        multi_input_dataset, validation_dataset, test_dataset = load_datasets(i, "0.6_40")

        model = train_model(multi_input_dataset, validation_dataset, transfer_learning=True)

        results = make_predictions_and_evaluation(model, multi_input_dataset, validation_dataset, test_dataset, i, results)
        
        results.to_csv("results_0_6_40_split.csv", index=False)

    for i in range(5):

        multi_input_dataset, validation_dataset, test_dataset = load_datasets(i, "0.6_40")

        model = train_model(multi_input_dataset, validation_dataset, transfer_learning=False)

        results = make_predictions_and_evaluation(model, multi_input_dataset, validation_dataset, test_dataset, i, results, mode="No TL")
        
        results.to_csv("results_0_6_40_split.csv", index=False)

def train_models(split):
    
    results = pd.DataFrame(columns=["model", "mode", "fold", "split", "accuracy", "precision", "recall", "f1", "roc_auc"])

    for i in range(5):

        multi_input_dataset, validation_dataset, test_dataset = load_datasets(i, split)

        model = train_model(multi_input_dataset, validation_dataset, transfer_learning=True)

        results = make_predictions_and_evaluation(model, multi_input_dataset, validation_dataset, test_dataset, i, results, split)
        
        results.to_csv(f"results_{split}_split.csv", index=False)

    for i in range(5):

        multi_input_dataset, validation_dataset, test_dataset = load_datasets(i, split)

        model = train_model(multi_input_dataset, validation_dataset, transfer_learning=False)

        results = make_predictions_and_evaluation(model, multi_input_dataset, validation_dataset, test_dataset, i, results, split, mode="No TL")
        
        results.to_csv(f"results_{split}_split.csv", index=False)

    return results


if __name__ == "__main__":
    results = train_models("random_reaction_holdout")
    results = pd.concat((results, train_models("0.6_40")))
    results = pd.concat((results, train_models("random")))
    results = pd.concat((results, train_models("0.8_60")))
    results = pd.concat((results, train_models("1_60")))
    results.to_csv("results_all_esi.csv", index=False)