


import traceback

import esm
from enzyme_substrate_prediction.models.esm_np_classifier import BatchConverterESMNPClassifierFP, ESM_NPClassifierFP, ProteinCompoundDataset
from sklearn.metrics import *

import pytorch_lightning as pl


from torch.utils.data import DataLoader

def train_model(train_dataset, validation_dataset):

    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

    train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=BatchConverterESMNPClassifierFP(alphabet, truncation_seq_length=884))
    val_dataloader = DataLoader(validation_dataset, batch_size=12, shuffle=False, collate_fn=BatchConverterESMNPClassifierFP(alphabet, truncation_seq_length=884))

    # Initialize the model
    model = ESM_NPClassifierFP(learning_rate=5.238402256370824e-05)

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=10, devices=[0, 1, 2], strategy='ddp_find_unused_parameters_true')  # Adjust max_epochs and gpus as needed

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

def test_train_model():

    from plants_sm.io.pickle import read_pickle
    ids_for_datasets = read_pickle("splits/splits_0_6_proteins.pkl")

    np_classifier_features = read_pickle("features_compounds_np_classifier_fp/features.pkl")["ligands"]

    import pandas as pd
    dataset = pd.read_csv("integrated_dataset_descriptors_available.csv")

    datasets = []

    protein_seqs = dict(zip(dataset["Enzyme ID"], dataset["Sequence"]))

    for train_ids, val_ids, test_ids in ids_for_datasets:

        train_dataset = dataset[dataset["Enzyme ID"].isin(train_ids)]

        interaction = list(zip(list(train_dataset["Enzyme ID"]), list(train_dataset["Substrate ID"])))
        labels = list(train_dataset["Binding"])

        from copy import copy

        train_dataset = ProteinCompoundDataset(np_classifier_features, protein_seqs, interaction, labels)

        import pandas as pd

        validation_dataset = dataset[dataset["Enzyme ID"].isin(val_ids)]
        interaction = list(zip(list(validation_dataset["Enzyme ID"]), list(validation_dataset["Substrate ID"])))
        labels = list(validation_dataset["Binding"])

        validation_dataset = ProteinCompoundDataset(np_classifier_features, protein_seqs, interaction, labels)

        test_dataset = dataset[dataset["Enzyme ID"].isin(test_ids)]
        interaction = list(zip(list(test_dataset["Enzyme ID"]), list(test_dataset["Substrate ID"])))
        labels = list(test_dataset["Binding"])

        test_dataset = ProteinCompoundDataset(np_classifier_features, protein_seqs, interaction, labels)

        datasets.append((train_dataset, validation_dataset, test_dataset))

    # f1_macro_validation = []
    # precision_validation = []
    # roc_auc_validation = []
    # accuracy_validation = []
    # mcc_validation = []

    # f1_macro_test = []
    # precision_test = []
    # roc_auc_test = []
    # accuracy_test = []
    # mcc_test = []

    for train_dataset, validation_dataset, test_dataset in datasets:
    #     try:
        model = train_model(train_dataset, validation_dataset)
    #     except Exception as e:
    #         print("An error occurred:")
    #         traceback.print_exc()
    #         continue  # Skip this fold if an error occurs

    #     # Evaluate on validation set
    #     predictions_probability, predictions, y_true = test_model(model, validation_dataset)


    #     f1_macro_validation.append(f1_score(y_true, predictions))
    #     precision_validation.append(precision_score(y_true, predictions))
    #     roc_auc_validation.append(roc_auc_score(y_true, predictions_probability))
    #     accuracy_validation.append(accuracy_score(y_true, predictions))
    #     mcc_validation.append(matthews_corrcoef(y_true, predictions))

    #     # Evaluate on test set
    #     predictions_probability, predictions, y_true = test_model(model, test_dataset)
    #     f1_macro_test.append(f1_score(y_true, predictions))
    #     precision_test.append(precision_score(y_true, predictions))
    #     roc_auc_test.append(roc_auc_score(y_true, predictions_probability))
    #     accuracy_test.append(accuracy_score(y_true, predictions))
    #     mcc_test.append(matthews_corrcoef(y_true, predictions))

test_train_model()