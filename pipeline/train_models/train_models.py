from copy import deepcopy
import os
from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from hyperopt import fmin, tpe, hp, Trials
import xgboost as xgb
from plants_sm.io.pickle import write_pickle, read_pickle
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset

def load_datasets(
    ids_for_datasets: List[Tuple[List[str], List[str], List[str]]],
    dataset: str = "curated_dataset.csv",
    random_state: int = 42,
    merge_validation_set: bool = False,
    enzymes_features: str = "esm2_3b_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp"
) -> List[Tuple[MultiInputDataset, ...]]:
    """
    Load and prepare enzyme and compound datasets based on provided IDs.

    Parameters
    ----------
    ids_for_datasets : List[Tuple[List[str], List[str], List[str]]]
        List of tuples, each containing (train_ids, val_ids, test_ids).
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset.csv".
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    merge_validation_set : bool, optional
        If True, merge validation set into the training set. Default is False.
    enzymes_features : str, optional
        Feature set for enzymes. Default is "esm2_3b_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp".

    Returns
    -------
    List[Tuple[MultiInputDataset, ...]]
        List of tuples, each containing (train_dataset, validation_dataset, test_dataset)
        or (train_dataset, test_dataset) if merge_validation_set is True.
    """
    dataset = pd.read_csv(dataset)
    datasets = []

    for train_ids, val_ids, test_ids in ids_for_datasets:
        train_dataset = dataset[dataset["Enzyme ID"].isin(train_ids)]
        validation_dataset = dataset[dataset["Enzyme ID"].isin(val_ids)]

        if merge_validation_set:
            train_dataset = pd.concat((train_dataset, validation_dataset), axis=0)
        else:
            validation_dataset = MultiInputDataset(
                dataframe=validation_dataset,
                representation_field={"proteins": "Sequence", "ligands": "SMILES"},
                instances_ids_field={"proteins": "Enzyme ID", "ligands": "Substrate ID"},
                labels_field="Binding"
            )
            validation_dataset.load_features(enzymes_features, "proteins")
            validation_dataset.load_features(compounds_features, "ligands")

        train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        train_dataset = MultiInputDataset(
            dataframe=train_dataset,
            representation_field={"proteins": "Sequence", "ligands": "SMILES"},
            instances_ids_field={"proteins": "Enzyme ID", "ligands": "Substrate ID"},
            labels_field="Binding"
        )
        train_dataset.load_features(enzymes_features, "proteins")
        train_dataset.load_features(compounds_features, "ligands")

        test_dataset = dataset[dataset["Enzyme ID"].isin(test_ids)]
        test_dataset = MultiInputDataset(
            dataframe=test_dataset,
            representation_field={"proteins": "Sequence", "ligands": "SMILES"},
            instances_ids_field={"proteins": "Enzyme ID", "ligands": "Substrate ID"},
            labels_field="Binding"
        )
        test_dataset.load_features(enzymes_features, "proteins")
        test_dataset.load_features(compounds_features, "ligands")

        if merge_validation_set:
            datasets.append((train_dataset, test_dataset))
        else:
            datasets.append((train_dataset, validation_dataset, test_dataset))
    return datasets

def load_datasets_compounds(
    ids_for_datasets: List[Tuple[List[str], List[str], List[str]]],
    dataset: str = "curated_dataset.csv",
    random_state: int = 42,
    merge_validation_set: bool = False,
    enzymes_features: str = "esm2_3b_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp"
) -> List[Tuple[MultiInputDataset, ...]]:
    """
    Load and prepare compound datasets based on provided IDs.

    Parameters
    ----------
    ids_for_datasets : List[Tuple[List[str], List[str], List[str]]]
        List of tuples, each containing (train_ids, val_ids, test_ids).
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset.csv".
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    merge_validation_set : bool, optional
        If True, merge validation set into the training set. Default is False.
    enzymes_features : str, optional
        Feature set for enzymes. Default is "esm2_3b_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp".

    Returns
    -------
    List[Tuple[MultiInputDataset, ...]]
        List of tuples, each containing (train_dataset, validation_dataset, test_dataset)
        or (train_dataset, test_dataset) if merge_validation_set is True.
    """
    dataset = pd.read_csv(dataset)
    datasets = []

    for train_ids, val_ids, test_ids in ids_for_datasets:
        train_dataset = dataset[dataset["Substrate ID"].isin(train_ids)]
        validation_dataset = dataset[dataset["Substrate ID"].isin(val_ids)]

        if merge_validation_set:
            train_dataset = pd.concat((train_dataset, validation_dataset), axis=0)
        else:
            validation_dataset = MultiInputDataset(
                dataframe=validation_dataset,
                representation_field={"proteins": "Sequence", "ligands": "SMILES"},
                instances_ids_field={"proteins": "Enzyme ID", "ligands": "Substrate ID"},
                labels_field="Binding"
            )
            validation_dataset.load_features(enzymes_features, "proteins")
            validation_dataset.load_features(compounds_features, "ligands")

        train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        train_dataset = MultiInputDataset(
            dataframe=train_dataset,
            representation_field={"proteins": "Sequence", "ligands": "SMILES"},
            instances_ids_field={"proteins": "Enzyme ID", "ligands": "Substrate ID"},
            labels_field="Binding"
        )
        train_dataset.load_features(enzymes_features, "proteins")
        train_dataset.load_features(compounds_features, "ligands")

        test_dataset = dataset[dataset["Substrate ID"].isin(test_ids)]
        test_dataset = MultiInputDataset(
            dataframe=test_dataset,
            representation_field={"proteins": "Sequence", "ligands": "SMILES"},
            instances_ids_field={"proteins": "Enzyme ID", "ligands": "Substrate ID"},
            labels_field="Binding"
        )
        test_dataset.load_features(enzymes_features, "proteins")
        test_dataset.load_features(compounds_features, "ligands")

        if merge_validation_set:
            datasets.append((train_dataset, test_dataset))
        else:
            datasets.append((train_dataset, validation_dataset, test_dataset))
    return datasets

depth_array = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
max_bin = [64, 128, 256, 512, 1024]

space_gradient_boosting = {
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
    "max_depth": hp.choice("max_depth", depth_array),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds": hp.uniform("num_rounds", 30, 1000),
    "weight": hp.uniform("weight", 0.01, 0.99),
    "max_bin": hp.choice("max_bin", max_bin)
}

def set_param_values_V2(
    param: Dict[str, Any],
    dtrain: xgb.DMatrix
) -> Tuple[Dict[str, Any], int, xgb.DMatrix]:
    """
    Set parameter values for XGBoost model.

    Parameters
    ----------
    param : Dict[str, Any]
        Dictionary of hyperparameters.
    dtrain : xgb.DMatrix
        XGBoost DMatrix for training.

    Returns
    -------
    Tuple[Dict[str, Any], int, xgb.DMatrix]
        Tuple of (updated parameters, number of rounds, updated DMatrix).
    """
    num_round = int(param["num_rounds"])
    param["tree_method"] = "hist"
    param["max_depth"] = int(depth_array[param["max_depth"]])
    param["max_bin"] = int(max_bin[param["max_bin"]])
    param["sampling_method"] = "gradient_based"
    param["device"] = "cuda:3"
    param['objective'] = 'binary:logistic'
    weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
    dtrain.set_weight(weights)
    del param["num_rounds"]
    del param["weight"]
    return param, num_round, dtrain

def get_performance(
    pred: np.ndarray,
    true: np.ndarray
) -> float:
    """
    Calculate Matthews Correlation Coefficient (MCC) for model performance.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values.
    true : np.ndarray
        True values.

    Returns
    -------
    float
        Negative MCC (for minimization in hyperopt).
    """
    MCC = matthews_corrcoef(true, np.round(pred))
    return -MCC

def train_with_batches(
    param: Dict[str, Any],
    num_round: int,
    dtrain: xgb.DMatrix,
    num_batches: int = 1
) -> Optional[xgb.Booster]:
    """
    Train XGBoost model in batches.

    Parameters
    ----------
    param : Dict[str, Any]
        Dictionary of hyperparameters.
    num_round : int
        Number of boosting rounds.
    dtrain : xgb.DMatrix
        XGBoost DMatrix for training.
    num_batches : int, optional
        Number of batches to split the data into. Default is 1.

    Returns
    -------
    Optional[xgb.Booster]
        Trained XGBoost model or None if training fails.
    """
    np.random.seed(42)
    try:
        total_size = len(dtrain.get_label())
        batch_size = total_size // num_batches
        rounds_per_batch = num_round // num_batches
        indices = np.random.permutation(total_size)
        bst = None
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size if i < num_batches - 1 else total_size
            batch_idx = indices[start:end]
            dtrain_batch = xgb.DMatrix(data=dtrain.get_data()[batch_idx], label=dtrain.get_label()[batch_idx])
            if bst is None:
                bst = xgb.train(param, dtrain_batch, num_boost_round=rounds_per_batch)
            else:
                bst = xgb.train(param, dtrain_batch, num_boost_round=rounds_per_batch, xgb_model=bst)
        param["num_rounds"] = num_round
        return bst
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def _predict_with_batches(
    dval: xgb.DMatrix,
    model: xgb.Booster,
    num_batches: int = 1
) -> Optional[np.ndarray]:
    """
    Predict using XGBoost model in batches.

    Parameters
    ----------
    dval : xgb.DMatrix
        XGBoost DMatrix for validation.
    model : xgb.Booster
        Trained XGBoost model.
    num_batches : int, optional
        Number of batches to split the data into. Default is 1.

    Returns
    -------
    Optional[np.ndarray]
        Predictions or None if prediction fails.
    """
    np.random.seed(42)
    try:
        total_size = len(dval.get_label())
        batch_size = total_size // num_batches
        indices = np.arange(total_size)
        predictions_all = np.array([])
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size if i < num_batches - 1 else total_size
            batch_idx = indices[start:end]
            dval_batch = xgb.DMatrix(data=dval.get_data()[batch_idx], label=dval.get_label()[batch_idx])
            predictions = model.predict(dval_batch)
            predictions_all = np.concatenate((predictions_all, predictions), axis=0)
        return predictions_all
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def predict_with_batches(
    dval: xgb.DMatrix,
    model: xgb.Booster
) -> Optional[np.ndarray]:
    """
    Predict using XGBoost model, with fallback to batch prediction if needed.

    Parameters
    ----------
    dval : xgb.DMatrix
        XGBoost DMatrix for validation.
    model : xgb.Booster
        Trained XGBoost model.

    Returns
    -------
    Optional[np.ndarray]
        Predictions or None if prediction fails.
    """
    try:
        y_val_pred = model.predict(dval)
        return y_val_pred
    except Exception as e:
        batches = 2
        print(f"Single batch prediction failed. Trying with {batches}")
        predictions = _predict_with_batches(dval, model)
        while predictions is None and batches < 30:
            batches += 1
            print(f"Trying with {batches}")
            predictions = _predict_with_batches(dval, model, num_batches=batches)
        if predictions is None:
            print("Failed to predict even with batching.")
            return None
        else:
            print(f"Succeeded with {batches} batches.")
            return predictions

def fit_xgboost(
    param: Dict[str, Any],
    dtrain: xgb.DMatrix,
    num_round: int
) -> Optional[xgb.Booster]:
    """
    Fit XGBoost model, with fallback to batch training if needed.

    Parameters
    ----------
    param : Dict[str, Any]
        Dictionary of hyperparameters.
    dtrain : xgb.DMatrix
        XGBoost DMatrix for training.
    num_round : int
        Number of boosting rounds.

    Returns
    -------
    Optional[xgb.Booster]
        Trained XGBoost model or None if training fails.
    """
    try:
        bst = xgb.train(param, dtrain, num_round)
        param["num_rounds"] = num_round
        return bst
    except Exception as e:
        batches = 2
        print(f"Single batch training failed. Trying with {batches}")
        bst = train_with_batches(param, num_round, dtrain, num_batches=batches)
        while bst is None and batches < 30:
            batches += 1
            print(f"Trying with {batches}")
            bst = train_with_batches(param, num_round, dtrain, num_batches=batches)
        if bst is None:
            print("Failed to train model even with batching.")
            return None
        else:
            print(f"Succeeded with {batches} batches.")
            return bst

def train_gb(
    train_dataset: MultiInputDataset,
    validation_dataset: MultiInputDataset,
    test_dataset: MultiInputDataset,
    save_pred_path: str,
    model_name: str,
    similarity: float,
    max_evals: int = 500
) -> Dict[str, Any]:
    """
    Train and optimize XGBoost model using hyperopt.

    Parameters
    ----------
    train_dataset : MultiInputDataset
        Training dataset.
    validation_dataset : MultiInputDataset
        Validation dataset.
    test_dataset : MultiInputDataset
        Test dataset.
    save_pred_path : str
        Path to save the best model.
    model_name : str
        Name of the model.
    similarity : float
        Similarity threshold.
    max_evals : int, optional
        Maximum number of hyperopt evaluations. Default is 500.

    Returns
    -------
    Dict[str, Any]
        Best hyperparameters found.
    """
    def set_param_values(param: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        num_round = int(param["num_rounds"])
        param["tree_method"] = "hist"
        param["sampling_method"] = "gradient_based"
        param["device"] = "cuda:3"
        param['objective'] = 'binary:logistic'
        weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
        dtrain.set_weight(weights)
        del param["num_rounds"]
        del param["weight"]
        return param, num_round

    def get_predictions(param: Dict[str, Any], dM_train: xgb.DMatrix, dM_val: xgb.DMatrix) -> np.ndarray:
        param, num_round, dM_train = set_param_values_V2(param=param, dtrain=dM_train)
        bst = fit_xgboost(param, dM_train, num_round)
        predictions = predict_with_batches(dM_val, bst)
        return predictions

    def get_performance_metrics(pred: np.ndarray, true: np.ndarray) -> None:
        acc = np.mean(np.round(pred) == np.array(true))
        roc_auc = roc_auc_score(np.array(true), pred)
        mcc = matthews_corrcoef(np.array(true), np.round(pred))
        print(f"accuracy: {acc}, ROC AUC: {roc_auc}, MCC: {mcc}")

    train_X_all = np.concatenate([train_dataset.X["proteins"], train_dataset.X["ligands"]], axis=1)
    test_X_all = np.concatenate([test_dataset.X["proteins"], test_dataset.X["ligands"]], axis=1)
    val_X_all = np.concatenate([validation_dataset.X["proteins"], validation_dataset.X["ligands"]], axis=1)

    dtrain = xgb.DMatrix(np.array(train_X_all), label=np.array(train_dataset.y).astype(float))
    dtest = xgb.DMatrix(np.array(test_X_all), label=np.array(test_dataset.y).astype(float))
    dvalid = xgb.DMatrix(np.array(val_X_all), label=np.array(validation_dataset.y).astype(float))
    dtrain_val = xgb.DMatrix(
        np.concatenate([np.array(train_X_all), np.array(val_X_all)], axis=0),
        label=np.concatenate([np.array(train_dataset.y).astype(float), np.array(validation_dataset.y).astype(float)], axis=0)
    )

    def train_xgboost_model_all(param: Dict[str, Any]) -> float:
        param, num_round = set_param_values(param)
        bst = fit_xgboost(param, dtrain, num_round)
        if bst is None:
            print("Failed to train model even with batching.")
            return 0
        else:
            return get_performance(pred=bst.predict(dvalid), true=validation_dataset.y)

    trials = Trials()
    best = fmin(
        fn=train_xgboost_model_all,
        space=space_gradient_boosting,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    best_copy = best.copy()

    y_val_pred_all = get_predictions(param=best_copy, dM_train=dtrain, dM_val=dvalid)
    get_performance_metrics(pred=y_val_pred_all, true=validation_dataset.y)

    best_copy = best.copy()
    y_test_pred_all = get_predictions(param=best_copy, dM_train=dtrain_val, dM_val=dtest)
    get_performance_metrics(pred=y_test_pred_all, true=test_dataset.y)

    os.makedirs(save_pred_path, exist_ok=True)
    write_pickle(os.path.join(save_pred_path, f"{model_name}_{similarity}_best.pkl"), best)
    return best

def evaluate_model(
    model_name: str,
    save_pred_path: str,
    train_dataset: MultiInputDataset,
    validation_dataset: MultiInputDataset,
    test_dataset: MultiInputDataset,
    similarity: float,
    proteins: bool = True,
    suffix_results: str = "",
    results_name: Optional[str] = None
) -> None:
    """
    Evaluate model performance on test dataset.

    Parameters
    ----------
    model_name : str
        Name of the model.
    save_pred_path : str
        Path to saved model.
    train_dataset : MultiInputDataset
        Training dataset.
    validation_dataset : MultiInputDataset
        Validation dataset.
    test_dataset : MultiInputDataset
        Test dataset.
    similarity : float
        Similarity threshold.
    proteins : bool, optional
        If True, use protein similarity. Default is True.
    suffix_results : str, optional
        Suffix for results file. Default is "".
    results_name : Optional[str], optional
        Name of results file. Default is None.
    """
    if results_name is None:
        if suffix_results == "":
            results_name = os.path.join(save_pred_path, f"results_{model_name}.csv")
        else:
            results_name = os.path.join(save_pred_path, f"results_{model_name}_{suffix_results}.csv")
    print("It will save the results in", results_name)
    if test_dataset.X["proteins"].shape[0] != 0:
        for seed in range(5):
            print("Seed:", seed)
            np.random.seed(seed)
            best = read_pickle(os.path.join(save_pred_path, f"{model_name}_{similarity}_best.pkl"))
            train_X_all = np.concatenate([train_dataset.X["proteins"], train_dataset.X["ligands"]], axis=1)
            test_X_all = np.concatenate([test_dataset.X["proteins"], test_dataset.X["ligands"]], axis=1)
            val_X_all = np.concatenate([validation_dataset.X["proteins"], validation_dataset.X["ligands"]], axis=1)

            train_X_all = np.concatenate([np.array(train_X_all), np.array(val_X_all)], axis=0)
            train_y_all = np.concatenate([np.array(train_dataset.y).astype(float), np.array(validation_dataset.y).astype(float)], axis=0)

            permutation = np.random.permutation(len(train_X_all))
            train_X_all = train_X_all[permutation]
            train_y_all = train_y_all[permutation]

            dtrain = xgb.DMatrix(train_X_all, label=train_y_all)
            dtest = xgb.DMatrix(np.array(test_X_all), label=np.array(test_dataset.y).astype(float))

            best["seed"] = seed
            best, num_round, dtrain = set_param_values_V2(best, dtrain)
            bst = fit_xgboost(best, dtrain, num_round)

            predictions_proba = bst.predict(dtest)
            predictions = np.round(predictions_proba)

            accuracy = accuracy_score(test_dataset.y, predictions)
            f1 = f1_score(test_dataset.y, predictions)
            recall = recall_score(test_dataset.y, predictions)
            precision = precision_score(test_dataset.y, predictions)
            roc_auc = roc_auc_score(test_dataset.y, predictions_proba)
            mcc = matthews_corrcoef(test_dataset.y, predictions)

            similarity_name = "identity" if proteins else "similarity"
            results = {
                'accuracy': accuracy,
                'f1_score': f1,
                'recall': recall,
                'precision': precision,
                'roc_auc': roc_auc,
                'mcc': mcc,
                'seed': seed,
                similarity_name: similarity,
            }
            df = pd.DataFrame([results])
            file_exists = os.path.exists(results_name)
            df.to_csv(results_name, mode='a', index=False, header=not file_exists)

def experiment_optimize(
    splits: List[Tuple[List[str], List[str], List[str]]],
    name: str,
    proteins_split: bool,
    similarity: float,
    dataset: str = "curated_dataset.csv",
    enzymes_features: str = "esm2_3b_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp",
    model_name: str = "xgb_np_esm2"
) -> None:
    """
    Run optimization experiment for a given split.

    Parameters
    ----------
    splits : List[Tuple[List[str], List[str], List[str]]]
        List of tuples, each containing (train_ids, val_ids, test_ids).
    name : str
        Name of the experiment.
    proteins_split : bool
        If True, split by proteins.
    similarity : float
        Similarity threshold.
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset.csv".
    enzymes_features : str, optional
        Feature set for enzymes. Default is "esm2_3b_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp".
    model_name : str, optional
        Name of the model. Default is "xgb_np_esm2".
    """
    if proteins_split:
        datasets = load_datasets(splits, dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features)
    else:
        datasets = load_datasets_compounds(splits, dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features)

    train_dataset, validation_dataset, test_dataset = datasets[0]
    train_gb(train_dataset, validation_dataset, test_dataset, model_name, model_name=name, similarity=similarity, max_evals=200)
    evaluate_model(name, model_name, train_dataset, validation_dataset, test_dataset, similarity=similarity, proteins=proteins_split)

def experiment_features(
    model_name: str,
    dataset: str = "curated_dataset.csv",
    enzymes_features: str = "esm2_3b_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp"
) -> None:
    """
    Run experiments for different feature sets and splits.

    Parameters
    ----------
    model_name : str
        Name of the model.
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset.csv".
    enzymes_features : str, optional
        Feature set for enzymes. Default is "esm2_3b_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp".
    """
    splits = read_pickle("../compounds_split/splits_compounds_08.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_compounds", proteins_split=False, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=80, dataset=dataset)

    splits = read_pickle("../compounds_split/splits_compounds_06.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_compounds", proteins_split=False, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=60, dataset=dataset)

    splits = read_pickle("../compounds_split/splits_compounds_04_corrected.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_compounds", proteins_split=False, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=40, dataset=dataset)

    splits = read_pickle("../compounds_split/splits_compounds_03.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_compounds", proteins_split=False, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=30, dataset=dataset)

    splits = read_pickle("../compounds_split/splits_compounds_02.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_compounds", proteins_split=False, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=20, dataset=dataset)

    splits = read_pickle("../protein_splits/splits_0_6_proteins_train_val_test.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_proteins", proteins_split=True, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=60, dataset=dataset)

    splits = read_pickle("../protein_splits/splits_0_8_proteins_train_val_test.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_proteins", proteins_split=True, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=80, dataset=dataset)

    splits = read_pickle("../protein_splits/splits_0_4_proteins_train_val_test.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_proteins", proteins_split=True, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=40, dataset=dataset)

    splits = read_pickle("../protein_splits/splits_0_2_proteins_train_val_test.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_proteins", proteins_split=True, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=20, dataset=dataset)

def experiment_np_esm2(
    dataset: str = "curated_dataset.csv",
    enzymes_features: str = "esm2_3b_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp"
) -> None:
    """
    Run experiments using ESM2 features.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset.csv".
    enzymes_features : str, optional
        Feature set for enzymes. Default is "esm2_3b_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp".
    """
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_esm2")

def experiment_np_esm2_augmented(
    dataset: str = "curated_dataset.csv",
    enzymes_features: str = "features_proteins_esm2_3b_ec_number_augmented_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp_augmented"
) -> None:
    """
    Run experiments using augmented ESM2 features.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset.csv".
    enzymes_features : str, optional
        Feature set for enzymes. Default is "features_proteins_esm2_3b_ec_number_augmented_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp_augmented".
    """
    experiment_features(enzymes_features=enzymes_features, dataset=dataset, compounds_features=compounds_features, model_name="xgb_np_esm2_augmented")

def experiment_prot_bert_np(
    dataset: str = "curated_dataset.csv",
    enzymes_features: str = "prot_bert_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp"
) -> None:
    """
    Run experiments using ProtBERT features.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset.csv".
    enzymes_features : str, optional
        Feature set for enzymes. Default is "prot_bert_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp".
    """
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_prot_bert")

def experiment_esm1b(
    dataset: str = "curated_dataset.csv",
    enzymes_features: str = "esm1b_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp"
) -> None:
    """
    Run experiments using ESM1b features.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset.csv".
    enzymes_features : str, optional
        Feature set for enzymes. Default is "esm1b_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp".
    """
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_esm1b")

def experiment_np_esm2_np_chirality(
    dataset: str = "curated_dataset.csv",
    enzymes_features: str = "esm2_3b_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp_chirality"
) -> None:
    """
    Run experiments using ESM2 features with chirality.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset.csv".
    enzymes_features : str, optional
        Feature set for enzymes. Default is "esm2_3b_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp_chirality".
    """
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_esm2_chirality")

def experiment_prot_bert_np_chirality(
    dataset: str = "curated_dataset.csv",
    enzymes_features: str = "prot_bert_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp_chirality"
) -> None:
    """
    Run experiments using ProtBERT features with chirality.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset.csv".
    enzymes_features : str, optional
        Feature set for enzymes. Default is "prot_bert_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp_chirality".
    """
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_prot_bert_chirality")

def experiment_esm1b_np_chirality(
    dataset: str = "curated_dataset.csv",
    enzymes_features: str = "esm1b_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp_chirality"
) -> None:
    """
    Run experiments using ESM1b features with chirality.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset.csv".
    enzymes_features : str, optional
        Feature set for enzymes. Default is "esm1b_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp_chirality".
    """
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_esm1b_chirality")

def train_model_and_save(
    model_params_path: str,
    save_model_path: str,
    enzymes_features: str = "prot_bert_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp"
) -> None:
    """
    Train a model and save it to disk.

    Parameters
    ----------
    model_params_path : str
        Path to the best model parameters.
    save_model_path : str
        Path to save the trained model.
    enzymes_features : str, optional
        Feature set for enzymes. Default is "prot_bert_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp".
    """
    splits = read_pickle("../protein_splits/splits_0_8_proteins_train_val_test.pkl")
    datasets = load_datasets(splits, enzymes_features=enzymes_features, compounds_features=compounds_features)
    train_dataset, validation_dataset, test_dataset = datasets[0]
    best = read_pickle(model_params_path)
    train_X_all = np.concatenate([train_dataset.X["proteins"], train_dataset.X["ligands"]], axis=1)
    test_X_all = np.concatenate([test_dataset.X["proteins"], test_dataset.X["ligands"]], axis=1)
    val_X_all = np.concatenate([validation_dataset.X["proteins"], validation_dataset.X["ligands"]], axis=1)

    dall = xgb.DMatrix(
        np.concatenate([np.array(train_X_all), np.array(val_X_all), np.array(test_X_all)], axis=0),
        label=np.concatenate([np.array(train_dataset.y).astype(float), np.array(validation_dataset.y).astype(float), np.array(test_dataset.y).astype(float)], axis=0)
    )

    param, num_round, _ = set_param_values_V2(best, dall)
    bst = fit_xgboost(param, dall, num_round)
    write_pickle(save_model_path, bst)

def load_datasets_compound_classes(
    ids_for_datasets: List[Tuple[List[str], List[str], List[str]]],
    compounds_split_datasets: Dict[str, List[str]],
    dataset: str = "curated_dataset_no_stereochemistry_duplicates.csv",
    random_state: int = 42,
    merge_validation_set: bool = False,
    enzymes_features: str = "esm2_3b_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp"
) -> Tuple[MultiInputDataset, MultiInputDataset, Dict[str, MultiInputDataset]]:
    """
    Load datasets for compound classes.

    Parameters
    ----------
    ids_for_datasets : List[Tuple[List[str], List[str], List[str]]]
        List of tuples, each containing (train_ids, val_ids, test_ids).
    compounds_split_datasets : Dict[str, List[str]]
        Dictionary mapping class names to lists of compound IDs.
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset_no_stereochemistry_duplicates.csv".
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    merge_validation_set : bool, optional
        If True, merge validation set into the training set. Default is False.
    enzymes_features : str, optional
        Feature set for enzymes. Default is "esm2_3b_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp".

    Returns
    -------
    Tuple[MultiInputDataset, MultiInputDataset, Dict[str, MultiInputDataset]]
        Tuple of (train_dataset, validation_dataset, test_datasets).
    """
    dataset = pd.read_csv(dataset)
    datasets = {}

    for train_ids, val_ids, test_ids in ids_for_datasets:
        train_dataset = dataset[dataset["Substrate ID"].isin(train_ids)]
        validation_dataset = dataset[dataset["Substrate ID"].isin(val_ids)]

        if merge_validation_set:
            train_dataset = pd.concat((train_dataset, validation_dataset), axis=0)
        else:
            validation_dataset = MultiInputDataset(
                dataframe=validation_dataset,
                representation_field={"proteins": "Sequence", "ligands": "SMILES"},
                instances_ids_field={"proteins": "Enzyme ID", "ligands": "Substrate ID"},
                labels_field="Binding"
            )
            validation_dataset.load_features(enzymes_features, "proteins")
            validation_dataset.load_features(compounds_features, "ligands")

        train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        train_dataset = MultiInputDataset(
            dataframe=train_dataset,
            representation_field={"proteins": "Sequence", "ligands": "SMILES"},
            instances_ids_field={"proteins": "Enzyme ID", "ligands": "Substrate ID"},
            labels_field="Binding"
        )
        train_dataset.load_features(enzymes_features, "proteins")
        train_dataset.load_features(compounds_features, "ligands")

        test_dataset_dataframe = dataset[dataset["Substrate ID"].isin(test_ids)]
        for class_ in compounds_split_datasets:
            test_dataset_class = test_dataset_dataframe[test_dataset_dataframe["Substrate ID"].isin(compounds_split_datasets[class_])]
            test_dataset = MultiInputDataset(
                dataframe=test_dataset_class,
                representation_field={"proteins": "Sequence", "ligands": "SMILES"},
                instances_ids_field={"proteins": "Enzyme ID", "ligands": "Substrate ID"},
                labels_field="Binding"
            )
            test_dataset.load_features(enzymes_features, "proteins")
            test_dataset.load_features(compounds_features, "ligands")
            test_dataset_copy = deepcopy(test_dataset)
            datasets[class_] = test_dataset_copy

    return train_dataset, validation_dataset, datasets

def train_model_and_evaluate_for_classes(
    split_path: str,
    general_split: str,
    model_name: str,
    save_pred_path: str,
    similarity: float,
    dataset: str = "curated_dataset_no_stereochemistry_duplicates.csv"
) -> None:
    """
    Train a model and evaluate it for different compound classes.

    Parameters
    ----------
    split_path : str
        Path to the split file.
    general_split : str
        Path to the general split file.
    model_name : str
        Name of the model.
    save_pred_path : str
        Path to save predictions.
    similarity : float
        Similarity threshold.
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset_no_stereochemistry_duplicates.csv".
    """
    split = read_pickle(split_path)
    general_split = read_pickle(general_split)
    train_dataset, validation_dataset, test_datasets = load_datasets_compound_classes(general_split, split, dataset=dataset)

    for class_ in test_datasets:
        evaluate_model(model_name, save_pred_path, train_dataset, validation_dataset, test_datasets[class_], similarity, proteins=False, suffix_results=class_)

def train_protbert_and_evaluate_for_classes(
    dataset: str = "curated_dataset_no_stereochemistry_duplicates.csv",
    save_pred_path: str = "xgb_np_prot_bert_no_stereo"
) -> None:
    """
    Train ProtBERT model and evaluate for different compound classes.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset_no_stereochemistry_duplicates.csv".
    save_pred_path : str, optional
        Path to save predictions. Default is "xgb_np_prot_bert_no_stereo".
    """
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_02.pkl", split_path="../protein_splits/pathway_to_compounds_split_02.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=20)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_04_corrected.pkl", split_path="../protein_splits/pathway_to_compounds_split_04.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=40)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_06.pkl", split_path="../protein_splits/pathway_to_compounds_split_06.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=60)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_08.pkl", split_path="../protein_splits/pathway_to_compounds_split_08.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=80)

def train_esm1b_and_evaluate_for_classes(
    dataset: str = "curated_dataset_no_stereochemistry_duplicates.csv",
    save_pred_path: str = "xgb_np_esm1b_no_stereo"
) -> None:
    """
    Train ESM1b model and evaluate for different compound classes.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset_no_stereochemistry_duplicates.csv".
    save_pred_path : str, optional
        Path to save predictions. Default is "xgb_np_esm1b_no_stereo".
    """
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_02.pkl", split_path="../protein_splits/pathway_to_compounds_split_02.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=20)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_04_corrected.pkl", split_path="../protein_splits/pathway_to_compounds_split_04.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=40)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_06.pkl", split_path="../protein_splits/pathway_to_compounds_split_06.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=60)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_08.pkl", split_path="../protein_splits/pathway_to_compounds_split_08.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=80)

def train_esm2_3b_and_evaluate_for_classes(
    dataset: str = "curated_dataset_no_stereochemistry_duplicates.csv",
    save_pred_path: str = "xgb_np_esm2_no_stereo"
) -> None:
    """
    Train ESM2 model and evaluate for different compound classes.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset CSV file. Default is "curated_dataset_no_stereochemistry_duplicates.csv".
    save_pred_path : str, optional
        Path to save predictions. Default is "xgb_np_esm2_no_stereo".
    """
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_02.pkl", split_path="../protein_splits/pathway_to_compounds_split_02.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=20)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_04_corrected.pkl", split_path="../protein_splits/pathway_to_compounds_split_04.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=40)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_06.pkl", split_path="../protein_splits/pathway_to_compounds_split_06.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=60)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="../compounds_split/splits_compounds_08.pkl", split_path="../protein_splits/pathway_to_compounds_split_08.pkl", save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=80)

def train_and_evaluate(
    dataset: str,
    splits: List[Tuple[List[str], List[str], List[str]]],
    proteins_split: bool,
    similarity: float,
    name: str,
    results_name: str,
    enzymes_features: str = "esm2_3b_ec_number_embedding",
    compounds_features: str = "features_compounds_np_classifier_fp",
    save_pred_path: str = "xgb_np_esm2_no_stereo"
) -> None:
    """
    Train and evaluate a model for a given split.

    Parameters
    ----------
    dataset : str
        Path to the dataset CSV file.
    splits : List[Tuple[List[str], List[str], List[str]]]
        List of tuples, each containing (train_ids, val_ids, test_ids).
    proteins_split : bool
        If True, split by proteins.
    similarity : float
        Similarity threshold.
    name : str
        Name of the model.
    results_name : str
        Name of the results file.
    enzymes_features : str, optional
        Feature set for enzymes. Default is "esm2_3b_ec_number_embedding".
    compounds_features : str, optional
        Feature set for compounds. Default is "features_compounds_np_classifier_fp".
    save_pred_path : str, optional
        Path to save predictions. Default is "xgb_np_esm2_no_stereo".
    """
    if proteins_split:
        datasets = load_datasets(splits, dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features)
    else:
        datasets = load_datasets_compounds(splits, dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features)

    train_dataset, validation_dataset, test_dataset = datasets[0]
    suffix_results = "proteins" if proteins_split else "compounds"
    evaluate_model(name, save_pred_path, train_dataset, validation_dataset, test_dataset, similarity=similarity, proteins=proteins_split, suffix_results=suffix_results, results_name=results_name)

def train_esm2_and_evaluate() -> None:
    """Train and evaluate ESM2 model."""
    dataset = "curated_dataset.csv"
    splits = read_pickle("../compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds", enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp", save_pred_path="xgb_np_esm2_stereo", results_name="results_xgb_np_esm2_compounds.csv")

    dataset = "curated_dataset_no_stereochemistry_duplicates.csv"
    splits = read_pickle("../compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds", enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp", save_pred_path="xgb_np_esm2_no_stereo", results_name="results_xgb_np_esm2_compounds.csv")

def train_esm1b_and_evaluate() -> None:
    """Train and evaluate ESM1b model."""
    dataset = "curated_dataset.csv"
    splits = read_pickle("../compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds", enzymes_features="esm1b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp", save_pred_path="xgb_np_esm1b_stereo", results_name="results_xgb_np_esm1b_compounds.csv")

    dataset = "curated_dataset_no_stereochemistry_duplicates.csv"
    splits = read_pickle("../compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds", enzymes_features="esm1b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp", save_pred_path="xgb_np_esm1b_no_stereo", results_name="results_xgb_np_esm1b_compounds.csv")

def train_protbert_and_evaluate() -> None:
    """Train and evaluate ProtBERT model."""
    dataset = "curated_dataset.csv"
    splits = read_pickle("../compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds", enzymes_features="prot_bert_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp", save_pred_path="xgb_np_prot_bert_stereo", results_name="results_xgb_np_prot_bert_compounds.csv")

    dataset = "curated_dataset_no_stereochemistry_duplicates.csv"
    splits = read_pickle("../compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds", enzymes_features="prot_bert_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp", save_pred_path="xgb_np_prot_bert_no_stereo", results_name="results_xgb_np_prot_bert_compounds.csv")

if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    experiment_np_esm2()
    experiment_esm1b()
    experiment_prot_bert_np()