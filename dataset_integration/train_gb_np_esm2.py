from copy import deepcopy
import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from hyperopt import fmin, tpe, hp, Trials, rand
import xgboost as xgb

from plants_sm.io.pickle import write_pickle, read_pickle


def load_datasets(ids_for_datasets, dataset="curated_dataset.csv", random_state=42, merge_validation_set=False, enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp"):
    import pandas as pd
    dataset = pd.read_csv(dataset)

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
            
            validation_dataset.load_features(enzymes_features, "proteins")
            validation_dataset.load_features(compounds_features, "ligands")

        train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_dataset = MultiInputDataset(dataframe=train_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")

        train_dataset.load_features(enzymes_features, "proteins")
        train_dataset.load_features(compounds_features, "ligands")

        test_dataset = dataset[dataset["Enzyme ID"].isin(test_ids)]
        
        test_dataset = MultiInputDataset(dataframe=test_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")
        test_dataset.load_features(enzymes_features, "proteins")
        test_dataset.load_features(compounds_features, "ligands")

        if merge_validation_set:
            datasets.append((train_dataset, test_dataset))
        else:
            datasets.append((train_dataset, validation_dataset, test_dataset))
    return datasets

def load_datasets_compounds(ids_for_datasets, dataset="curated_dataset.csv", random_state=42, merge_validation_set=False, enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp"):
    import pandas as pd
    dataset = pd.read_csv(dataset)

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
            
            validation_dataset.load_features(enzymes_features, "proteins")
            validation_dataset.load_features(compounds_features, "ligands")

        train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_dataset = MultiInputDataset(dataframe=train_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")

        train_dataset.load_features(enzymes_features, "proteins")
        train_dataset.load_features(compounds_features, "ligands")

        test_dataset = dataset[dataset["Substrate ID"].isin(test_ids)]
        
        test_dataset = MultiInputDataset(dataframe=test_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")
        test_dataset.load_features(enzymes_features, "proteins")
        test_dataset.load_features(compounds_features, "ligands")

        if merge_validation_set:
            datasets.append((train_dataset, test_dataset))
        else:
            datasets.append((train_dataset, validation_dataset, test_dataset))
    return datasets

depth_array = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
max_bin = [64, 128, 256, 512, 1024]

space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
    "max_depth": hp.choice("max_depth", depth_array),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds":  hp.uniform("num_rounds", 30, 1000),
    "weight" : hp.uniform("weight", 0.01,0.99),
    "max_bin": hp.choice("max_bin", max_bin)
}

def set_param_values_V2(param, dtrain):
    num_round = int(param["num_rounds"])
    param["tree_method"] = "hist"
    param["max_depth"] = int(depth_array[param["max_depth"]])
    param["max_bin"] = int(max_bin[param["max_bin"]])
    # param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param["device"] = "cuda:3"

    param['objective'] = 'binary:logistic'
    weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
    dtrain.set_weight(weights)
    del param["num_rounds"]
    del param["weight"]
    return(param, num_round, dtrain)

def get_performance(pred, true):
    MCC = matthews_corrcoef(true, np.round(pred))
    return(-MCC)

def train_with_batches(param, num_round, dtrain, num_batches=1):
    np.random.seed(42)
    try:
        total_size = len(dtrain.get_label())
        batch_size = total_size // num_batches
        rounds_per_batch = num_round // num_batches

        # Random permutation of indices
        indices = np.random.permutation(total_size)

        bst = None
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size if i < num_batches - 1 else total_size
            batch_idx = indices[start:end]

            dtrain_batch = xgb.DMatrix(
                data=dtrain.get_data()[batch_idx],
                label=dtrain.get_label()[batch_idx]
            )

            if bst is None:
                # First batch
                bst = xgb.train(param, dtrain_batch, num_boost_round=rounds_per_batch)
            else:
                # Continue training
                bst = xgb.train(param, dtrain_batch, num_boost_round=rounds_per_batch, xgb_model=bst)

        param["num_rounds"] = num_round
        return bst

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def _predict_with_batches(dval, model, num_batches=1):
    np.random.seed(42)
    try:
        total_size = len(dval.get_label())
        batch_size = total_size // num_batches

        indices = np.arange(total_size)

        bst = None
        predictions_all = np.array([])
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size if i < num_batches - 1 else total_size
            batch_idx = indices[start:end]

            dval_batch = xgb.DMatrix(
                data=dval.get_data()[batch_idx],
                label=dval.get_label()[batch_idx]
            )

            if bst is None:
                # First batch
                predictions = model.predict(dval_batch)
            else:
                # Continue training
                predictions = model.predict(dval_batch)

            predictions_all = np.concatenate((predictions_all, predictions), axis=0)

        return predictions_all

    except Exception as e:
        print(f"Error occurred: {e}")
        return None
    
def predict_with_batches(dval, model):
    try:
        #Training:
        y_val_pred = model.predict(dval)
        return y_val_pred
    except Exception as e:
        
        batches = 2
        print(f"Single batch training failed. Trying with {batches}")
        predictions = _predict_with_batches(dval, model)
        while predictions is None and batches < 30:
            batches += 1
            print(f"Trying with {batches}")
            predictions = _predict_with_batches(dval, model, num_batches=batches)
        if predictions is None:
            print("Failed to train model even with batching.")
            return None
        else:
            print(f"Succeeded with {batches} batches.")
            return predictions
    
def fit_xgboost(param, dtrain, num_round):
    try:
        #Training:
        bst = xgb.train(param,  dtrain, num_round)
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

def train_gb(train_dataset, validation_dataset, test_dataset, save_pred_path, model_name, similarity, max_evals=500):

    def set_param_values(param):
        num_round = int(param["num_rounds"])
        param["tree_method"] = "hist"
        param["sampling_method"] = "gradient_based"
        param["device"] = "cuda:3"

        param['objective'] = 'binary:logistic'
        weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
        dtrain.set_weight(weights)

        del param["num_rounds"]
        del param["weight"]
        return(param, num_round)

    def get_predictions(param, dM_train, dM_val):
        param, num_round, dM_train = set_param_values_V2(param = param, dtrain = dM_train)
        bst = fit_xgboost(param, dM_train, num_round)
        predictions = predict_with_batches(dM_val, bst)
        return predictions
    

    def get_performance_metrics(pred, true):
        acc = np.mean(np.round(pred) == np.array(true))
        roc_auc = roc_auc_score(np.array(true), pred)
        mcc = matthews_corrcoef(np.array(true),np.round(pred))
        print("accuracy: %s,ROC AUC: %s, MCC: %s" % (acc, roc_auc, mcc))

    train_X_all = np.concatenate([train_dataset.X["proteins"], train_dataset.X["ligands"]], axis = 1)
    test_X_all = np.concatenate([test_dataset.X["proteins"], test_dataset.X["ligands"]], axis = 1)
    val_X_all = np.concatenate([validation_dataset.X["proteins"], validation_dataset.X["ligands"]], axis = 1)

    dtrain = xgb.DMatrix(np.array(train_X_all), label = np.array(train_dataset.y).astype(float))
    dtest = xgb.DMatrix(np.array(test_X_all), label = np.array(test_dataset.y).astype(float))
    dvalid = xgb.DMatrix(np.array(val_X_all), label = np.array(validation_dataset.y).astype(float))
    dtrain_val = xgb.DMatrix(np.concatenate([np.array(train_X_all), np.array(val_X_all)], axis = 0),
                                    label = np.concatenate([np.array(train_dataset.y).astype(float),np.array(validation_dataset.y).astype(float)], axis = 0))
    
    def train_xgboost_model_all(param):
        param, num_round = set_param_values(param)
        bst = fit_xgboost(param, dtrain, num_round)
        if bst is None:
            print("Failed to train model even with batching.")
            return 0
        else:
            return get_performance(pred=bst.predict(dvalid), true=validation_dataset.y)

    trials = Trials()
    best = fmin(fn = train_xgboost_model_all, space = space_gradient_boosting,
                algo = tpe.suggest, max_evals = max_evals, trials = trials)
    best_copy = best.copy()

    y_val_pred_all = get_predictions(param = best_copy, dM_train = dtrain, dM_val = dvalid)
    get_performance_metrics(pred = y_val_pred_all, true = validation_dataset.y)

    best_copy = best.copy()
    y_test_pred_all = get_predictions(param = best_copy, dM_train = dtrain_val, dM_val = dtest)
    get_performance_metrics(pred = y_test_pred_all, true = test_dataset.y)

    os.makedirs(save_pred_path, exist_ok=True)

    write_pickle(os.path.join(save_pred_path, f"{model_name}_{similarity}_best.pkl"), best)

    return best

def evaluate_model(model_name, save_pred_path, train_dataset, validation_dataset, test_dataset, similarity, proteins=True, suffix_results ="", results_name=None):
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, r2_score, mean_squared_error, accuracy_score

    print("Evaluating model...", f"{model_name}_{similarity}_best.pkl")
    if results_name is None:
        if suffix_results == "":
            results_name = os.path.join(save_pred_path, f"results_{model_name}.csv")
        else:
            results_name = os.path.join(save_pred_path, f"results_{model_name}_{suffix_results}.csv")
    print("It will save the results in ", results_name)
    if test_dataset.X["proteins"].shape[0] != 0:

        for seed in range(5):
            print("Seed:", seed)
            np.random.seed(seed)
            best = read_pickle(os.path.join(save_pred_path, f"{model_name}_{similarity}_best.pkl"))
            train_X_all = np.concatenate([train_dataset.X["proteins"], train_dataset.X["ligands"]], axis = 1)
            test_X_all = np.concatenate([test_dataset.X["proteins"], test_dataset.X["ligands"]], axis = 1)
            val_X_all = np.concatenate([validation_dataset.X["proteins"], validation_dataset.X["ligands"]], axis = 1)

            train_X_all = np.concatenate([np.array(train_X_all), np.array(val_X_all)], axis = 0)
            train_y_all = np.concatenate([np.array(train_dataset.y).astype(float), np.array(validation_dataset.y).astype(float)], axis = 0)

            permutation = np.random.permutation(len(train_X_all))

            # Shuffle both arrays using the same permutation
            train_X_all = train_X_all[permutation]
            train_y_all = train_y_all[permutation]

            dtrain = xgb.DMatrix(train_X_all, label = train_y_all)

            dtest = xgb.DMatrix(np.array(test_X_all), label = np.array(test_dataset.y).astype(float))

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

            if proteins:
                similarity_name = "identity"
            else:
                similarity_name = "similarity"

            results = {
                'accuracy': accuracy,
                'f1_score': f1,
                'recall': recall,
                'precision': precision,
                'roc_auc': roc_auc,
                'mcc': mcc,
                'seed':seed,
                similarity_name: similarity,
            }
            df = pd.DataFrame([results])
            if os.path.exists(results_name):
                file_exists=True
            else:
                file_exists=False

            df.to_csv(results_name, mode='a', index=False, header=not file_exists)



def experiment_optimize(splits, name, proteins_split, similarity, dataset="curated_dataset.csv", enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp", model_name="xgb_np_esm2"):
    print(dataset)
    if proteins_split:
        datasets = load_datasets(splits, dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features)
    else:
        datasets = load_datasets_compounds(splits, dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features)
    
    train_dataset, validation_dataset, test_dataset = datasets[0]
    train_gb(train_dataset, validation_dataset, test_dataset, model_name, model_name=name, similarity=similarity,
             max_evals=200)
    evaluate_model(name, model_name, train_dataset, validation_dataset, test_dataset, similarity=similarity, proteins=proteins_split)

def experiment_features(model_name, dataset = "curated_dataset.csv",enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp"):

    splits = read_pickle("compounds_split/splits_compounds_08.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_compounds", 
                        proteins_split=False, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=80, dataset=dataset)
    
    splits = read_pickle("compounds_split/splits_compounds_06.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_compounds", 
                        proteins_split=False, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=60, dataset=dataset)
    
    splits = read_pickle("compounds_split/splits_compounds_04_corrected.pkl")

    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_compounds", 
                        proteins_split=False, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=40, dataset=dataset)
    
    splits = read_pickle("compounds_split/splits_compounds_03.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_compounds", 
                        proteins_split=False, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=30, dataset=dataset)
    
    splits = read_pickle("compounds_split/splits_compounds_02.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_compounds", 
                        proteins_split=False, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=20, dataset=dataset)


    splits = read_pickle("splits/splits_0_6_proteins_train_val_test.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_proteins", 
                        proteins_split=True, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=60, dataset=dataset)
    
    splits = read_pickle("splits/splits_0_8_proteins_train_val_test.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_proteins", 
                        proteins_split=True, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=80, dataset=dataset)
    
    splits = read_pickle("splits/splits_0_4_proteins_train_val_test.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_proteins", 
                        proteins_split=True, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=40, dataset=dataset)

    splits = read_pickle("splits/splits_0_2_proteins_train_val_test.pkl")
    experiment_optimize(splits, model_name=model_name, name="binding_np_classifier_proteins", 
                        proteins_split=True, enzymes_features=enzymes_features, compounds_features=compounds_features, similarity=20, dataset=dataset)
    


def experiment_np_esm2(dataset = "curated_dataset.csv", enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp"):
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_esm2")

def experiment_np_esm2_augmented(dataset = "curated_dataset.csv", enzymes_features="features_proteins_esm2_3b_ec_number_augmented_embedding", compounds_features="features_compounds_np_classifier_fp_augmented"):
    experiment_features(enzymes_features=enzymes_features, dataset=dataset, compounds_features=compounds_features, model_name="xgb_np_esm2_augmented")

def experiment_prot_bert_np(dataset="curated_dataset.csv", enzymes_features="prot_bert_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp"):
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_prot_bert")

def experiment_esm1b(dataset="curated_dataset.csv", enzymes_features="esm1b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp"):
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_esm1b")

def experiment_np_esm2_np_chirality(dataset="curated_dataset.csv", enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp_chirality"):
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_esm2_chirality")

def experiment_prot_bert_np_chirality(dataset="curated_dataset.csv", enzymes_features="prot_bert_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp_chirality"):
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_prot_bert_chirality")

def experiment_esm1b_np_chirality(dataset="curated_dataset.csv", enzymes_features="esm1b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp_chirality"):
    experiment_features(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_esm1b_chirality")

def train_model_and_save(model_params_path, save_model_path, enzymes_features="prot_bert_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp"):
    splits = read_pickle("splits/splits_0_8_proteins_train_val_test.pkl")
    datasets = load_datasets(splits, enzymes_features=enzymes_features, compounds_features=compounds_features)
    
    train_dataset, validation_dataset, test_dataset = datasets[0]
    # read pickle
    best = read_pickle(model_params_path)
    train_X_all = np.concatenate([train_dataset.X["proteins"], train_dataset.X["ligands"]], axis = 1)
    test_X_all = np.concatenate([test_dataset.X["proteins"], test_dataset.X["ligands"]], axis = 1)
    val_X_all = np.concatenate([validation_dataset.X["proteins"], validation_dataset.X["ligands"]], axis = 1)

    dall = xgb.DMatrix(np.concatenate([np.array(train_X_all), np.array(val_X_all), np.array(test_X_all)], axis = 0),
                                    label = np.concatenate([np.array(train_dataset.y).astype(float),np.array(validation_dataset.y).astype(float), np.array(test_dataset.y).astype(float)], axis = 0))

    param, num_round, _ = set_param_values_V2(best, dall)
    bst = fit_xgboost(param, dall, num_round)

    write_pickle(save_model_path, bst)

def load_datasets_compound_classes(ids_for_datasets, compounds_split_datasets, 
                                   dataset="curated_dataset_no_stereochemistry_duplicates.csv", random_state=42, 
                                   merge_validation_set=False, enzymes_features="esm2_3b_ec_number_embedding", 
                                   compounds_features="features_compounds_np_classifier_fp"):
    import pandas as pd
    dataset = pd.read_csv(dataset)

    datasets = {}

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
            
            validation_dataset.load_features(enzymes_features, "proteins")
            validation_dataset.load_features(compounds_features, "ligands")

        train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_dataset = MultiInputDataset(dataframe=train_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={
                                                                    "proteins": "Enzyme ID",
                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")

        train_dataset.load_features(enzymes_features, "proteins")
        train_dataset.load_features(compounds_features, "ligands")

        test_dataset_dataframe = dataset[dataset["Substrate ID"].isin(test_ids)]
        for class_ in compounds_split_datasets:
            test_dataset_class = test_dataset_dataframe[test_dataset_dataframe["Substrate ID"].isin(compounds_split_datasets[class_])]
        
            test_dataset = MultiInputDataset(dataframe=test_dataset_class, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                    instances_ids_field={"proteins": "Enzyme ID",
                                                                                        "ligands": "Substrate ID", 
                                                                    },
                                                                    labels_field="Binding")
            test_dataset.load_features(enzymes_features, "proteins")
            test_dataset.load_features(compounds_features, "ligands")
            test_dataset_copy = deepcopy(test_dataset)
            datasets[class_] = test_dataset_copy

    return train_dataset, validation_dataset, datasets

def train_model_and_evaluate_for_classes(split_path, general_split, model_name, 
                                         save_pred_path, similarity, 
                                         dataset="curated_dataset_no_stereochemistry_duplicates.csv"):
    split = read_pickle(split_path)
    general_split = read_pickle(general_split)

    train_dataset, validation_dataset, test_datasets = load_datasets_compound_classes(general_split, split, dataset=dataset)

    for class_ in test_datasets:

        evaluate_model(model_name, save_pred_path, train_dataset, validation_dataset, test_datasets[class_], similarity, proteins=False, suffix_results=class_)

def train_protbert_and_evaluate_for_classes(dataset="curated_dataset_no_stereochemistry_duplicates.csv",
                                            save_pred_path="xgb_np_prot_bert_no_stereo"):
    
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_02.pkl", 
                        split_path="splits/pathway_to_compounds_split_02.pkl", 
                        save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=20)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_04_corrected.pkl", 
                        split_path="splits/pathway_to_compounds_split_04.pkl", 
                        save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=40)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_06.pkl", 
                        split_path="splits/pathway_to_compounds_split_06.pkl", 
                        save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=60)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_08.pkl", 
                        split_path="splits/pathway_to_compounds_split_08.pkl", 
                        save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=80)
def train_esm1b_and_evaluate_for_classes(dataset="curated_dataset_no_stereochemistry_duplicates.csv",
                                        save_pred_path="xgb_np_esm1b_no_stereo"):
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_02.pkl", 
                        split_path="splits/pathway_to_compounds_split_02.pkl", 
                        save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=20)

    train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_04_corrected.pkl", 
                        split_path="splits/pathway_to_compounds_split_04.pkl", 
                        save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=40)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_06.pkl", 
                        split_path="splits/pathway_to_compounds_split_06.pkl", 
                        save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=60)
    train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_08.pkl", 
                        split_path="splits/pathway_to_compounds_split_08.pkl", 
                        save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=80)

def train_esm2_3b_and_evaluate_for_classes(dataset="curated_dataset_no_stereochemistry_duplicates.csv",
                                           save_pred_path="xgb_np_esm2_no_stereo"):
    # train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_02.pkl", 
    #                     split_path="splits/pathway_to_compounds_split_02.pkl", 
    #                     save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=20)

    train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_04_corrected.pkl", 
                        split_path="splits/pathway_to_compounds_split_04.pkl", 
                        save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=40)
    # train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_06.pkl", 
    #                     split_path="splits/pathway_to_compounds_split_06.pkl", 
    #                     save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=60)
    # train_model_and_evaluate_for_classes(dataset=dataset, general_split="compounds_split/splits_compounds_08.pkl", 
    #                     split_path="splits/pathway_to_compounds_split_08.pkl", 
    #                     save_pred_path=save_pred_path, model_name="binding_np_classifier_compounds", similarity=80)

def train_and_evaluate(dataset, splits, proteins_split, similarity, name,results_name,
                               enzymes_features="esm2_3b_ec_number_embedding", 
                               compounds_features="features_compounds_np_classifier_fp", 
                               save_pred_path="xgb_np_esm2_no_stereo"):

    if proteins_split:
        datasets = load_datasets(splits, dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features)
    else:
        datasets = load_datasets_compounds(splits, dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features)
    
    train_dataset, validation_dataset, test_dataset = datasets[0]

    if proteins_split:
        suffix_results = "proteins"
    else:
        suffix_results = "compounds"

    evaluate_model(name, save_pred_path, train_dataset, validation_dataset, test_dataset, similarity=similarity, proteins=proteins_split,
                   suffix_results=suffix_results, results_name=results_name)

def train_esm2_and_evaluate():
    dataset = "curated_dataset.csv"
    splits = read_pickle("compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds",
                       enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp",
                       save_pred_path="xgb_np_esm2_stereo", results_name = "results_xgb_np_esm2_compounds.csv")
    
    
    dataset = "curated_dataset_no_stereochemistry_duplicates.csv"
    splits = read_pickle("compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds",
                       enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp",
                       save_pred_path="xgb_np_esm2_no_stereo", results_name = "results_xgb_np_esm2_compounds.csv")
    

def train_esm1b_and_evaluate():
    dataset = "curated_dataset.csv"
    splits = read_pickle("compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds",
                       enzymes_features="esm1b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp",
                       save_pred_path="xgb_np_esm1b_stereo", results_name = "results_xgb_np_esm1b_compounds.csv")
    
    dataset = "curated_dataset_no_stereochemistry_duplicates.csv"
    splits = read_pickle("compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds",
                       enzymes_features="esm1b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp",
                       save_pred_path="xgb_np_esm1b_no_stereo", results_name = "results_xgb_np_esm1b_compounds.csv")
    

def train_protbert_and_evaluate():
    dataset = "curated_dataset.csv"
    splits = read_pickle("compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds",
                       enzymes_features="prot_bert_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp",
                       save_pred_path="xgb_np_prot_bert_stereo", results_name = "results_xgb_np_prot_bert_compounds.csv")
    
    dataset = "curated_dataset_no_stereochemistry_duplicates.csv"
    splits = read_pickle("compounds_split/splits_compounds_04_corrected.pkl")
    train_and_evaluate(dataset, splits, proteins_split=False, similarity=40, name="binding_np_classifier_compounds",
                          enzymes_features="prot_bert_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp",
                            save_pred_path="xgb_np_prot_bert_no_stereo", results_name = "results_xgb_np_prot_bert_compounds.csv")
    
    
def load_docking_datasets(enzymes_features="prot_bert_ec_number_embedding", compounds_features="features_compounds_docking_fp"):
    from plants_sm.io.pickle import read_pickle
    pairs = read_pickle("docking_augmentation/pairs_to_select.pkl")
    dataset = pd.read_csv("curated_dataset_no_stereochemistry_duplicates.csv")
    
    
def train_protbert_for_docking_classes():
    pass

if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # train_esm2_and_evaluate()
    # train_esm1b_and_evaluate()
    # train_protbert_and_evaluate()

    # experiment_np_esm2_augmented()
    # experiment_np_esm2()
    # experiment_esm1b()
    # experiment_prot_bert_np()
    # experiment_np_esm2(dataset="curated_dataset_no_stereochemistry_duplicates.csv",)
    # experiment_esm1b(dataset="curated_dataset_no_stereochemistry_duplicates.csv",)
    # experiment_prot_bert_np(dataset="curated_dataset_no_stereochemistry_duplicates.csv",)
    # train_model_and_save("xgb_np_prot_bert/binding_np_classifier_compounds_02_best.pkl", "xgb_prot_bert_20.pkl")
    # train_model_and_save("xgb_np_esm2_tpe/binding_np_classifier_compounds_02_best.pkl", "xgb_esm2_20.pkl",enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp_chirality")
    # train_model_and_save("xgb_np_esm1b/binding_np_classifier_compounds_20_best.pkl", "xgb_esm1b_20.pkl",enzymes_features="esm1b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp_chirality")
    
    
    # train_protbert_and_evaluate_for_classes(dataset="curated_dataset.csv", 
    #                                         save_pred_path="xgb_np_prot_bert_stereo")
    # train_esm1b_and_evaluate_for_classes(dataset="curated_dataset.csv", 
    #                                         save_pred_path="xgb_np_esm1b_stereo")
    train_esm2_3b_and_evaluate_for_classes(dataset="curated_dataset.csv", 
                                            save_pred_path="xgb_np_esm2_stereo")
    





