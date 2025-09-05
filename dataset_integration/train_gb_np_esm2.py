import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from hyperopt import fmin, tpe, hp, Trials, rand
import xgboost as xgb

from plants_sm.io.pickle import write_pickle


def load_datasets(ids_for_datasets, random_state=42, merge_validation_set=False):
    import pandas as pd
    dataset = pd.read_csv("curated_dataset.csv")

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
            
            validation_dataset.load_features("esm2_3b_ec_number_embedding", "proteins")
            validation_dataset.load_features("features_compounds_np_classifier_fp", "ligands")

        train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_dataset = MultiInputDataset(dataframe=train_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")

        train_dataset.load_features("esm2_3b_ec_number_embedding", "proteins")
        train_dataset.load_features("features_compounds_np_classifier_fp", "ligands")

        test_dataset = dataset[dataset["Enzyme ID"].isin(test_ids)]
        
        test_dataset = MultiInputDataset(dataframe=test_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")
        test_dataset.load_features("esm2_3b_ec_number_embedding", "proteins")
        test_dataset.load_features("features_compounds_np_classifier_fp", "ligands")
        
        if merge_validation_set:
            datasets.append((train_dataset, test_dataset))
        else:
            datasets.append((train_dataset, validation_dataset, test_dataset))
    return datasets

def load_datasets_compounds(ids_for_datasets, random_state=42, merge_validation_set=False):
    import pandas as pd
    dataset = pd.read_csv("curated_dataset.csv")

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
            
            validation_dataset.load_features("esm2_3b_ec_number_embedding", "proteins")
            validation_dataset.load_features("features_compounds_np_classifier_fp", "ligands")

        train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_dataset = MultiInputDataset(dataframe=train_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")

        train_dataset.load_features("esm2_3b_ec_number_embedding", "proteins")
        train_dataset.load_features("features_compounds_np_classifier_fp", "ligands")

        test_dataset = dataset[dataset["Substrate ID"].isin(test_ids)]
        
        test_dataset = MultiInputDataset(dataframe=test_dataset, representation_field={"proteins": "Sequence","ligands": "SMILES"},
                                                                instances_ids_field={"proteins": "Enzyme ID",
                                                                                    "ligands": "Substrate ID", 
                                                                },
                                                                labels_field="Binding")
        test_dataset.load_features("esm2_3b_ec_number_embedding", "proteins")
        test_dataset.load_features("features_compounds_np_classifier_fp", "ligands")
        
        if merge_validation_set:
            datasets.append((train_dataset, test_dataset))
        else:
            datasets.append((train_dataset, validation_dataset, test_dataset))
    return datasets

depth_array = [6,7,8,9,10,11,12,13,14]
space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
    "max_depth": hp.choice("max_depth", depth_array),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds":  hp.uniform("num_rounds", 30, 1000),
    "weight" : hp.uniform("weight", 0.01,0.99),
    "max_bin": hp.choice("max_bin", [64, 128, 256, 512]),}

def set_param_values_V2(param, dtrain):
    num_round = int(param["num_rounds"])
    param["tree_method"] = "hist"
    param["max_depth"] = int(depth_array[param["max_depth"]])
    # param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param["device"] = "cuda:2"

    param['objective'] = 'binary:logistic'
    weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
    dtrain.set_weight(weights)
    del param["num_rounds"]
    del param["weight"]
    return(param, num_round, dtrain)

def train_gb(train_dataset, validation_dataset, test_dataset, save_pred_path, model_name):

    def set_param_values(param):
        num_round = int(param["num_rounds"])
        param["tree_method"] = "hist"
        param["sampling_method"] = "gradient_based"
        param["device"] = "cuda:2"

        param['objective'] = 'binary:logistic'
        weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
        dtrain.set_weight(weights)

        del param["num_rounds"]
        del param["weight"]
        return(param, num_round)

    def get_predictions(param, dM_train, dM_val):
        param, num_round, dM_train = set_param_values_V2(param = param, dtrain = dM_train)
        bst = xgb.train(param,  dM_train, num_round)
        y_val_pred = bst.predict(dM_val)
        return(y_val_pred)
    

    def get_performance(pred, true):
        MCC = matthews_corrcoef(true, np.round(pred))
        return(-MCC)
    
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
        try:
            #Training:
            bst = xgb.train(param,  dtrain, num_round)
            param["num_rounds"] = num_round
            return(get_performance(pred = bst.predict(dvalid), true =validation_dataset.y))
        except Exception as e:
            # param, num_round = set_param_values(param)
            param["grow_policy"] = "lossguide"
            param["max_bin"] = 64
            print(e)
            #Training:
            bst = xgb.train(param,  dtrain, num_round)
            param["num_rounds"] = num_round
            return(get_performance(pred = bst.predict(dvalid), true =validation_dataset.y))

    trials = Trials()
    best = fmin(fn = train_xgboost_model_all, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = 500, trials = trials)
    print(best)
    
    y_val_pred_all = get_predictions(param = trials.argmin, dM_train = dtrain, dM_val = dvalid)
    get_performance_metrics(pred = y_val_pred_all, true = validation_dataset.y)
    y_test_pred_all = get_predictions(param = trials.argmin, dM_train = dtrain_val, dM_val = dtest)
    get_performance_metrics(pred = y_test_pred_all, true = test_dataset.y)

    os.makedirs(save_pred_path, exist_ok=True)

    write_pickle(os.path.join(save_pred_path, f"{model_name}_best.pkl"), best)

    return best

def evaluate_model(model_name, save_pred_path, train_dataset, validation_dataset, test_dataset):
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, r2_score, mean_squared_error, accuracy_score


    for seed in range(5):
        np.random.seed(seed)
        best = read_pickle(os.path.join(save_pred_path, f"{model_name}_best.pkl"))
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
        bst = xgb.train(best,  dtrain, num_round)

        predictions_proba = bst.predict(dtest)
        predictions = np.round(predictions_proba)

        accuracy = accuracy_score(test_dataset.y, predictions)
        f1 = f1_score(test_dataset.y, predictions)
        recall = recall_score(test_dataset.y, predictions)
        precision = precision_score(test_dataset.y, predictions)
        roc_auc = roc_auc_score(test_dataset.y, predictions_proba)
        mcc = matthews_corrcoef(test_dataset.y, predictions)

        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'recall': recall,
            'precision': precision,
            'roc_auc': roc_auc,
            'mcc': mcc,
            'seed':seed
        }
        df = pd.DataFrame([results])
        if os.path.exists(os.path.join(save_pred_path, f"results_{model_name}.csv")):
            file_exists=True
        else:
            file_exists=False

        df.to_csv(os.path.join(save_pred_path, f"results_{model_name}.csv"), mode='a', index=False, header=not file_exists)



def experiment_optimize(splits, name, proteins_split):
    if proteins_split:
        datasets = load_datasets(splits)
    else:
        datasets = load_datasets_compounds(splits)
    
    train_dataset, validation_dataset, test_dataset = datasets[0]
    train_gb(train_dataset, validation_dataset, test_dataset, "xgb_np_esm2", model_name=name)
    evaluate_model(name, "xgb_np_esm2", train_dataset, validation_dataset, test_dataset)


if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    from plants_sm.io.pickle import read_pickle

    splits = read_pickle("compounds_split/splits_compounds_08.pkl")

    experiment_optimize(splits, name="binding_np_classifier_compounds_08", proteins_split=False)

    splits = read_pickle("compounds_split/splits_compounds_06.pkl")

    experiment_optimize(splits, name="binding_np_classifier_compounds_06", proteins_split=False)

    splits = read_pickle("compounds_split/splits_compounds_04.pkl")

    experiment_optimize(splits, name="binding_np_classifier_compounds_04", proteins_split=False)

    splits = read_pickle("compounds_split/splits_compounds_02.pkl")

    experiment_optimize(splits, name="binding_np_classifier_compounds_02", proteins_split=False)

    splits = read_pickle("splits/splits_0_6_proteins_train_val_test.pkl")

    experiment_optimize(splits, name="binding_np_classifier_06_proteins_3", proteins_split=True)

    splits = read_pickle("splits/splits_0_8_proteins_train_val_test.pkl")

    experiment_optimize(splits, name="binding_np_classifier_08_proteins_3", proteins_split=True)

    splits = read_pickle("splits/splits_0_4_proteins_train_val_test.pkl")

    experiment_optimize(splits, name="binding_np_classifier_04_proteins_3", proteins_split=True)





