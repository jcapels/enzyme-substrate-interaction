import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from hyperopt import fmin, tpe, hp, Trials, rand
import xgboost as xgb

from plants_sm.io.pickle import write_pickle, read_pickle



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
    param["device"] = "cuda:2"

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

def train_gb(train_dataset, validation_dataset, test_dataset, save_pred_path, model_name, max_evals=500):

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
        bst = fit_xgboost(param, dM_train, num_round)
        y_val_pred = bst.predict(dM_val)
        return(y_val_pred)
    

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
        bst = fit_xgboost(best, dtrain, num_round)


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
            'seed':seed,
        }
        df = pd.DataFrame([results])
        if os.path.exists(os.path.join(save_pred_path, f"results_{model_name}.csv")):
            file_exists=True
        else:
            file_exists=False

        df.to_csv(os.path.join(save_pred_path, f"results_{model_name}.csv"), mode='a', index=False, header=not file_exists)

def load_datasets(enzymes_features="prot_bert_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp"):

    from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
    # load datasets

    train_dataset = pd.read_csv("train_dataset_w_split.csv")
    test_dataset = pd.read_csv("test_dataset_w_representation_filtered.csv")
    validation_dataset = pd.read_csv("validation_dataset.csv")

    # train_dataset = pd.concat([train_dataset, validation_dataset], ignore_index=True)

    train_dataset = MultiInputDataset(train_dataset, representation_field={"proteins": "sequence",
                                                                                                        "ligands": "smiles"},
                                                                                            instances_ids_field={"proteins": "Uniprot ID",
                                                                                                                "ligands": "molecule ID", 
                                                                                            },
                                                                                            labels_field="Binding")
    
    validation_dataset = MultiInputDataset(validation_dataset, representation_field={"proteins": "sequence", "ligands": "smiles",
                                                                                            },
                                                                                            instances_ids_field={"proteins": "Uniprot ID",
                                                                                                                "ligands": "molecule ID", 
                                                                                            },
                                                                                            labels_field="Binding")

    test_dataset = MultiInputDataset(test_dataset, representation_field={"proteins": "sequence",
                                                                                                        "ligands": "smiles", 
                                                                                            },
                                                                                            instances_ids_field={"proteins": "Uniprot ID",
                                                                                                                "ligands": "molecule ID", 
                                                                                            },
                                                                                            labels_field="Binding")

    train_dataset.load_features(enzymes_features, "proteins")
    train_dataset.load_features(compounds_features, "ligands")

    validation_dataset.load_features(enzymes_features, "proteins")
    validation_dataset.load_features(compounds_features, "ligands")

    test_dataset.load_features(enzymes_features, "proteins")
    test_dataset.load_features(compounds_features, "ligands")

    return [(train_dataset, validation_dataset, test_dataset)]


def experiment_optimize(name, enzymes_features="prot_bert_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp", model_name="xgb_np_esm2"):
    datasets = load_datasets(enzymes_features=enzymes_features, compounds_features=compounds_features)
    
    train_dataset, validation_dataset, test_dataset = datasets[0]
    train_gb(train_dataset, validation_dataset, test_dataset, model_name, model_name=name,
             max_evals=200)
    evaluate_model(name, model_name, train_dataset, validation_dataset, test_dataset)


def experiment_np_esm2(enzymes_features="esm2_3b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp"):
    experiment_optimize(enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_esm2", name="xgb_binding_np_classifier")

def experiment_prot_bert_np(enzymes_features="prot_bert_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp"):
    experiment_optimize(enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_prot_bert", name="xgb_binding_np_classifier")

def experiment_esm1b(enzymes_features="esm1b_ec_number_embedding", compounds_features="features_compounds_np_classifier_fp"):
    experiment_optimize(enzymes_features=enzymes_features, compounds_features=compounds_features, model_name="xgb_np_esm1b", name="xgb_binding_np_classifier")

if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    experiment_prot_bert_np()

    





