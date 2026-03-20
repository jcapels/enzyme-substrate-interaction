import datetime
import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from hurry.filesize import size
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset

from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_standardization.truncation import Truncator

from generate_features import process_probert, process_esm1b, process_to_esm2, generate_features_for_compounds
from train_gb_np_esm2 import *


def generate_features_protbert(dataset):

    multi_input_dataset = MultiInputDataset.from_csv(dataset, representation_field={"ligands": "SMILES", 
                                                                                        "proteins": "Sequence"},
                                                                                        instances_ids_field={"ligands": "Substrate ID", 
                                                                                        "proteins": "Enzyme ID"})
    
    process_probert(multi_input_dataset, dataset.replace(".csv", f"proteins_protbert"))
    


def benchmark_resources_generating_features():
    pipelines = [process_probert, process_esm1b, process_to_esm2]
    datasets = [
        "dataset_100_100.csv", "dataset_300_1000.csv", "dataset_700_5000.csv", 
                "dataset_7000.csv", 
                "../dataset_integration/curated_dataset.csv"]
    
    if os.path.exists("benchmark_results.csv"):
        results = pd.read_csv("benchmark_results.csv")
    else:
        results = pd.DataFrame()

    for pipeline in pipelines:
        for dataset in datasets:
            dataset_df = pd.read_csv(dataset)
            tracemalloc.start()
            start = time.time()

            truncator = Truncator(max_length=884)
            protein_standardizer = ProteinStandardizer()

            multi_input_dataset = MultiInputDataset.from_csv(dataset, representation_field={"ligands": "SMILES", 
                                                                                        "proteins": "Sequence"},
                                                                                        instances_ids_field={"ligands": "Substrate ID", 
                                                                                        "proteins": "Enzyme ID"})

            multi_input_dataset = protein_standardizer.fit_transform(multi_input_dataset, "proteins")
            multi_input_dataset = truncator.fit_transform(multi_input_dataset, "proteins")
            pipeline(multi_input_dataset, f"proteins_{pipeline.__name__}")
            generate_features_for_compounds(multi_input_dataset, "compounds_{pipeline.__name__}")

            end = time.time()
            print("Time spent: ", end - start)
            print("Memory needed: ", tracemalloc.get_traced_memory())
            unique_substrates_dataset = np.unique(dataset_df["Substrate ID"])
            num_unique_substrates = len(unique_substrates_dataset)
            unique_enzymes_dataset = np.unique(dataset_df["Enzyme ID"])
            num_unique_enzymes = len(unique_enzymes_dataset)
            num_rows = dataset_df.shape[0]

            results = pd.concat((results, 
                                    pd.DataFrame({"pipeline": [pipeline.__name__], 
                                                  "unique_enzymes": [num_unique_enzymes],
                                                  "unique_substrates": [num_unique_substrates],
                                                  "num_pairs": [num_rows],
                                                  "time": [str(datetime.timedelta(seconds=end - start))], 
                                                  "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                  ignore_index=True, axis=0)
            tracemalloc.stop()

            results.to_csv("benchmark_results.csv", index=False)

def benchmark_resources_training():
    pipelines = [
        ("experiment_np_esm2", experiment_np_esm2, "../dataset_integration/curated_dataset.csv", "../dataset_integration/esm2_3b_ec_number_embedding", "../dataset_integration/features_compounds_np_classifier_fp"),
        ("experiment_prot_bert_np", experiment_prot_bert_np, "../dataset_integration/curated_dataset.csv", "../dataset_integration/prot_bert_ec_number_embedding", "../dataset_integration/features_compounds_np_classifier_fp"),
        ("experiment_esm1b", experiment_esm1b, "../dataset_integration/curated_dataset.csv", "../dataset_integration/esm1b_ec_number_embedding", "../dataset_integration/features_compounds_np_classifier_fp")
    ]

    results = pd.DataFrame(columns=["pipeline", "time", "memory"])

    for name, pipeline, dataset, enzymes_features, compounds_features in pipelines:
        tracemalloc.start()
        start = time.time()
        pipeline(dataset=dataset, enzymes_features=enzymes_features, compounds_features=compounds_features)
        end = time.time()
        tracemalloc.stop()

        results = pd.concat(
            (
                results,
                pd.DataFrame({
                    "pipeline": [name],
                    "time": [str(datetime.timedelta(seconds=end - start))], 
                    "memory": [size(int(tracemalloc.get_traced_memory()[1]))]
                })
            ),
            ignore_index=True
        )

    results.to_csv("benchmark_results_optimize_pipeline.csv", index=False)


if __name__=="__main__":
    # benchmark_resources_generating_features()
    benchmark_resources_training()
            