import luigi
import pandas as pd

from enzyme_substrate_prediction.datasets_integration.datasets_gather import DatasetsGatherer
import csv

def csv_to_fasta(csv_file_path, fasta_file_path):
    with open(csv_file_path, mode='r', newline='') as csv_file, open(fasta_file_path, mode='w') as fasta_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row if there is one

        for row in csv_reader:
            if len(row) >= 2:
                sequence_id = row[1].replace(" ", "_")
                sequence = row[0]
                fasta_file.write(f">{sequence_id}\n")
                fasta_file.write(f"{sequence}\n")

class DatasetAugmenter(luigi.Task):

    def requires(self):
        return [DatasetsGatherer()]

    def output(self):
        return [luigi.LocalTarget('augmented_dataset_validated.csv'), luigi.LocalTarget('augmented_dataset_not_validated.csv')]
    
    def run(self):
        import pandas as pd
        rhea = pd.read_csv("RHEA_final_dataset_random.csv")
        rhea_positives = rhea[rhea["interaction"]==1]

        # Rename columns
        rhea_positives = rhea_positives.rename(columns={'uniprot_id': 'Enzyme ID', 'CHEBI_ID': 'Substrate ID', "interaction":"Binding",
                                                "sequence":"Sequence"})
        
        dataset = pd.read_csv("integrated_dataset.csv").drop_duplicates(subset=["Enzyme ID", "Substrate ID"], keep="first")
        dataset["Validated"] = True

        rhea_positives["Validated"] = False

        dataset = pd.concat([dataset, rhea_positives], axis=0).drop_duplicates(subset=["Enzyme ID", "Substrate ID"], keep="first")

        train_dataset = pd.read_csv("ESP_train_df.csv")
        validation_dataset = pd.read_csv("ESP_val_df.csv")
        test_dataset = pd.read_csv("ESP_test_df.csv")

        dataset_prosmith = pd.concat([train_dataset, validation_dataset, test_dataset])
        dataset_prosmith["Validated"] = False

        dataset_prosmith = dataset_prosmith.rename(columns={'Uniprot ID': 'Enzyme ID', 'molecule ID': 'Substrate ID', "output":"Binding",
                                                "Protein sequence":"Sequence"})
        
        dataset = pd.concat([dataset, dataset_prosmith], axis=0).drop_duplicates(subset=["Enzyme ID", "Substrate ID"], keep="first")

        dataset[dataset["Validated"]==False].drop_duplicates(subset=["Sequence"]).loc[:, ["Sequence", "Enzyme ID"]].sample(frac=1).to_csv("unique_enzymes.csv", index=False)
        dataset[dataset["Validated"]==False].to_csv("augmented_dataset_not_validated.csv", index=False)

        dataset[dataset["Validated"]==True].to_csv("augmented_dataset_validated.csv", index=False)

        csv_file_path = 'unique_enzymes.csv'  # Path to your CSV file
        fasta_file_path = 'all_sequences.fasta'  # Path to save the FASTA file
        csv_to_fasta(csv_file_path, fasta_file_path)