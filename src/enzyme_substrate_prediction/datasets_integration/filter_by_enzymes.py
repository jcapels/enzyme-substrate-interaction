
from Bio import SeqIO
import luigi

from enzyme_substrate_prediction.datasets_integration.merge_with_augmented_data import DatasetAugmenter

def read_fasta(file_path):
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


class EnzymeFilter(luigi.Task):

    def requires(self):
        return DatasetAugmenter()
    
    def output(self):
        return luigi.LocalTarget("augmented_dataset_not_validated_60.csv")

    def run(self):

        file_path = 'all_sequences_clustered_sequences_60.fasta'
        sequences = read_fasta(file_path)

        import pandas as pd

        dataset = pd.read_csv("augmented_dataset_not_validated.csv")
        dataset_60 = dataset[dataset["Enzyme ID"].isin(sequences.keys())]
        dataset_60.to_csv("augmented_dataset_not_validated_60.csv", index=False)