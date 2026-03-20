
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
        return luigi.LocalTarget("augmented_dataset_not_validated_90.csv")

    def run(self):

        file_path = 'all_sequences_clustered_sequences_90.fasta'
        sequences = read_fasta(file_path)

        import pandas as pd

        dataset = pd.read_csv("augmented_dataset_not_validated.csv")
        dataset_60 = dataset[dataset["Enzyme ID"].isin(sequences.keys())]
        dataset_60.to_csv("augmented_dataset_not_validated_90.csv", index=False)


class Merger(luigi.Task):

    def requires(self):
        return EnzymeFilter()
    
    def output(self):
        return luigi.LocalTarget("augmented_dataset.csv")

    def run(self):

        import pandas as pd

        augmented_dataset_not_validated_60 = pd.read_csv("augmented_dataset_not_validated_90.csv")
        augmented_dataset_validated = pd.read_csv("augmented_dataset_validated.csv")

        augmented_dataset = pd.concat([augmented_dataset_validated, augmented_dataset_not_validated_60], axis=0)
        augmented_dataset.to_csv("augmented_dataset.csv")
