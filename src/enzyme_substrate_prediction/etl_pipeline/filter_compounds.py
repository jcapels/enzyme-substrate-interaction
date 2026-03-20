import luigi
import pandas as pd

from enzyme_substrate_prediction.etl_pipeline.dataset_with_sequences_assembly import SequencesAssembler

class FilterCompounds(luigi.Task):

    def requires(self):
        return [SequencesAssembler(directionality="MetaCyc")]
    
    def input(self):
        return [luigi.LocalTarget('cofactors.tsv'), 
                luigi.LocalTarget("compounds_to_remove.csv"), 
                luigi.LocalTarget('RHEA_enzyme_compound_pairs_sequence_smiles.csv')]
    
    def output(self):
        return luigi.LocalTarget('RHEA_enzyme_compound_pairs_sequence_smiles_filtered.csv')
    

    def run(self):
        """
        Filter compounds based on CHEBI ontology (all the compounds under the category "cofactor" are removed).
        Remove other irrelevant compounds that are solvents, reactants or products but are not specific substrates
        based on a list of compounds to remove.
        """

        cofactors = pd.read_csv(self.input()[0].path, sep="\t")
        cofactors = cofactors.index.values
        
        df_RHEA_smiles_sequences = pd.read_csv(self.input()[2].path)
        df_RHEA_smiles_sequences_without_cofactors =  df_RHEA_smiles_sequences[~df_RHEA_smiles_sequences["CHEBI_ID"].isin(cofactors)]

        compounds_to_remove = pd.read_csv(self.input()[1].path, sep="\t")
        compounds_to_remove = compounds_to_remove[compounds_to_remove["remove"]=="yes"]["CHEBI_ID"].values

        df_RHEA_smiles_sequences_without_cofactors = df_RHEA_smiles_sequences_without_cofactors[~df_RHEA_smiles_sequences_without_cofactors["CHEBI_ID"].isin(compounds_to_remove)]
        df_RHEA_smiles_sequences_without_cofactors.to_csv("RHEA_enzyme_compound_pairs_sequence_smiles_filtered.csv", index=False)