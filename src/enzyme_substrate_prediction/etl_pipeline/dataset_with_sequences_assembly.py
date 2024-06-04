import luigi
import pandas as pd
from luigi.parameter import ParameterVisibility

from enzyme_substrate_prediction.etl_pipeline.enzyme_compound_pairs_assembly import EnzymeCompoundPairs, EnzymeCompoundPairsAssemblyMetaCyc

class SequencesAssembler(luigi.Task):
    directionality = luigi.Parameter(default="UNK", visibility=ParameterVisibility.PUBLIC)

    def requires(self):
        if self.directionality == "UNK":
            return [EnzymeCompoundPairs()]
        elif self.directionality == "MetaCyc":
            return [EnzymeCompoundPairsAssemblyMetaCyc()]
        else:
            raise ValueError("Invalid directionality parameter. It should be 'UNK' or 'MetaCyc'.")

    def input(self):
        return [luigi.LocalTarget('RHEA_enzyme_compound_pairs.csv'), 
                luigi.LocalTarget("rhea-chebi-smiles.tsv"), 
                luigi.LocalTarget("swiss_prot_enzymes.csv")]
    
    def output(self):
        return luigi.LocalTarget('RHEA_enzyme_compound_pairs_sequence_smiles.csv')
    
    def run(self):
        """
        This method starts by reading the RHEA dataset and the RHEA-CHEBI dataset with the SMILES string. It then merges the two datasets on the CHEBI_ID column.
        Next, it reads the SwissProt dataset with the amino-acid sequences and merges it with the previous dataset on the uniprot_id column.

        Finally, it saves the resulting dataset to the output file.
        """

        df_RHEA = pd.read_csv(self.input()[0].path)
        
        rhea_chebi = pd.read_csv(self.input()[1].path, sep="\t", header=None)
        rhea_chebi.columns = ["CHEBI_ID", "SMILES"]
        rhea_chebi.drop_duplicates(inplace=True)

        df_RHEA_smiles = pd.merge(df_RHEA, rhea_chebi, on = "CHEBI_ID", how = "inner")

        swiss_prot_enzymes = pd.read_csv(self.input()[2].path)
        swiss_prot_enzymes.drop(["name", "enzyme"], axis=1, inplace=True)
        swiss_prot_enzymes.columns = ["uniprot_id", "sequence"]

        df_RHEA_smiles_sequences = pd.merge(df_RHEA_smiles, swiss_prot_enzymes, on = "uniprot_id", how = "inner")
        df_RHEA_smiles_sequences.to_csv(self.output().path, index=False)
