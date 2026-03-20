import luigi
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from enzyme_substrate_prediction.etl_pipeline.dataset_assembly import DatasetAssemblyIDs
from enzyme_substrate_prediction.etl_pipeline._utils import tqdm_joblib

class EnzymeCompoundPairs(luigi.Task):

    def requires(self):
        return [DatasetAssemblyIDs()]

    def input(self):
        return [luigi.LocalTarget('RHEA_IDs.csv'), luigi.LocalTarget('rhea2ec.tsv'), luigi.LocalTarget('rhea2uniprot_sprot.tsv'),
                luigi.LocalTarget('rhea-reaction-smiles.tsv')]
    
    def output(self):
        return luigi.LocalTarget('RHEA_enzyme_compound_pairs.csv')

    @staticmethod
    def add_line(j: int, row: pd.Series, compound: str, decomposed_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Add a line to a newly created dataset (decomposed_dataset).

        Parameters
        ----------
        j : int
            The index of the line to add.
        row : pd.Series
            The row of the original dataset.
        compound : str
            The compound to add.
        decomposed_dataset : pd.DataFrame
            The dataset to add the line to.
        
        Returns
        -------
        pd.DataFrame
            The dataset with the new line added.
        """
        decomposed_dataset.at[j, "RHEA_ID"] = row["RHEA_ID"]
        decomposed_dataset.at[j, "EC number"] = row["EC number"]
        decomposed_dataset.at[j, "CHEBI_ID"] = compound
        decomposed_dataset.at[j, "uniprot_id"] = row["ID"]
        decomposed_dataset.at[j, "reaction_SMILES"] = row["reaction SMILES"]
        return decomposed_dataset

    def create_decomposed_dataset(self, row: pd.Series) -> pd.DataFrame:
        """
        Create a enzyme-compound dataset from a row of the original dataset.

        Parameters
        ----------
        row : pd.Series
            The row of the original dataset.

        Returns
        -------
        pd.DataFrame
            The enzyme-compound dataset.
        """

        decomposed_dataset = pd.DataFrame(columns = ["RHEA_ID", "EC number", "CHEBI_ID", "uniprot_id"])
        j = 0
        reactants = row["reactants"].split(";")
        products = row["products"].split(";")
        if row["DIRECTION"] == "UN":
            for reactant in reactants:
                decomposed_dataset = self.add_line(j, row, reactant, decomposed_dataset)
                j += 1
            for product in products:
                decomposed_dataset = self.add_line(j, row, product, decomposed_dataset)
                j += 1
        elif row["DIRECTION"] == "LR":
            for reactant in reactants:
                decomposed_dataset = self.add_line(j, row, reactant, decomposed_dataset)
                j += 1
        elif row["DIRECTION"] == "RL":
            for product in products:
                decomposed_dataset = self.add_line(j, row, product, decomposed_dataset)
                j += 1
        return decomposed_dataset

    def run(self):
        """
        This method will create the dataset with the enzyme-compound pairs.

        It starts by reading the RHEA_IDs and the RHEA to EC mapping. Then, it reads the RHEA to UniProt mapping.
        It then associates each RHEA_ID with its EC number and UniProt ID. Finally, it creates the dataset with the enzyme-compound pairs.
        
        """

        df_RHEA = pd.read_csv(self.input()[0].path)

        rhea_to_ec = pd.read_csv(self.input()[1].path, sep="\t")

        rhea2uniprot = pd.read_csv(self.input()[2].path, sep="\t")

        rhea2metacyc = pd.read_csv(self.input()[3].path, sep="\t")

        reaction_to_smiles = pd.read_csv("rhea-reaction-smiles.tsv", sep="\t", header=None)
        reaction_to_smiles.columns = ["RHEA_ID", "reaction SMILES"]

        # convert into a dict where the keys are rhea_id and the values are the reaction SMILES
        rhea_to_reaction_smarts_dict = {}
        for _, row in reaction_to_smiles.iterrows():
            rhea_to_reaction_smarts_dict[row["RHEA_ID"]] = row["reaction SMILES"]

        
        # convert the MASTER_ID to keys and the direction into values in a dictionary
        rhea_to_reaction_smarts_dict_master_id = {}
        metacyc_dict = {}
        for _, row in rhea2metacyc.iterrows():
            metacyc_dict[row["MASTER_ID"]] = row["DIRECTION"]
            if row["RHEA_ID"] in rhea_to_reaction_smarts_dict:
                rhea_to_reaction_smarts_dict_master_id[row["MASTER_ID"]] = rhea_to_reaction_smarts_dict[row["RHEA_ID"]]

        # convert rhea_to_reaction_smarts_dict_master_id into a dataframe
        rhea_to_reaction_smarts = pd.DataFrame(rhea_to_reaction_smarts_dict_master_id.items(), columns = ["RHEA_ID", "reaction SMILES"])
                
        # get the EC number for each RHEA_ID
        df_RHEA_copy = df_RHEA.copy()
        for i, row in df_RHEA.iterrows():
            values = rhea_to_ec[rhea_to_ec["RHEA_ID"] == row["RHEA_ID"]]["ID"].values
            if values.size > 0:
                df_RHEA_copy.at[i, "EC number"] = values[0]
        
        # remove rows with missing EC number
        # df_RHEA_copy = df_RHEA_copy[~df_RHEA_copy["EC number"].isna()]
        
        df_RHEA_copy = pd.merge(df_RHEA_copy, rhea2uniprot, on = "RHEA_ID", how = "inner")
        df_RHEA_copy = pd.merge(df_RHEA_copy, rhea_to_reaction_smarts, on = "RHEA_ID")
        
        # multiprocessing to create the dataset faster
        parallel_callback = Parallel(n_jobs=-1, backend="multiprocessing", prefer="threads")
        with tqdm_joblib(tqdm(desc="create dataset", total=len(df_RHEA))):
            res = parallel_callback(
                delayed(self.create_decomposed_dataset)(row)
                for _, row in df_RHEA_copy.iterrows())
            
        df_RHEA = pd.concat(res, ignore_index = True)

        df_RHEA.to_csv(self.output().path, index = False)


class EnzymeCompoundPairsAssemblyMetaCyc(EnzymeCompoundPairs):

    def input(self):
        return [luigi.LocalTarget('RHEA_IDs.csv'), luigi.LocalTarget('rhea2ec.tsv'), 
                luigi.LocalTarget('rhea2uniprot_sprot.tsv'), luigi.LocalTarget('rhea2metacyc.tsv'),
                luigi.LocalTarget('rhea-reaction-smiles.tsv')]
    
    def create_decomposed_dataset(self, row: pd.Series, metacyc_dict: dict) -> pd.DataFrame:
        """
        Create a enzyme-compound dataset from a row of the original dataset.

        Parameters
        ----------
        row : pd.Series
            The row of the original dataset.

        Returns
        -------
        pd.DataFrame
            The enzyme-compound dataset.
        """

        decomposed_dataset = pd.DataFrame(columns = ["RHEA_ID", "EC number", "CHEBI_ID", "uniprot_id", "reaction_SMILES"])
        j = 0
        reactants = row["reactants"].split(";")
        products = row["products"].split(";")
        rhea_id = row["RHEA_ID"]
        
        keys = metacyc_dict.keys()
        if rhea_id in keys:
            direction = metacyc_dict[rhea_id]
        else:
            return decomposed_dataset
            # direction = row["DIRECTION"]

        if direction == "UN":
            for reactant in reactants:
                decomposed_dataset = self.add_line(j, row, reactant, decomposed_dataset)
                j += 1
            for product in products:
                decomposed_dataset = self.add_line(j, row, product, decomposed_dataset)
                j += 1
        elif direction == "LR":
            for reactant in reactants:
                decomposed_dataset = self.add_line(j, row, reactant, decomposed_dataset)
                j += 1
        elif direction == "RL":
            for product in products:
                decomposed_dataset = self.add_line(j, row, product, decomposed_dataset)
                j += 1
        return decomposed_dataset

    def run(self):
        df_RHEA = pd.read_csv(self.input()[0].path)

        rhea_to_ec = pd.read_csv(self.input()[1].path, sep="\t")

        rhea2uniprot = pd.read_csv(self.input()[2].path, sep="\t")

        rhea2metacyc = pd.read_csv(self.input()[3].path, sep="\t")

        reaction_to_smiles = pd.read_csv("rhea-reaction-smiles.tsv", sep="\t", header=None)
        reaction_to_smiles.columns = ["RHEA_ID", "reaction SMILES"]

        # convert into a dict where the keys are rhea_id and the values are the reaction SMILES
        rhea_to_reaction_smarts_dict = {}
        for _, row in reaction_to_smiles.iterrows():
            rhea_to_reaction_smarts_dict[row["RHEA_ID"]] = row["reaction SMILES"]

        
        # convert the MASTER_ID to keys and the direction into values in a dictionary
        rhea_to_reaction_smarts_dict_master_id = {}
        metacyc_dict = {}
        for _, row in rhea2metacyc.iterrows():
            metacyc_dict[row["MASTER_ID"]] = row["DIRECTION"]
            if row["RHEA_ID"] in rhea_to_reaction_smarts_dict:
                rhea_to_reaction_smarts_dict_master_id[row["MASTER_ID"]] = rhea_to_reaction_smarts_dict[row["RHEA_ID"]]

        # convert rhea_to_reaction_smarts_dict_master_id into a dataframe
        rhea_to_reaction_smarts = pd.DataFrame(rhea_to_reaction_smarts_dict_master_id.items(), columns = ["RHEA_ID", "reaction SMILES"])
                
        # get the EC number for each RHEA_ID
        df_RHEA_copy = df_RHEA.copy()
        for i, row in df_RHEA.iterrows():
            values = rhea_to_ec[rhea_to_ec["RHEA_ID"] == row["RHEA_ID"]]["ID"].values
            if values.size > 0:
                df_RHEA_copy.at[i, "EC number"] = values[0]
        
        # remove rows with missing EC number
        # df_RHEA_copy = df_RHEA_copy[~df_RHEA_copy["EC number"].isna()]
        
        df_RHEA_copy = pd.merge(df_RHEA_copy, rhea2uniprot, on = "RHEA_ID", how = "inner")
        df_RHEA_copy = pd.merge(df_RHEA_copy, rhea_to_reaction_smarts, on = "RHEA_ID")

        # multiprocessing to create the dataset faster
        parallel_callback = Parallel(n_jobs=-1, backend="multiprocessing", prefer="threads")
        with tqdm_joblib(tqdm(desc="create dataset", total=len(df_RHEA_copy))):
            res = parallel_callback(
                delayed(self.create_decomposed_dataset)(row, metacyc_dict)
                for _, row in df_RHEA_copy.iterrows())
            
        df_RHEA = pd.concat(res, ignore_index = True)

        df_RHEA.to_csv(self.output().path, index = False)





    
