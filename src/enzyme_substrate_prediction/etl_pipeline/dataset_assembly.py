from typing import Tuple, List
import luigi
import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed

from enzyme_substrate_prediction.etl_pipeline.download import DownloadRheaData, DownloadSwissProt
from enzyme_substrate_prediction.etl_pipeline._utils import tqdm_joblib

class DatasetAssemblyIDs(luigi.Task):

    def requires(self):
        return [DownloadRheaData(), DownloadSwissProt()]

    def input(self):
        return [luigi.LocalTarget('rhea-reactions.txt')]
    
    def output(self):
        return luigi.LocalTarget('RHEA_IDs.csv')
    

    @staticmethod
    def extract_RHEA_ID_and_CHEBI_IDs(entry: str) -> Tuple[List[str], List[str]]:
        """
        Extracts the RHEA ID and the CHEBI IDs from a RHEA entry.

        Parameters
        ----------
        entry : str
            The RHEA entry.
        
        Returns
        -------
        Tuple[List[str], List[str]]
            The RHEA ID and the CHEBI IDs.
        """
        RHEA_ID = entry[0][len("ENTRY"): -1]
        RHEA_ID = RHEA_ID.split(" ")[-1]
        CHEBI_IDs = entry[2][len("EQUATION"): -1]
        CHEBI_IDs = CHEBI_IDs[CHEBI_IDs.index("CHEBI"):]
        return(RHEA_ID, CHEBI_IDs)

    @staticmethod
    def get_substrate_IDs(IDs: List[str]) -> Tuple[str, str]:
        """
        Get substrate IDs from a list of IDs.

        Parameters
        ----------
        IDs : List[str]
            The list of IDs.
        
        Returns
        -------
        Tuple[str, str]
            The reactants IDs and the products IDs.
        """
        possible_separations = [" = ", " => ", " <=> "]
        for separation in possible_separations:
            if separation in IDs:
                IDs = IDs.split(separation)
                reactants_ids = IDs[0]
                reactants_ids = reactants_ids.replace(" + ", ";")
                reactants_ids = reactants_ids.split(";")
                reactants_ids = ";".join([ID.split(" ")[-1] for ID in reactants_ids])
                products_ids = IDs[1]
                products_ids = products_ids.replace(" + ", ";")
                products_ids = products_ids.split(";")
                products_ids = ";".join([ID.split(" ")[-1] for ID in products_ids])
                return(reactants_ids, products_ids)

    def get_entries(self) -> List[str]:
        """
        Get entries from the RHEA reactions file.

        Returns
        -------
        List[str]
            The entries.
        """

        file1 = open(os.path.join(self.input()[0]), 'r')
        lines = file1.readlines()
        entries = []
        entry = []
        
        for line in lines:
            if '///\n' in line:
                entry.append(line)
                entries.append(entry)
                entry=[]
            else:
                entry.append(line)

        return entries
    
    def get_substrates(self, entry: str) -> pd.DataFrame:
        """
        Get substrates from a RHEA entry.

        Parameters
        ----------
        entry : str
            The RHEA entry.
        
        Returns
        -------
        pd.DataFrame
            The substrates.
        """
        RHEA_ID, CHEBI_IDs = self.extract_RHEA_ID_and_CHEBI_IDs(entry)
        reactants, products = self.get_substrate_IDs(IDs = CHEBI_IDs)
        return pd.DataFrame({"RHEA_ID" : [RHEA_ID], "reactants" : [reactants], "products": [products]})
    
    def run(self):
        """
        Get entries from the RHEA reactions file and get the substrates.
        """
        rhea_entries = self.get_entries()
        parallel_callback = Parallel(n_jobs=-1, backend="multiprocessing", prefer="threads")
        with tqdm_joblib(tqdm(desc="get substrates", total=len(rhea_entries))):
            res = parallel_callback(
                delayed(self.get_substrates)(entry)
                for entry in rhea_entries)
            
        df_RHEA = pd.concat(res, ignore_index = True)
        df_RHEA["RHEA_ID"] = [float(ID.split(":")[-1]) for ID in df_RHEA["RHEA_ID"]]

        df_RHEA.to_csv(self.output().path, index = False)
