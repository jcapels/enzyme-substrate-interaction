import pickle
import luigi
from .filter_compounds import FilterCompounds
from enzyme_substrate_prediction.etl_pipeline.cdhit_clusters import ClustersIdentifier
from tqdm import tqdm
from enzyme_substrate_prediction.etl_pipeline._utils import tqdm_joblib

import pandas as pd
from typing import Dict, Tuple, List

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ExplicitBitVect

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

class NegativeCasesGenerator(luigi.Task):
    
    def requires(self):
        return FilterCompounds()
    
    def input(self):
        return [luigi.LocalTarget('RHEA_enzyme_compound_pairs_sequence_smiles_filtered.csv')]
    
    def output(self):
        return [luigi.LocalTarget("similarity_matrix.csv"),
                luigi.LocalTarget("CHEBI_ID_uniprot_id.pkl"), 
                luigi.LocalTarget("uniprot_ids_for_negative_cases.pkl"), 
                luigi.LocalTarget('RHEA_negative_cases_random.csv')]
    
    def np_array_to_bit_vector(fv: np.array):
        bv = ExplicitBitVect(len(fv))
        for i,v in enumerate(fv):
            if v:
                bv.SetBit(i)
        return bv
    
    def get_members_of_cluster_for_negative_cases(self, clusters: ClustersIdentifier, 
                                                  uniprot_id: str, 
                                                  uniprot_ids_for_negative_cases: Dict[str, ClustersIdentifier]) -> Dict[str, ClustersIdentifier]:

        cluster = clusters.get_cluster_by_member(uniprot_id)
        uniprot_ids_for_negative_cases[uniprot_id] = cluster
        return uniprot_ids_for_negative_cases
    
    def get_uniprot_ids_for_negative_cases(self) -> Tuple[dict, list]:

        uniprot_ids_for_negative_cases = {}
        not_considered_sequences = []
        for uniprot_id in self.positive_dataset.uniprot_id.unique():
            cluster = self.clust_ident_90.get_cluster_by_member(uniprot_id)
            if cluster is None:
                not_considered_sequences.append(uniprot_id)
            else:
                if len(cluster) > 1:
                    uniprot_ids_for_negative_cases = \
                        self.get_members_of_cluster_for_negative_cases(self.clust_ident_90, uniprot_id, uniprot_ids_for_negative_cases)

                elif len(self.clust_ident_80.get_cluster_by_member(uniprot_id)) > 1:
                    uniprot_ids_for_negative_cases = \
                        self.get_members_of_cluster_for_negative_cases(self.clust_ident_80, uniprot_id, uniprot_ids_for_negative_cases)

                elif len(self.clust_ident_60.get_cluster_by_member(uniprot_id)) > 1:
                    uniprot_ids_for_negative_cases = \
                        self.get_members_of_cluster_for_negative_cases(self.clust_ident_60, uniprot_id, uniprot_ids_for_negative_cases)

                elif len(self.clust_ident_40.get_cluster_by_member(uniprot_id)) > 1:
                    uniprot_ids_for_negative_cases = self.get_members_of_cluster_for_negative_cases(self.clust_ident_40, uniprot_id, uniprot_ids_for_negative_cases)
        return uniprot_ids_for_negative_cases, not_considered_sequences
    
    def set_clusters_of_protein_identity_threshold(self):

        self.clust_ident_40 = ClustersIdentifier.from_files(identity_threshold=40, folder="../clusters/", filename='all_sequences')
        self.clust_ident_80 = ClustersIdentifier.from_files(identity_threshold=80, folder="../clusters/", filename='all_sequences')
        self.clust_ident_60 = ClustersIdentifier.from_files(identity_threshold=60, folder="../clusters/", filename='all_sequences')
        self.clust_ident_90 = ClustersIdentifier.from_files(identity_threshold=90, folder="../clusters/", filename='all_sequences')

    def set_rhea_to_xrefs(self):

        self.rhea_id_to_uniprot_ids = {}
        for rhea_id in self.positive_dataset.RHEA_ID.unique():
            self.rhea_id_to_uniprot_ids[rhea_id] = self.positive_dataset[self.positive_dataset.RHEA_ID == rhea_id].uniprot_id.unique()

        self.rhea_id_to_compound_ids = {}
        for rhea_id in self.positive_dataset.RHEA_ID.unique():
            self.rhea_id_to_compound_ids[rhea_id] = self.positive_dataset[self.positive_dataset.RHEA_ID == rhea_id].CHEBI_ID.unique()

    def get_similarity_matrix(self, smiles: np.array) -> np.array:
        # Convert SMILES to molecular objects and compute fingerprints
        mol_list = [Chem.MolFromSmiles(smiles_) for smiles_ in smiles]
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mol_list]
        #fps = list(fingerprints.X)

        # Compute Tanimoto similarity matrix
        similarity_matrix = np.zeros((len(fps), len(fps)))
        for i in range(len(fps)):
            for j in range(i, len(fps)):
                #intersection = np.logical_and(fps[i], fps[j])
                #union = np.logical_or(fps[i], fps[j])

                # Calculate the Tanimoto similarity
                #similarity = np.sum(intersection) / np.sum(union)
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        return similarity_matrix
    
    def get_negative_cases(self, reaction_id: str, uniprot_ids_for_negative_cases: Dict[str, ClustersIdentifier], 
                           similarity_matrix_pd: pd.DataFrame, 
                            CHEBI_ID_uniprot_id: List[str]):
            
        negative_cases = []
        
        entries = self.positive_dataset[self.positive_dataset.RHEA_ID == reaction_id]
        for _, entry in entries.iterrows():
            uniprot_id = entry.uniprot_id
            chebi_id = entry.CHEBI_ID
            try:
                members = uniprot_ids_for_negative_cases[uniprot_id].member_to_other_members(uniprot_id)
                chebi_similarities = similarity_matrix_pd.loc[chebi_id, :]
                chebi_similar_compounds = chebi_similarities[chebi_similarities > 0.6]
                if len(members) > 0 and len(chebi_similar_compounds) > 0:
                    for member in members:
                        for chebi_id_, similarity in chebi_similar_compounds.items():
                            pair = f"{chebi_id_}_{member}"
                            if pair not in CHEBI_ID_uniprot_id:
                                negative_cases.append([f"fake_{reaction_id}", chebi_id_, member, uniprot_ids_for_negative_cases[uniprot_id].identity_threshold, similarity])
            except KeyError:
                continue
        return negative_cases

    def _generate_challenging_negative_cases(self):
        
        self.set_clusters_of_protein_identity_threshold()

        self.positive_dataset = pd.read_csv(self.input()[0].path)
        
        chebi_ids = self.positive_dataset.loc[:, ["CHEBI_ID", "SMILES"]]
        # select only unique CHEBI_IDs
        chebi_ids = chebi_ids.drop_duplicates()

        similarity_matrix = self.get_similarity_matrix(chebi_ids.SMILES)
        similarity_matrix_pd = pd.DataFrame(similarity_matrix, columns=chebi_ids.CHEBI_ID, index=chebi_ids.CHEBI_ID)

        uniprot_ids_for_negative_cases, _ = self.get_uniprot_ids_for_negative_cases()

        # get chebi ids concatenated with uniprot ids
        chebi_ids_uniprot_ids = self.positive_dataset.loc[:, ["CHEBI_ID", "uniprot_id"]]
        chebi_ids_uniprot_ids = chebi_ids_uniprot_ids.drop_duplicates()
        chebi_ids_uniprot_ids["CHEBI_ID"] = chebi_ids_uniprot_ids["CHEBI_ID"].astype(str)   
        chebi_ids_uniprot_ids["uniprot_id"] = chebi_ids_uniprot_ids["uniprot_id"].astype(str)
        CHEBI_ID_uniprot_id = chebi_ids_uniprot_ids["CHEBI_ID"] + "_" + chebi_ids_uniprot_ids["uniprot_id"]

        bar = tqdm(total=len(self.positive_dataset.RHEA_ID.unique()), desc="Generating negative cases for each reaction")
        negative_cases = []
        for reaction_id in self.positive_dataset.RHEA_ID.unique():
            bar.update(1)
            negative_cases_ = self.get_negative_cases(reaction_id, uniprot_ids_for_negative_cases, similarity_matrix_pd, CHEBI_ID_uniprot_id)
            negative_cases.extend(negative_cases_)

        bar.close()

        negative_cases = pd.DataFrame(negative_cases, columns=["RHEA_ID", "CHEBI_ID", "uniprot_id", "protein_identity_threshold", "compound_similarity"])

        negative_cases.sort_values(by=["protein_identity_threshold", "compound_similarity"], ascending=False, inplace=True)
        negative_cases.drop_duplicates(subset=["CHEBI_ID", "uniprot_id"], inplace=True)
        negative_cases["protein_identity_"] = negative_cases["protein_identity_threshold"] * 0.01

        negative_cases.to_csv(self.output()[0].path, index=False)
        
        similarity_matrix_pd.to_csv(self.output()[1].path, index=False)

        # save CHEBI_ID_uniprot_id in pickle
        with open(self.output()[2].path, "wb") as f:
            pickle.dump(CHEBI_ID_uniprot_id, f)

        with open(self.output()[3].path, "wb") as f:
            pickle.dump(uniprot_ids_for_negative_cases, f)

    def _generate_randomly(self):
            
        uniprot_ids = self.positive_dataset.uniprot_id.unique()
        chebi_ids = self.positive_dataset.CHEBI_ID.unique()
        
        uniprot_ids_chebi_ids = list((self.positive_dataset.CHEBI_ID + "_" + self.positive_dataset.uniprot_id).values)

        number_of_positive_cases = len(uniprot_ids_chebi_ids)
        negative_cases = []
        for i in tqdm(range(number_of_positive_cases), desc="Generating negative cases randomly"):
            stop = False
            while not stop:
                uniprot_id = np.random.choice(uniprot_ids)
                chebi_id = np.random.choice(chebi_ids)
                rhea_reactions = self.chebi_ids_rhea_ids[chebi_id]
                j = 0

                # just to ensure that we don't get the same reaction
                rhea_reaction = rhea_reactions[j]
                while f"{rhea_reaction}_{uniprot_id}" in self.uniprot_ids_rhea_ids:
                    j += 1
                    if j == len(rhea_reactions):
                        rhea_reaction = None
                    else:
                        rhea_reaction = rhea_reactions[j]

                if rhea_reaction is not None and f"{chebi_id}_{uniprot_id}" not in uniprot_ids_chebi_ids:
                    stop = True

            uniprot_ids_chebi_ids.append(f"{chebi_id}_{uniprot_id}")
            self.uniprot_ids_rhea_ids.append(f"{rhea_reaction}_{uniprot_id}")

            negative_cases.append([f"fake_{rhea_reaction}", chebi_id, uniprot_id, np.nan, np.nan])
        
        negative_cases = pd.DataFrame(negative_cases, columns=["RHEA_ID", "CHEBI_ID", "uniprot_id", "protein_identity_threshold", "compound_similarity"])
        negative_cases.to_csv(self.output()[-1].path, index=False)                         
    
    def run(self):
        self.positive_dataset = pd.read_csv(self.input()[0].path)
        self.uniprot_ids_rhea_ids = list((self.positive_dataset.RHEA_ID.astype(str) + "_" + self.positive_dataset.uniprot_id).values)
        # get a dictionary with chebi_ids as keys and rhea_ids as values
        self.chebi_ids_rhea_ids = {}
        for chebi_id in tqdm(self.positive_dataset.CHEBI_ID.unique()):
            self.chebi_ids_rhea_ids[chebi_id] = list(self.positive_dataset[self.positive_dataset.CHEBI_ID == chebi_id].RHEA_ID.unique())

        import pickle
        with open("chebi_ids_rhea_ids.pkl", "wb") as f:
            pickle.dump(self.chebi_ids_rhea_ids, f)

        self._generate_randomly()
        # self._generate_challenging_negative_cases()


        
        