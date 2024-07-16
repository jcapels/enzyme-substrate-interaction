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
        return [luigi.LocalTarget('RHEA_enzyme_compound_pairs_sequence_smiles_filtered.csv'),
                luigi.LocalTarget('../clusters')]
    
    def output(self):
        return [luigi.LocalTarget('RHEA_negative_cases.csv'), luigi.LocalTarget("similarity_matrix.csv"),
                luigi.LocalTarget("CHEBI_ID_uniprot_id.pkl"), luigi.LocalTarget("uniprot_ids_for_negative_cases.pkl")]
    
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
    
    def run(self):
        
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

        negative_cases = pd.DataFrame(negative_cases, columns=["RHEA_ID", "CHEBI_ID", "uniprot_id", "protein_identity_threshold", "compound_similarity"])

        negative_cases.sort_values(by=["protein_identity_threshold", "compound_similarity"], ascending=False, inplace=True)
        negative_cases.drop_duplicates(subset=["CHEBI_ID", "uniprot_id"], inplace=True)
        negative_cases["protein_identity_"] = negative_cases["protein_identity_threshold"] * 0.01

        negative_cases.to_csv(self.output()[0].path, index=False)
        bar.close()
        similarity_matrix_pd.to_csv(self.output()[1].path, index=False)

        # save CHEBI_ID_uniprot_id in pickle
        with open(self.output()[2].path, "wb") as f:
            pickle.dump(CHEBI_ID_uniprot_id, f)

        with open(self.output()[3].path, "wb") as f:
            pickle.dump(uniprot_ids_for_negative_cases, f)


        
        