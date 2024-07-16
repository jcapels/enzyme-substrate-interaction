import luigi

from .generate_negative_cases import NegativeCasesGenerator
from .filter_compounds import FilterCompounds

import pickle
import pandas as pd

from numpy import NaN

import numpy as np
import random

from tqdm import tqdm 

class NegativeCasesAssembler(luigi.Task):

    def requires(self):
        return [NegativeCasesGenerator()]
    
    def output(self):
        return luigi.LocalTarget('RHEA_negative_cases_with_sequences_smiles.csv')
    
    def input(self):
        return [luigi.LocalTarget('RHEA_negative_cases.csv'), luigi.LocalTarget("similarity_matrix.csv"),
                luigi.LocalTarget("CHEBI_ID_uniprot_id.pkl"), luigi.LocalTarget("uniprot_ids_for_negative_cases.pkl"),
                luigi.LocalTarget("RHEA_enzyme_compound_pairs_sequence_smiles_filtered.csv")]
    
    def get_sample_for_negative_cases(self, number_of_data_points, percentage, protein_identity, lb_compound_similarity, up_compound_similarity):
        if lb_compound_similarity == up_compound_similarity:
            negative_cases_ = self.negative_cases[(self.negative_cases.protein_identity_threshold == protein_identity) & (self.negative_cases.compound_similarity == lb_compound_similarity)]
        else:
            negative_cases_ = self.negative_cases[(self.negative_cases.protein_identity_threshold == protein_identity) & (self.negative_cases.compound_similarity < up_compound_similarity) & (self.negative_cases.compound_similarity >= lb_compound_similarity)]
        return negative_cases_.sample(n=int(percentage * number_of_data_points), random_state=123)

    def get_negative_cases_for_reactions_left_out(self, reaction_id):
        
        negative_cases = []
        
        entries = self.positive_dataset[self.positive_dataset.RHEA_ID == reaction_id]
        for _, entry in entries.iterrows():
            uniprot_id = entry.uniprot_id
            chebi_id = entry.CHEBI_ID
            try:
                members = self.uniprot_ids_for_negative_cases[uniprot_id].member_to_other_members(uniprot_id)
                if len(members) > 0:
                    for member in members:
                        pair = f"{chebi_id}_{member}"
                        if pair not in self.CHEBI_ID_uniprot_id:
                            negative_cases.append([f"fake_{reaction_id}", chebi_id, member, 
                                                   self.uniprot_ids_for_negative_cases[uniprot_id].identity_threshold, 1])
                            self.CHEBI_ID_uniprot_id.append(pair)
            except KeyError:
                continue
        return negative_cases

    def get_negative_cases_for_reactions_left_out_late(self, reaction_id, n_pairs_per_reaction):
        # Check for global seeds
        np.random.seed(123)
        random.seed(123)
        
        negative_cases = []
        
        entries = self.positive_dataset[self.positive_dataset.RHEA_ID == reaction_id]
        uniprot_ids = list(self.positive_dataset.uniprot_id.unique())
        for _, entry in entries.iterrows():
            chebi_id = entry.CHEBI_ID
            try:
                for i in range(0, n_pairs_per_reaction):
                    member = random.choice(uniprot_ids)
                    pair = f"{chebi_id}_{member}"
                    while pair in self.CHEBI_ID_uniprot_id:
                        member = random.choice(uniprot_ids)
                        pair = f"{chebi_id}_{member}"
                    
                    negative_cases.append([f"fake_{reaction_id}", chebi_id, member, NaN, 1])
                    self.CHEBI_ID_uniprot_id.append(pair)
                    assert pair in self.CHEBI_ID_uniprot_id
                    
            except KeyError:
                continue
        
        return negative_cases

    def get_negative_cases_for_compounds_left_out(self, compound_id, n_pairs_per_reaction):
        # Check for global seeds
        np.random.seed(123)
        random.seed(123)
        
        negative_cases = []
        
        entries = self.positive_dataset[self.positive_dataset.CHEBI_ID == compound_id]
        uniprot_ids = list(self.positive_dataset.uniprot_id.unique())
        for _, entry in entries.iterrows():
            chebi_id = entry.CHEBI_ID
            reaction_id = entry.RHEA_ID
            try:
                for i in range(0, n_pairs_per_reaction):
                    member = random.choice(uniprot_ids)
                    pair = f"{chebi_id}_{member}"
                    while pair in self.CHEBI_ID_uniprot_id:
                        member = random.choice(uniprot_ids)
                        pair = f"{chebi_id}_{member}"
                    
                    negative_cases.append([f"fake_{reaction_id}", chebi_id, member, NaN, 1])
                    self.CHEBI_ID_uniprot_id.append(pair)
                    assert pair in self.CHEBI_ID_uniprot_id
                    
            except KeyError:
                continue
        
        return negative_cases
    
    def run(self):
        
        # read pickle files
        with open(self.input()[2].path, "rb") as f:
            self.CHEBI_ID_uniprot_id = pickle.load(f)

        with open(self.input()[3].path, "rb") as f:
            self.uniprot_ids_for_negative_cases = pickle.load(f)

        self.negative_cases = pd.read_csv(self.input()[0].path)
        self.similarity_matrix_pd = pd.read_csv(self.input()[1].path)
        self.positive_dataset = pd.read_csv(self.input()[4].path)

        # get the number of data points
        number_of_data_points = len(self.positive_dataset)

        final_negative_cases = pd.concat([self.get_sample_for_negative_cases(number_of_data_points, 0.45, 90, 1, 1),
                                    self.get_sample_for_negative_cases(number_of_data_points, 0.1, 90, 0.9, 1),
                                    self.get_sample_for_negative_cases(number_of_data_points, 0.02, 80, 0.9, 1),
                                    self.get_sample_for_negative_cases(number_of_data_points, 0.1, 80, 0.8, 0.9),
                                    self.get_sample_for_negative_cases(number_of_data_points, 0.02, 60, 0.6, 0.8),
                                    self.get_sample_for_negative_cases(number_of_data_points, 0.02, 40, 0.6, 0.8)])
        
        final_negative_cases.drop_duplicates(subset=["CHEBI_ID", "uniprot_id"], inplace=True)
        final_negative_cases_CHEBI_ID_uniprot_id = final_negative_cases["CHEBI_ID"] + "_" + final_negative_cases["uniprot_id"]
        self.CHEBI_ID_uniprot_id = pd.concat((self.CHEBI_ID_uniprot_id, final_negative_cases_CHEBI_ID_uniprot_id))
        self.CHEBI_ID_uniprot_id = list(self.CHEBI_ID_uniprot_id)

        rhea_ids_in_negative_dataset = final_negative_cases.RHEA_ID.unique()
        rhea_ids_in_negative_dataset = pd.Series(rhea_ids_in_negative_dataset)

        rhea_ids_in_negative_dataset = rhea_ids_in_negative_dataset.str.replace("fake_", "").astype(np.float64)
        leftout_reactions = self.positive_dataset[~self.positive_dataset["RHEA_ID"].isin(rhea_ids_in_negative_dataset)].RHEA_ID.unique()

        negative_cases_leftout_reactions = []
        bar = tqdm(total=len(leftout_reactions), desc="Generating negative cases for leftout reactions")
        bar.reset()
        for reaction in leftout_reactions:
            bar.update(1)
            negative_cases_ = self.get_negative_cases_for_reactions_left_out(reaction)
            negative_cases_leftout_reactions.extend(negative_cases_)

        negative_cases_leftout_reactions = pd.DataFrame(negative_cases_leftout_reactions, columns=["RHEA_ID", "CHEBI_ID", "uniprot_id", "protein_identity_threshold", "compound_similarity"])

        rhea_ids_in_negative_dataset_2 = negative_cases_leftout_reactions.RHEA_ID.unique()
        rhea_ids_in_negative_dataset_2 = pd.Series(rhea_ids_in_negative_dataset_2)
        rhea_ids_in_negative_dataset_2 = rhea_ids_in_negative_dataset_2.str.replace("fake_", "").astype(np.float64)
        rhea_ids_in_negative_dataset_2 = pd.concat((rhea_ids_in_negative_dataset, pd.Series(rhea_ids_in_negative_dataset_2)))
        leftout_reactions_2 = self.positive_dataset[~self.positive_dataset["RHEA_ID"].isin(rhea_ids_in_negative_dataset_2)].RHEA_ID.unique()

        final_negative_cases_CHEBI_ID_uniprot_id = negative_cases_leftout_reactions["CHEBI_ID"] + "_" + negative_cases_leftout_reactions["uniprot_id"]

        negative_cases_leftout_reactions_late = []
        bar = tqdm(total=len(leftout_reactions_2), desc="Generating negative cases for leftout reactions 2")
        bar.reset()
        for reaction in leftout_reactions_2:
            bar.update(1)
            negative_cases_ = self.get_negative_cases_for_reactions_left_out_late(reaction, 10)
            negative_cases_leftout_reactions_late.extend(negative_cases_)

        negative_cases_leftout_reactions_late = pd.DataFrame(negative_cases_leftout_reactions_late, columns=["RHEA_ID", "CHEBI_ID", "uniprot_id", "protein_identity_threshold", "compound_similarity"])

        rhea_ids_in_negative_dataset_3 = negative_cases_leftout_reactions_late.RHEA_ID.unique()
        rhea_ids_in_negative_dataset_3 = pd.Series(rhea_ids_in_negative_dataset_3)
        rhea_ids_in_negative_dataset_3 = rhea_ids_in_negative_dataset_3.str.replace("fake_", "").astype(np.float64)
        rhea_ids_in_negative_dataset_3 = pd.concat((rhea_ids_in_negative_dataset_2, pd.Series(rhea_ids_in_negative_dataset_3)))
        leftout_reactions_3 = self.positive_dataset[~self.positive_dataset["RHEA_ID"].isin(rhea_ids_in_negative_dataset_3)].RHEA_ID.unique()
        assert len(leftout_reactions_3) == 0

        if negative_cases_leftout_reactions.shape[0] > int(0.12*self.positive_dataset.shape[0]):
            negative_cases_leftout_reactions_ = negative_cases_leftout_reactions.sample(int(0.12*self.positive_dataset.shape[0]))
        else:
            negative_cases_leftout_reactions_ = negative_cases_leftout_reactions

        final_negative_cases_2 = pd.concat((negative_cases_leftout_reactions_late, final_negative_cases, negative_cases_leftout_reactions_))
        
        leftout_compounds = self.positive_dataset[~self.positive_dataset["CHEBI_ID"].isin(final_negative_cases_2.CHEBI_ID)].CHEBI_ID.unique()
        in_compounds = self.positive_dataset[self.positive_dataset["CHEBI_ID"].isin(final_negative_cases_2.CHEBI_ID)].CHEBI_ID.unique()

        negative_cases_leftout_compounds = []
        bar = tqdm(total=len(leftout_compounds), desc="Generating negative cases for leftout compounds")
        bar.reset()
        for compound in leftout_compounds:
            bar.update(1)
            negative_cases_ = self.get_negative_cases_for_compounds_left_out(compound, 10)
            negative_cases_leftout_compounds.extend(negative_cases_)

        negative_cases_leftout_compounds = pd.DataFrame(negative_cases_leftout_compounds, columns=["RHEA_ID", "CHEBI_ID", "uniprot_id", "protein_identity_threshold", "compound_similarity"])

        chebi_ids_in_negative_dataset = negative_cases_leftout_compounds.CHEBI_ID.unique()
        chebi_ids_in_negative_dataset = pd.Series(chebi_ids_in_negative_dataset)
        chebi_ids_in_negative_dataset = pd.concat((chebi_ids_in_negative_dataset, pd.Series(in_compounds)))
        leftout_compounds_2 = self.positive_dataset[~self.positive_dataset["CHEBI_ID"].isin(chebi_ids_in_negative_dataset)].RHEA_ID.unique()
        assert leftout_compounds_2.shape[0] == 0

        final_negative_cases_2 = pd.concat((final_negative_cases_2, negative_cases_leftout_compounds))

        rhea_chebi = pd.read_csv("rhea-chebi-smiles.tsv", sep="\t", header=None)
        rhea_chebi.columns = ["CHEBI_ID", "SMILES"]
        rhea_chebi.drop_duplicates(inplace=True)

        df_RHEA_smiles = pd.merge(final_negative_cases_2, rhea_chebi, on = "CHEBI_ID", how = "inner")

        swiss_prot_enzymes = pd.read_csv("swiss_prot_enzymes.csv")
        swiss_prot_enzymes.drop(["name", "EC"], axis=1, inplace=True)
        swiss_prot_enzymes.columns = ["uniprot_id", "sequence"]

        df_RHEA_smiles_sequences = pd.merge(df_RHEA_smiles, swiss_prot_enzymes, on = "uniprot_id", how = "inner")

        df_RHEA_smiles_sequences.to_csv(self.output().path, index=False)

class FinalDatasetAssembler(luigi.Task):

    def requires(self):
        return [NegativeCasesAssembler(), FilterCompounds()]
    
    def output(self):
        return luigi.LocalTarget('RHEA_final_dataset.csv')
    
    def input(self):
        return [luigi.LocalTarget('RHEA_negative_cases_with_sequences_smiles.csv'), luigi.LocalTarget('RHEA_enzyme_compound_pairs_sequence_smiles_filtered.csv')]
    
    def run(self):

        positives = pd.read_csv(self.input()[1].path)
        negatives = pd.read_csv(self.input()[0].path)

        negatives.drop(columns=["protein_identity_threshold", "compound_similarity", "protein_identity_"], inplace=True)
        negatives["interaction"] = 0
        positives["interaction"] = 1
        
        final_dataset = pd.concat([positives, negatives])
        final_dataset.to_csv(self.output().path, index=False)