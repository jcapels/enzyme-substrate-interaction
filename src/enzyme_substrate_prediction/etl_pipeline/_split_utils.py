from enzyme_substrate_prediction.etl_pipeline.cdhit_clusters import ClustersIdentifier
import numpy as np
import pandas as pd

from deepmol.datasets import SmilesDataset

from deepmol.splitters import SimilaritySplitter

def split_protein_clusters(train_frac=0.8, test_frac=0.2, identity_threshold=40, seed=123):

    np.random.seed(seed)
    assert train_frac + test_frac == 1
    clust_ident = ClustersIdentifier.from_files(identity_threshold=identity_threshold, folder="../clusters/", filename='all_sequences')
    clusters = list(clust_ident.cluster_to_members.keys())
    clusters = np.array(clusters)
    train_clusters = np.random.choice(clusters, int(len(clust_ident.cluster_to_members) * train_frac), replace=False)
    
    clusters = np.setdiff1d(clusters, train_clusters)
    test_clusters = np.random.choice(clusters, int(len(clust_ident.cluster_to_members) * test_frac), replace=False)

    train_proteins = []
    test_proteins = []

    for cluster in train_clusters:
        train_proteins.extend(clust_ident.cluster_to_members[cluster].members)

    for cluster in test_clusters:
        test_proteins.extend(clust_ident.cluster_to_members[cluster].members)

    return train_proteins, test_proteins

def split_compounds(unique_compounds, train_frac=0.8, test_frac=0.2, similarity_threshold=0.6):

    smiles_dataset = SmilesDataset(smiles = unique_compounds["SMILES"].values, ids = unique_compounds["CHEBI_ID"].values)

    similarity_splitter = SimilaritySplitter()

    train_compounds, test_dataset = similarity_splitter.train_test_split(smiles_dataset, frac_train=train_frac, homogenous_threshold=similarity_threshold)

    return train_compounds.ids, test_dataset.ids

def get_split(final_dataset, report_path, train_frac=0.8, test_frac=0.2, similarity_threshold=0.6, identity_threshold=40, seed=123, repeat_reactions=False):

    train_proteins, test_proteins = split_protein_clusters(train_frac=train_frac, test_frac=test_frac, identity_threshold=identity_threshold, seed=seed)
    
    unique_compounds = final_dataset[["CHEBI_ID", "SMILES"]].drop_duplicates()

    if similarity_threshold != 1:
        train_compounds, test_compounds = split_compounds(unique_compounds, train_frac=train_frac, test_frac=test_frac, similarity_threshold=similarity_threshold)
    
    train_dataset = final_dataset[final_dataset["uniprot_id"].isin(train_proteins)]
    test_dataset = final_dataset[final_dataset["uniprot_id"].isin(test_proteins)]

    report = open(report_path, "a+")

    report.write(f"Train dataset after protein restriction: {train_dataset.shape}\n")
    report.write(f'Unique compounds {len(train_dataset["CHEBI_ID"].unique())}\n')
    report.write(f'Unique proteins {len(train_dataset["uniprot_id"].unique())}\n')
    report.write("\n")

    report.write(f"Test dataset after protein restriction: {test_dataset.shape}")
    report.write(f'Unique compounds: {len(test_dataset["CHEBI_ID"].unique())}\n')
    report.write(f'Unique proteins {len(test_dataset["uniprot_id"].unique())}\n')
    report.write("\n")

    if similarity_threshold != 1:
        train_dataset = train_dataset[train_dataset["CHEBI_ID"].isin(train_compounds)]
        test_dataset = test_dataset[test_dataset["CHEBI_ID"].isin(test_compounds)]

        report.write(f"Train dataset after compound restriction: {train_dataset.shape}\n")
        report.write(f'Unique compounds {len(train_dataset["CHEBI_ID"].unique())}\n')
        report.write(f'Unique proteins {len(train_dataset["uniprot_id"].unique())}\n')
        report.write("\n")

        report.write(f"Test dataset after compound restriction: {test_dataset.shape}")
        report.write(f'Unique compounds: {len(test_dataset["CHEBI_ID"].unique())}\n')
        report.write(f'Unique proteins {len(test_dataset["uniprot_id"].unique())}\n')
        report.write("\n")

        # to_add_to_the_validation_set = train_dataset[(~train_dataset["CHEBI_ID"].isin(train_compounds)) & (~train_dataset["CHEBI_ID"].isin(test_compounds))]
        # validation_dataset = pd.concat([validation_dataset, to_add_to_the_validation_set])

    # report.write("Validation dataset after compound restriction + leftout entries", validation_dataset.shape)
    # report.write("Unique compounds", len(validation_dataset["CHEBI_ID"].unique()))
    # report.write("Unique proteins", len(validation_dataset["uniprot_id"].unique()))
    # report.write()

    if not repeat_reactions:
        train_dataset = train_dataset[~train_dataset["RHEA_ID"].isin(test_dataset.RHEA_ID.values)]

    report.write(f"Train dataset after reaction restriction: {train_dataset.shape}\n")
    report.write(f'Unique compounds {len(train_dataset["CHEBI_ID"].unique())}\n')
    report.write(f'Unique proteins {len(train_dataset["uniprot_id"].unique())}\n')
    report.write("\n")

    report.write(f"Test dataset after reaction restriction: {test_dataset.shape}\n")
    report.write(f'Unique compounds: {len(test_dataset["CHEBI_ID"].unique())}\n')
    report.write(f'Unique proteins {len(test_dataset["uniprot_id"].unique())}\n')
    report.write("\n")

    train_reactions_set = set(train_dataset["RHEA_ID"].values)
    test_reactions_set = set(test_dataset["RHEA_ID"].values)

    train_compounds_set = set(train_dataset["CHEBI_ID"].values)
    test_compounds_set = set(test_dataset["CHEBI_ID"].values)

    train_protein_set = set(train_dataset["uniprot_id"].values)
    test_protein_set = set(test_dataset["uniprot_id"].values)

    report.write("Intersections between sets\n")

    report.write(f"Intersection between train and test reactions {len(train_reactions_set.intersection(test_reactions_set))}\n")

    report.write(f"Intersection between train and test compounds {len(train_compounds_set.intersection(test_compounds_set))}\n")
          
    report.write(f"Intersection between train and test proteins {len(train_protein_set.intersection(test_protein_set))}\n")

    final_dataset_2 = pd.concat([train_dataset, test_dataset])

    final_dataset_2_reactions_set = set(final_dataset_2["RHEA_ID"].values)
    final_dataset_2_compounds_set = set(final_dataset_2["CHEBI_ID"].values)
    final_dataset_2_protein_set = set(final_dataset_2["uniprot_id"].values)

    final_dataset_reactions_set = set(final_dataset["RHEA_ID"].values)
    final_dataset_compounds_set = set(final_dataset["CHEBI_ID"].values)
    final_dataset_protein_set = set(final_dataset["uniprot_id"].values)

    report.write(f"Difference between final dataset and final dataset 2 reactions: \
          {len(final_dataset_reactions_set.intersection(final_dataset_2_reactions_set))} out of {len(final_dataset_reactions_set)}\n")
    fake_reactions = 0
    for reaction in final_dataset_reactions_set.intersection(final_dataset_2_reactions_set):
        if type(reaction) == str and reaction.startswith("fake_"):
            fake_reactions += 1

    report.write(f"Fake reactions: {fake_reactions}\n")

    report.write(f"Difference between final dataset and final dataset 2 compounds: \
            {len(final_dataset_compounds_set.intersection(final_dataset_2_compounds_set))} out of {len(final_dataset_compounds_set)}\n")
    
    report.write(f"Difference between final dataset and final dataset 2 proteins: \
            {len(final_dataset_protein_set.intersection(final_dataset_2_protein_set))} out of {len(final_dataset_protein_set)}\n")
    
    report.write(f"-------------------------------\n")
    report.write("\n")
    report.close()
    return train_dataset, test_dataset
