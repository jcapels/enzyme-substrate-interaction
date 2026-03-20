import luigi

from enzyme_substrate_prediction.datasets_integration.datasets_gather import DatasetsGatherer
from enzyme_substrate_prediction.datasets_integration.filter_by_enzymes import EnzymeFilter, Merger

import pandas as pd

import os
from deepmol.loaders import CSVLoader
from deepmol.compound_featurization import ThreeDimensionalMoleculeGenerator

from deepmol.loaders import SDFLoader
from deepmol.compound_featurization import All3DDescriptors, MixedFeaturizer, TwoDimensionDescriptors
from plants_sm.io.pickle import write_pickle

from enzyme_substrate_prediction.datasets_integration.merge_with_augmented_data import csv_to_fasta


class DataAugmenterAnd3DGenerator(luigi.Task):

    def requires(self):
        return Merger()

    def run(self):

        dataset = pd.read_csv("augmented_dataset.csv")

        dataset.drop_duplicates(subset="SMILES").to_csv("unique_compounds.csv", index=False)

        # Processing parameters
        timeout = 200
        threads = 50
        n_conformations = 1
        max_iterations = 100
        etkdg_version = 3
        mode = "MMFF94"

        dataset = CSVLoader("unique_compounds.csv", id_field="Substrate ID", smiles_field="SMILES").create_dataset()
                    
        # Generate 3D conformers
        generator = ThreeDimensionalMoleculeGenerator(
            timeout_per_molecule=timeout, threads=threads,
            n_conformations=n_conformations, max_iterations=max_iterations
        )
        generator.generate(dataset, etkdg_version=etkdg_version, mode=mode)

        # Save as SDF
        output_sdf_path = os.path.join("unique_compounds_conformers.sdf")
        dataset.to_sdf(output_sdf_path)

        from deepmol.loaders._utils import load_sdf_file

        mols = load_sdf_file("unique_compounds_conformers.sdf")
        ids = []
        for mol in mols:
            id_ = mol.GetProp("_ID")
            ids.append(id_)

        df = pd.DataFrame({"ids":ids, "mols": mols})
        df = df.drop_duplicates(subset="ids")
        from deepmol.datasets import SmilesDataset

        dataset = SmilesDataset(smiles=df.mols, mols=df.mols, ids=df.ids)
        dataset.to_sdf("unique_compounds_conformers.sdf")

        compounds_dataset = SDFLoader("unique_compounds_conformers.sdf", id_field="_ID").create_dataset()

        MixedFeaturizer([All3DDescriptors(mandatory_generation_of_conformers=False), TwoDimensionDescriptors()]).featurize(compounds_dataset, inplace=True) 

        compounds_dataset.to_sdf(path="unique_compounds_with_features.sdf")

        dataset = pd.read_csv("augmented_dataset.csv")

        dataset[dataset["Substrate ID"].isin(compounds_dataset.ids)].to_csv("augmented_dataset_descriptors_available.csv", index=False)
        compounds_dataset.to_csv("unique_compounds_with_features.csv", index=False)


        compounds_with_features = pd.read_csv("unique_compounds_with_features.csv")
        features_dict = compounds_with_features.set_index('ids').drop(columns='smiles').apply(lambda row: row.to_numpy(), axis=1).to_dict()
        write_pickle("compounds_features.pkl", features_dict)

        dataset = pd.read_csv("augmented_dataset_descriptors_available.csv")
        dataset["Enzyme ID"] = dataset["Enzyme ID"].str.replace(" ", "_")
        dataset.to_csv("augmented_dataset_descriptors_available.csv", index=False)

        dataset.drop_duplicates(subset=["Enzyme ID"]).loc[:, ["Sequence", "Enzyme ID"]].to_csv("unique_enzymes_augmented.csv", index=False)

        csv_file_path = 'unique_enzymes_augmented.csv'  # Path to your CSV file
        fasta_file_path = 'unique_enzymes_augmented.fasta'  # Path to save the FASTA file
        csv_to_fasta(csv_file_path, fasta_file_path)

        dataset[dataset["Validated"]==True].drop_duplicates(subset=["Enzyme ID"]).loc[:, ["Sequence", "Enzyme ID"]].to_csv("unique_enzymes_curated.csv", index=False)
        fasta_file_path = 'unique_enzymes_curated.fasta'
        csv_file_path = 'unique_enzymes_curated.csv'
        csv_to_fasta(csv_file_path, fasta_file_path)

        dataset[dataset["Validated"]==True].to_csv("curated_dataset.csv", index=False)


class DataFilter(luigi.Task):

    def requires(self):
        return DatasetsGatherer()
    
    def drop_stereochemistry(self, dataset):
        # get the inchikeys
        from rdkit import Chem
        from rdkit.Chem import AllChem

        df_unique_compounds = dataset.drop_duplicates(subset=['SMILES']).loc[:, ["Substrate ID", 'SMILES']].reset_index(drop=True)	

        def smiles_to_inchikey(smiles):
            if pd.isna(smiles):
                return None
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            inchikey = AllChem.MolToInchiKey(mol)
            return inchikey

        df_unique_compounds['inchikey'] = df_unique_compounds['SMILES'].apply(smiles_to_inchikey)
        # separate in different cells for clarity by -
        df_unique_compounds["inchikey_skeleton"] = df_unique_compounds['inchikey'].apply(lambda x: x.split('-')[0] if pd.notna(x) else None)
        df_unique_compounds["inchikey_stereochemistry"] = df_unique_compounds['inchikey'].apply(lambda x: '-'.join(x.split('-')[:2]) if pd.notna(x) else None)
        unique_compounds = df_unique_compounds.drop_duplicates(subset=['inchikey_skeleton'])

        dataset[dataset["Substrate ID"].isin(unique_compounds["Substrate ID"])].to_csv("curated_dataset_no_stereochemistry_duplicates.csv", index=False)


    def run(self):

        dataset = pd.read_csv("integrated_dataset.csv")

        dataset.drop_duplicates(subset="SMILES").to_csv("unique_compounds.csv", index=False)

        # Processing parameters
        timeout = 200
        threads = 50
        n_conformations = 1
        max_iterations = 100
        etkdg_version = 3
        mode = "MMFF94"

        dataset = CSVLoader("unique_compounds.csv", id_field="Substrate ID", smiles_field="SMILES").create_dataset()
                    
        # Generate 3D conformers
        generator = ThreeDimensionalMoleculeGenerator(
            timeout_per_molecule=timeout, threads=threads,
            n_conformations=n_conformations, max_iterations=max_iterations
        )
        generator.generate(dataset, etkdg_version=etkdg_version, mode=mode)

        # Save as SDF
        output_sdf_path = os.path.join("unique_compounds_conformers.sdf")
        dataset.to_sdf(output_sdf_path)

        from deepmol.loaders._utils import load_sdf_file

        mols = load_sdf_file("unique_compounds_conformers.sdf")
        ids = []
        for mol in mols:
            id_ = mol.GetProp("_ID")
            ids.append(id_)

        df = pd.DataFrame({"ids":ids, "mols": mols})
        df = df.drop_duplicates(subset="ids")
        from deepmol.datasets import SmilesDataset

        dataset = SmilesDataset(smiles=df.mols, mols=df.mols, ids=df.ids)
        dataset.to_sdf("unique_compounds_conformers.sdf")

        compounds_dataset = SDFLoader("unique_compounds_conformers.sdf", id_field="_ID").create_dataset()

        MixedFeaturizer([All3DDescriptors(mandatory_generation_of_conformers=False), TwoDimensionDescriptors()]).featurize(compounds_dataset, inplace=True) 

        compounds_dataset.to_sdf(path="unique_compounds_with_features.sdf")

        dataset = pd.read_csv("integrated_dataset.csv")

        dataset[dataset["Substrate ID"].isin(compounds_dataset.ids)].to_csv("dataset_descriptors_available.csv", index=False)
        compounds_dataset.to_csv("unique_compounds_with_features.csv", index=False)


        compounds_with_features = pd.read_csv("unique_compounds_with_features.csv")
        features_dict = compounds_with_features.set_index('ids').drop(columns='smiles').apply(lambda row: row.to_numpy(), axis=1).to_dict()
        write_pickle("compounds_features.pkl", features_dict)

        dataset = pd.read_csv("dataset_descriptors_available.csv")
        dataset["Enzyme ID"] = dataset["Enzyme ID"].str.replace(" ", "_")
        dataset.to_csv("dataset_descriptors_available.csv", index=False)

        dataset.drop_duplicates(subset=["Enzyme ID"]).loc[:, ["Sequence", "Enzyme ID"]].to_csv("unique_enzymes.csv", index=False)

        csv_file_path = 'unique_enzymes.csv'  # Path to your CSV file
        fasta_file_path = 'unique_enzymes.fasta'  # Path to save the FASTA file
        csv_to_fasta(csv_file_path, fasta_file_path)

        dataset.drop_duplicates(subset=["Enzyme ID"]).loc[:, ["Sequence", "Enzyme ID"]].to_csv("unique_enzymes_curated.csv", index=False)
        fasta_file_path = 'unique_enzymes_curated.fasta'
        csv_file_path = 'unique_enzymes_curated.csv'
        csv_to_fasta(csv_file_path, fasta_file_path)

        dataset.to_csv("curated_dataset.csv", index=False)

        self.drop_stereochemistry(dataset)

        

