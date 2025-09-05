
def process_to_spawn(multi_input_dataset):

    from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
    transformer = ESMEncoder(esm_function="esm2_t36_3B_UR50D", batch_size=1, num_gpus=4)

    multi_input_dataset = transformer.fit_transform(multi_input_dataset, "proteins")

    multi_input_dataset.save_features("features_proteins_esm2_3b")

def generate_features_for_compounds(multi_input_dataset):
    from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors

    featurizer = DeepMolDescriptors(preset="np_classifier_fp", n_jobs=10)
    featurizer.fit_transform(multi_input_dataset, "ligands")
    multi_input_dataset.save_features("features_compounds_np_classifier_fp")

if __name__ == "__main__":
    from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
    # load datasets

    multi_input_dataset = MultiInputDataset.from_csv("curated_dataset.csv", representation_field={"ligands": "SMILES", 
                                                                                        "proteins": "Sequence"},
                                                                                        instances_ids_field={"ligands": "Substrate ID", 
                                                                                        "proteins": "Enzyme ID"})


    from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
    from plants_sm.data_standardization.truncation import Truncator
    from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper

    truncator = Truncator(max_length=884)
    protein_standardizer = ProteinStandardizer()
    propythia_descriptors = PropythiaWrapper(preset="all-no-aac")

    multi_input_dataset = protein_standardizer.fit_transform(multi_input_dataset, "proteins")
    multi_input_dataset = truncator.fit_transform(multi_input_dataset, "proteins")

    process_to_spawn(multi_input_dataset)
    generate_features_for_compounds(multi_input_dataset)

    multi_input_dataset = MultiInputDataset.from_csv("curated_dataset.csv", representation_field={"ligands": "SMILES", 
                                                                                        "proteins": "Sequence"},
                                                                                        instances_ids_field={"ligands": "Substrate ID", 
                                                                                        "proteins": "Enzyme ID"})
    multi_input_dataset = protein_standardizer.fit_transform(multi_input_dataset, "proteins")

    multi_input_dataset = propythia_descriptors.fit_transform(multi_input_dataset, "proteins")
    multi_input_dataset.save_features("propythia_descriptors")
    