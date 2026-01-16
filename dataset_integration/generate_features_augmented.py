
def process_to_spawn(multi_input_dataset):

    from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
    transformer = ESMEncoder(esm_function="esm2_t36_3B_UR50D", batch_size=1, num_gpus=4)

    multi_input_dataset = transformer.fit_transform(multi_input_dataset, "proteins")

    multi_input_dataset.save_features("features_proteins_esm2_3b_augmented")

def process_probert(multi_input_dataset):
    from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert
    transformer = ProtBert(batch_size=1, device="cuda:3")

    multi_input_dataset = transformer.fit_transform(multi_input_dataset, "proteins")

    multi_input_dataset.save_features("features_proteins_probert_augmented")

def process_esm1b(multi_input_dataset):
    from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
    transformer = ESMEncoder(esm_function="esm1b_t33_650M_UR50S", batch_size=1, device="cuda:3")

    multi_input_dataset = transformer.fit_transform(multi_input_dataset, "proteins")

    multi_input_dataset.save_features("features_proteins_esm1b_augmented")

def generate_features_for_compounds(multi_input_dataset):
    from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors

    featurizer = DeepMolDescriptors(preset="np_classifier_fp", n_jobs=10, kwargs={"useChirality": False})
    featurizer.fit_transform(multi_input_dataset, "ligands")
    multi_input_dataset.save_features("features_compounds_np_classifier_fp_augmented")

if __name__ == "__main__":
    from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
    # load datasets

    multi_input_dataset = MultiInputDataset.from_csv("augmented_dataset_descriptors_available.csv", representation_field={"ligands": "SMILES", 
                                                                                        "proteins": "Sequence"},
                                                                                        instances_ids_field={"ligands": "Substrate ID", 
                                                                                        "proteins": "Enzyme ID"})


    from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
    from plants_sm.data_standardization.truncation import Truncator
    from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper

    truncator = Truncator(max_length=884)
    protein_standardizer = ProteinStandardizer()

    # multi_input_dataset = protein_standardizer.fit_transform(multi_input_dataset, "proteins")
    # multi_input_dataset = truncator.fit_transform(multi_input_dataset, "proteins")

    # process_to_spawn(multi_input_dataset)
    # process_probert(multi_input_dataset)
    # process_esm1b(multi_input_dataset)
    generate_features_for_compounds(multi_input_dataset)

    # multi_input_dataset = MultiInputDataset.from_csv("curated_dataset.csv", representation_field={"ligands": "SMILES", 
    #                                                                                     "proteins": "Sequence"},
    #                                                                                     instances_ids_field={"ligands": "Substrate ID", 
    #                                                                                     "proteins": "Enzyme ID"})
    # multi_input_dataset = protein_standardizer.fit_transform(multi_input_dataset, "proteins")

    # multi_input_dataset = propythia_descriptors.fit_transform(multi_input_dataset, "proteins")
    # multi_input_dataset.save_features("propythia_descriptors")
    