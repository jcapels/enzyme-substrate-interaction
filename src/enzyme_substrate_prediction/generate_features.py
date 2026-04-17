
import os

import torch
from torch import nn
from plants_sm.models.fc.fc import DNN
from plants_sm.models.pytorch_model import PyTorchModel
from plants_sm.pathway_prediction.ec_numbers_annotator_utils._utils import _download_pipeline_to_cache

def apply_transfer_learning_to_features(features_file, model, out_dir):
    import pandas as pd

    pandas_dataset = pd.read_csv("curated_dataset.csv")
    pandas_dataset.drop_duplicates(subset="Enzyme ID", inplace=True)

    from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset

    dataset = SingleInputDataset(dataframe=pandas_dataset, representation_field="Sequence", instances_ids_field="Enzyme ID")

    dataset.load_features(features_file, "proteins")
    dataset._features["place_holder"] = dataset._features["proteins"]
    embedding = model.get_embeddings(dataset)
    from plants_sm.io.pickle import write_pickle
    import os

    features = {"proteins": {}}

    for ids, emb in zip(dataset.identifiers, embedding):
        features["proteins"][ids] = emb

    os.makedirs(out_dir, exist_ok=True)
    write_pickle(os.path.join(out_dir, "features.pkl"), features)

def apply_transfer_learning_to_features_esm2_3b():
    pipeline_path = _download_pipeline_to_cache("ESM2 pipeline")
    protein_model_ = torch.load(os.path.join(pipeline_path, "esm2_3b.pt"), map_location="cpu")
    protein_model = DNN(2560, [2560], 5743, batch_norm=True, last_sigmoid=True)

    protein_model.load_state_dict(protein_model_)
    model = PyTorchModel(model=protein_model, loss_function=nn.BCELoss, model_name="ec_number", device="cpu")

    apply_transfer_learning_to_features("features_proteins_esm2_3b", model, "esm2_3b_ec_number_embedding")

def apply_transfer_learning_to_features_protbert():
    pipeline_path = _download_pipeline_to_cache("ProtBERT pipeline")
    protein_model_ = torch.load(os.path.join(pipeline_path, "prot_bert.pt"), map_location="cpu")
    protein_model = DNN(1024, [2560], 5743, batch_norm=True, last_sigmoid=True)

    protein_model.load_state_dict(protein_model_)
    model = PyTorchModel(model=protein_model, loss_function=nn.BCELoss, model_name="ec_number", device="cpu")

    apply_transfer_learning_to_features("features_proteins_prot_bert", model, "prot_bert_ec_number_embedding")

def apply_transfer_learning_to_features_esm1b():
    pipeline_path = _download_pipeline_to_cache("ESM1b pipeline")
    protein_model_ = torch.load(os.path.join(pipeline_path, "esm1b.pt"), map_location="cpu")
    protein_model = DNN(1280, [2560, 5120], 5743, batch_norm=True, last_sigmoid=True)

    protein_model.load_state_dict(protein_model_)
    model = PyTorchModel(model=protein_model, loss_function=nn.BCELoss, model_name="ec_number", device="cuda:3")

    apply_transfer_learning_to_features("features_proteins_esm1b", model, "esm1b_ec_number_embedding")


def preprocess_dataset(multi_input_dataset_path):
    from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
    # load datasets

    multi_input_dataset = MultiInputDataset.from_csv(multi_input_dataset_path, representation_field={"ligands": "SMILES", 
                                                                                        "proteins": "Sequence"},
                                                                                        instances_ids_field={"ligands": "Substrate ID", 
                                                                                        "proteins": "Enzyme ID"})


    from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
    from plants_sm.data_standardization.truncation import Truncator

    truncator = Truncator(max_length=884)
    protein_standardizer = ProteinStandardizer()

    multi_input_dataset = protein_standardizer.fit_transform(multi_input_dataset, "proteins")
    multi_input_dataset = truncator.fit_transform(multi_input_dataset, "proteins")
    return multi_input_dataset

def process_esm2_to_spawn(multi_input_dataset_path):

    multi_input_dataset = preprocess_dataset(multi_input_dataset_path)

    from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
    transformer = ESMEncoder(esm_function="esm2_t36_3B_UR50D", batch_size=1, num_gpus=4)

    multi_input_dataset = transformer.fit_transform(multi_input_dataset, "proteins")

    multi_input_dataset.save_features("features_proteins_esm2_3b")
    apply_transfer_learning_to_features_esm2_3b()

def process_probert(multi_input_dataset_path):

    multi_input_dataset = preprocess_dataset(multi_input_dataset_path)

    from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert
    transformer = ProtBert(batch_size=1, device="cuda")

    multi_input_dataset = transformer.fit_transform(multi_input_dataset, "proteins")

    multi_input_dataset.save_features("features_proteins_prot_bert")
    apply_transfer_learning_to_features_protbert()

def process_esm1b(multi_input_dataset_path):
    from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder

    multi_input_dataset = preprocess_dataset(multi_input_dataset_path)

    transformer = ESMEncoder(esm_function="esm1b_t33_650M_UR50S", batch_size=1, device="cuda")

    multi_input_dataset = transformer.fit_transform(multi_input_dataset, "proteins")

    multi_input_dataset.save_features("features_proteins_esm1b")
    apply_transfer_learning_to_features_esm1b()

def generate_features_for_compounds(multi_input_dataset_path):
    multi_input_dataset = preprocess_dataset(multi_input_dataset_path)

    from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors

    featurizer = DeepMolDescriptors(preset="np_classifier_fp", n_jobs=10)
    featurizer.fit_transform(multi_input_dataset, "ligands")
    multi_input_dataset.save_features("features_compounds_np_classifier_fp")
