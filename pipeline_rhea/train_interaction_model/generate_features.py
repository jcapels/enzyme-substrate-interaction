
from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_standardization.truncation import Truncator
import os
import pandas as pd

def generate_esm1b_data():

    from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset

    

    if not os.path.exists("RHEA_all_data.csv"):
        multi_input_dataset_csv = "../RHEA_final_dataset.csv"
        random_dataset = "../RHEA_final_dataset_random.csv"
        multi_input_dataset = pd.read_csv(multi_input_dataset_csv)
        random_dataset = pd.read_csv(random_dataset)

        pd.concat([multi_input_dataset, random_dataset]).to_csv("RHEA_all_data.csv", index=False)

        

    multi_input_dataset = MultiInputDataset.from_csv(file_path="RHEA_all_data.csv",
                                                    representation_field={"proteins": "sequence",
                                                                        "ligands": "SMILES",
                                                                        "reactions": "reaction_SMILES"},
                                                    instances_ids_field={"proteins": "uniprot_id",
                                                                        "ligands": "CHEBI_ID",
                                                                        "reactions": "RHEA_ID"},
                                                    labels_field="interaction")
    
    truncator = Truncator(max_length=884)
    protein_standardizer = ProteinStandardizer()
    transformer = ESMEncoder(esm_function="esm1b_t33_650M_UR50S", batch_size=1, device="cuda:0")

    multi_input_dataset = protein_standardizer.fit_transform(multi_input_dataset, "proteins")
    multi_input_dataset = truncator.fit_transform(multi_input_dataset, "proteins")
    multi_input_dataset = transformer.fit_transform(multi_input_dataset, "proteins")

    from plants_sm.featurization.reactions.reaction_bert import ReactionBERT

    ReactionBERT(bert_model_path="/home/jcapela/enzyme-substrate-interaction/pipeline_rhea/train_interaction_model/smiles_reaction_bert_model.pt", device="cuda:0").fit_transform(multi_input_dataset, "reactions")

    multi_input_dataset.save_features("RHEA_all_data_features")

generate_esm1b_data()