import luigi
import pandas as pd


class DatasetsGatherer(luigi.Task):

    def output(self):
        return luigi.LocalTarget('integrated_dataset.csv')
    
    def run(self):
        goldman_data = pd.read_csv("../Goldman_data/goldman_data.csv")
        kroll_data = pd.read_csv("../kroll_data_experiment/kroll_data.csv")
        mou_data = pd.read_csv("../Mou_data/mou_data.csv")
        yang_data = pd.read_csv("../Yang_data/yang_training_set_w_sequences.csv")
        oat_data = pd.read_csv("../Yang_data/oat_data.csv")
        berry_data = pd.read_csv("../Yang_data/berry_data.csv")
        robinson_data = pd.read_csv("../Robinson_data/final_dataset.csv")
        data = pd.concat([goldman_data, kroll_data, mou_data, yang_data, oat_data, berry_data, robinson_data], axis=0)

        data_cleaned = data.loc[:, ~data.isna().any()]
        data_cleaned.drop_duplicates(subset=["Enzyme ID", "Substrate ID"], inplace=True)
        from rdkit import Chem

        smiles = data_cleaned.SMILES.unique()

        # Function to generate InChI keys
        def generate_inchi_key(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                inchi = Chem.MolToInchi(mol)
                inchi_key = Chem.InchiToInchiKey(inchi)
                return inchi_key
            return None

        # Apply the function to the SMILES column
        inchikeys = []

        for smile in smiles:
            inchi_key = generate_inchi_key(smile)
            inchikeys.append(inchi_key)

        # Function to search PubChem using InChI key
        import requests
        from tqdm import tqdm

        def search_pubchem(inchi_key):
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchi_key}/JSON"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                return None

        pubchem_ids = []
        for inchi_key in tqdm(inchikeys):
            info = search_pubchem(inchi_key)
            if info != None:
                cid = info['PC_Compounds'][0]["id"]["id"]["cid"]
                pubchem_ids.append(cid)
            else:
                pubchem_ids.append(None)

        # pubchem_ids_df = pd.DataFrame({"SMILES":smiles, "pubchem_id": pubchem_ids})
        # data_cleaned = pd.merge(data_cleaned, pubchem_ids_df, on="SMILES")
        # data_cleaned.drop(columns="pubchem_id_y", inplace=True)
        # data_cleaned = data_cleaned.rename(columns={"pubchem_id_x": "pubchem_id"})

        data_cleaned.to_csv("integrated_dataset.csv", index=False)