import os
import pandas as pd
import subprocess
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import BulkTanimotoSimilarity

from plants_sm.io.pickle import read_pickle

RDLogger.DisableLog('rdApp.*')

SEQUENCE_COL = "Sequence"
SMILES_COL = "SMILES"

ENZYME_ID_COL = "Enzyme ID"
SUBSTRATE_ID_COL = "Substrate ID"

database_dir = "database"
results_dir = "results"
os.makedirs(database_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def get_unique_sequences(csv_path, col):
    df = pd.read_csv(csv_path)
    return df[[col]].drop_duplicates().reset_index(drop=True)

def save_fasta(df, col, fasta_path, prefix):
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(df[col]):
            f.write(f">{prefix}_{i}\n{seq}\n")
    print(f"FASTA created: {fasta_path} ({len(df)} sequences)")

def make_blast_db(fasta, db_prefix):
    cmd = ["makeblastdb", "-in", fasta, "-dbtype", "prot", "-out", db_prefix]
    subprocess.run(cmd, check=True)
    print(f" BLAST DB created: {db_prefix}")

def run_blast(query_fasta, db_prefix, output_tsv):
    cmd = [
        "blastp",
        "-query", query_fasta,
        "-db", db_prefix,
        "-out", output_tsv,
        "-outfmt", "6 qseqid sseqid pident evalue bitscore",
        "-num_threads", "4"
    ]
    subprocess.run(cmd, check=True)
    print(f"BLAST was executed: {output_tsv}")

    try:
        df = pd.read_csv(output_tsv, sep="\t",
                         names=["query_id", "subject_id", "identity", "evalue", "bitscore"])
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["query_id", "subject_id", "identity", "evalue", "bitscore"])

    return df

def get_unique_chemicals(df, column="SMILES"):
    return df[column].dropna().unique().tolist()


def generate_fingerprints(smiles_list, radius=2, nBits=1024):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits))
        else:
            fps.append(None)
    return fps

def compute_similarity(test_fps, train_fps):
    matrix = []
    valid_train_fps = [fp for fp in train_fps if fp is not None]

    valid_train_indices = [i for i, fp in enumerate(train_fps) if fp is not None]

    for tfp in test_fps:
        if tfp is None:
            matrix.append([0.0] * len(valid_train_fps))
        else:
            similarities = BulkTanimotoSimilarity(tfp, valid_train_fps)
            full_scores = [0.0] * len(train_fps)
            for i, score in zip(valid_train_indices, similarities):
                full_scores[i] = score

            matrix.append(full_scores)

    return matrix

def get_similarity_results(splits_pickle_file, compounds = True, 
                           similarity=20, 
                           dataset_csv="curated_dataset_no_stereochemistry_duplicates.csv",
                           results_dir="results"):

    dataset_df = pd.read_csv(dataset_csv)

    os.makedirs(results_dir, exist_ok=True)

    from plants_sm.io.pickle import read_pickle

    splits = read_pickle(splits_pickle_file)

    from plants_sm.io.pickle import read_pickle

    splits = pd.read_pickle(splits_pickle_file)
    train_indices, val_indices, test_indices = splits[0]

    if not isinstance(train_indices, set):
        train_indices = set(train_indices)
        val_indices = set(val_indices)
        test_indices = set(test_indices)

    train_indices = train_indices.union(val_indices)
    if compounds:
        train_df = dataset_df[dataset_df[SUBSTRATE_ID_COL].isin(train_indices)]
        test_df = dataset_df[dataset_df[SUBSTRATE_ID_COL].isin(test_indices)]
    else:
        train_df = dataset_df[dataset_df[ENZYME_ID_COL].isin(train_indices)]
        test_df = dataset_df[dataset_df[ENZYME_ID_COL].isin(test_indices)]

    train_seq_unique = train_df[[ENZYME_ID_COL, SEQUENCE_COL]].drop_duplicates().reset_index(drop=True)
    test_seq_unique = test_df[[ENZYME_ID_COL, SEQUENCE_COL]].drop_duplicates().reset_index(drop=True)
    # train_seq_unique[ENZYME_ID_COL] = [f"E_train_{i}" for i in train_seq_unique.index]
    # test_seq_unique[ENZYME_ID_COL] = [f"E_test_{i}" for i in test_seq_unique.index]

    # train_id_to_seq = train_seq_unique.set_index(ENZYME_ID_COL)[SEQUENCE_COL].to_dict()
    train_seq_to_id = train_seq_unique.set_index(SEQUENCE_COL)[ENZYME_ID_COL].to_dict()

    # test_id_to_seq = test_seq_unique.set_index(ENZYME_ID_COL)[SEQUENCE_COL].to_dict()
    test_seq_to_id = test_seq_unique.set_index(SEQUENCE_COL)[ENZYME_ID_COL].to_dict()

    # train_fasta_map = train_seq_unique.set_index(SEQUENCE_COL)[ENZYME_ID_COL].to_dict()
    # test_fasta_map = test_seq_unique.set_index(SEQUENCE_COL)[ENZYME_ID_COL].to_dict()

    train_fasta = os.path.join(database_dir, "train_unique.fasta")
    test_fasta = os.path.join(database_dir, "test_unique.fasta")


    def save_fasta_with_id(df, seq_col, id_col, fasta_path):
        os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
        with open(fasta_path, "w") as f:
            for seq, seq_id in zip(df[seq_col], df[id_col]):
                f.write(f">{seq_id}\n{seq}\n")
        print(f"FASTA created: {fasta_path} ({len(df)} sequences)")


    save_fasta_with_id(train_seq_unique, SEQUENCE_COL, ENZYME_ID_COL, train_fasta)
    save_fasta_with_id(test_seq_unique, SEQUENCE_COL, ENZYME_ID_COL, test_fasta)

    blast_db_prefix = os.path.join(database_dir, "train_db")
    make_blast_db(train_fasta, blast_db_prefix)

    blast_output = os.path.join(results_dir, "blast_results.tsv")
    blast_df = run_blast(test_fasta, blast_db_prefix, blast_output)

    train_smiles_unique = train_df[[SUBSTRATE_ID_COL, SMILES_COL]].drop_duplicates().reset_index(drop=True)
    test_smiles_unique = test_df[[SUBSTRATE_ID_COL, SMILES_COL]].drop_duplicates().reset_index(drop=True)

    # train_smiles_unique[SUBSTRATE_ID_COL] = [f"S_train_{i}" for i in train_smiles_unique.index]
    # test_smiles_unique[SUBSTRATE_ID_COL] = [f"S_test_{i}" for i in test_smiles_unique.index]

    train_smiles_to_id = train_smiles_unique.set_index(SMILES_COL)[SUBSTRATE_ID_COL].to_dict()
    train_id_to_smiles = train_smiles_unique.set_index(SUBSTRATE_ID_COL)[SMILES_COL].to_dict()
    test_smiles_to_id = test_smiles_unique.set_index(SMILES_COL)[SUBSTRATE_ID_COL].to_dict()

    train_smiles_list = train_smiles_unique[SMILES_COL].tolist()
    test_smiles_list = test_smiles_unique[SMILES_COL].tolist()

    train_fps = generate_fingerprints(train_smiles_list)
    test_fps = generate_fingerprints(test_smiles_list)

    similarity_matrix = compute_similarity(test_fps, train_fps)

    similarity_df = pd.DataFrame(similarity_matrix,
                                index=test_smiles_list,
                                columns=train_smiles_list)

    similarity_df = similarity_df[~similarity_df.index.duplicated(keep="first")]

    train_pairs_set = set(zip(
        train_df[SEQUENCE_COL].apply(lambda x: train_seq_to_id.get(x)),
        train_df[SMILES_COL].apply(lambda x: train_smiles_to_id.get(x))
    ))

    final_results = []

    for _, test_row in test_df.iterrows():

        test_seq_original = test_row[SEQUENCE_COL]
        test_ligand = test_row[SMILES_COL]
        test_enzyme_id = test_seq_to_id.get(test_seq_original)
        test_substrate_id = test_smiles_to_id.get(test_ligand)


        hits = blast_df[blast_df["query_id"] == test_enzyme_id].sort_values("evalue", ascending=True)

        blast_train_ids = hits["subject_id"].tolist()
        blast_scores = hits["evalue"].tolist()

        chem_train_smiles = []
        chem_scores_values = []

        if test_ligand in similarity_df.index:
            row = similarity_df.loc[test_ligand]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]


            chem_scores = row.sort_values(ascending=False)

            chem_train_smiles_original = chem_scores.index.tolist()
            chem_scores_values = chem_scores.values.tolist()

            chem_train_ids = [train_smiles_to_id.get(s, None) for s in chem_train_smiles_original]
        else:
            chem_train_ids = []

        max_pairs = min(len(blast_train_ids), len(chem_train_ids))

        for i in range(max_pairs):
            enzyme_id_train = blast_train_ids[i]
            substrate_id_train = chem_train_ids[i]

            score_seq = blast_scores[i]
            score_comp = chem_scores_values[i]

            match = 1 if (enzyme_id_train, substrate_id_train) in train_pairs_set else 0

            final_results.append({
                "Test_Enzyme_ID": test_enzyme_id,
                "Test_Substrate_ID": test_substrate_id,
                "Train_Enzyme_ID_Hit": enzyme_id_train,
                "Train_Substrate_ID_Hit": substrate_id_train,
                "evalue": score_seq,
                "Tanimoto": score_comp,
                "Match_Found_in_Train": match
            })

    final_df = pd.DataFrame(final_results)
    final_df = final_df.drop_duplicates(
        subset=["Test_Enzyme_ID", "Test_Substrate_ID", "Train_Enzyme_ID_Hit", "Train_Substrate_ID_Hit"],
        keep="first"
    )

    final_excel = os.path.join(results_dir, "test_dataset_final.csv")
    final_df.drop_duplicates(subset=["Test_Enzyme_ID", "Test_Substrate_ID"],inplace=True)
    final_df.to_csv(final_excel, index=False)

    total_matches = final_df["Match_Found_in_Train"].sum()

    print("---")
    print(f"Final Excel created: {final_excel} ({len(final_df)} lines)")
    print(f"Total de Matches (Match_Found_in_Train = 1): **{int(total_matches)}**")
    print("---")

    results = pd.read_csv(os.path.join(results_dir, "test_dataset_final.csv"))
    # Set the two columns as a multi-index for both DataFrames
    df_reference = test_df.set_index(['Enzyme ID', 'Substrate ID'])
    df_target = results.set_index(['Test_Enzyme_ID', 'Test_Substrate_ID'])

    # Reindex df_target to match df_reference's index
    df_target_reordered = df_target.reindex(df_reference.index)

    # Reset index if needed
    df_target_reordered = df_target_reordered.reset_index()
    df_target_reordered = df_target.reindex(df_reference.index).fillna(0).reset_index()

    y_true = df_reference['Binding'].values
    y_pred = df_target_reordered['Match_Found_in_Train'].values

    from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    if compounds:
        similarity_type = "Compound_Similarity"
    else:
        similarity_type = "Enzyme_Identity"

    if os.path.exists(f"{results_dir}/similarity_evaluation_results.csv"):
        results = pd.read_csv(f"{results_dir}/similarity_evaluation_results.csv")
    else:
        
        results = pd.DataFrame(columns=[
            similarity_type, "MCC", "Accuracy", "Precision", "Recall", "F1_Score"
        ])
    new_row = {
        similarity_type: similarity,
        "MCC": mcc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }
    results = results.append(new_row, ignore_index=True)
    results.to_csv(f"{results_dir}/similarity_evaluation_results.csv", index=False)


def get_similarity_results_compound_classes(splits_pickle_file, pathway_class_compounds, compounds = True, 
                           similarity=20,
                           dataset_csv="curated_dataset_no_stereochemistry_duplicates.csv",
                           results_dir="results"):

    dataset_df = pd.read_csv(dataset_csv)

    os.makedirs(results_dir, exist_ok=True)

    from plants_sm.io.pickle import read_pickle

    splits = read_pickle(splits_pickle_file)

    from plants_sm.io.pickle import read_pickle

    splits = pd.read_pickle(splits_pickle_file)
    train_indices, val_indices, test_indices = splits[0]

    if not isinstance(train_indices, set):
        train_indices = set(train_indices)
        val_indices = set(val_indices)
        test_indices = set(test_indices)

    train_indices = train_indices.union(val_indices)
    if compounds:
        train_df = dataset_df[dataset_df[SUBSTRATE_ID_COL].isin(train_indices)]
        test_df = dataset_df[dataset_df[SUBSTRATE_ID_COL].isin(test_indices)]
    else:
        train_df = dataset_df[dataset_df[ENZYME_ID_COL].isin(train_indices)]
        test_df = dataset_df[dataset_df[ENZYME_ID_COL].isin(test_indices)]

    train_seq_unique = train_df[[ENZYME_ID_COL, SEQUENCE_COL]].drop_duplicates().reset_index(drop=True)
    test_seq_unique = test_df[[ENZYME_ID_COL, SEQUENCE_COL]].drop_duplicates().reset_index(drop=True)
    # train_seq_unique[ENZYME_ID_COL] = [f"E_train_{i}" for i in train_seq_unique.index]
    # test_seq_unique[ENZYME_ID_COL] = [f"E_test_{i}" for i in test_seq_unique.index]

    # train_id_to_seq = train_seq_unique.set_index(ENZYME_ID_COL)[SEQUENCE_COL].to_dict()
    train_seq_to_id = train_seq_unique.set_index(SEQUENCE_COL)[ENZYME_ID_COL].to_dict()

    # test_id_to_seq = test_seq_unique.set_index(ENZYME_ID_COL)[SEQUENCE_COL].to_dict()
    test_seq_to_id = test_seq_unique.set_index(SEQUENCE_COL)[ENZYME_ID_COL].to_dict()

    # train_fasta_map = train_seq_unique.set_index(SEQUENCE_COL)[ENZYME_ID_COL].to_dict()
    # test_fasta_map = test_seq_unique.set_index(SEQUENCE_COL)[ENZYME_ID_COL].to_dict()

    train_fasta = os.path.join(database_dir, "train_unique.fasta")
    test_fasta = os.path.join(database_dir, "test_unique.fasta")


    def save_fasta_with_id(df, seq_col, id_col, fasta_path):
        os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
        with open(fasta_path, "w") as f:
            for seq, seq_id in zip(df[seq_col], df[id_col]):
                f.write(f">{seq_id}\n{seq}\n")
        print(f"FASTA created: {fasta_path} ({len(df)} sequences)")


    save_fasta_with_id(train_seq_unique, SEQUENCE_COL, ENZYME_ID_COL, train_fasta)
    save_fasta_with_id(test_seq_unique, SEQUENCE_COL, ENZYME_ID_COL, test_fasta)

    blast_db_prefix = os.path.join(database_dir, "train_db")
    make_blast_db(train_fasta, blast_db_prefix)

    blast_output = os.path.join(results_dir, "blast_results.tsv")
    blast_df = run_blast(test_fasta, blast_db_prefix, blast_output)

    train_smiles_unique = train_df[[SUBSTRATE_ID_COL, SMILES_COL]].drop_duplicates().reset_index(drop=True)
    test_smiles_unique = test_df[[SUBSTRATE_ID_COL, SMILES_COL]].drop_duplicates().reset_index(drop=True)

    # train_smiles_unique[SUBSTRATE_ID_COL] = [f"S_train_{i}" for i in train_smiles_unique.index]
    # test_smiles_unique[SUBSTRATE_ID_COL] = [f"S_test_{i}" for i in test_smiles_unique.index]

    train_smiles_to_id = train_smiles_unique.set_index(SMILES_COL)[SUBSTRATE_ID_COL].to_dict()
    train_id_to_smiles = train_smiles_unique.set_index(SUBSTRATE_ID_COL)[SMILES_COL].to_dict()
    test_smiles_to_id = test_smiles_unique.set_index(SMILES_COL)[SUBSTRATE_ID_COL].to_dict()

    train_smiles_list = train_smiles_unique[SMILES_COL].tolist()
    test_smiles_list = test_smiles_unique[SMILES_COL].tolist()

    train_fps = generate_fingerprints(train_smiles_list)
    test_fps = generate_fingerprints(test_smiles_list)

    similarity_matrix = compute_similarity(test_fps, train_fps)

    similarity_df = pd.DataFrame(similarity_matrix,
                                index=test_smiles_list,
                                columns=train_smiles_list)

    similarity_df = similarity_df[~similarity_df.index.duplicated(keep="first")]

    train_pairs_set = set(zip(
        train_df[SEQUENCE_COL].apply(lambda x: train_seq_to_id.get(x)),
        train_df[SMILES_COL].apply(lambda x: train_smiles_to_id.get(x))
    ))

    final_results = []

    for _, test_row in test_df.iterrows():

        test_seq_original = test_row[SEQUENCE_COL]
        test_ligand = test_row[SMILES_COL]
        test_enzyme_id = test_seq_to_id.get(test_seq_original)
        test_substrate_id = test_smiles_to_id.get(test_ligand)


        hits = blast_df[blast_df["query_id"] == test_enzyme_id].sort_values("evalue", ascending=True)

        blast_train_ids = hits["subject_id"].tolist()
        blast_scores = hits["evalue"].tolist()

        chem_train_smiles = []
        chem_scores_values = []

        if test_ligand in similarity_df.index:
            row = similarity_df.loc[test_ligand]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]


            chem_scores = row.sort_values(ascending=False)

            chem_train_smiles_original = chem_scores.index.tolist()
            chem_scores_values = chem_scores.values.tolist()

            chem_train_ids = [train_smiles_to_id.get(s, None) for s in chem_train_smiles_original]
        else:
            chem_train_ids = []

        max_pairs = min(len(blast_train_ids), len(chem_train_ids))

        for i in range(max_pairs):
            enzyme_id_train = blast_train_ids[i]
            substrate_id_train = chem_train_ids[i]

            score_seq = blast_scores[i]
            score_comp = chem_scores_values[i]

            match = 1 if (enzyme_id_train, substrate_id_train) in train_pairs_set else 0

            final_results.append({
                "Test_Enzyme_ID": test_enzyme_id,
                "Test_Substrate_ID": test_substrate_id,
                "Train_Enzyme_ID_Hit": enzyme_id_train,
                "Train_Substrate_ID_Hit": substrate_id_train,
                "evalue": score_seq,
                "Tanimoto": score_comp,
                "Match_Found_in_Train": match
            })

    final_df = pd.DataFrame(final_results)
    final_df = final_df.drop_duplicates(
        subset=["Test_Enzyme_ID", "Test_Substrate_ID", "Train_Enzyme_ID_Hit", "Train_Substrate_ID_Hit"],
        keep="first"
    )

    final_excel = os.path.join(results_dir, "test_dataset_final.csv")
    final_df.drop_duplicates(subset=["Test_Enzyme_ID", "Test_Substrate_ID"],inplace=True)
    final_df.to_csv(final_excel, index=False)

    total_matches = final_df["Match_Found_in_Train"].sum()

    print("---")
    print(f"Final Excel created: {final_excel} ({len(final_df)} lines)")
    print(f"Total de Matches (Match_Found_in_Train = 1): **{int(total_matches)}**")
    print("---")

    for class_ in pathway_class_compounds:
        results = pd.read_csv(os.path.join(results_dir, "test_dataset_final.csv"))
        df_reference_class_ = test_df[test_df["Substrate ID"].isin(pathway_class_compounds[class_])]
        df_target_reordered_class_ = results[results["Test_Substrate_ID"].isin(pathway_class_compounds[class_])]

        # Set the two columns as a multi-index for both DataFrames
        df_reference = df_reference_class_.set_index(['Enzyme ID', 'Substrate ID'])
        df_target = df_target_reordered_class_.set_index(['Test_Enzyme_ID', 'Test_Substrate_ID'])

        # Reindex df_target to match df_reference's index
        df_target_reordered = df_target.reindex(df_reference.index)

        # Reset index if needed
        df_target_reordered = df_target_reordered.reset_index()
        df_target_reordered = df_target.reindex(df_reference.index).fillna(0).reset_index()

        y_true = df_reference_class_['Binding'].values
        y_pred = df_target_reordered_class_['Match_Found_in_Train'].values

        from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
        mcc = matthews_corrcoef(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        similarity_type = "Compound_Similarity"

        if os.path.exists(f"{results_dir}/similarity_evaluation_results_{class_}.csv"):
            results = pd.read_csv(f"{results_dir}/similarity_evaluation_results_{class_}.csv")
        else:
            
            results = pd.DataFrame(columns=[
                similarity_type, "mcc", "accuracy", "precision", "recall", "f1_score"
            ])
        new_row = {
            similarity_type: similarity,
            "mcc": mcc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        results = results.append(new_row, ignore_index=True)
        results.to_csv(f"{results_dir}/similarity_evaluation_results_{class_}.csv", index=False)

def get_results_for_stereo():
    results_dir = "results_stereo"
    dataset_csv = "curated_dataset.csv"
    get_similarity_results("splits_compounds_02.pkl", compounds=True, similarity=20, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_compounds_04_corrected.pkl", compounds=True, similarity=40, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_compounds_06.pkl", compounds=True, similarity=60, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_compounds_08.pkl", compounds=True, similarity=80, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_0_8_proteins_train_val_test.pkl", compounds=False, similarity=80, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_0_6_proteins_train_val_test.pkl", compounds=False, similarity=60, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_0_4_proteins_train_val_test.pkl", compounds=False, similarity=40, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_0_2_proteins_train_val_test.pkl", compounds=False, similarity=20, dataset_csv=dataset_csv, results_dir=results_dir)


def get_results_for_no_stereo():
    results_dir = "results_no_stereo"
    dataset_csv = "curated_dataset_no_stereochemistry_duplicates.csv"
    get_similarity_results("splits_compounds_02.pkl", compounds=True, similarity=20, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_compounds_04_corrected.pkl", compounds=True, similarity=40, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_compounds_06.pkl", compounds=True, similarity=60, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_compounds_08.pkl", compounds=True, similarity=80, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_0_8_proteins_train_val_test.pkl", compounds=False, similarity=80, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_0_6_proteins_train_val_test.pkl", compounds=False, similarity=60, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_0_4_proteins_train_val_test.pkl", compounds=False, similarity=40, dataset_csv=dataset_csv, results_dir=results_dir)
    get_similarity_results("splits_0_2_proteins_train_val_test.pkl", compounds=False, similarity=20, dataset_csv=dataset_csv, results_dir=results_dir)

def pathways_to_compounds():
    results_dir = "results_stereo"
    dataset_csv = "curated_dataset.csv"

    for similarity, string in [(20, "02"), (40, "04"), (60, "06"), (80, "08")]:

        pathways_to_compounds = read_pickle(f"pathway_to_compounds_split_{string}.pkl")

        get_similarity_results_compound_classes(f"splits_compounds_{string}.pkl", compounds=True, similarity=similarity, dataset_csv=dataset_csv, results_dir=results_dir, pathway_class_compounds=pathways_to_compounds)


if __name__ == "__main__":
    # get_results_for_no_stereo()
    # get_results_for_stereo()
    pathways_to_compounds()