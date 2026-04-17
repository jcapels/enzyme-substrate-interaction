from plants_sm.pathway_prediction.esi_annotator import ESM1bESIAnnotator


def test_esi_esm1b_inference(data_path):
    annotator = ESM1bESIAnnotator()
    return annotator.annotate_from_file(data_path, "csv")

if __name__ == "__main__":
    print(test_esi_esm1b_inference("curated_dataset_test.csv"))