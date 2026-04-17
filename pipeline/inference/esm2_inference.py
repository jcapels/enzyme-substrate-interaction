from plants_sm.pathway_prediction.esi_annotator import ESM2ESIAnnotator


def test_esi_esm2_inference(data_path):
    annotator = ESM2ESIAnnotator()
    return annotator.annotate_from_file(data_path, "csv")

test_esi_esm2_inference("curated_dataset_test.csv")