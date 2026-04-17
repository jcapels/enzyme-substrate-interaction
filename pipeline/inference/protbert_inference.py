from plants_sm.pathway_prediction.esi_annotator import ProtBertESIAnnotator


def test_esi_protbert_inference(data_path):
    annotator = ProtBertESIAnnotator()
    return annotator.annotate_from_file(data_path, "csv")

test_esi_protbert_inference("curated_dataset_test.csv")