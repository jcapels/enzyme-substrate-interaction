
from enzyme_substrate_prediction.generate_features import process_esm2_to_spawn, process_esm1b, generate_features_for_compounds, process_probert

process_esm2_to_spawn("curated_dataset.csv")
process_esm1b("curated_dataset.csv")
generate_features_for_compounds("curated_dataset.csv")
process_probert("curated_dataset.csv")