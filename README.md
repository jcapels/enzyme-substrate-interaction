# Enzyme-substrate interaction prediction

### Table of contents:

- [Requirements](#requirements)
- [Installation](#installation)
    - [Pip](#pip)
    - [From github](#From-github)
- [Run pipeline to obtain the data](#Run-pipeline-to-obtain-the-data)
- [Extract features](#extract-features)
- [Train models](#Train-models)
  - [Train baselines](#Train-baselines)
  - [Train models](#Train-models)
  - [Train models with both training and validation sets](#Train-models-with-both-training-and-validation-sets)
  - [Train models with the whole data](#Train-models-with-the-whole-data)
- [Predict EC numbers](#predict-ec-numbers)
  - [Predict with model](#predict-with-model)
  - [Predict with BLAST](#predict-with-blast)
  - [Predict with an ensemble of BLAST and DL models](#predict-with-an-ensemble-of-blast-and-dl-models)
- [Data availability](#data-availability)
- [Post analysis - generate results and plots](#post-analysis---generate-results-and-plots)

## Create conda environment
    
```bash
conda create -n esi python=3.11
conda activate esi

# Make sure you add these channels to your conda configuration
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge

conda install bioconda::blast==2.12.0
```

## Installation

### Pip

```bash
pip install enzyme-substrate-interaction
```

### From github

```bash
pip install git+https://github.com/jcapels/enzyme-substrate-interaction.git
```

## Run pipeline to obtain the data

The steps of this pipeline are the following:

1. The data was downloaded from each publication and preprocessed.
2. Filter compounds based on whether a 3D structure and rdkit descriptors can be generated.
3. Drop same isomers. 

To run the data integration pipeline:

```bash
cd pipeline/dataset_integration_pipeline
python pipeline.py
```

## Extract features

The following functions will take the data you have in your working directory (please pass its path as 
input to the following functions) and generate features using ESM, ProtBERT and NPClassifierFP.

You can either run this:

```python
from enzyme_substrate_prediction.generate_features import process_esm2_to_spawn, process_esm1b, generate_features_for_compounds, process_probert

process_esm2_to_spawn("curated_dataset.csv")
process_esm1b("curated_dataset.csv")
generate_features_for_compounds("curated_dataset.csv")
process_probert("curated_dataset.csv")
```

or go to 

```bash
cd pipeline/generate_features
python generate_features.py
```

## Split the data - enzymes

### Prerequisites
- **Docker**: Required to run the pipeline (used in this workflow).
- **Input File**: Ensure the `unique_enzymes.fasta` file is placed in the `docker_volume` folder before proceeding.

---

### Steps to Execute the Split

1. **Navigate to the Pipeline Directory**:
   ```bash
   cd pipeline
   ```

2. **Run the Split Script**:
   ```bash
   sh run_split.sh
   ```

---

#### Notes
- While Docker is not strictly mandatory, it was used in this implementation for consistency and reproducibility.


## Split the data - substrates

Check out the following notebook: **[pipeline/split_data/compounds_split/perform_compounds_split.ipynb](pipeline/split_data/compounds_split/perform_compounds_split.ipynb)**.

## Train models

To train and optimize the models run:

```bash
cd pipeline/train_models.py
python train_models.py
```

## Predict Enzyme-Substrate Specificity



## Efficiency estimation

For the efficiency estimation (for memory and runtime requirements) refer to folder **[efficiency_evaluation](efficiency_evaluation)**.

## All results and plot generation

For the results and plot generation (performance) refer to folder **[results_analysis](results_analysis)**.

## ProSmith and ESP

- ProSmith retraining and evaluation: **[ProSmith](https://github.com/jcapels/ProSmith.git)**

- ESP retraining and evaluation: **[ESP](https://github.com/jcapels/ESP_prediction_function.git)**

