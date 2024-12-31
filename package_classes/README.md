
# Preprocessing Multi-Omics Data and Generating Embeddings at Inference-time

SeNMo is a foundational deep learning model designed to analyze multi-omics data across 33 cancer types. It integrates diverse molecular modalities such as gene expression, miRNA expression, DNA methylation, protein expression, DNA Mutation, and clinical data to predict patient outcomes with high accuracy. This page provides instructions to preprocess these data modalities and generate embeddings using the SeNMo model.

---

## Step 1: Uni-modal Data Pre-processing

Preprocessing each modality individually is crucial to ensure that the data is cleaned, normalized, and structured in a way that preserves its unique characteristics. This step ensures that features from different modalities are properly aligned and contribute meaningfully to the final multi-omics integration and embedding generation. Preprocess individual modalities using the scripts provided for each type of data.

### 1. miRNA Expression
Use [miRNA_preprocess.py](https://github.com/lab-rasool/SeNMo/blob/main/package_classes/miRNA_preprocess.py) to preprocess miRNA expression data. The script standardizes and formats the raw miRNA data for downstream analysis.

**Example command:**
```bash
python miRNA_preprocess.py input_mirna.tsv output_mirna.csv
```

---

### 2. DNA Methylation
Use [DNAmethyl_preprocess.py](https://github.com/lab-rasool/SeNMo/blob/main/package_classes/DNAmethyl_preprocess.py) to preprocess DNA methylation data. The script filters and normalizes methylation profiles from raw input files.

**Example command:**
```bash
python DNAmethyl_preprocess.py methylation450.tsv output_file.csv
```

---

### 3. Gene Expression
Use [GeneExpresn_preprocess.py](https://github.com/lab-rasool/SeNMo/blob/main/package_classes/GeneExpresn_preprocess.py) to preprocess gene expression data. The script converts raw gene expression data into a standardized format compatible with multi-omics integration.

**Example command:**
```bash
python GeneExpresn_preprocess.py gene-expr-RNAhtseq_fpkm.tsv output_file.csv
```

---

### 4. Protein Expression
Use [ProteinExpresn_preprocess.py](https://github.com/lab-rasool/SeNMo/blob/main/package_classes/ProteinExpresn_preprocess.py) to preprocess protein expression data. This script processes RPPA data into a usable format for analysis.

**Example command:**
```bash
python ProteinExpresn_preprocess.py sample_RPPA_data.tsv output_file.csv
```

---

### 5. DNA Mutation
Use [DNAMut_preprocess.py](https://github.com/lab-rasool/SeNMo/blob/main/package_classes/DNAMut_preprocess.py) along with [Hugo_symbols.tsv](https://github.com/lab-rasool/SeNMo/blob/main/package_classes/Hugo_symbols.tsv) to preprocess DNA mutation data. The script maps mutation profiles to a standard gene nomenclature and extracts relevant features.

**Example command:**
```bash
python DNAMut_preprocess.py wxs.aliquot_ensemble_masked.maf Hugo_symbols.tsv output_file.csv
```

---

### 6. Clinical Data
Use [Clinical_preprocess.py](https://github.com/lab-rasool/SeNMo/blob/main/package_classes/Clinical_preprocess.py) to preprocess clinical data. This script processes phenotype files into structured datasets for multi-omics integration.

**Example command:**
```bash
python Clinical_preprocess.py GDC_phenotype.tsv output_file.csv
```

---

## Step 2: Combine Preprocessed Unimodal Data

Use [combine_features.py](https://github.com/lab-rasool/SeNMo/blob/main/package_classes/combine_features.py) to merge preprocessed data from all modalities into a single multi-omics feature file. The script performs checks to ensure proper alignment of samples and consistency in feature formatting, preventing mismatches or missing data during integration. The script ensures proper alignment and formatting of the combined dataset, and ignores any missing data modality so the inference can be performed with missing data modalities.

**Example command:**
```bash
python combine_features.py clinical.csv dnamut.csv protein.csv gene.csv methylation.csv mirna.csv multiomic_features.pkl
```

---

## Step 3: Generate Embeddings

Once the multi-omics features are prepared, use the SeNMo model to generate embeddings and survival hazard scores. For example, the output is a pickle file containing 48-dimensional embedding vector for input sample, corresponding to feature dimensions (e.g., `dim_1`, `dim_2`, ..., `dim_48`), and associated hazard score in the dictionary format: `output['Features', 'hazard score']`. The python file [generate_embeddings.py](https://github.com/lab-rasool/SeNMo/blob/main/generate_embeddings.py) utilizes an ensemble of 10 SeNMo checkpoints to produce a 48-dimensional embedding vector. The pre-trained model checkpoints are available [here](https://huggingface.co/Lab-Rasool/SeNMo/tree/main), and separately in two parts [part-1](https://doi.org/10.5281/zenodo.14219799) and [part-2](https://doi.org/10.5281/zenodo.14286190).

**Example command:**
```bash
python generate_embeddings.py ./checkpoints ./multiomic_features.pkl ./outputs
```

---

Follow these steps sequentially to preprocess the data, combine features, and generate embeddings for your multi-omics analysis. Ensure that the input files match the expected formats for each script.
