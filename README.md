# AutomatingAbercrombie
Replication code for "Automating Abercrobie: Machine-learning Trademark Distinctiveness" by Adarsh, Ash, Bechtold, Beebe, &amp; Fromer, Journal of Empirical Legal Studies (2024).

[Zenodo Link](https://doi.org/10.5281/zenodo.13846292) for downloading datasets and trained models. 

### Directory Structure 
```bash
code/
├── appendix/
│   ├── bootstrap/
│   ├── court-decisions/
│   ├── oov/
│   ├── prob-vs-publication-delay/
│   ├── robustness-checks/
│   └── xai/
├── main-paper/
│   ├── data-creation/
│   ├── distilbert/
│   ├── roberta/
│   └── xgboost/
└── requirements.txt

README.md
``` 

## Project Replication Guide

--- 

### 1. Install Requirements

To begin, please install the necessary Python packages by running the following command:

```bash
pip install -r requirements.txt
```

---

### 2. Data Creation

Follow the steps below to generate the training data from the case files dataset:

1. **Download the Dataset**: 
- Download the **Trademark Case Files Dataset** from [USPTO](https://www.uspto.gov/ip-policy/economic-research/research-datasets/trademark-case-files-dataset).

2. **Data Segmentation (2012-2019)**:    
- Run the following script to extract data from the 2012-2019 timeframe:
```bash
python data-creation/data_segmentation_2012_2019.py
```

3. **Data Preparation**:      
- Prepare the extracted case files with basic preprocessing and features required from case files:
```bash
python data-creation/data_prep.py
```

4. **Add WordNet Indicator**:    
- Extract marks in the dataset that are present in WordNet by running:
```bash
python data-creation/add_wordnet_indicator.py
```

5. **Extract Pseudo Marks**:     
- Extract pseudo marks (if available) from the data using:
```bash
python data-creation/statement_proc_pseudo_mark.py
```

6. **Append English Translation**:    
- Translate foreign language marks into English
```bash
python data-creation/append_translation.py
```

---

### 3. Model Training

1. **Prepare Model Input Data**: 
    - For distilbert and roberta models: 
    ```bash
    python model_directory/{model}_data_prep.py
    ```
    - For XGBoost: 
    ```bash
    python xgboost/gen_fasttext_emb.py
    ```

2. **Training**:
    - For distilbert and roberta models: 
    ```bash
    python model_directory/{model}.py
    ```
    - For XGBoost: 
    ```bash
    python xgboost/xgb_ft.py
    ```

3. **Evaluate Model Performance**:
    - Evaluate the trained model using:
    ```bash
    python model_directory/{model}_report.py
    ```

Replace `{model}` with the specific model name (e.g., `distilbert`, `roberta`, `xgboost`).

---

### 4. Explainable AI Analysis

1. **SHAP and BERTViz Analysis**:
- Run the following Jupyter notebook to reproduce SHAP examples as presented in the main text of the paper:
```bash
xai/shap_bertviz_examples.ipynb
```

2. **Top Mark-Word Attributions**:
- Run the following Jupyter notebook for extracting top mark words contributing towards SHAP scores for NICE Classes:
```bash
xai/word_attributions_nice_classes_shap.ipynb
```

---

#### Additional Notes:
- main-paper directory consists of code used for generating results in the main text of the paper, while appendix directory consists of code used for generating results in the appendix section. 
- Trained models could be downlaoded from [Zenodo](https://doi.org/10.5281/zenodo.13846292)
- df_distilbert_input_unprocessed.pkl and data/distilbert/df_bert_input.pkl have been used interchangably. This corresponds to the dataframe after Step 3.1 
