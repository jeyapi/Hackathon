# Technical Documentation

## 1. Overview

This project is a Streamlit application for explainable HR feedback analysis and 9-box talent category prediction.

Main capabilities:
- Rule-based text analytics (sentiment, topics, bias flags, quality scoring)
- Interactive visual dashboards
- DistilBERT-based multi-class classification to predict `nine_box_category`
- Persistent model artifacts (save/load fine-tuned BERT model across restarts)

## 2. Repository Structure

```text
Hackathon/
  app.py
  README.md
  requirements.txt
  TECHNICAL_DOCUMENTATION.md
  dataset/
    employee_review_mturk_dataset_v10_kaggle.csv
    employee_review_mturk_dataset_test_v6_kaggle.csv
  artifacts/
    bert_classifier/
      model/
      tokenizer/
      label_classes.json
      metadata.json
```

## 3. Runtime and Dependencies

- Python: 3.14+
- App framework: Streamlit
- Data handling: pandas
- Visualization: plotly
- NLP sentiment: nltk (VADER)
- ML/DL: scikit-learn, torch, transformers

Install dependencies:

```bash
pip install -r requirements.txt
```

Run app:

```bash
streamlit run app.py
```

## 4. Data Contracts

### 4.1 Required input columns
- `feedback`

### 4.2 Optional input columns
- `id`
- `person_name`
- `nine_box_category` (required for BERT training/evaluation blocks)

### 4.3 Built-in datasets
- Train: `dataset/employee_review_mturk_dataset_v10_kaggle.csv`
- Test: `dataset/employee_review_mturk_dataset_test_v6_kaggle.csv`

## 5. Processing Pipeline

### 5.1 Analysis pipeline (rule-based)
1. Load CSV from built-in dataset or upload.
2. Optional display anonymization for PII (`anonymize_text`, `anonymize_name`).
3. Sentiment scoring with VADER (`get_sentiment`).
4. Topic tagging via keyword matching (`detect_topics`).
5. Bias marker tagging (`detect_bias_flags`).
6. Quality scoring and label assignment (`quality_score`).
7. Aggregate outputs for charts/tables and CSV export.

### 5.2 BERT preprocessing
Before training/evaluation, comments are normalized by `anonymize_for_bert`:
- Replace person names with `John`
- Replace she/her/hers/herself with he/his/his/himself

This preprocessing is applied to both train and test sets for consistent behavior.

## 6. BERT Classification Module

### 6.1 Model configuration
- Base model: `distilbert-base-uncased`
- Task: multi-class classification (`nine_box_category`)
- Max sequence length: 128
- Batch size: 16
- Epochs: 3
- Optimizer: AdamW
- Learning rate: `2e-5`

### 6.2 Training and evaluation flow
Implemented in `train_bert_classifier(...)`:
1. Load train/test CSVs.
2. Apply BERT anonymization.
3. Encode labels with `LabelEncoder`.
4. Train DistilBERT (unless valid saved artifacts are available).
5. Evaluate on the test set.
6. Return:
   - accuracy
   - classification report
   - confusion matrix
   - per-row prediction table
   - model/tokenizer/encoder handles for inference

### 6.3 Inference flow
The prediction sandbox:
1. Tokenizes user text
2. Runs model forward pass
3. Applies softmax to get class probabilities
4. Selects argmax class as predicted box
5. Renders:
   - predicted box label
   - confidence
   - 9-box position highlight
   - probability bar chart

## 7. Persistent Model Artifacts

### 7.1 Artifact location
`artifacts/bert_classifier/`

Saved files:
- `model/` (Hugging Face model files)
- `tokenizer/` (tokenizer files)
- `label_classes.json` (ordered class names)
- `metadata.json` (fingerprint, config, history, timestamps)

### 7.2 Fingerprint strategy
A fingerprint is computed from:
- Training/test dataset SHA256 hashes
- BERT config (model name, max length, batch size, epochs, learning rate)
- Preprocessing mode identifier

On app run:
- If fingerprint matches saved metadata: load artifacts from disk
- If fingerprint differs: retrain and overwrite artifacts (when requested)

### 7.3 UI controls
- `Train / Load DistilBERT`: loads saved model if valid; otherwise trains
- `Retrain and overwrite`: forces training and replaces saved artifacts

## 8. UI Modules

1. Summary KPIs (entries, avg sentiment, sentiment counts)
2. Sentiment visuals (pie + histogram)
3. 9-box sentiment heatmap
4. Topic diagnostics
5. Bias diagnostics
6. Quality and consistency
7. Comment explorer (collapsed by default)
8. BERT section:
   - metrics
   - training history curve
   - confusion matrix
   - 9-box prediction overview matrix
   - per-class metrics table
   - test comments actual vs predicted
   - prediction sandbox
9. CSV export

## 9. 9-Box Mapping Logic

- The 9-box visual layout is defined in `NINE_BOX_LAYOUT`.
- Overview counts map labels by parsed box number (not fuzzy string matching) to avoid count mismatch.
- Box colors are defined per cell in the layout configuration.

## 10. Caching Strategy

- `@st.cache_data` for CSV loading
- `@st.cache_resource` for VADER analyzer
- `@st.cache_resource` for BERT train/load routine

Note: Streamlit memory cache resets when process restarts; disk artifacts preserve BERT model state.

## 11. Known Limitations

- Rule-based signals can miss context/sarcasm.
- Pronoun-based bias proxy is heuristic only.
- DistilBERT generalization depends on dataset quality and size.
- CPU training may be slow on low-spec machines.

## 12. Operational Notes

### 12.1 If Streamlit port is busy
Use another port:

```bash
streamlit run app.py --server.port 8502
```

### 12.2 If model imports fail in editor
Static analyzer warnings can appear if the selected Python interpreter differs from the runtime environment. Ensure VS Code uses the same interpreter where `torch` and `transformers` are installed.

## 13. Extension Points

- Add model explainability (SHAP/LIME) for class-level rationale
- Add experiment tracking and model version registry
- Add multilingual preprocessing/tokenization
- Add automated unit/integration tests
