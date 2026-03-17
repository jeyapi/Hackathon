# HR Feedback Intelligence (Hackathon)

This project is an HR feedback dashboard built for the Explainability AI hackathon.

It helps teams analyze written employee feedback at group level, explore patterns in comments, and predict 9-box talent categories from text.

## Team Members

- ALLORY Tom
- BHATT Ankit
- HADJ-SAID Léa
- IBRAHIM WAIS Abdoussamad
- JEYACHANDRAN Apinayean

## Objectives

- Turn raw employee comments into actionable, easy-to-read insights.
- Support HR and managers with group-level trend analysis.
- Provide transparent outputs (charts, counts, comparisons) instead of black-box decisions.
- Enable 9-box category prediction from feedback text.

## Scope

### In Scope

- Sentiment overview and distributions.
- Topic, bias-marker, and quality diagnostics.
- Comment exploration with filters.
- 9-box prediction with overview matrix and test-set comparison table.
- CSV export of enriched analysis.

### Out of Scope

- Individual-level HR decisions.
- Fully automated performance decisions.
- Policy or legal compliance judgments.

## Personas

### HR Analyst

- Wants to monitor feedback quality and sentiment trends over teams.
- Uses dashboard views and export features for reporting.

### Team Manager

- Wants quick visibility into strengths, risks, and coaching priorities.
- Uses comment explorer and 9-box overview to guide conversations.

### Data/AI Reviewer

- Wants to inspect prediction outcomes and model behavior.
- Uses test comparison tables, confusion matrix, and per-class metrics.

## Instructions

### 1) Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2) Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

### 3) Use the dashboard

1. Choose data source:
	- built-in dataset, or
	- upload your own CSV.
2. Confirm your CSV has at least a `feedback` column.
3. Use sidebar toggles for anonymization and group-level mode.
4. Review sentiment, topic, bias, quality, and comment explorer sections.

### 4) Run 9-box prediction

1. Open the BERT Employee Classification section.
2. Click `Train / Load DistilBERT`.
3. Review:
	- test accuracy and F1 metrics,
	- confusion matrix,
	- 9-box prediction overview,
	- actual vs predicted test comments.
4. Use the prediction sandbox to test custom text.

### 5) Export

- Download enriched CSV outputs from the app using the download buttons.

## Dataset Requirements

### Required column

- `feedback`

### Optional columns

- `id`
- `person_name`
- `nine_box_category` (needed for model training/evaluation views)

## Project Files

```text
Hackathon/
  app.py
  README.md
  TECHNICAL_DOCUMENTATION.md
  requirements.txt
  dataset/
	 employee_review_mturk_dataset_v10_kaggle.csv
	 employee_review_mturk_dataset_test_v6_kaggle.csv
```

## Technical Details

Technical implementation details are documented separately in `TECHNICAL_DOCUMENTATION.md`.
