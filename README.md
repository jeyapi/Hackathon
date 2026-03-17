# HR Feedback Intelligence (Hackathon)

We built an explainable and responsible NLP dashboard to analyze employee feedback at group level and classify employees into the McKinsey 9-box talent matrix using a fine-tuned DistilBERT model.

This project was developed during the Explainability AI hackathon. Our goal was to combine useful HR text analytics with clear, inspectable logic so users can understand why a result appears — and to push beyond simple sentiment into actionable talent classification.

## Team Members

- ALLORY Tom
- BHATT Ankit
- HADJ-SAID Léa
- IBRAHIM WAIS Samo
- JEYACHANDRAN Apinayean

## What Was Implemented

In our Streamlit app (`app.py`), we implemented:

### Visual Analytics
- Sentiment analysis on HR feedback with VADER (`Positive`, `Neutral`, `Negative`) and compound score.
- Summary metrics bar (comment count, sentiment breakdown).
- Sentiment distribution pie chart and score histogram.
- 9-box sentiment heatmap (performance × potential).
- Topic-level charts: comment volume and negative-rate ranking per topic.
- Quality score and label distribution charts.
- Bias diagnostics: pronoun group comparison (she/her vs he/him), vague language flags, very short comments, and potentially harsh wording.
- Rule-based topic detection with a visible keyword dictionary.
- Review quality scoring (`Excellent`, `Good`, `Fair`, `Poor`) using explicit heuristics.

### Comment Explorer
- Collapsible explorer (collapsed by default) with filters for sentiment, topic, and quality.
- Color-coded comment cards with keyword highlighting.

### Anonymization
- Personal names transformed to initials for display.
- Emails, phone numbers, and long numeric IDs masked.
- Pre-BERT anonymization: all person names replaced with "John" and female pronouns (she/her/hers/herself) mapped to male equivalents before fine-tuning, to reduce gender bias in the classifier.

### BERT Employee Classification
- Fine-tuning of `distilbert-base-uncased` on `employee_review_mturk_dataset_v10_kaggle.csv` (878 rows).
- Evaluation on the held-out test set `employee_review_mturk_dataset_test_v6_kaggle.csv` (225 rows).
- 3 training epochs, batch size 16, learning rate 2e-5, max sequence length 128, CPU inference.
- Post-training metrics: accuracy, macro F1, weighted F1.
- Training loss curve and confusion matrix displayed side by side.
- **9-box prediction overview matrix**: a 3×3 HTML table summarizing predicted counts, actual counts, and correctly classified counts per box, shown above the per-class metrics.
- Per-class precision/recall/F1 expander.
- Test comments table: side-by-side actual vs predicted category for all 225 test comments, with a filter (All / Correct only / Errors only) and a CSV download button.
- Prediction sandbox: type any free-text comment and get the predicted 9-box category with the classified cell highlighted in an interactive matrix, plus a class probability bar chart.

### Export
- Download the full enriched analysis (with sentiment, topics, quality, bias flags) as a CSV.

Everything above reflects what is currently in the codebase, not planned features.

## Repository Structure

```
Hackathon/
	app.py
	README.md
	requirements.txt
	dataset/
		employee_review_mturk_dataset_v10_kaggle.csv   # training set (878 rows)
		employee_review_mturk_dataset_test_v6_kaggle.csv  # test set (225 rows)
```

## Dataset and Expected Columns

Minimum required input column:

- `feedback`

Optional columns used when available:

- `id`
- `person_name`
- `nine_box_category` (required for BERT training and evaluation)

By default, we load the built-in CSVs from the `dataset/` folder. Users can also upload their own CSV.

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
pip install -r requirements.txt
```

> `torch` and `transformers` are required for BERT classification. Training runs on CPU and takes approximately 5–10 minutes for 3 epochs on the provided dataset.

## Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal (usually `http://localhost:8501`).

Click the **Fine-tune DistilBERT** button to train the classifier. The model is cached in memory for the duration of the server session; restarting the app requires retraining.

## Explainability and Responsible AI Choices

To align with the hackathon explainability and trust requirements, we deliberately favored transparent methods:

- Rules and thresholds are explicit and inspectable (topics, bias flags, quality score).
- Visual outputs are descriptive and traceable to text patterns that users can review.
- The BERT block shows evaluation metrics, per-class breakdown, a confusion matrix, and class probabilities instead of only hard predictions.
- Pre-training anonymization removes gendered pronouns and person names before fine-tuning to reduce bias encoded in the model weights.
- Privacy support through anonymization and PII masking at display time.
- Group-level analysis mode with a clear warning against individual-level HR decision making.
- The 9-box matrix display makes model predictions spatially interpretable by placing the result in the talent grid context.

## Limitations

- Sentiment and bias signals are heuristic and may miss context or sarcasm.
- The pronoun-based fairness proxy is coarse and is not demographic ground truth.
- The BERT model is fine-tuned on a relatively small dataset (878 training examples across 9 classes) and may not generalize well to other organizations or comment styles.
- Training runs on CPU only; the model cache is lost on every server restart.
- We designed this as a decision-support and analysis tool, not an automated HR decision system.

## Future Improvements

- Add SHAP/LIME explanations for BERT predictions.
- Add calibration and uncertainty analysis for the classifier.
- Expand fairness diagnostics with stronger, validated metrics.
- Add robust preprocessing for multilingual and noisy text.
- Add automated tests and experiment tracking.
- GPU support and model persistence (save/load fine-tuned weights to disk).
