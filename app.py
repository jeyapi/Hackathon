import os
import re
import json
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="HR Feedback Intelligence", page_icon="HR", layout="wide")

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
DATASET_CANDIDATES = [
    "employee_review_mturk_dataset_v10_kaggle.csv",
    "employee_review_mturk_dataset_test_v6_kaggle.csv",
]

SENTIMENT_COLORS = {"Positive": "#16a34a", "Neutral": "#d97706", "Negative": "#dc2626"}
BADGE_COLOR = {"Positive": "#16a34a", "Neutral": "#d97706", "Negative": "#dc2626"}

TOPICS = {
    "Management": ["manager", "leadership", "supervisor", "boss", "management", "director"],
    "Teamwork": ["team", "collaboration", "colleague", "peer", "interpersonal"],
    "Performance": ["performance", "delivery", "results", "target", "quality", "deadline", "task"],
    "Communication": ["communication", "feedback", "listen", "share", "explain"],
    "Workload": ["workload", "pressure", "stress", "overwhelm", "hours", "multitask"],
    "Growth": ["skills", "training", "development", "learn", "potential", "improve"],
    "Motivation": ["motivation", "initiative", "effort", "attitude", "lazy", "apathy"],
    "Attendance": ["late", "attendance", "arrive", "show up", "absent", "punctual"],
}

POSITIVE_WORDS = {
    "great", "excellent", "strong", "best", "reliable", "consistent", "dedicated", "valuable",
    "asset", "talented", "efficient", "effective", "stellar", "phenomenal", "proactive", "solid",
}
NEGATIVE_WORDS = {
    "poor", "bad", "worst", "lazy", "apathy", "reckless", "failed", "lacking", "struggle",
    "underperform", "inconsistent", "unreliable", "mediocre", "terrible", "awful", "substandard", "slow",
}

BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 3
BERT_LR = 2e-5
BERT_ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "bert_classifier")
BERT_MODEL_DIR = os.path.join(BERT_ARTIFACT_DIR, "model")
BERT_TOKENIZER_DIR = os.path.join(BERT_ARTIFACT_DIR, "tokenizer")
BERT_LABELS_PATH = os.path.join(BERT_ARTIFACT_DIR, "label_classes.json")
BERT_METADATA_PATH = os.path.join(BERT_ARTIFACT_DIR, "metadata.json")

NINE_BOX_LAYOUT = [
    [
        {"box": 7, "name": "Potential Gem", "performance": "Low", "potential": "High", "color": "#fde047"},
        {"box": 8, "name": "High Potential", "performance": "Moderate", "potential": "High", "color": "#a3e635"},
        {"box": 9, "name": "Star", "performance": "High", "potential": "High", "color": "#a3e635"},
    ],
    [
        {"box": 4, "name": "Inconsistent Player", "performance": "Low", "potential": "Moderate", "color": "#f97316"},
        {"box": 5, "name": "Core Player", "performance": "Moderate", "potential": "Moderate", "color": "#facc15"},
        {"box": 6, "name": "High Performer", "performance": "High", "potential": "Moderate", "color": "#a3e635"},
    ],
    [
        {"box": 1, "name": "Risk", "performance": "Low", "potential": "Low", "color": "#ef4444"},
        {"box": 2, "name": "Average Performer", "performance": "Moderate", "potential": "Low", "color": "#f97316"},
        {"box": 3, "name": "Solid Performer", "performance": "High", "potential": "Low", "color": "#fde047"},
    ],
]

GENDERED_SHE = {"she", "her", "hers", "herself"}
GENDERED_HE = {"he", "him", "his", "himself"}
VAGUE_WORDS = {"good", "nice", "okay", "fine", "average", "decent", "adequate", "satisfactory", "moderate", "ok"}


@st.cache_resource
def get_vader():
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def resolve_builtin_dataset_path() -> str | None:
    for name in DATASET_CANDIDATES:
        candidate = os.path.join(DATASET_DIR, name)
        if os.path.exists(candidate):
            return candidate

    if os.path.isdir(DATASET_DIR):
        for name in os.listdir(DATASET_DIR):
            if name.lower().endswith(".csv"):
                return os.path.join(DATASET_DIR, name)
    return None


def anonymize_text(text: str) -> str:
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b", "[EMAIL]", str(text))
    text = re.sub(r"\b(?:\+?\d[\s.-]?){7,15}\b", "[PHONE]", text)
    text = re.sub(r"\b\d{5,}\b", "[ID]", text)
    return text


def anonymize_name(name: str) -> str:
    parts = str(name).strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1][0]}."
    if parts:
        return f"{parts[0][0]}."
    return "[ANON]"


def get_sentiment(text: str, sid) -> tuple[float, str]:
    score = sid.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return round(score, 3), label


def detect_topics(text: str) -> list[str]:
    lower = str(text).lower()
    found = [topic for topic, kws in TOPICS.items() if any(k in lower for k in kws)]
    return found or ["Other"]


def detect_bias_flags(text: str) -> list[str]:
    text = str(text)
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    words = text.lower().split()
    flags = []

    has_she = bool(tokens & GENDERED_SHE)
    has_he = bool(tokens & GENDERED_HE)
    if has_she and not has_he:
        flags.append("Gendered she/her")
    if has_he and not has_she:
        flags.append("Gendered he/him")

    if sum(1 for w in words if w in VAGUE_WORDS) >= 3:
        flags.append("Vague language")

    if len(words) < 15:
        flags.append("Too brief")

    if any(w in text.lower() for w in ["worst", "useless", "pathetic", "hate", "disgusting"]):
        flags.append("Potentially harsh")

    return flags


def quality_score(text: str) -> tuple[int, str]:
    text = str(text)
    words = text.split()
    score = 50

    if len(words) >= 50:
        score += 15
    elif len(words) >= 30:
        score += 8
    elif len(words) < 15:
        score -= 20

    action_words = ["should", "could", "recommend", "suggest", "improve", "focus", "develop", "consider"]
    score += sum(5 for token in action_words if token in text.lower())

    if re.search(r"\b\d+\b", text):
        score += 10

    if any(token in text.lower() for token in ["for example", "specifically", "in particular", "such as"]):
        score += 10

    score -= sum(1 for w in words if w.lower() in VAGUE_WORDS) * 3
    score = max(0, min(100, score))

    if score >= 75:
        label = "Excellent"
    elif score >= 55:
        label = "Good"
    elif score >= 35:
        label = "Fair"
    else:
        label = "Poor"

    return int(score), label


def extract_perf_pot(category: str) -> tuple[str, str]:
    c = str(category).lower()
    if "low performance" in c:
        perf = "Low"
    elif "moderate performance" in c or "average performer" in c:
        perf = "Moderate"
    elif "high performance" in c:
        perf = "High"
    else:
        perf = "Unknown"

    if "low potential" in c:
        pot = "Low"
    elif "moderate potential" in c:
        pot = "Moderate"
    elif "high potential" in c:
        pot = "High"
    else:
        pot = "Unknown"

    return perf, pot


def anonymize_for_bert(text: str, names: list[str] | None = None) -> str:
    """Replace person names with 'John Doe' and neutralise female pronouns."""
    text = str(text)

    # Replace any provided names (first / last / full)
    if names:
        for name in names:
            for part in str(name).strip().split():
                if len(part) > 2:
                    text = re.sub(rf"\b{re.escape(part)}\b", "John", text, flags=re.IGNORECASE)

    # Pronoun neutralisation (order matters: longest forms first)
    replacements = [
        (r"\bherself\b", "himself"),
        (r"\bshe\b", "he"),
        (r"\bher\b", "his"),
        (r"\bhers\b", "his"),
        (r"\bShe\b", "He"),
        (r"\bHer\b", "His"),
        (r"\bHers\b", "His"),
        (r"\bHerself\b", "Himself"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    return text


def parse_nine_box_category(category: str) -> dict:
    category = str(category)
    box_match = re.search(r"Category\s+(\d+)", category)
    name_match = re.search(r"'([^']+)'", category)
    performance, potential = extract_perf_pot(category)
    return {
        "box": int(box_match.group(1)) if box_match else None,
        "name": name_match.group(1) if name_match else category,
        "performance": performance,
        "potential": potential,
        "raw": category,
    }


def format_box_label(category: str) -> str:
    parsed = parse_nine_box_category(category)
    if parsed["box"] is None:
        return parsed["name"]
    return f"Box {parsed['box']} - {parsed['name']}"


def render_nine_box_matrix(predicted_category: str):
    parsed = parse_nine_box_category(predicted_category)
    predicted_box = parsed["box"]
    html_rows = []

    for row in NINE_BOX_LAYOUT:
        row_html = []
        for cell in row:
            is_selected = cell["box"] == predicted_box
            border = "4px solid #111827" if is_selected else "1px solid #d1d5db"
            shadow = "box-shadow: 0 0 0 3px rgba(17,24,39,0.15);" if is_selected else ""
            row_html.append(
                f"<div style='background:{cell['color']};border:{border};{shadow}padding:12px;border-radius:8px;min-height:110px;display:flex;flex-direction:column;justify-content:center;text-align:center'>"
                f"<div style='font-size:14px;font-weight:700;color:#111827'>Box {cell['box']}</div>"
                f"<div style='font-size:16px;font-weight:700;color:#111827'>{cell['name']}</div>"
                f"<div style='font-size:12px;color:#111827'>{cell['potential']} potential</div>"
                f"<div style='font-size:12px;color:#111827'>{cell['performance']} performance</div>"
                "</div>"
            )
        html_rows.append("".join(row_html))

    st.markdown(
        "<div style='margin-top:6px'>"
        "<div style='font-size:13px;font-weight:700;margin-bottom:8px'>Predicted position on the 9-box matrix</div>"
        "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px'>"
        f"{''.join(html_rows)}"
        "</div>"
        "<div style='display:flex;justify-content:space-between;margin-top:6px;font-size:12px;color:#374151'>"
        "<span>Performance: Low to High</span>"
        "<span>Potential: Low to High</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def build_prediction_results_df(source_df: pd.DataFrame, actual_labels, predicted_labels, probabilities) -> pd.DataFrame:
    results = source_df[[col for col in ["id", "person_name", "feedback", "nine_box_category"] if col in source_df.columns]].copy()
    results = results.reset_index(drop=True)
    results["actual_category"] = list(actual_labels)
    results["predicted_category"] = list(predicted_labels)
    results["actual_box"] = results["actual_category"].apply(format_box_label)
    results["predicted_box"] = results["predicted_category"].apply(format_box_label)
    results["match"] = results["actual_category"] == results["predicted_category"]
    results["confidence"] = [round(float(max(row)), 4) for row in probabilities]
    return results


def highlight_feedback(text: str) -> str:
    out = []
    for token in str(text).split():
        word = re.sub(r"[^\w]", "", token.lower())
        if word in POSITIVE_WORDS:
            out.append(f"<mark style='background:#dcfce7;padding:1px 4px;border-radius:4px'>{token}</mark>")
        elif word in NEGATIVE_WORDS:
            out.append(f"<mark style='background:#fee2e2;padding:1px 4px;border-radius:4px'>{token}</mark>")
        else:
            out.append(token)
    return " ".join(out)


def build_analysis(df_raw: pd.DataFrame, anonymize: bool) -> pd.DataFrame:
    sid = get_vader()
    rows = []
    for _, r in df_raw.iterrows():
        text = str(r.get("feedback", ""))
        scored_text = text
        shown_text = anonymize_text(text) if anonymize else text
        name = str(r.get("person_name", "Unknown"))
        shown_name = anonymize_name(name) if anonymize else name

        score, label = get_sentiment(scored_text, sid)
        topics = detect_topics(scored_text)
        flags = detect_bias_flags(scored_text)
        q_score, q_label = quality_score(scored_text)
        category = str(r.get("nine_box_category", "Unknown"))
        perf, pot = extract_perf_pot(category)

        rows.append(
            {
                "id": str(r.get("id", "")),
                "name": shown_name,
                "raw_name": name,
                "feedback": shown_text,
                "raw_feedback": scored_text,
                "category": category,
                "performance": perf,
                "potential": pot,
                "sentiment_score": score,
                "sentiment": label,
                "topics": ", ".join(topics),
                "bias_flags": flags,
                "quality_score": q_score,
                "quality_label": q_label,
            }
        )
    return pd.DataFrame(rows)



def file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bert_fingerprint(train_path: str, test_path: str) -> tuple[str, dict]:
    payload = {
        "model_name": BERT_MODEL_NAME,
        "max_len": BERT_MAX_LEN,
        "batch_size": BERT_BATCH_SIZE,
        "epochs": BERT_EPOCHS,
        "learning_rate": BERT_LR,
        "anonymization": "names_to_john_and_she_her_to_he_his",
        "train_sha256": file_sha256(train_path),
        "test_sha256": file_sha256(test_path),
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return fingerprint, payload


def save_bert_artifacts(model, tokenizer, label_classes: list[str], metadata: dict):
    os.makedirs(BERT_ARTIFACT_DIR, exist_ok=True)
    model.save_pretrained(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_TOKENIZER_DIR)

    with open(BERT_LABELS_PATH, "w", encoding="utf-8") as fh:
        json.dump(label_classes, fh, indent=2)

    save_meta = dict(metadata)
    save_meta["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(BERT_METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(save_meta, fh, indent=2)


def load_bert_artifacts(expected_fingerprint: str):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        return {"ok": False, "reason": f"Missing dependency: {exc}"}

    required_paths = [BERT_MODEL_DIR, BERT_TOKENIZER_DIR, BERT_LABELS_PATH, BERT_METADATA_PATH]
    if not all(os.path.exists(path) for path in required_paths):
        return {"ok": False, "reason": "No saved BERT artifacts found."}

    try:
        with open(BERT_METADATA_PATH, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        if metadata.get("fingerprint") != expected_fingerprint:
            return {"ok": False, "reason": "Saved model is outdated for current datasets/config."}

        with open(BERT_LABELS_PATH, "r", encoding="utf-8") as fh:
            label_classes = json.load(fh)

        tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
        le = LabelEncoder()
        le.classes_ = np.array(label_classes, dtype=object)
        return {
            "ok": True,
            "model": model,
            "tokenizer": tokenizer,
            "label_encoder": le,
            "metadata": metadata,
        }
    except Exception as exc:
        return {"ok": False, "reason": f"Unable to load saved model: {exc}"}


def evaluate_bert_model(model, tokenizer, label_encoder: LabelEncoder, df_test: pd.DataFrame, device):
    import torch
    from torch.utils.data import Dataset, DataLoader

    class FeedbackDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=BERT_MAX_LEN,
                return_tensors=None,
            )
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    y_test = label_encoder.transform(df_test["nine_box_category"])
    x_test = df_test["feedback"].astype(str).tolist()
    test_loader = DataLoader(FeedbackDataset(x_test, y_test), batch_size=BERT_BATCH_SIZE, shuffle=False)

    model.to(device)
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    predicted_labels = label_encoder.inverse_transform(all_preds)
    actual_labels = label_encoder.inverse_transform(y_test)
    acc = accuracy_score(y_test, all_preds)
    report = classification_report(y_test, all_preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, all_preds)
    test_results = build_prediction_results_df(df_test, actual_labels, predicted_labels, all_probs)
    return acc, report, cm, test_results


@st.cache_resource(show_spinner=False)
def train_bert_classifier(force_retrain: bool = False, run_id: int = 0):
    """Fine-tune or load DistilBERT and evaluate on the test CSV."""
    _ = run_id
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        return {"ok": False, "reason": f"Missing dependency: {exc}"}

    train_path = os.path.join(DATASET_DIR, "employee_review_mturk_dataset_v10_kaggle.csv")
    test_path = os.path.join(DATASET_DIR, "employee_review_mturk_dataset_test_v6_kaggle.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return {"ok": False, "reason": "Both dataset CSVs must be present in dataset/"}

    df_tr = pd.read_csv(train_path).dropna(subset=["feedback", "nine_box_category"])
    df_te = pd.read_csv(test_path).dropna(subset=["feedback", "nine_box_category"])
    fingerprint, fingerprint_payload = compute_bert_fingerprint(train_path, test_path)

    tr_names = df_tr["person_name"].dropna().tolist() if "person_name" in df_tr.columns else []
    te_names = df_te["person_name"].dropna().tolist() if "person_name" in df_te.columns else []
    all_names = list(set(tr_names + te_names))
    df_tr["feedback"] = df_tr["feedback"].apply(lambda t: anonymize_for_bert(t, all_names))
    df_te["feedback"] = df_te["feedback"].apply(lambda t: anonymize_for_bert(t, all_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not force_retrain:
        loaded = load_bert_artifacts(expected_fingerprint=fingerprint)
        if loaded["ok"]:
            model = loaded["model"]
            tokenizer = loaded["tokenizer"]
            le = loaded["label_encoder"]
            acc, report, cm, test_results = evaluate_bert_model(model, tokenizer, le, df_te, device)
            return {
                "ok": True,
                "model": model,
                "tokenizer": tokenizer,
                "label_encoder": le,
                "device": device,
                "accuracy": acc,
                "report": report,
                "confusion_matrix": cm,
                "labels": le.classes_.tolist(),
                "history": loaded.get("metadata", {}).get("history", []),
                "test_results": test_results,
                "source": "saved_artifact",
                "fingerprint": fingerprint,
            }

    le = LabelEncoder()
    le.fit(pd.concat([df_tr["nine_box_category"], df_te["nine_box_category"]]))
    y_train = le.transform(df_tr["nine_box_category"])
    x_train = df_tr["feedback"].astype(str).tolist()
    num_labels = len(le.classes_)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=num_labels)
    model.to(device)

    class FeedbackDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=BERT_MAX_LEN,
                return_tensors=None,
            )
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_loader = DataLoader(FeedbackDataset(x_train, y_train), batch_size=BERT_BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BERT_LR)

    history = []
    for epoch in range(BERT_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
        history.append({"epoch": epoch + 1, "loss": round(total_loss / len(train_loader), 4)})

    save_bert_artifacts(
        model=model,
        tokenizer=tokenizer,
        label_classes=le.classes_.tolist(),
        metadata={
            "fingerprint": fingerprint,
            "config": fingerprint_payload,
            "history": history,
            "train_rows": int(len(df_tr)),
            "test_rows": int(len(df_te)),
        },
    )

    acc, report, cm, test_results = evaluate_bert_model(model, tokenizer, le, df_te, device)
    return {
        "ok": True,
        "model": model,
        "tokenizer": tokenizer,
        "label_encoder": le,
        "device": device,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "labels": le.classes_.tolist(),
        "history": history,
        "test_results": test_results,
        "source": "trained_now",
        "fingerprint": fingerprint,
    }


def get_group_from_pronouns(text: str) -> str:
    tokens = set(re.findall(r"\b\w+\b", str(text).lower()))
    has_she = bool(tokens & GENDERED_SHE)
    has_he = bool(tokens & GENDERED_HE)
    if has_she and not has_he:
        return "She/Her only"
    if has_he and not has_she:
        return "He/Him only"
    if has_she and has_he:
        return "Both"
    return "Neutral/none"


def main():
    st.title("HR Feedback Intelligence")
    st.caption("Responsible NLP for HR trend analysis and explainable decision support")

    with st.sidebar:
        st.subheader("Settings")
        source = st.radio("Data", ["Built-in dataset", "Upload CSV"])
        anonymize = st.toggle("Anonymize names and PII", value=True)
        group_only = st.toggle("Group-level mode", value=False)
        st.info("Use for group-level insights. Do not use this app for individual HR decisions.")

    if source == "Built-in dataset":
        dataset_path = resolve_builtin_dataset_path()
        if not dataset_path:
            st.error("Built-in dataset not found.")
            return
        df_raw = load_csv(dataset_path)
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if not uploaded:
            st.stop()
        df_raw = pd.read_csv(uploaded)

    required = {"feedback"}
    if not required.issubset(df_raw.columns):
        st.error("CSV must contain at least a feedback column.")
        return

    with st.spinner("Analyzing feedback..."):
        df = build_analysis(df_raw, anonymize)

    counts = df["sentiment"].value_counts()
    total = len(df)
    avg_score = df["sentiment_score"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Entries", f"{total:,}")
    c2.metric("Avg sentiment", f"{avg_score:+.2f}")
    c3.metric("Positive", f"{counts.get('Positive', 0):,}")
    c4.metric("Neutral", f"{counts.get('Neutral', 0):,}")
    c5.metric("Negative", f"{counts.get('Negative', 0):,}")

    left, right = st.columns(2)
    with left:
        fig_pie = px.pie(
            values=counts.values,
            names=counts.index,
            title="Sentiment distribution",
            color=counts.index,
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.45,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        fig_hist = px.histogram(
            df,
            x="sentiment_score",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment score distribution",
            nbins=40,
        )
        fig_hist.add_vline(x=0.05, line_dash="dash", line_color="#16a34a")
        fig_hist.add_vline(x=-0.05, line_dash="dash", line_color="#dc2626")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("9-box overview")
    box_df = df[df["performance"].isin(["Low", "Moderate", "High"]) & df["potential"].isin(["Low", "Moderate", "High"])]
    if not box_df.empty:
        pivot = (
            box_df.groupby(["potential", "performance"])["sentiment_score"]
            .mean()
            .reset_index()
            .pivot(index="potential", columns="performance", values="sentiment_score")
            .reindex(index=["High", "Moderate", "Low"], columns=["Low", "Moderate", "High"])
        )
        hm = px.imshow(
            pivot,
            text_auto=True,
            color_continuous_scale="RdYlGn",
            zmin=-0.6,
            zmax=0.6,
            title="Average sentiment by performance x potential",
        )
        st.plotly_chart(hm, use_container_width=True)

    st.subheader("Topic diagnostics")
    topic_count = defaultdict(lambda: Counter())
    for _, row in df.iterrows():
        for topic in row["topics"].split(", "):
            topic_count[topic][row["sentiment"]] += 1

    topic_rows = []
    for topic, d in topic_count.items():
        total_t = sum(d.values())
        topic_rows.append(
            {
                "topic": topic,
                "total": total_t,
                "positive": d.get("Positive", 0),
                "neutral": d.get("Neutral", 0),
                "negative": d.get("Negative", 0),
                "negative_pct": round(d.get("Negative", 0) / total_t * 100, 1),
            }
        )
    df_topics = pd.DataFrame(topic_rows).sort_values("total", ascending=False)

    t1, t2 = st.columns(2)
    with t1:
        topic_melt = df_topics.melt(id_vars="topic", value_vars=["positive", "neutral", "negative"])
        fig_topic = px.bar(topic_melt, x="value", y="topic", color="variable", orientation="h", title="Topic sentiment volume")
        st.plotly_chart(fig_topic, use_container_width=True)
    with t2:
        fig_neg = px.bar(df_topics.sort_values("negative_pct"), x="negative_pct", y="topic", orientation="h", title="Topics ranked by negative rate")
        st.plotly_chart(fig_neg, use_container_width=True)

    st.dataframe(df_topics, use_container_width=True, hide_index=True)

    st.subheader("Bias diagnostics")
    pronoun_group = df["raw_feedback"].apply(get_group_from_pronouns)
    fair_df = pd.DataFrame({"group": pronoun_group, "score": df["sentiment_score"]})
    fair_summary = fair_df.groupby("group").agg(count=("score", "size"), avg_sentiment=("score", "mean")).reset_index()

    b1, b2 = st.columns(2)
    with b1:
        fig_fair = px.bar(fair_summary, x="group", y="avg_sentiment", color="group", title="Average sentiment by pronoun group")
        st.plotly_chart(fig_fair, use_container_width=True)
    with b2:
        bias_flags = Counter(flag for flags in df["bias_flags"] for flag in flags)
        if bias_flags:
            bf = pd.DataFrame({"flag": list(bias_flags.keys()), "count": list(bias_flags.values())}).sort_values("count", ascending=True)
            fig_bias = px.bar(bf, x="count", y="flag", orientation="h", title="Potential bias markers")
            st.plotly_chart(fig_bias, use_container_width=True)
        else:
            st.success("No major bias marker detected with current heuristics.")

    st.subheader("Quality and consistency")
    q1, q2 = st.columns(2)
    with q1:
        fig_q = px.histogram(df, x="quality_score", nbins=20, title="Review quality score distribution")
        st.plotly_chart(fig_q, use_container_width=True)
    with q2:
        q_label = df["quality_label"].value_counts().reset_index()
        q_label.columns = ["quality", "count"]
        fig_q2 = px.bar(q_label, x="quality", y="count", title="Quality label counts")
        st.plotly_chart(fig_q2, use_container_width=True)

    if not group_only:
        with st.expander("Comment explorer", expanded=False):
            f1, f2, f3 = st.columns(3)
            sentiment_filter = f1.multiselect("Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
            topic_filter = f2.multiselect("Topics", list(TOPICS.keys()) + ["Other"])
            quality_filter = f3.multiselect("Quality", ["Excellent", "Good", "Fair", "Poor"])

            mask = df["sentiment"].isin(sentiment_filter)
            if topic_filter:
                mask &= df["topics"].apply(lambda x: any(t in x for t in topic_filter))
            if quality_filter:
                mask &= df["quality_label"].isin(quality_filter)

            view = df[mask]
            st.caption(f"Showing {min(60, len(view))} / {len(view)} comments")

            for _, row in view.head(60).iterrows():
                flags = ", ".join(row["bias_flags"]) if row["bias_flags"] else "none"
                color = BADGE_COLOR[row["sentiment"]]
                st.markdown(
                    f"<div style='border-left:4px solid {color};padding:10px 12px;margin-bottom:8px;background:#fafafa;border-radius:6px'>"
                    f"<div style='font-size:13px;color:#111827'><b>{row['name']}</b> | <b>{row['sentiment']}</b> ({row['sentiment_score']:+.2f}) | "
                    f"Quality: {row['quality_label']} ({row['quality_score']}) | Topics: {row['topics']} | Flags: {flags}</div>"
                    f"<div style='margin-top:6px;font-size:14px;color:#1f2937'>{highlight_feedback(row['feedback'])}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

    # ── BERT Employee Classification ─────────────────────────────────────────
    st.divider()
    st.subheader("BERT Employee Classification")
    st.caption(
        f"DistilBERT fine-tuned on the training set (878 samples) and evaluated on the held-out test set (225 samples) "
        f"over {BERT_EPOCHS} epochs — CPU training."
    )

    bert_train_ok = os.path.exists(os.path.join(DATASET_DIR, "employee_review_mturk_dataset_v10_kaggle.csv"))
    bert_test_ok = os.path.exists(os.path.join(DATASET_DIR, "employee_review_mturk_dataset_test_v6_kaggle.csv"))

    if not (bert_train_ok and bert_test_ok):
        st.warning("Both dataset CSVs must be present in the dataset/ folder to run BERT training.")
    else:
        if "bert_done" not in st.session_state:
            st.session_state["bert_done"] = False
        if "bert_run_id" not in st.session_state:
            st.session_state["bert_run_id"] = 0

        if not st.session_state["bert_done"]:
            st.info(
                "Click to train DistilBERT. If a compatible saved model exists, it loads instantly from disk."
            )
            bt1, bt2 = st.columns(2)
            train_clicked = bt1.button("Train / Load DistilBERT", key="bert_btn")
            retrain_clicked = bt2.button("Retrain and overwrite", key="bert_btn_retrain")
            if train_clicked or retrain_clicked:
                if retrain_clicked:
                    st.session_state["bert_run_id"] += 1
                    train_bert_classifier.clear()
                with st.spinner("Downloading model weights and fine-tuning DistilBERT…"):
                    _res = train_bert_classifier(
                        force_retrain=retrain_clicked,
                        run_id=st.session_state["bert_run_id"],
                    )
                if retrain_clicked:
                    train_bert_classifier.clear()
                st.session_state["bert_done"] = True
                st.rerun()
        else:
            import torch
            bert_res = train_bert_classifier(force_retrain=False, run_id=st.session_state["bert_run_id"])

            if not bert_res["ok"]:
                st.error(f"BERT training failed: {bert_res['reason']}")
            else:
                if bert_res.get("source") == "saved_artifact":
                    st.success("Loaded saved DistilBERT model from disk. Retraining was skipped.")
                else:
                    st.success("DistilBERT trained and saved to disk for future sessions.")

                if st.button("Retrain model and overwrite saved artifacts", key="bert_btn_retrain_live"):
                    st.session_state["bert_run_id"] += 1
                    train_bert_classifier.clear()
                    with st.spinner("Retraining DistilBERT and updating saved artifacts…"):
                        _res = train_bert_classifier(
                            force_retrain=True,
                            run_id=st.session_state["bert_run_id"],
                        )
                    train_bert_classifier.clear()
                    st.rerun()

                bm1, bm2, bm3 = st.columns(3)
                bm1.metric("Test accuracy", f"{bert_res['accuracy'] * 100:.1f}%")
                macro_f1 = bert_res["report"].get("macro avg", {}).get("f1-score", 0.0)
                bm2.metric("Macro F1", f"{macro_f1 * 100:.1f}%")
                weighted_f1 = bert_res["report"].get("weighted avg", {}).get("f1-score", 0.0)
                bm3.metric("Weighted F1", f"{weighted_f1 * 100:.1f}%")

                bl, br = st.columns(2)
                with bl:
                    hist_df = pd.DataFrame(bert_res["history"])
                    if not hist_df.empty:
                        fig_loss = px.line(
                            hist_df, x="epoch", y="loss", markers=True,
                            title="Training loss per epoch",
                            labels={"loss": "Cross-entropy loss", "epoch": "Epoch"},
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                    else:
                        st.info("Training history unavailable for this loaded model.")
                with br:
                    short_labels = [
                        lbl.split("'")[1] if "'" in lbl else lbl
                        for lbl in bert_res["labels"]
                    ]
                    cm_df = pd.DataFrame(
                        bert_res["confusion_matrix"],
                        index=short_labels,
                        columns=short_labels,
                    )
                    fig_cm = px.imshow(
                        cm_df, text_auto=True,
                        title="Confusion matrix — test set",
                        color_continuous_scale="Blues",
                    )
                    fig_cm.update_layout(height=520)
                    st.plotly_chart(fig_cm, use_container_width=True)

                st.subheader("9-box prediction overview")
                pred_counts = bert_res["test_results"]["predicted_category"].value_counts().to_dict()
                actual_counts = bert_res["test_results"]["actual_category"].value_counts().to_dict()
                correct_counts = bert_res["test_results"][bert_res["test_results"]["match"]]["predicted_category"].value_counts().to_dict()
                label_by_box = {
                    parsed["box"]: lbl
                    for lbl in bert_res["labels"]
                    for parsed in [parse_nine_box_category(lbl)]
                    if parsed["box"] is not None
                }
                matrix_html_rows = []
                for row in NINE_BOX_LAYOUT:
                    row_cells = []
                    for cell in row:
                        full_label = label_by_box.get(cell["box"])
                        n_pred = pred_counts.get(full_label, 0) if full_label else 0
                        n_actual = actual_counts.get(full_label, 0) if full_label else 0
                        n_correct = correct_counts.get(full_label, 0) if full_label else 0
                        row_cells.append(
                            f"<div style='background:{cell['color']};border:1px solid #d1d5db;padding:10px 6px;border-radius:8px;"
                            f"min-height:110px;display:flex;flex-direction:column;justify-content:center;text-align:center'>"
                            f"<div style='font-size:13px;font-weight:700;color:#111827'>Box {cell['box']} — {cell['name']}</div>"
                            f"<div style='font-size:11px;color:#374151;margin-top:4px'>{cell['potential']} potential / {cell['performance']} perf.</div>"
                            f"<div style='font-size:12px;font-weight:600;color:#111827;margin-top:6px'>Predicted: {n_pred}</div>"
                            f"<div style='font-size:12px;color:#374151'>Actual: {n_actual}</div>"
                            f"<div style='font-size:12px;color:#166534'>Correct: {n_correct}</div>"
                            "</div>"
                        )
                    matrix_html_rows.append("".join(row_cells))
                st.markdown(
                    "<div style='margin-bottom:16px'>"
                    "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px'>"
                    + "".join(matrix_html_rows) +
                    "</div>"
                    "<div style='display:flex;justify-content:space-between;margin-top:6px;font-size:12px;color:#6b7280'>"
                    "<span>← Performance: Low &nbsp; Moderate &nbsp; High →</span>"
                    "<span>Potential: Low ↓ Moderate ↑ High</span>"
                    "</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                with st.expander("Per-class metrics"):
                    metric_rows = []
                    for lbl in bert_res["labels"]:
                        r = bert_res["report"].get(lbl, {})
                        metric_rows.append({
                            "category": lbl,
                            "precision": round(r.get("precision", 0), 3),
                            "recall": round(r.get("recall", 0), 3),
                            "f1": round(r.get("f1-score", 0), 3),
                            "support": int(r.get("support", 0)),
                        })
                    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

                with st.expander("Test comments: actual vs predicted"):
                    results_df = bert_res["test_results"].copy()
                    show_matches_only = st.selectbox(
                        "Filter rows",
                        ["All", "Correct only", "Errors only"],
                        index=0,
                        key="bert_results_filter",
                    )
                    if show_matches_only == "Correct only":
                        results_df = results_df[results_df["match"]]
                    elif show_matches_only == "Errors only":
                        results_df = results_df[~results_df["match"]]

                    st.caption(f"Showing {len(results_df)} test comments from employee_review_mturk_dataset_test_v6_kaggle.csv")
                    
                    summary_df = results_df[[
                        col for col in [
                            "id",
                            "person_name",
                            "actual_box",
                            "predicted_box",
                            "match",
                            "confidence",
                        ]
                        if col in results_df.columns
                    ]].copy()
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                    st.subheader("Full feedback review")
                    for idx, row in results_df.iterrows():
                        match_badge = "✅ Correct" if row["match"] else "❌ Error"
                        with st.expander(f"{match_badge} — {row.get('person_name', 'Unknown')} | Actual: {row['actual_box']} → Predicted: {row['predicted_box']}"):
                            st.markdown(f"**Feedback:** {row['feedback']}")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Actual", row["actual_box"])
                            col2.metric("Predicted", row["predicted_box"])
                            col3.metric("Confidence", f"{row['confidence']*100:.1f}%")

                    test_csv = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download test predictions CSV",
                        data=test_csv,
                        file_name="bert_test_predictions.csv",
                        mime="text/csv",
                        key="bert_test_predictions_download",
                    )

                st.markdown("**Prediction sandbox**")
                sample_bert = st.text_area(
                    "Enter employee review text",
                    placeholder="She consistently delivered high-quality results and proactively mentored junior team members…",
                    height=100,
                    key="bert_sandbox",
                )
                if sample_bert.strip():
                    model_b = bert_res["model"]
                    tokenizer_b = bert_res["tokenizer"]
                    le_b = bert_res["label_encoder"]
                    device_b = bert_res["device"]
                    enc = tokenizer_b(
                        sample_bert,
                        return_tensors="pt",
                        truncation=True,
                        max_length=BERT_MAX_LEN,
                        padding=True,
                    )
                    enc = {k: v.to(device_b) for k, v in enc.items()}
                    with torch.no_grad():
                        out = model_b(**enc)
                    probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
                    pred_idx = int(probs.argmax())
                    pred_label = le_b.classes_[pred_idx]
                    conf = float(probs[pred_idx])
                    pred_info = parse_nine_box_category(pred_label)
                    st.success(f"Predicted box: {format_box_label(pred_label)}")
                    st.caption(f"Model confidence: {conf * 100:.1f}%")
                    p1, p2 = st.columns(2)
                    p1.metric("Performance", pred_info["performance"])
                    p2.metric("Potential", pred_info["potential"])
                    render_nine_box_matrix(pred_label)
                    prob_df = pd.DataFrame({
                        "category": [
                            format_box_label(lbl)
                            for lbl in le_b.classes_
                        ],
                        "probability": probs,
                    }).sort_values("probability", ascending=True)
                    fig_prob = px.bar(
                        prob_df, x="probability", y="category", orientation="h",
                        title="Class probabilities",
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

    export_df = df.copy()
    export_df["bias_flags"] = export_df["bias_flags"].apply(lambda x: "; ".join(x))
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download analysis CSV", data=csv_bytes, file_name="hr_feedback_analysis.csv", mime="text/csv")


if __name__ == "__main__":
    main()
