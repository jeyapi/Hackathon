import os
import re
from collections import Counter, defaultdict

import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

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


@st.cache_data
def train_text_model(df: pd.DataFrame) -> dict:
    if "nine_box_category" not in df.columns or "feedback" not in df.columns:
        return {"ok": False, "reason": "Missing required columns"}

    model_df = df[["feedback", "nine_box_category"]].dropna().copy()
    model_df = model_df[model_df["feedback"].astype(str).str.len() > 5]

    if model_df["nine_box_category"].nunique() < 2 or len(model_df) < 60:
        return {"ok": False, "reason": "Insufficient class diversity"}

    x_train, x_test, y_train, y_test = train_test_split(
        model_df["feedback"].astype(str),
        model_df["nine_box_category"].astype(str),
        test_size=0.25,
        random_state=42,
        stratify=model_df["nine_box_category"].astype(str),
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    clf = LogisticRegression(max_iter=1500, n_jobs=None)
    clf.fit(x_train_vec, y_train)

    pred = clf.predict(x_test_vec)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=sorted(y_test.unique()))
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)

    return {
        "ok": True,
        "vectorizer": vectorizer,
        "model": clf,
        "accuracy": acc,
        "labels": sorted(y_test.unique()),
        "confusion_matrix": cm,
        "report": report,
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
        st.subheader("Comment explorer")
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

    if "nine_box_category" in df_raw.columns:
        st.subheader("ML module: 9-box category prediction")
        model_pack = train_text_model(df_raw)

        if not model_pack["ok"]:
            st.warning(f"Model not trained: {model_pack['reason']}")
        else:
            m1, m2 = st.columns([1, 2])
            with m1:
                st.metric("Hold-out accuracy", f"{model_pack['accuracy'] * 100:.1f}%")
                macro_f1 = model_pack["report"].get("macro avg", {}).get("f1-score", 0.0)
                st.metric("Macro F1", f"{macro_f1 * 100:.1f}%")
            with m2:
                cm = pd.DataFrame(
                    model_pack["confusion_matrix"],
                    index=model_pack["labels"],
                    columns=model_pack["labels"],
                )
                fig_cm = px.imshow(cm, text_auto=True, title="Confusion matrix")
                st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("Prediction sandbox")
            sample = st.text_area(
                "Enter synthetic review text",
                placeholder="Delivered strong technical work and helped teammates, but needs better communication under pressure.",
                height=110,
            )
            if sample.strip():
                vec = model_pack["vectorizer"].transform([sample])
                pred = model_pack["model"].predict(vec)[0]
                probs = model_pack["model"].predict_proba(vec)[0]
                classes = model_pack["model"].classes_
                conf = float(probs.max())

                st.success(f"Predicted category: {pred}")
                st.caption(f"Model confidence: {conf * 100:.1f}%")

                top_idx = probs.argsort()[-5:][::-1]
                top_df = pd.DataFrame(
                    {"category": classes[top_idx], "probability": [float(probs[i]) for i in top_idx]}
                )
                fig_prob = px.bar(top_df, x="probability", y="category", orientation="h", title="Top predicted categories")
                st.plotly_chart(fig_prob, use_container_width=True)

    export_df = df.copy()
    export_df["bias_flags"] = export_df["bias_flags"].apply(lambda x: "; ".join(x))
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download analysis CSV", data=csv_bytes, file_name="hr_feedback_analysis.csv", mime="text/csv")


if __name__ == "__main__":
    main()
