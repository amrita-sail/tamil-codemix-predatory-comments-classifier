"""Streamlit interface for the Tamil/Tanglish predatory-comment classifier."""
import os

import streamlit as st

from explainer import CommentExplainer, lime_evidence_html
from predictor import PredatoryClassifier

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.pkl")


@st.cache_resource(show_spinner="Loading the language model…")
def load_classifier(model_path):
    """Load the large model once per Streamlit server process."""
    return PredatoryClassifier(model_path)


def show_prediction(classifier, text):
    result = classifier.predict([text])[0]
    probability = result["predatory_probability"]
    predatory_label = classifier.id2label[1]

    left, right = st.columns([1, 2])
    with left:
        st.metric("Predatory probability", f"{probability:.1%}")
    with right:
        st.progress(int(probability * 100), text="Model confidence")

    if result["label"] == predatory_label:
        st.error(f"Flagged as **{predatory_label}**. A human review is recommended.")
    else:
        st.success("Looks non-predatory.")


def main():
    st.set_page_config(page_title="Predatory Comment Detector", page_icon="🛡️", layout="centered")
    classifier = load_classifier(MODEL_PATH)
    explainer = CommentExplainer(classifier)

    st.title("🛡️ Predatory Comment Detector")
    st.caption(
        "Tamil · Tanglish · English  |  Research and moderation support only — "
        "always use human review."
    )

    single_tab, batch_tab = st.tabs(["Single comment", "Batch check"])

    with single_tab:
        text = st.text_area(
            "Comment",
            placeholder="Paste a comment here…",
            height=130,
            label_visibility="collapsed",
        )
        classify_col, explain_col = st.columns(2)
        with classify_col:
            classify_clicked = st.button("Classify", type="primary", use_container_width=True)
        with explain_col:
            explain_clicked = st.button("Classify & explain", use_container_width=True)

        if classify_clicked or explain_clicked:
            if not text or not text.strip():
                st.warning("Enter a comment to classify.")
            else:
                show_prediction(classifier, text)
                if explain_clicked:
                    with st.spinner("Generating LIME evidence…"):
                        try:
                            evidence = explainer.lime(text)
                            st.markdown(lime_evidence_html(evidence), unsafe_allow_html=True)
                        except Exception as exc:
                            st.warning(f"Could not generate the LIME explanation: {exc}")

    with batch_tab:
        batch_text = st.text_area(
            "Comments — one per line",
            height=220,
            placeholder="First comment\nSecond comment",
        )
        if st.button("Classify all comments", type="primary"):
            comments = [line for line in batch_text.splitlines() if line.strip()]
            if not comments:
                st.warning("Enter at least one comment.")
            else:
                results = classifier.predict(comments)
                st.dataframe(
                    [
                        {
                            "Comment": result["text"],
                            "Label": result["label"],
                            "Predatory probability": result["predatory_probability"],
                        }
                        for result in results
                    ],
                    hide_index=True,
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
