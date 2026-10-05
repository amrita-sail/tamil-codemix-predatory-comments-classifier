"""Readable, local LIME explanations for :class:`PredatoryClassifier`."""
from __future__ import annotations

import html

import numpy as np

from predictor import clean_text


class CommentExplainer:
    """Create a local, word-level LIME explanation for one comment."""

    def __init__(self, classifier, num_features=8):
        self.classifier = classifier
        self.num_features = num_features

    def predict_proba(self, texts):
        """Return the two-column probability array expected by LIME/SHAP."""
        predatory = self.classifier.predict_proba(texts)
        return np.column_stack((1 - predatory, predatory))

    def lime(self, text):
        """Return ``[(word, weight), ...]`` for the predatory class."""
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError as exc:
            raise RuntimeError("LIME is not installed. Run `pip install -r requirements.txt`.") from exc

        explainer = LimeTextExplainer(
            class_names=[self.classifier.id2label[0], self.classifier.id2label[1]],
            bow=True,
            random_state=42,
        )
        explanation = explainer.explain_instance(
            clean_text(text),
            self.predict_proba,
            labels=(1,),
            num_features=self.num_features,
        )
        return explanation.as_list(label=1)

def lime_evidence_html(items):
    """Render a theme-safe, compact LIME evidence panel."""
    supporting = sorted(
        ((word, score) for word, score in items if score > 0),
        key=lambda pair: pair[1], reverse=True,
    )
    opposing = sorted(
        ((word, score) for word, score in items if score <= 0),
        key=lambda pair: pair[1],
    )

    def chips(features, background, border, accent, empty_message):
        if not features:
            return f'<span style="color:#94a3b8;font-size:14px">{empty_message}</span>'
        rendered = []
        for word, score in features:
            rendered.append(
                f'<span style="display:inline-flex;align-items:center;gap:8px;margin:0 7px 8px 0;'
                f'padding:7px 10px;border:1px solid {border};border-radius:8px;background:{background};'
                f'color:#f8fafc;font-size:14px;line-height:1.2">'
                f'<strong style="color:#ffffff;font-weight:600">{html.escape(str(word))}</strong>'
                f'<span style="color:{accent};font-variant-numeric:tabular-nums">{score:+.3f}</span></span>'
            )
        return "".join(rendered)

    return (
        '<section style="max-width:860px;padding:18px 20px;border:1px solid #334155;'
        'border-radius:14px;background:#0f172a;color:#f8fafc;font-family:ui-sans-serif,system-ui,sans-serif">'
        '<div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:18px">'
        '<h3 style="margin:0;color:#ffffff;font-size:18px;font-weight:650">Key signals</h3>'
        '<span style="color:#94a3b8;font-size:12px;letter-spacing:.06em;text-transform:uppercase">LIME</span></div>'
        '<div style="margin-bottom:16px"><div style="margin-bottom:8px;color:#fca5a5;font-size:13px;'
        'font-weight:700;letter-spacing:.04em;text-transform:uppercase">Raises concern</div>'
        + chips(supporting, "#3f1d2e", "#7f1d1d", "#fca5a5", "No strong signals.")
        + '</div><div><div style="margin-bottom:8px;color:#86efac;font-size:13px;font-weight:700;'
        'letter-spacing:.04em;text-transform:uppercase">Reduces concern</div>'
        + chips(opposing, "#12352a", "#166534", "#86efac", "No strong signals.")
        + '</div></section>'
    )
