# Predatory Comment Detector

A Streamlit prototype for detecting potentially predatory comments in Tamil,
Tanglish, and English. It uses a fine-tuned MuRIL classifier and optionally
shows a concise LIME explanation for each single-comment prediction.

> This is a research and moderation-support tool, not an automated moderation
> decision system. Review flagged comments with a human moderator.

## Run locally

```bash
python3 -m pip install --user -r requirements.txt
python3 -m streamlit run app.py
```

`best_model.pkl` must be in the project root. It is intentionally tracked with
Git LFS because it is too large for normal GitHub storage.

## Publish to GitHub

Install Git LFS and commit the model through LFS before your first push:

```bash
brew install git-lfs
git lfs install
git add .
git commit -m "Initial Streamlit classifier"
git lfs ls-files
```

The final command should list `best_model.pkl`.

## Deploy to Streamlit Community Cloud

1. Push the repository, including `.gitattributes`, to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io) and create an app.
3. Select the repository, its `main` branch, and `app.py` as the entry point.
4. In **Advanced settings**, select Python 3.12, then deploy.

Streamlit Community Cloud supports repositories that use Git LFS, so it will
retrieve `best_model.pkl` as part of deployment.
