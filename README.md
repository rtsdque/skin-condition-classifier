# SkinScan — ML Skin Condition Classifier

**[🔬 Live App →](https://skinconditionclassification.streamlit.app/)**

An educational machine learning project that classifies skin conditions from uploaded photos using a fine-tuned MobileNetV2 convolutional neural network. Built with PyTorch and deployed with Streamlit.

> ⚕️ **Medical disclaimer:** SkinScan is an educational project and is not a medical diagnostic tool. Results are not a substitute for professional medical advice. Always consult a qualified dermatologist. Skin conditions are medical issues — not hygiene failures.

## What it does

- Upload a photo of a skin condition and receive an instant prediction across 15 conditions
- Confidence breakdown showing the top 5 most likely conditions
- Severity grading for acne (clear → mild → moderate → severe)
- Ingredient recommendations — what to look for and what to avoid — specific to the predicted condition
- AI-powered personalised skincare advice via the Claude API
- Model insights page with training data distribution, per-class F1 accuracy, notable confusions, and honest limitations

## Model

| Detail | Value |
|---|---|
| Architecture | MobileNetV2 (pretrained on ImageNet) |
| Training approach | Transfer learning — frozen base + custom classifier head |
| Fine-tuning | Top layers unfrozen in Phase B at lr=0.0001 |
| Training epochs | 20 (10 Phase A + 10 Phase B) |
| Test accuracy | 59.3% across 15 classes |
| Best class | Nail Fungus — F1 0.86 |
| Weakest class | Cellulitis — F1 0.31 |

### Why 59.3% is meaningful

Random guessing across 15 classes would yield 6.7%. The model performs nearly 9× better than chance. Many of the misclassifications are clinically logical — eczema and atopic dermatitis are the same condition under different names, and psoriasis/tinea/seborrheic keratoses all involve scaly patches that challenge even dermatologists.

## Classes

| Class | Source | Training images |
|---|---|---|
| Acne — Clear | ACNE04 | 400 |
| Acne — Mild | ACNE04 | 400 |
| Acne — Moderate | ACNE04 | 186 |
| Acne — Severe | ACNE04 | 137 |
| Acne & Rosacea | DermNet | 1,000 |
| Atopic Dermatitis | DermNet | 612 |
| Cellulitis | DermNet | 361 |
| Contact Dermatitis | DermNet | 325 |
| Eczema | DermNet | 1,000 |
| Nail Fungus | DermNet | 1,000 |
| Psoriasis | DermNet | 1,000 |
| Seborrheic Keratoses | DermNet | 1,000 |
| Tinea / Ringworm | DermNet | 1,000 |
| Urticaria (Hives) | DermNet | 265 |
| Warts | DermNet | 1,000 |

## Datasets

- **[DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)** — 23 categories of skin disease images sourced from DermNet NZ via Kaggle
- **[ACNE04](https://www.kaggle.com/datasets/jincyjis/acne04)** — Acne severity dataset with dermatologist annotations. Wu et al., *Joint Acne Image Grading and Counting*, ICCV 2019

## Tech stack

- **Model:** PyTorch · TorchVision · MobileNetV2
- **Training:** Google Colab (T4 GPU)
- **App:** Streamlit
- **AI advice:** Anthropic Claude API
- **Model hosting:** Hugging Face Hub
- **Deployment:** Streamlit Community Cloud

## Run locally

```bash
# Clone the repo
git clone https://github.com/rtsdque/skin-condition-classifier
cd skin-condition-classifier

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Anthropic API key
echo ANTHROPIC_API_KEY=your-key-here > .env

# Run the app
streamlit run app/app.py
```

The model file (`skin_classifier.pt`) is hosted on Hugging Face and downloads automatically on first run.

To retrain the model, open `notebooks/skin_classifier_pipeline.ipynb` in Google Colab with a T4 GPU runtime and run all cells.

## Limitations

- **Skin tone bias** — Both datasets skew toward lighter skin tones. Performance on darker skin tones has not been systematically evaluated and is likely lower.
- **Image quality** — Trained on clinical photographs. Blurry or poorly lit photos will produce less reliable results.
- **Small classes** — Cellulitis (361 images) and urticaria (265 images) are underrepresented and the model is measurably weaker on these.
- **Not a diagnostic tool** — This is an educational ML project. No prediction should be used as a substitute for professional medical evaluation.

## Related projects

This project is part of a broader portfolio focus on healthcare and data science:

- **[Accutane Analytics](https://accutaneanalytics.streamlit.app/)** — Data analytics dashboard exploring Accutane (isotretinoin) usage, side effects, and outcomes. Built with Python and Streamlit. [GitHub →](https://github.com/rtsdque/accutane-analytics)

## Privacy

Uploaded photos are never stored or transmitted. All inference happens in-memory and images are discarded immediately after prediction. The only external call made is to the Claude API, which receives only the predicted condition name and confidence score — not the photo.

*Built with PyTorch, MobileNetV2, and Streamlit. Educational portfolio project.*