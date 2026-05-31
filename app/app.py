import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import plotly.graph_objects as go
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinScan",
    page_icon="🔬",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #FAFAF8; }
.title-block { text-align: center; padding: 2rem 0 1rem 0; }
.title-block h1 { font-family: 'DM Serif Display', serif; font-size: 2.8rem; color: #1a1a1a; margin-bottom: 0.2rem; letter-spacing: -0.5px; }
.title-block p { color: #6b7280; font-size: 1rem; font-weight: 300; }
.disclaimer-box { background: #FEF9EC; border-left: 3px solid #F59E0B; border-radius: 0 8px 8px 0; padding: 0.75rem 1rem; font-size: 0.85rem; color: #78350F; margin: 1rem 0; }
.result-card { background: white; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
.condition-name { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #1a1a1a; margin: 0; }
.severity-badge { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 500; margin-left: 8px; vertical-align: middle; }
.badge-clear { background: #D1FAE5; color: #065F46; }
.badge-mild { background: #FEF9C3; color: #713F12; }
.badge-moderate { background: #FEE2E2; color: #991B1B; }
.badge-severe { background: #7F1D1D; color: #FEE2E2; }
.badge-general { background: #EDE9FE; color: #4C1D95; }
.confidence-label { font-size: 0.82rem; color: #374151; margin-bottom: 2px; }
.section-heading { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #9CA3AF; margin: 1.2rem 0 0.6rem 0; }
.pill { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; margin: 3px 3px 3px 0; border: 1px solid #E5E7EB; color: #374151; background: #F9FAFB; }
.pill-avoid { border-color: #FECACA; color: #991B1B; background: #FFF5F5; }
.ai-response { background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.9rem; color: #14532D; line-height: 1.7; margin-top: 0.5rem; }
.stat-card { background: white; border: 1px solid #E5E7EB; border-radius: 10px; padding: 1.2rem; text-align: center; }
.stat-number { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #1a1a1a; margin: 0; }
.stat-label { font-size: 0.8rem; color: #6B7280; margin: 4px 0 0 0; }
.insight-box { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem 1.2rem; margin: 0.5rem 0; font-size: 0.88rem; color: #374151; line-height: 1.7; }
.footer-note { text-align: center; font-size: 0.78rem; color: #9CA3AF; margin-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), '..', 'model', 'skin_classifier.pt')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Analytics data ────────────────────────────────────────────────────────────
CLASS_DISPLAY = {
    "acne_clear": "Clear skin", "acne_mild": "Mild acne", "acne_moderate": "Moderate acne",
    "acne_rosacea": "Acne & Rosacea", "acne_severe": "Severe acne",
    "atopic_dermatitis": "Atopic Dermatitis", "cellulitis": "Cellulitis",
    "contact_dermatitis": "Contact Dermatitis", "eczema": "Eczema",
    "nail_fungus": "Nail Fungus", "psoriasis": "Psoriasis",
    "seborrheic_keratoses": "Seborrheic Keratoses", "tinea_ringworm": "Tinea / Ringworm",
    "urticaria": "Urticaria (Hives)", "warts": "Warts",
}

TRAINING_COUNTS = {
    "acne_clear": 400, "acne_mild": 400, "acne_moderate": 186, "acne_rosacea": 1000,
    "acne_severe": 137, "atopic_dermatitis": 612, "cellulitis": 361,
    "contact_dermatitis": 325, "eczema": 1000, "nail_fungus": 1000,
    "psoriasis": 1000, "seborrheic_keratoses": 1000, "tinea_ringworm": 1000,
    "urticaria": 265, "warts": 1000,
}

PER_CLASS_METRICS = {
    "acne_clear": {"f1": 0.79}, "acne_mild": {"f1": 0.69}, "acne_moderate": {"f1": 0.54},
    "acne_rosacea": {"f1": 0.70}, "acne_severe": {"f1": 0.78},
    "atopic_dermatitis": {"f1": 0.53}, "cellulitis": {"f1": 0.31},
    "contact_dermatitis": {"f1": 0.54}, "eczema": {"f1": 0.48},
    "nail_fungus": {"f1": 0.86}, "psoriasis": {"f1": 0.40},
    "seborrheic_keratoses": {"f1": 0.59}, "tinea_ringworm": {"f1": 0.53},
    "urticaria": {"f1": 0.54}, "warts": {"f1": 0.58},
}

CONFUSION_NOTES = [
    ("Eczema → Atopic Dermatitis", "These two are medically the same condition under different names. The model's confusion here isn't a failure — it reflects a real labeling overlap in the dataset."),
    ("Psoriasis → Seborrheic Keratoses & Tinea", "All three involve scaly, patchy skin. Even dermatologists can struggle to differentiate these visually without additional context."),
    ("Moderate Acne → Mild & Severe", "Adjacent severity levels naturally share visual features. The model correctly identifies the acne family but struggles at the severity boundaries — which mirrors how dermatologists grade acne too."),
    ("Cellulitis → Acne & Rosacea", "Both involve facial redness and inflammation. Cellulitis is also the model's weakest class, partly due to its smaller training set of 361 images."),
]

# ── Recommendations ───────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "acne_clear": {
        "display": "Clear skin (no active acne)", "severity": "clear",
        "look_for": ["Gentle cleanser", "SPF 30+ moisturizer", "Antioxidant serum (Vitamin C)"],
        "avoid": ["Heavy occlusive oils", "Skipping sunscreen"],
        "note": "Maintain your routine and focus on prevention."
    },
    "acne_mild": {
        "display": "Mild acne", "severity": "mild",
        "look_for": ["Salicylic acid (BHA) 0.5–2%", "Niacinamide 5–10%", "Gentle foaming cleanser", "Oil-free moisturizer"],
        "avoid": ["Heavy creams", "Alcohol-heavy toners", "Picking or squeezing"],
        "note": "Consistency is key — give products 6–8 weeks to work."
    },
    "acne_moderate": {
        "display": "Moderate acne", "severity": "moderate",
        "look_for": ["Benzoyl peroxide 2.5–5%", "Salicylic acid", "Niacinamide", "Azelaic acid 10%"],
        "avoid": ["Comedogenic moisturizers", "Heavy makeup", "Harsh scrubs"],
        "note": "Consider consulting a dermatologist if OTC products aren't helping after 8 weeks."
    },
    "acne_severe": {
        "display": "Severe acne", "severity": "severe",
        "look_for": ["Benzoyl peroxide 5–10%", "Azelaic acid", "Gentle non-comedogenic moisturizer"],
        "avoid": ["DIY treatments", "Over-exfoliating", "Oil-based products"],
        "note": "Severe acne typically benefits significantly from professional dermatologist care."
    },
    "acne_rosacea": {
        "display": "Acne & Rosacea", "severity": "general",
        "look_for": ["Azelaic acid 10–15%", "Niacinamide", "Gentle mineral sunscreen", "Fragrance-free moisturizer"],
        "avoid": ["Alcohol", "Menthol", "Spicy foods (trigger)", "Extreme temperatures"],
        "note": "Rosacea is a chronic condition — a dermatologist can recommend prescription options like metronidazole."
    },
    "atopic_dermatitis": {
        "display": "Atopic Dermatitis", "severity": "general",
        "look_for": ["Heavy fragrance-free moisturizer", "Ceramide creams", "Colloidal oatmeal", "Gentle soap-free cleanser"],
        "avoid": ["Fragrances", "Harsh soaps", "Wool fabrics", "Hot showers"],
        "note": "Keeping skin well-moisturized is the cornerstone of atopic dermatitis management."
    },
    "cellulitis": {
        "display": "Cellulitis", "severity": "general",
        "look_for": ["Keep area clean and elevated", "Gentle wound care if applicable"],
        "avoid": ["Self-treating with topical creams", "Delaying medical care"],
        "note": "⚠️ Cellulitis is a bacterial skin infection that typically requires antibiotic treatment. Please consult a doctor promptly."
    },
    "contact_dermatitis": {
        "display": "Contact Dermatitis", "severity": "general",
        "look_for": ["Hydrocortisone 1% cream (short-term)", "Fragrance-free moisturizer", "Colloidal oatmeal bath", "Cold compress"],
        "avoid": ["The identified irritant or allergen", "Fragrances", "Nickel jewelry", "Latex"],
        "note": "Identifying and avoiding the trigger is the most important step."
    },
    "eczema": {
        "display": "Eczema", "severity": "general",
        "look_for": ["Ceramide-rich moisturizer", "Fragrance-free products", "Colloidal oatmeal", "Gentle cleanser"],
        "avoid": ["Fragrances", "Harsh soaps", "Hot water", "Synthetic fabrics"],
        "note": "Moisturize immediately after bathing while skin is still slightly damp."
    },
    "nail_fungus": {
        "display": "Nail Fungus", "severity": "general",
        "look_for": ["Antifungal nail treatment (clotrimazole, terbinafine)", "Keep nails trimmed and dry", "Breathable footwear"],
        "avoid": ["Nail polish over infected nails", "Walking barefoot in public areas", "Sharing nail tools"],
        "note": "Nail fungus is slow to treat — OTC treatments take 3–6 months."
    },
    "psoriasis": {
        "display": "Psoriasis", "severity": "general",
        "look_for": ["Coal tar shampoo/cream", "Salicylic acid (scale removal)", "Heavy fragrance-free moisturizer"],
        "avoid": ["Skin injury (can trigger flares)", "Stress", "Alcohol", "Smoking"],
        "note": "Psoriasis is a chronic autoimmune condition — a dermatologist can offer prescription treatments."
    },
    "seborrheic_keratoses": {
        "display": "Seborrheic Keratoses", "severity": "general",
        "look_for": ["Gentle moisturizer", "SPF 30+ sunscreen"],
        "avoid": ["Picking or scratching lesions", "Harsh exfoliants on affected areas"],
        "note": "Seborrheic keratoses are benign growths and do not require treatment."
    },
    "tinea_ringworm": {
        "display": "Tinea / Ringworm", "severity": "general",
        "look_for": ["Antifungal cream (clotrimazole, miconazole)", "Keep area clean and dry", "Loose breathable clothing"],
        "avoid": ["Sharing towels or clothing", "Tight clothing over affected area", "Scratching (spreads infection)"],
        "note": "Ringworm is a fungal infection (not a worm) and is contagious."
    },
    "urticaria": {
        "display": "Urticaria (Hives)", "severity": "general",
        "look_for": ["OTC antihistamine (cetirizine, loratadine)", "Cool compress", "Fragrance-free soothing lotion"],
        "avoid": ["Known triggers (foods, medications, stress)", "Hot showers", "Tight clothing"],
        "note": "If hives are accompanied by difficulty breathing or throat swelling, seek emergency care immediately."
    },
    "warts": {
        "display": "Warts", "severity": "general",
        "look_for": ["Salicylic acid treatment (OTC)", "Keep area clean and covered", "Cryotherapy (dermatologist)"],
        "avoid": ["Picking or cutting warts", "Sharing towels", "Walking barefoot in public (plantar warts)"],
        "note": "Warts are caused by HPV and can resolve on their own over months to years."
    },
}

SEVERITY_LABELS = {
    "acne_clear":    ("Clear",    "badge-clear"),
    "acne_mild":     ("Mild",     "badge-mild"),
    "acne_moderate": ("Moderate", "badge-moderate"),
    "acne_severe":   ("Severe",   "badge-severe"),
}

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    checkpoint  = torch.load(MODEL_PATH, map_location='cpu')
    class_names = checkpoint['class_names']
    num_classes  = checkpoint['num_classes']
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_features, 512),
        nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(512, num_classes)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names

def predict(image, model, class_names):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
    top5_probs, top5_idx = torch.topk(probs, 5)
    return [(class_names[idx.item()], prob.item()) for idx, prob in zip(top5_idx, top5_probs)]

def get_ai_advice(condition, confidence):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "API key not configured."
    client = anthropic.Anthropic(api_key=api_key)
    rec = RECOMMENDATIONS.get(condition, {})
    display_name = rec.get("display", condition)
    prompt = f"""You are a helpful skincare assistant. A user uploaded a photo and the ML model predicted: {display_name} (confidence: {confidence*100:.0f}%).

Provide 3-4 sentences of friendly, practical skincare advice for this condition.
- Focus on daily habits and OTC options
- Mention when to see a dermatologist
- Be empathetic and non-judgmental
- Do NOT diagnose or replace medical advice
Do not repeat the condition name in the opening sentence."""
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem 0;">
        <span style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:#1a1a1a;">SkinScan</span><br>
        <span style="font-size:0.75rem;color:#9CA3AF;">ML skin condition classifier</span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", ["🔬 Classifier", "📊 Model insights"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#9CA3AF;line-height:1.8;">
        <strong>Datasets</strong><br>
        DermNet (Kaggle)<br>
        ACNE04 (ICCV 2019)<br><br>
        <strong>Model</strong><br>
        MobileNetV2<br>
        Transfer learning · PyTorch<br><br>
        <strong>Classes:</strong> 15<br>
        <strong>Test accuracy:</strong> 59.3%
    </div>
    """, unsafe_allow_html=True)

# ── Page: Classifier ──────────────────────────────────────────────────────────
if page == "🔬 Classifier":
    st.markdown("""
    <div class="title-block">
        <h1>SkinScan</h1>
        <p>ML-powered skin condition classifier · Educational use only</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        ⚕️ <strong>Medical disclaimer:</strong> SkinScan is an educational ML project and is <strong>not a medical diagnostic tool</strong>. Results are not a substitute for professional medical advice. Always consult a qualified dermatologist. Skin conditions are medical issues — not hygiene failures.
    </div>
    """, unsafe_allow_html=True)

    try:
        model, class_names = load_model()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.info("Make sure skin_classifier.pt is in the model/ folder.")
        st.stop()

    uploaded_file = st.file_uploader("Upload a photo of the affected skin area", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing..."):
            results = predict(image, model, class_names)

        top_class, top_conf = results[0]
        rec          = RECOMMENDATIONS.get(top_class, {})
        display_name = rec.get("display", top_class.replace("_", " ").title())

        if top_class in SEVERITY_LABELS:
            sev_text, sev_class = SEVERITY_LABELS[top_class]
            badge_html = f'<span class="severity-badge {sev_class}">{sev_text}</span>'
        else:
            badge_html = '<span class="severity-badge badge-general">Detected</span>'

        st.markdown(f"""
        <div class="result-card">
            <p class="section-heading">Condition detected</p>
            <p class="condition-name">{display_name}{badge_html}</p>
            <p style="color:#6B7280;font-size:0.88rem;margin-top:4px">Confidence: {top_conf*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-heading">Confidence breakdown</p>', unsafe_allow_html=True)
        for cls, prob in results:
            label = RECOMMENDATIONS.get(cls, {}).get("display", cls.replace("_", " ").title())
            st.markdown(f'<p class="confidence-label">{label} — {prob*100:.1f}%</p>', unsafe_allow_html=True)
            st.progress(float(prob))

        if rec:
            st.markdown('<p class="section-heading">Recommended ingredients & habits</p>', unsafe_allow_html=True)
            look_for = rec.get("look_for", [])
            avoid    = rec.get("avoid", [])
            note     = rec.get("note", "")
            if look_for:
                st.markdown("**Look for**")
                st.markdown(" ".join([f'<span class="pill">{i}</span>' for i in look_for]), unsafe_allow_html=True)
            if avoid:
                st.markdown("**Avoid**")
                st.markdown(" ".join([f'<span class="pill pill-avoid">{i}</span>' for i in avoid]), unsafe_allow_html=True)
            if note:
                st.caption(f"💡 {note}")

        st.markdown("---")
        if st.button("✨ Get AI-powered skincare advice for my specific case"):
            with st.spinner("Getting personalised advice..."):
                advice = get_ai_advice(top_class, top_conf)
            st.markdown(f'<div class="ai-response">{advice}</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer-note">SkinScan · PyTorch · MobileNetV2 · Streamlit · Educational project<br>DermNet & ACNE04 datasets · 15 classes · 59.3% test accuracy</div>', unsafe_allow_html=True)

# ── Page: Model insights ──────────────────────────────────────────────────────
elif page == "📊 Model insights":
    st.markdown("""
    <div class="title-block">
        <h1>Model insights</h1>
        <p>How SkinScan works, what it learned, and where it struggles</p>
    </div>
    """, unsafe_allow_html=True)

    # Overview stats
    st.markdown("### Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="stat-card"><p class="stat-number">59.3%</p><p class="stat-label">Test accuracy</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-card"><p class="stat-number">15</p><p class="stat-label">Conditions</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-card"><p class="stat-number">9,686</p><p class="stat-label">Training images</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="stat-card"><p class="stat-number">20</p><p class="stat-label">Training epochs</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # How it works
    st.markdown("### How the model works")
    st.markdown("""
    <div class="insight-box">
        <strong>Transfer learning with MobileNetV2</strong><br><br>
        Rather than training from scratch — which would require millions of images — SkinScan uses <em>transfer learning</em>. MobileNetV2 is a CNN pretrained by Google on 1.2 million images. It already knows how to detect edges, textures, colors, and shapes.<br><br>
        We replaced only the final classification layer with a new one trained to recognise 15 skin conditions. In Phase A, the base model was frozen and only the new classifier head trained (10 epochs). In Phase B, the top layers were unfrozen and fine-tuned at a lower learning rate (0.0001), boosting val accuracy from 47.8% to 60.2%.<br><br>
        Each image is resized to 224×224 pixels, normalised using ImageNet statistics, and passed through the network. The final layer produces 15 confidence scores converted to probabilities via softmax.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Training data distribution
    st.markdown("### Training data distribution")
    st.caption("Images per class after preprocessing. Most classes capped at 1,000 images to reduce imbalance.")

    labels = [CLASS_DISPLAY[k] for k in TRAINING_COUNTS]
    counts = list(TRAINING_COUNTS.values())
    colors = ["#6366F1" if c == 1000 else "#F59E0B" if c >= 300 else "#EF4444" for c in counts]

    fig_dist = go.Figure(go.Bar(
        x=counts, y=labels, orientation='h',
        marker_color=colors,
    ))
    fig_dist.update_layout(
        height=500, margin=dict(l=180, r=40, t=10, b=10),
        xaxis_title="Number of images",
        yaxis=dict(autorange="reversed", tickfont=dict(color="#1a1a1a", size=12)),
        xaxis=dict(gridcolor="#F3F4F6", tickfont=dict(color="#555555")),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="DM Sans", size=12, color="#1a1a1a"),
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown('<div style="font-size:0.8rem;color:#6B7280;margin-top:-0.5rem;"><span style="color:#6366F1">■</span> 1,000 (capped) &nbsp;<span style="color:#F59E0B">■</span> 300–612 &nbsp;<span style="color:#EF4444">■</span> Under 300</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Per-class F1
    st.markdown("### Per-class accuracy (F1 score)")
    st.caption("F1 balances precision and recall. Higher is better. 1.0 = perfect.")

    sorted_cls = sorted(PER_CLASS_METRICS.keys(), key=lambda k: PER_CLASS_METRICS[k]["f1"], reverse=True)
    f1_labels  = [CLASS_DISPLAY[k] for k in sorted_cls]
    f1_scores  = [PER_CLASS_METRICS[k]["f1"] for k in sorted_cls]
    f1_colors  = ["#10B981" if s >= 0.70 else "#6366F1" if s >= 0.50 else "#EF4444" for s in f1_scores]

    fig_f1 = go.Figure(go.Bar(
        x=f1_scores, y=f1_labels, orientation='h',
        marker_color=f1_colors,
    ))
    fig_f1.update_layout(
        height=500, margin=dict(l=180, r=40, t=10, b=10),
        xaxis=dict(range=[0, 1.05], gridcolor="#F3F4F6", tickformat=".0%", tickfont=dict(color="#555555")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#1a1a1a", size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="DM Sans", size=12, color="#1a1a1a"),
    )
    st.plotly_chart(fig_f1, use_container_width=True)
    st.markdown('<div style="font-size:0.8rem;color:#6B7280;margin-top:-0.5rem;"><span style="color:#10B981">■</span> Strong (≥0.70) &nbsp;<span style="color:#6366F1">■</span> Moderate (0.50–0.69) &nbsp;<span style="color:#EF4444">■</span> Weak (<0.50)</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Notable confusions
    st.markdown("### Notable confusions — and why they make sense")
    st.caption("Where the model makes mistakes, the mistakes are usually clinically logical.")
    for title, explanation in CONFUSION_NOTES:
        with st.expander(title):
            st.markdown(explanation)

    st.markdown("---")

    # Limitations
    st.markdown("### Honest limitations")
    st.markdown("""
    <div class="insight-box">
        <strong>Skin tone bias</strong> — Both DermNet and ACNE04 skew toward lighter skin tones. Performance on darker skin tones is likely lower and has not been systematically evaluated. This is a known and serious problem in dermatology AI research.<br><br>
        <strong>Image quality sensitivity</strong> — The model was trained on clinical photographs. Blurry or poorly lit photos will produce less reliable predictions.<br><br>
        <strong>Small classes</strong> — Cellulitis (361 training images) and urticaria (265 images) are underrepresented. The model is measurably weaker on these conditions.<br><br>
        <strong>Overlapping conditions</strong> — Eczema and atopic dermatitis are the same condition. Psoriasis, seborrheic keratoses, and tinea ringworm all involve scaly patches and are frequently confused — by the model and by clinicians.<br><br>
        <strong>59.3% is the ceiling for now</strong> — A larger, more diverse, professionally labelled dataset would meaningfully improve performance. This is an educational project, not a clinical tool.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset credits
    st.markdown("### Dataset sources")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**DermNet**\n\n23 categories of skin disease images · Sourced via Kaggle\n\n[View dataset](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)")
    with col2:
        st.markdown("**ACNE04**\n\nAcne severity dataset · ICCV 2019 · Wu et al.\n\n[View dataset](https://www.kaggle.com/datasets/jincyjis/acne04)")

    st.markdown('<div class="footer-note">SkinScan · PyTorch · MobileNetV2 · Streamlit · Educational portfolio project</div>', unsafe_allow_html=True)
