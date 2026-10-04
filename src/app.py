"""Streamlit app: upload a brain MRI and get the predicted tumour type with confidence."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing import val_test_transform  # noqa: E402
from model_registry import (  # noqa: E402
    CLASS_NAMES, DISPLAY_NAMES, MODELS_DIR, available_models, load_trained_model,
)

CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary",
}
COMPARISON_PATH = MODELS_DIR / "comparison.json"

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")


@st.cache_resource
def get_model(name):
    return load_trained_model(name, device="cpu")


@st.cache_data
def load_comparison():
    if COMPARISON_PATH.exists():
        return json.loads(COMPARISON_PATH.read_text())
    return None


models_found = available_models()
if not models_found:
    st.error(f"No trained models found in {MODELS_DIR}. Train one first (see README).")
    st.stop()

comparison = load_comparison()
best_model = comparison["best_model"] if comparison else models_found[0]
default_index = models_found.index(best_model) if best_model in models_found else 0

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox(
        "Choose a model",
        models_found,
        index=default_index,
        format_func=lambda n: DISPLAY_NAMES[n] + ("  ★ best" if n == best_model else ""),
    )
    if comparison:
        row = next((r for r in comparison["results"] if r["model"] == model_name), None)
        if row:
            st.metric("Test accuracy", f"{row['accuracy']:.2%}")
            st.caption(f"F1 {row['f1']:.3f} · {row['size_mb']:.1f} MB · {row['params_millions']:.1f}M params")

model = get_model(model_name)

st.title("🧠 Brain Tumor MRI Classifier")
st.caption("Classifies a brain MRI scan as glioma, meningioma, pituitary tumor, or no tumor. "
           "For educational use only — not a medical diagnosis.")

predict_tab, compare_tab, quality_tab = st.tabs(["Predict", "Model Comparison", "Model Quality"])

with predict_tab:
    uploaded = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        tensor = val_test_transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        pred_idx = int(probs.argmax())
        predicted = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        col_img, col_result = st.columns([1, 1])
        with col_img:
            st.image(img, caption=uploaded.name, use_column_width=True)
        with col_result:
            st.subheader("Prediction")
            if predicted == "notumor":
                st.success(f"**{CLASS_LABELS[predicted]}**")
            else:
                st.warning(f"**{CLASS_LABELS[predicted]}**")
            st.metric("Confidence", f"{confidence:.2%}")
            if confidence < 0.6:
                st.info("Low confidence — the model is unsure about this image.")
            st.caption(f"Model: {DISPLAY_NAMES[model_name]}")

        st.subheader("Class probabilities")
        prob_df = pd.DataFrame({
            "Class": [CLASS_LABELS[c] for c in CLASS_NAMES],
            "Probability": [float(p) for p in probs],
        }).set_index("Class")
        st.bar_chart(prob_df)

with compare_tab:
    if not comparison:
        st.info("Run `python src/evaluation.py` to generate the model comparison.")
    else:
        df = pd.DataFrame(comparison["results"])
        df["model"] = df["model"].map(DISPLAY_NAMES)
        st.dataframe(
            df.rename(columns={
                "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
                "recall": "Recall", "f1": "F1", "params_millions": "Params (M)",
                "size_mb": "Size (MB)", "ms_per_image": "ms / image",
            }).style.format(precision=4),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Best model on the test set: **{DISPLAY_NAMES[comparison['best_model']]}**")

        for name in [r["model"] for r in comparison["results"]]:
            cm_path = MODELS_DIR / f"{name}_cm.png"
            hist_path = MODELS_DIR / f"{name}_history.png"
            with st.expander(DISPLAY_NAMES[name]):
                if hist_path.exists():
                    st.image(str(hist_path), caption="Training history")
                if cm_path.exists():
                    st.image(str(cm_path), caption="Confusion matrix (test set)")

with quality_tab:
    st.write("Run the selected model on a balanced random sample of the Testing set.")
    samples_per_class = st.slider("Testing images per class", min_value=2, max_value=30, value=8, step=2)

    if st.button("Run quality check", type="primary"):
        from model_quality import evaluate_model_quality

        with st.spinner("Evaluating model on the test dataset sample..."):
            result = evaluate_model_quality(model, val_test_transform,
                                            samples_per_class=samples_per_class, batch_size=8)

        st.metric("Sample accuracy", f"{result['accuracy']:.2%}")
        st.caption(f"Evaluated {result['sample_count']} testing images.")

        class_labels = [CLASS_LABELS.get(c, c) for c in result["classes"]]
        st.subheader("Confusion Matrix")
        st.dataframe(
            pd.DataFrame(result["confusion_matrix"], index=class_labels, columns=class_labels),
            use_container_width=True,
        )

        st.subheader("Per-Class Scores")
        rows = []
        for class_name in result["classes"]:
            metrics = result["report"][class_name]
            rows.append({
                "Class": CLASS_LABELS.get(class_name, class_name),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1": round(metrics["f1-score"], 4),
                "Support": int(metrics["support"]),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
