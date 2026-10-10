import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ExifTags
from huggingface_hub import hf_hub_download
import onnxruntime as ort

APP_TITLE = "SAFER Verify"
MODEL_REPO = "TensorFold/detectra-v1"
MODEL_FILE = "model.onnx"
EXPECTED_MODEL_SHA256 = "1414b9aafaa01a644ed706224973f09b53a1388282104409862df9893b1b962b"
MODEL_THRESHOLD = 0.65
MAX_UPLOAD_MB = 15
ALLOWED_TYPES = {"png", "jpg", "jpeg", "webp"}

st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="wide")

st.title("🛡️ SAFER Verify")
st.caption("Research prototype · Image signal assessment, not a proof-of-authenticity service")

st.warning(
    "Important: this experimental tool cannot prove that an image is genuine or fake. "
    "It estimates whether the image resembles AI-generated examples seen by one model. "
    "Do not use its output alone to accuse someone, remove content, make legal decisions, "
    "or approve high-stakes transactions."
)

@st.cache_resource(show_spinner="Loading the research model (first run downloads it)...")
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    if digest.lower() != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "The downloaded model checksum did not match the published model-card checksum. "
            "For safety, inference was stopped."
        )
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
    return session, digest

def preprocess(image: Image.Image) -> np.ndarray:
    """Model-card preprocessing: shortest edge 440, center crop 384, ImageNet normalization."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = 440 / min(width, height)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (new_w - 384) // 2
    top = (new_h - 384) // 2
    image = image.crop((left, top, left + 384, top + 384))
    array = np.asarray(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)

def sigmoid(x):
    x = float(np.asarray(x).reshape(-1)[0])
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)

def predict(session, image):
    inputs = preprocess(image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inputs})
    # Model card documents one logit, with sigmoid(logit) as the AI-generated score.
    return float(sigmoid(outputs[0]))

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def inspect_c2pa(upload_bytes: bytes, suffix: str):
    """Optional C2PA inspection via the official c2patool CLI, if installed locally."""
    exe = shutil.which("c2patool")
    if not exe:
        return {
            "status": "unavailable",
            "summary": "c2patool is not installed. Provenance was not checked.",
            "details": None,
        }
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(upload_bytes)
            path = f.name
        proc = subprocess.run(
            [exe, path],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        raw = (proc.stdout or "").strip()
        if not raw:
            return {
                "status": "unknown",
                "summary": "The provenance tool returned no report.",
                "details": (proc.stderr or "")[:1200],
            }
        try:
            report = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "status": "unknown",
                "summary": "The provenance tool returned a non-JSON report.",
                "details": raw[:2500],
            }

        if not report or not report.get("manifests"):
            return {
                "status": "not_found",
                "summary": "No embedded C2PA manifest was reported. This does not mean the image is fake.",
                "details": report,
            }

        validation = report.get("validation_status")
        active = report.get("active_manifest")
        summary = "A C2PA manifest was found. Review its validation status and claims."
        if validation:
            summary = "A C2PA manifest was found, but the report includes validation-status entries that need review."
        return {
            "status": "found",
            "summary": summary,
            "details": {
                "active_manifest": active,
                "validation_status": validation,
                "manifest_keys": list(report.get("manifests", {}).keys())[:10],
            },
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "summary": "C2PA inspection timed out.", "details": None}
    except Exception as exc:
        return {"status": "error", "summary": "C2PA inspection could not be completed.", "details": str(exc)[:1200]}
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

with st.sidebar:
    st.header("About this prototype")
    st.markdown(
        "- **Model:** Detectra v3 ONNX model published as `TensorFold/detectra-v1`\n"
        "- **Inference:** local CPU processing\n"
        "- **Provenance:** optional local `c2patool` check\n"
        "- **Uploads:** not intentionally sent to a remote inference API\n"
        "- **Model threshold:** 0.65, as described in the model card"
    )
    st.markdown("[Model card](https://huggingface.co/TensorFold/detectra-v1)")
    st.markdown("[C2PA tool](https://github.com/contentauth/c2patool)")
    st.caption("The model is third-party research software. Its published evaluation is not an independent evaluation by SAFER AI.")

uploaded = st.file_uploader(
    "Choose an image to assess",
    type=sorted(ALLOWED_TYPES),
    help="Supported: PNG, JPG/JPEG, WebP. Maximum file size: 15 MB.",
)

if uploaded is not None:
    data = uploaded.getvalue()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File is larger than {MAX_UPLOAD_MB} MB. Choose a smaller image.")
        st.stop()

    try:
        image = Image.open(uploaded)
        image.verify()
        uploaded.seek(0)
        image = Image.open(uploaded)
        if image.width < 32 or image.height < 32:
            st.error("Image is too small for this prototype.")
            st.stop()
    except Exception:
        st.error("This file could not be decoded as a supported image.")
        st.stop()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption=f"{uploaded.name} · {image.width} × {image.height}", use_container_width=True)
        st.caption(f"SHA-256: `{sha256_bytes(data)}`")
        st.caption("The hash identifies these exact file bytes; it does not establish who created the image.")

    with col2:
        st.subheader("1. Synthetic-image signal")
        try:
            session, model_hash = load_model()
            with st.spinner("Assessing image locally..."):
                score = predict(session, image)
            st.metric("Model score", f"{score * 100:.1f}%")
            st.progress(min(max(score, 0.0), 1.0))

            if score >= MODEL_THRESHOLD:
                st.error("AI-generated signal detected by this model")
                st.write(
                    "The score is at or above the model card's 0.65 operating threshold. "
                    "This is a signal for further review, not proof that the image is fake."
                )
            elif score >= 0.35:
                st.warning("Inconclusive / intermediate model score")
                st.write(
                    "The score is below the model's stated threshold but is not sufficiently low "
                    "to treat as reassuring. Verify the source independently."
                )
            else:
                st.info("No strong AI-generated signal from this model")
                st.write(
                    "A low score does not prove the image is genuine. The model may miss new "
                    "generators, edits, or manipulated content."
                )

            with st.expander("Model details and limitations"):
                st.write(f"Model repository: `{MODEL_REPO}`")
                st.write(f"Model file SHA-256 verified: `{model_hash}`")
                st.write("Input preprocessing follows the model card: resize shortest edge to 440, center-crop 384, ImageNet normalization.")
                st.write("This model assesses AI-generated-image signals. It is not a general-purpose forensic detector for every face swap, composite, altered photograph, or deceptive caption.")
        except Exception as exc:
            st.error("The model could not be loaded or run.")
            st.code(str(exc)[:2500])
            st.info("Check internet access for the initial model download and confirm that the required packages are installed.")

        st.subheader("2. Content provenance")
        c2pa = inspect_c2pa(data, Path(uploaded.name).suffix.lower() or ".img")
        if c2pa["status"] == "found":
            st.success(c2pa["summary"])
        elif c2pa["status"] == "not_found":
            st.info(c2pa["summary"])
        elif c2pa["status"] == "unavailable":
            st.info(c2pa["summary"])
        else:
            st.warning(c2pa["summary"])
        if c2pa.get("details") is not None:
            with st.expander("C2PA report details"):
                st.json(c2pa["details"])

    st.subheader("3. Safe interpretation")
    st.markdown(
        "- **A detector score is not a probability that a person lied.** It is a model output for this file.\n"
        "- **No C2PA credentials does not mean fake.** Many genuine files have no credentials.\n"
        "- **Valid credentials do not prove the scene is truthful.** They provide provenance information under the credential's trust and validation conditions.\n"
        "- **For consequential decisions, seek corroborating evidence and human review.** Preserve the original file and document where it came from."
    )
else:
    st.info("Upload an image to start. The app will not produce a result until a file is selected.")

st.divider()
st.caption(
    "SAFER AI research prototype v0.1 · Not for high-stakes decisions · "
    "Do not upload intimate, confidential, or otherwise sensitive images to a prototype unless you understand your local environment and data handling."
)
