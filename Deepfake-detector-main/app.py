"""
Deepfake Image & Audio Detection — Multi-Agent AI System (Streamlit)
======================================================================
Stack: LangGraph (multi-agent orchestration) + Hugging Face Transformers
       (detection models) + Groq / LangChain (explanation LLM) + Streamlit (UI)

Run locally:
    pip install -r requirements.txt
    export GROQ_API_KEY="your-key-here"   # optional — see note below
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push this repo, point the app at app.py, and add GROQ_API_KEY under
    "Secrets" in the app settings (Settings -> Secrets):
        GROQ_API_KEY = "your-key-here"
    The app also works with no key at all — it falls back to a
    template-based explanation instead of the LLM.

Models used (verified live on the Hugging Face Hub, Aug 2026):
  Image : prithivMLmods/Deep-Fake-Detector-v2-Model   (ViT, "Realism"/"Deepfake")
          fallback -> Wvolf/ViT_Deepfake_Detection
  Audio : MelodyMachine/Deepfake-audio-detection-V2    (Wav2Vec2-based)
          fallback -> mo-thecreator/Deepfake-audio-detection
  LLM   : openai/gpt-oss-20b on Groq (llama-3.1-8b-instant was deprecated
          by Groq in June 2026 — gpt-oss-20b is its recommended replacement)
"""

import os
import io
import gc
import uuid
import tempfile
import traceback
from datetime import datetime
from typing import TypedDict, Optional, Literal, Any, Dict

import numpy as np
import torch
from PIL import Image
import cv2

import librosa
import soundfile as sf  # noqa: F401  (kept for explicit wav I/O support)
from pydub import AudioSegment

from transformers import pipeline

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from fpdf import FPDF
import streamlit as st

# =========================================================
# CONFIGURATION
# =========================================================


class Config:
    IMAGE_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    IMAGE_MODEL_FALLBACK = "Wvolf/ViT_Deepfake_Detection"

    # There have been user reports of prithivMLmods/Deep-Fake-Detector-v2-Model
    # shipping an inverted id2label mapping, but this cannot be verified from
    # here (no network access to huggingface.co in this environment) and a
    # blind guess at the direction has already proven WRONG in practice (it
    # made deepfakes show up as REAL). Do NOT hardcode a flip based on
    # unverified forum reports again. Instead this is now a runtime-toggleable,
    # per-model-id setting that defaults to "not inverted" (trust the pipeline's
    # raw label) and is only flipped if the sidebar calibration check (see
    # `calibrate_image_label_orientation` and the "Detect" page's calibration
    # expander) empirically proves the model is backwards for THIS deployment.
    IMAGE_MODEL_LABELS_INVERTED_DEFAULT = False

    AUDIO_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
    AUDIO_MODEL_FALLBACK = "mo-thecreator/Deepfake-audio-detection"

    # llama-3.1-8b-instant was deprecated by Groq (June 2026).
    # openai/gpt-oss-20b is Groq's recommended replacement; llama-3.3-70b-versatile
    # is kept as a secondary fallback in case gpt-oss-20b is unavailable on your account.
    GROQ_MODEL = "openai/gpt-oss-20b"
    GROQ_MODEL_FALLBACK = "llama-3.3-70b-versatile"

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    AUDIO_SAMPLE_RATE = 16000
    AUDIO_MAX_SECONDS = 10
    IMAGE_TARGET_SIZE = (224, 224)
    # Conservative decision policy. These are NOT claimed as universal thresholds;
    # they are intentionally conservative defaults until calibrated on a labeled
    # validation set with evaluate_detector.py.
    IMAGE_MIN_CONFIDENCE = 0.75
    IMAGE_MIN_MARGIN = 0.20
    IMAGE_UNCERTAIN_CONFIDENCE = 0.60
    IMAGE_MAX_BLUR_SCORE = 45.0
    IMAGE_MIN_BRIGHTNESS = 15.0
    IMAGE_MAX_BRIGHTNESS = 245.0
    IMAGE_MIN_DIMENSION = 64

    REPORT_DIR = os.environ.get("REPORT_DIR", tempfile.gettempdir())


os.makedirs(Config.REPORT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Which image model actually ended up loaded (set by load_image_model()).
# image_detection_agent() reads this to know whether to apply the label-inversion
# correction for the model that is currently active.
_ACTIVE_IMAGE_MODEL_ID: Optional[str] = None

# =========================================================
# UTILITY FUNCTIONS
# =========================================================


def get_file_extension(filepath: str) -> str:
    return os.path.splitext(filepath)[1].lower()


def safe_load_image(filepath: str) -> Optional[Image.Image]:
    try:
        img = Image.open(filepath)
        img.verify()
        img = Image.open(filepath).convert("RGB")
        return img
    except Exception as e:
        print(f"[image loader] failed: {e}")
        return None


def safe_load_audio(filepath: str, sr: int = Config.AUDIO_SAMPLE_RATE):
    try:
        ext = get_file_extension(filepath)
        if ext in {".mp3", ".m4a", ".ogg"}:
            audio = AudioSegment.from_file(filepath)
            audio = audio.set_channels(1).set_frame_rate(sr)
            buf = io.BytesIO()
            audio.export(buf, format="wav")
            buf.seek(0)
            y, _ = librosa.load(buf, sr=sr, mono=True)
        else:
            y, _ = librosa.load(filepath, sr=sr, mono=True)
        if y is None or len(y) == 0:
            raise ValueError("Decoded audio is empty.")
        return y
    except Exception as e:
        print(f"[audio loader] failed: {e}")
        return None


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_confidence(score: float) -> str:
    return f"{score * 100:.2f}%"


def normalize_label(label: str) -> str:
    """Map known model labels to REAL / DEEPFAKE.

    Unknown labels remain UNKNOWN. We never silently interpret an
    unrecognized model label as REAL or DEEPFAKE.
    """
    l = (
        str(label)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    fake_keywords = (
        "fake",
        "spoof",
        "deepfake",
        "synthetic",
        "generated",
        "ai generated",
        "manipulated",
        "forged",
    )

    real_keywords = (
        "real",
        "realism",
        "bonafide",
        "genuine",
        "authentic",
        "human",
        "original",
    )

    if any(keyword in l for keyword in fake_keywords):
        return "DEEPFAKE"

    if any(keyword in l for keyword in real_keywords):
        return "REAL"

    return "UNKNOWN"


def correct_inverted_labels(results, invert: bool):
    """Optionally flip REAL<->DEEPFAKE labels.

    `invert` should come from a value the user has empirically confirmed via
    the calibration check in the Detect page (or manually via the sidebar
    toggle) — never hardcoded from an unverified source. Only the semantic
    REAL/DEEPFAKE meaning is flipped — the raw label text is otherwise left
    untouched so debugging/logging still shows exactly what the model
    literally returned before correction.
    """
    if not invert:
        return results

    flip = {"REAL": "DEEPFAKE", "DEEPFAKE": "REAL"}
    corrected = []
    for item in results:
        semantic = normalize_label(item.get("label", ""))
        new_item = dict(item)
        if semantic in flip:
            new_item["label"] = flip[semantic]
        corrected.append(new_item)
    return corrected


def _prepare_image_for_model(img: Image.Image) -> Image.Image:
    """Apply deterministic preprocessing before model inference."""
    img = img.convert("RGB")

    return img.resize(
        Config.IMAGE_TARGET_SIZE,
        Image.Resampling.LANCZOS
    )
def _image_quality(img: Image.Image) -> Dict[str, Any]:
    """Cheap quality checks used to avoid overconfident predictions."""
    arr = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    w, h = img.size
    issues = []
    if min(w, h) < Config.IMAGE_MIN_DIMENSION:
        issues.append("image resolution is too small")
    if blur_score < Config.IMAGE_MAX_BLUR_SCORE:
        issues.append("image may be too blurry")
    if brightness < Config.IMAGE_MIN_BRIGHTNESS:
        issues.append("image is extremely dark")
    if brightness > Config.IMAGE_MAX_BRIGHTNESS:
        issues.append("image is extremely bright")
    return {
        "width": w,
        "height": h,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "usable": not issues,
        "issues": issues,
    }


def _aggregate_image_scores(results) -> Dict[str, float]:
    """Convert classifier outputs into REAL/DEEPFAKE probabilities.

    Only recognized labels are used. Unknown labels are ignored rather
    than being guessed as REAL or DEEPFAKE.
    """
    scores = {
        "REAL": 0.0,
        "DEEPFAKE": 0.0,
    }

    for item in results:
        label = item.get("label", "")
        score = float(item.get("score", 0.0))

        semantic = normalize_label(label)

        if semantic == "REAL":
            scores["REAL"] += max(0.0, score)

        elif semantic == "DEEPFAKE":
            scores["DEEPFAKE"] += max(0.0, score)

    total = scores["REAL"] + scores["DEEPFAKE"]

    if total > 0:
        scores["REAL"] /= total
        scores["DEEPFAKE"] /= total

    return scores


def _decide_image_label(
    real_p: float,
    fake_p: float,
    quality: Dict[str, Any]
) -> tuple[str, float, str]:
    """Make a conservative REAL / DEEPFAKE / UNKNOWN decision."""

    confidence = max(real_p, fake_p)
    margin = abs(real_p - fake_p)

    # Poor-quality images should not receive a confident verdict.
    if not quality.get("usable", True):
        return (
            "UNKNOWN",
            confidence,
            "Image quality is insufficient for a reliable classification."
        )

    # Very low model confidence.
    if confidence < Config.IMAGE_UNCERTAIN_CONFIDENCE:
        return (
            "UNKNOWN",
            confidence,
            "Model confidence is too low."
        )

    # Confidence or difference between classes is insufficient.
    if (
        confidence < Config.IMAGE_MIN_CONFIDENCE
        or margin < Config.IMAGE_MIN_MARGIN
    ):
        return (
            "UNKNOWN",
            confidence,
            "Model evidence is not decisive enough."
        )

    if real_p > fake_p:
        return (
            "REAL",
            confidence,
            "Model evidence passed the conservative decision policy."
        )

    return (
        "DEEPFAKE",
        confidence,
        "Model evidence passed the conservative decision policy."
    )

# =========================================================
# LANGGRAPH STATE
# =========================================================


class DeepfakeState(TypedDict, total=False):
    filepath: str
    file_name: str
    is_valid: bool
    validation_message: str
    media_type: Literal["image", "audio", "unknown"]
    prediction: Literal["REAL", "DEEPFAKE", "UNKNOWN"]
    raw_scores: Dict[str, float]
    raw_model_labels_uncorrected: Dict[str, float]
    real_probability: float
    deepfake_probability: float
    decision_margin: float
    image_quality: Dict[str, Any]
    confidence: float
    confidence_label: str
    explanation: str
    recommendation: str
    report_paths: Dict[str, str]
    error: Optional[str]
    status: str


# =========================================================
# MODEL LOADING (cached across Streamlit reruns, with fallback)
# =========================================================


@st.cache_resource(show_spinner="Loading image deepfake model...")
def load_image_model():
    global _ACTIVE_IMAGE_MODEL_ID
    for model_id in (Config.IMAGE_MODEL_ID, Config.IMAGE_MODEL_FALLBACK):
        try:
            print(f"[model] loading image model: {model_id}")
            clf = pipeline("image-classification", model=model_id, device=0 if DEVICE == "cuda" else -1)
            print(f"[model] loaded: {model_id} (device={DEVICE})")
            _ACTIVE_IMAGE_MODEL_ID = model_id
            return clf
        except Exception as e:
            print(f"[model] failed to load '{model_id}': {e}")
    return None


@st.cache_resource(show_spinner="Loading audio deepfake model...")
def load_audio_model():
    for model_id in (Config.AUDIO_MODEL_ID, Config.AUDIO_MODEL_FALLBACK):
        try:
            print(f"[model] loading audio model: {model_id}")
            clf = pipeline("audio-classification", model=model_id, device=0 if DEVICE == "cuda" else -1)
            print(f"[model] loaded: {model_id} (device={DEVICE})")
            return clf
        except Exception as e:
            print(f"[model] failed to load '{model_id}': {e}")
    return None


# =========================================================
# AGENT NODES
# =========================================================


def supervisor_agent(state: DeepfakeState) -> DeepfakeState:
    print("=" * 60)
    print(f"[Supervisor] starting analysis for: {state.get('file_name', 'unknown')}")
    state["status"] = "started"
    state["error"] = None
    return state


def input_validation_agent(state: DeepfakeState) -> DeepfakeState:
    filepath = state.get("filepath")
    if not filepath or not os.path.exists(filepath):
        state["is_valid"] = False
        state["validation_message"] = "File not found or path is invalid."
        state["error"] = state["validation_message"]
        return state

    ext = get_file_extension(filepath)
    if ext not in Config.IMAGE_EXTENSIONS and ext not in Config.AUDIO_EXTENSIONS:
        state["is_valid"] = False
        state["validation_message"] = (
            f"Unsupported file type '{ext}'. Supported: "
            f"{sorted(Config.IMAGE_EXTENSIONS | Config.AUDIO_EXTENSIONS)}"
        )
        state["error"] = state["validation_message"]
        return state

    if os.path.getsize(filepath) == 0:
        state["is_valid"] = False
        state["validation_message"] = "File is empty (0 bytes) — possibly corrupted upload."
        state["error"] = state["validation_message"]
        return state

    state["is_valid"] = True
    state["validation_message"] = "File passed validation."
    return state


def media_detection_agent(state: DeepfakeState) -> DeepfakeState:
    if not state.get("is_valid"):
        state["media_type"] = "unknown"
        return state
    ext = get_file_extension(state["filepath"])
    if ext in Config.IMAGE_EXTENSIONS:
        state["media_type"] = "image"
    elif ext in Config.AUDIO_EXTENSIONS:
        state["media_type"] = "audio"
    else:
        state["media_type"] = "unknown"
    return state


def image_detection_agent(state: DeepfakeState) -> DeepfakeState:
    if state.get("media_type") != "image":
        return state

    try:
        # ---------------------------------------------------------
        # Load image
        # ---------------------------------------------------------
        img = safe_load_image(state["filepath"])

        if img is None:
            state["error"] = (
                "Could not decode the uploaded image "
                "(possibly corrupted)."
            )
            state["prediction"] = "UNKNOWN"
            state["raw_scores"] = {}
            return state

        # ---------------------------------------------------------
        # Image quality analysis
        # ---------------------------------------------------------
        quality = _image_quality(img)
        state["image_quality"] = quality

        # ---------------------------------------------------------
        # Load model
        # ---------------------------------------------------------
        clf = load_image_model()

        if clf is None:
            state["error"] = (
                "Image deepfake model failed to load "
                "(check internet connection)."
            )
            state["prediction"] = "UNKNOWN"
            state["raw_scores"] = {}
            return state

        # ---------------------------------------------------------
        # Prepare image
        # ---------------------------------------------------------
        model_img = _prepare_image_for_model(img)

        # ---------------------------------------------------------
        # Run classifier
        # ---------------------------------------------------------
        try:
            results = clf(model_img, top_k=None)
        except TypeError:
            results = clf(model_img)

        # ---------------------------------------------------------
        # Store the raw, uncorrected labels BEFORE any inversion so the
        # calibration check / debug view can show exactly what the model
        # literally returned, independent of the invert toggle below.
        # ---------------------------------------------------------
        state["raw_model_labels_uncorrected"] = {
            str(r.get("label")): float(r.get("score", 0.0)) for r in results
        }

        # ---------------------------------------------------------
        # Apply the invert-labels toggle if the user has enabled it
        # (set empirically via calibration, see calibrate_image_label_orientation
        # and the sidebar/Detect-page toggle — never hardcoded).
        # ---------------------------------------------------------
        try:
            invert_labels = bool(st.session_state.get("invert_image_labels", Config.IMAGE_MODEL_LABELS_INVERTED_DEFAULT))
        except Exception:
            invert_labels = Config.IMAGE_MODEL_LABELS_INVERTED_DEFAULT
        results = correct_inverted_labels(results, invert_labels)

        # ---------------------------------------------------------
        # Sort results by confidence
        # ---------------------------------------------------------
        results = sorted(
            results,
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True
        )

        # ---------------------------------------------------------
        # DEBUG: print the actual model output
        # ---------------------------------------------------------
        print("\n" + "=" * 50)
        print(f"RAW IMAGE MODEL OUTPUT (model={_ACTIVE_IMAGE_MODEL_ID}, "
              f"inversion_corrected={invert_labels})")
        print("=" * 50)

        for result in results:
            print(
                f"Label: {result.get('label')} | "
                f"Score: {float(result.get('score', 0.0)):.6f}"
            )

        print("=" * 50 + "\n")

        # ---------------------------------------------------------
        # Store original (post-correction) model scores
        # ---------------------------------------------------------
        raw_scores = {
            str(result["label"]): float(result["score"])
            for result in results
        }

        # ---------------------------------------------------------
        # Convert model labels into REAL / DEEPFAKE
        # ---------------------------------------------------------
        semantic = _aggregate_image_scores(results)

        real_p = float(semantic.get("REAL", 0.0))
        fake_p = float(semantic.get("DEEPFAKE", 0.0))

        # ---------------------------------------------------------
        # Unknown model labels
        # ---------------------------------------------------------
        if real_p == 0.0 and fake_p == 0.0:
            state["prediction"] = "UNKNOWN"
            state["confidence"] = 0.0
            state["confidence_label"] = "Low"
            state["raw_scores"] = raw_scores
            state["real_probability"] = 0.0
            state["deepfake_probability"] = 0.0
            state["decision_margin"] = 0.0

            state["error"] = (
                "The selected model returned labels that could not "
                "be mapped to REAL or DEEPFAKE."
            )

            return state

        # ---------------------------------------------------------
        # Final conservative decision
        # ---------------------------------------------------------
        prediction, confidence, reason = _decide_image_label(
            real_p,
            fake_p,
            quality
        )

        # ---------------------------------------------------------
        # Store results
        # ---------------------------------------------------------
        state["prediction"] = prediction
        state["raw_scores"] = raw_scores

        state["real_probability"] = real_p
        state["deepfake_probability"] = fake_p

        state["decision_margin"] = abs(real_p - fake_p)

        state["confidence"] = confidence
        state["decision_reason"] = reason

        if confidence >= 0.75:
            state["confidence_label"] = "High"
        elif confidence >= 0.60:
            state["confidence_label"] = "Medium"
        else:
            state["confidence_label"] = "Low"

    except Exception as e:
        traceback.print_exc()

        state["error"] = f"Image detection failed: {e}"
        state["prediction"] = "UNKNOWN"
        state["raw_scores"] = {}

    finally:
        gc.collect()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return state


def audio_detection_agent(state: DeepfakeState) -> DeepfakeState:
    if state.get("media_type") != "audio":
        return state
    try:
        y = safe_load_audio(state["filepath"], sr=Config.AUDIO_SAMPLE_RATE)
        if y is None:
            state["error"] = "Could not decode the uploaded audio (possibly corrupted)."
            state["prediction"] = "UNKNOWN"
            state["raw_scores"] = {}
            return state

        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if len(y_trimmed) == 0:
            y_trimmed = y
        max_len = Config.AUDIO_SAMPLE_RATE * Config.AUDIO_MAX_SECONDS
        if len(y_trimmed) > max_len:
            y_trimmed = y_trimmed[:max_len]

        clf = load_audio_model()
        if clf is None:
            state["error"] = "Audio deepfake model failed to load (check internet connection)."
            state["prediction"] = "UNKNOWN"
            state["raw_scores"] = {}
            return state

        results = clf({"array": y_trimmed, "sampling_rate": Config.AUDIO_SAMPLE_RATE})
        raw_scores = {r["label"]: float(r["score"]) for r in results}
        top = results[0]
        prediction = normalize_label(top["label"])
        state["prediction"] = prediction
        state["raw_scores"] = raw_scores
        state["confidence"] = float(top["score"])
        state["decision_reason"] = "Audio model top-class prediction."
    except Exception as e:
        traceback.print_exc()
        state["error"] = f"Audio detection failed: {e}"
        state["prediction"] = "UNKNOWN"
        state["raw_scores"] = {}
    finally:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return state


def confidence_agent(state: DeepfakeState) -> DeepfakeState:
    confidence = state.get("confidence", 0.0) or 0.0
    if state.get("prediction") == "UNKNOWN":
        label = "Uncertain"
    elif confidence >= 0.90:
        label = "Very High"
    elif confidence >= 0.75:
        label = "High"
    elif confidence >= 0.60:
        label = "Moderate"
    else:
        label = "Low"
    state["confidence_label"] = label
    return state


def _template_explanation(state: DeepfakeState) -> str:
    media = state.get("media_type", "file")
    pred = state.get("prediction", "UNKNOWN")
    conf = format_confidence(state.get("confidence", 0.0) or 0.0)
    scores = state.get("raw_scores", {})
    scores_str = ", ".join(f"{k}: {format_confidence(v)}" for k, v in scores.items()) or "n/a"
    reason = state.get("decision_reason", "")
    if pred == "DEEPFAKE":
        return (f"The {media} was classified as likely DEEPFAKE with {conf} confidence. "
                f"This is a model prediction, not forensic proof. {reason} Raw model scores: {scores_str}.")
    if pred == "REAL":
        return (f"The {media} was classified as likely REAL with {conf} confidence. "
                f"This is a model prediction, not proof of authenticity. {reason} Raw model scores: {scores_str}.")
    return (f"The system returned UNCERTAIN for this {media}. {reason} "
            f"Do not treat this result as evidence of a deepfake. Raw model scores: {scores_str}.")


def explanation_agent(state: DeepfakeState) -> DeepfakeState:
    if state.get("error") and not state.get("prediction"):
        state["explanation"] = f"Analysis could not be completed: {state['error']}"
        return state

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        state["explanation"] = _template_explanation(state)
        return state

    system_prompt = (
        "You are a digital forensics assistant explaining deepfake detection results to a "
        "non-technical end user. Be concise (3-4 sentences), clear, and avoid jargon where "
        "possible. Never claim certainty — describe results as model predictions."
    )
    user_prompt = (
        f"Media type: {state.get('media_type')}\n"
        f"Prediction: {state.get('prediction')}\n"
        f"Confidence: {format_confidence(state.get('confidence', 0.0) or 0.0)}\n"
        f"Raw model scores: {state.get('raw_scores')}\n\n"
        "Explain this result to the user in plain language."
    )

    for model_name in (Config.GROQ_MODEL, Config.GROQ_MODEL_FALLBACK):
        try:
            llm = ChatGroq(model=model_name, api_key=groq_key, temperature=0.3, max_tokens=250)
            response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            state["explanation"] = response.content.strip()
            return state
        except Exception as e:
            print(f"[Explanation Agent] Groq model '{model_name}' failed: {e}")
            continue

    state["explanation"] = _template_explanation(state)
    return state


def recommendation_agent(state: DeepfakeState) -> DeepfakeState:
    pred = state.get("prediction", "UNKNOWN")
    conf_label = state.get("confidence_label", "Low")

    if pred == "DEEPFAKE" and conf_label in ("High", "Very High"):
        rec = (
            "High likelihood of manipulation detected. Do not trust or share this media without "
            "independent verification. Consider reverse image/audio search, metadata inspection, "
            "and consultation with a digital forensics expert before acting on it."
        )
    elif pred == "DEEPFAKE":
        rec = (
            "Possible manipulation detected, but confidence is not high. Treat with caution and "
            "seek a second opinion or additional forensic tools before drawing conclusions."
        )
    elif pred == "REAL" and conf_label in ("High", "Very High"):
        rec = (
            "No strong signs of manipulation detected. The media appears authentic, but automated "
            "detectors are not infallible — apply normal source-verification practices for sensitive content."
        )
    elif pred == "REAL":
        rec = (
            "The media appears likely authentic, but confidence is moderate/low. If this content is "
            "high-stakes (legal, financial, identity-related), corroborate with additional evidence."
        )
    else:
        rec = ("The system could not confidently classify this media. Treat the result as UNCERTAIN; "
               "use a clearer image or an independent forensic method rather than assuming it is fake.")

    state["recommendation"] = rec
    return state


def _build_report_dict(state: DeepfakeState) -> Dict[str, str]:
    return {
        "File Name": state.get("file_name", "N/A"),
        "Media Type": str(state.get("media_type", "N/A")).upper(),
        "Prediction": state.get("prediction", "N/A"),
        "Confidence Score": format_confidence(state.get("confidence", 0.0) or 0.0),
        "Confidence Label": state.get("confidence_label", "N/A"),
        "Real Probability": format_confidence(state.get("real_probability", 0.0) or 0.0),
        "Deepfake Probability": format_confidence(state.get("deepfake_probability", 0.0) or 0.0),
        "Decision Margin": format_confidence(state.get("decision_margin", 0.0) or 0.0),
        "Image Quality": str(state.get("image_quality", {})),
        "Decision Reason": state.get("decision_reason", "N/A"),
        "Explanation": state.get("explanation", "N/A"),
        "Recommendation": state.get("recommendation", "N/A"),
        "Timestamp": timestamp_now(),
    }


def report_generator_agent(state: DeepfakeState) -> DeepfakeState:
    report = _build_report_dict(state)
    run_id = uuid.uuid4().hex[:8]
    base_name = f"deepfake_report_{run_id}"
    paths = {}

    try:
        txt_path = os.path.join(Config.REPORT_DIR, f"{base_name}.txt")
        with open(txt_path, "w") as f:
            f.write("DEEPFAKE DETECTION REPORT\n" + "=" * 40 + "\n")
            for k, v in report.items():
                f.write(f"{k}: {v}\n")
        paths["txt"] = txt_path

        md_path = os.path.join(Config.REPORT_DIR, f"{base_name}.md")
        with open(md_path, "w") as f:
            f.write("# Deepfake Detection Report\n\n")
            for k, v in report.items():
                f.write(f"**{k}:** {v}\n\n")
        paths["md"] = md_path

        pdf_path = os.path.join(Config.REPORT_DIR, f"{base_name}.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Deepfake Detection Report", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(4)
        for k, v in report.items():
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 7, f"{k}:")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 7, str(v))
            pdf.ln(1)
        pdf.output(pdf_path)
        paths["pdf"] = pdf_path

        state["report_paths"] = paths
    except Exception as e:
        state["error"] = f"Report generation failed: {e}"
        state["report_paths"] = paths
    return state


# =========================================================
# WORKFLOW (LangGraph assembly, cached — built once per session)
# =========================================================


def route_after_validation(state: DeepfakeState) -> str:
    return "media_detection" if state.get("is_valid") else "report_generator"


def route_by_media_type(state: DeepfakeState) -> str:
    media_type = state.get("media_type")
    if media_type == "image":
        return "image_detection"
    if media_type == "audio":
        return "audio_detection"
    return "confidence"


@st.cache_resource(show_spinner=False)
def build_workflow():
    graph = StateGraph(DeepfakeState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("input_validation", input_validation_agent)
    graph.add_node("media_detection", media_detection_agent)
    graph.add_node("image_detection", image_detection_agent)
    graph.add_node("audio_detection", audio_detection_agent)
    graph.add_node("confidence", confidence_agent)
    graph.add_node("explanation", explanation_agent)
    graph.add_node("recommendation", recommendation_agent)
    graph.add_node("report_generator", report_generator_agent)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "input_validation")
    graph.add_conditional_edges(
        "input_validation",
        route_after_validation,
        {"media_detection": "media_detection", "report_generator": "report_generator"},
    )
    graph.add_conditional_edges(
        "media_detection",
        route_by_media_type,
        {"image_detection": "image_detection", "audio_detection": "audio_detection", "confidence": "confidence"},
    )
    graph.add_edge("image_detection", "confidence")
    graph.add_edge("audio_detection", "confidence")
    graph.add_edge("confidence", "explanation")
    graph.add_edge("explanation", "recommendation")
    graph.add_edge("recommendation", "report_generator")
    graph.add_edge("report_generator", END)

    return graph.compile()


def run_pipeline(filepath: str) -> DeepfakeState:
    workflow = build_workflow()
    initial_state: DeepfakeState = {
        "filepath": filepath,
        "file_name": os.path.basename(filepath) if filepath else "N/A",
    }
    return workflow.invoke(initial_state)


def calibrate_image_label_orientation(real_sample_filepath: str, fake_sample_filepath: str) -> Dict[str, Any]:
    """Run one known-REAL and one known-DEEPFAKE image through the model with NO
    inversion applied, and report whether the raw pipeline output matches or is
    backwards. This is the only reliable way to know the correct orientation for
    a given model/environment — do not hardcode it.

    Returns a dict with per-sample raw predictions and a `recommended_invert`
    bool the caller can use to set st.session_state.invert_image_labels.
    """
    clf = load_image_model()
    if clf is None:
        return {"ok": False, "reason": "Image model failed to load."}

    def _raw_top_label(filepath: str) -> Optional[str]:
        img = safe_load_image(filepath)
        if img is None:
            return None
        model_img = _prepare_image_for_model(img)
        try:
            results = clf(model_img, top_k=None)
        except TypeError:
            results = clf(model_img)
        results = sorted(results, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return normalize_label(results[0].get("label", "")) if results else None

    real_pred = _raw_top_label(real_sample_filepath)
    fake_pred = _raw_top_label(fake_sample_filepath)

    if real_pred is None or fake_pred is None:
        return {"ok": False, "reason": "Could not decode one or both calibration images."}

    # Correctly-oriented model: real sample -> REAL, fake sample -> DEEPFAKE.
    correct_as_is = (real_pred == "REAL") and (fake_pred == "DEEPFAKE")
    # Backwards model: real sample -> DEEPFAKE, fake sample -> REAL.
    backwards = (real_pred == "DEEPFAKE") and (fake_pred == "REAL")

    return {
        "ok": True,
        "real_sample_raw_prediction": real_pred,
        "fake_sample_raw_prediction": fake_pred,
        "correct_as_is": correct_as_is,
        "backwards": backwards,
        "recommended_invert": bool(backwards),
        "ambiguous": not correct_as_is and not backwards,
    }


def save_uploaded_file(uploaded_file) -> str:
    """Persist a Streamlit UploadedFile to a real path on disk for the pipeline to read."""
    ext = os.path.splitext(uploaded_file.name)[1]
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex[:8]}{ext}")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path


# =========================================================
# REDESIGNED MODERN STREAMLIT DASHBOARD UI
# =========================================================

st.set_page_config(
    page_title="Deepfake Shield AI — Multi-Agent Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Custom Glassmorphism CSS ---
def load_css():
    # Resolve styles.css relative to this script, with fallback to cwd
    _here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    css_path = os.path.join(_here, "styles.css")
    if not os.path.exists(css_path):
        css_path = os.path.join(os.getcwd(), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --- Groq API key resolution ---
if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""
    if secret_key:
        os.environ["GROQ_API_KEY"] = secret_key

# --- Session State Management ---
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

# --- Helper: Render Circular SVG Confidence Gauge ---
def render_circular_gauge(confidence: float, prediction: str):
    pct = max(0.0, min(100.0, confidence * 100.0))
    stroke_class = "fake" if prediction == "DEEPFAKE" else ("real" if prediction == "REAL" else "unknown")
    svg_html = f"""
    <div class="circular-gauge-container">
        <div class="single-chart">
            <svg viewBox="0 0 36 36" class="circular-chart">
                <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <path class="circle {stroke_class}" stroke-dasharray="{pct:.1f}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <text x="18" y="20.35" class="percentage">{pct:.1f}%</text>
            </svg>
        </div>
        <div class="gauge-label">Confidence Score</div>
    </div>
    """
    st.markdown(svg_html, unsafe_allow_html=True)


# --- Top Navigation Bar ---
def render_top_nav():
    cols = st.columns([2.5, 1, 1, 1, 1])
    
    with cols[0]:
        st.markdown(
            """
            <div class="nav-brand">
                <div class="nav-brand-icon">🛡️</div>
                <div>DEEPFAKE SHIELD <span style="font-weight:400; font-size:0.9rem; color:var(--accent-cyan); display:block; margin-top:-4px;">AI Forensics Lab</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    nav_pages = [("🏠 Home", "Home"), ("🔍 Detect", "Detect"), ("📊 Reports", "Reports"), ("ℹ️ About", "About")]

    for idx, (label, page_key) in enumerate(nav_pages, start=1):
        with cols[idx]:
            is_active = st.session_state.active_page == page_key
            btn_type = "primary" if is_active else "tertiary"
            if st.button(label, key=f"nav_btn_{page_key}", type=btn_type, use_container_width=True):
                st.session_state.active_page = page_key
                st.rerun()

    st.markdown("<hr style='border:0; height:1px; background:rgba(255,255,255,0.08); margin:15px 0 25px 0;'>", unsafe_allow_html=True)


# --- Sidebar Setup ---
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    
    st.markdown(
        f"""
        <div class="sidebar-badge">
            <div class="sidebar-status-dot"></div>
            <span>Hardware Engine: <b>{DEVICE.upper()}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    key_input = st.text_input(
        "Groq API key (Optional)",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Powers LLM Explanation Agent. If omitted, standard rule-based explanations will be generated.",
    )
    if key_input:
        os.environ["GROQ_API_KEY"] = key_input

    st.markdown("<hr style='border:0; height:1px; background:rgba(255,255,255,0.08); margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown("### 🔁 Image Label Orientation")
    if "invert_image_labels" not in st.session_state:
        st.session_state.invert_image_labels = Config.IMAGE_MODEL_LABELS_INVERTED_DEFAULT
    st.session_state.invert_image_labels = st.checkbox(
        "Invert REAL/DEEPFAKE for image model",
        value=st.session_state.invert_image_labels,
        help=(
            "Only enable this if you've confirmed via the calibration check on the "
            "Detect page (or by testing known real/fake images) that the model's "
            "raw output is backwards for this deployment. Don't guess — verify first."
        ),
    )
    st.caption("⚠️ Don't toggle this blindly. Use the calibration check on the Detect page with a known-real and known-deepfake image first.")

    st.markdown("<hr style='border:0; height:1px; background:rgba(255,255,255,0.08); margin:20px 0;'>", unsafe_allow_html=True)
    
    st.markdown("### 🔬 Multi-Agent Engine")
    st.caption("• **Input Validation Agent**\n• **Media Type Agent**\n• **ViT / Wav2Vec2 Detectors**\n• **Conservative Confidence + Quality Gate**\n• **Groq LLM Explainer**\n• **PDF/MD Report Generator**")
    
    st.markdown("<hr style='border:0; height:1px; background:rgba(255,255,255,0.08); margin:20px 0;'>", unsafe_allow_html=True)
    st.caption("🔒 **Security Disclaimer**: Probabilistic multi-agent analysis. Verify high-stakes media with official forensic channels.")


# =========================================================
# PAGE 1: HOME PAGE
# =========================================================
def render_home_page():
    # Hero Section
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">⚡ MULTI-AGENT AI SYSTEM v2.0</div>
            <div class="hero-title">Next-Gen <span class="hero-highlight">Deepfake Detection</span> & Digital Forensics</div>
            <div class="hero-desc">
                Identify manipulated media in seconds using autonomous multi-agent orchestration. Powered by Hugging Face Vision Transformers, Wav2Vec2 acoustic feature models, and Groq LLM synthesis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col_cta1, col_cta2, col_cta3 = st.columns([1.5, 1, 1])
    with col_cta1:
        if st.button("🚀 Launch Deepfake Detector", key="home_cta_btn", type="primary", use_container_width=True):
            st.session_state.active_page = "Detect"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Statistics Grid
    st.markdown("<h3 style='color:var(--text-main); font-weight:700; margin-bottom:15px;'>Platform Architecture Overview</h3>", unsafe_allow_html=True)
    
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-icon">🖼️</div>
                <div>
                    <div class="stat-value">ViT v2</div>
                    <div class="stat-label">Image Artifact Classification</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-icon">🎙️</div>
                <div>
                    <div class="stat-value">Wav2Vec2</div>
                    <div class="stat-label">Audio Acoustic Verification</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-icon">🤖</div>
                <div>
                    <div class="stat-value">LangGraph</div>
                    <div class="stat-label">9-Node Autonomous Pipeline</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Highlight Cards
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:var(--accent-cyan); margin-top:0;">📸 Image Forensic Detection</h4>
                <p style="color:var(--text-muted); font-size:0.95rem; line-height:1.6;">
                    Analyzes micro-level facial pixel inconsistencies, boundary noise, lighting incongruities, and ViT embeddings across JPG, PNG, WEBP, and BMP images.
                </p>
                <ul style="color:var(--text-sub); font-size:0.9rem; padding-left:20px;">
                    <li>Detects AI face-swap & diffusion generation</li>
                    <li>Evaluates classification confidence tiers</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:var(--accent-purple); margin-top:0;">🎵 Audio Synthetic Analysis</h4>
                <p style="color:var(--text-muted); font-size:0.95rem; line-height:1.6;">
                    Inspects acoustic waveforms, spectral anomalies, and synthetic voice cloning artifacts using Wav2Vec2 neural architecture across WAV, MP3, FLAC, OGG, and M4A.
                </p>
                <ul style="color:var(--text-sub); font-size:0.9rem; padding-left:20px;">
                    <li>Detects AI voice synthesis & neural voice cloning</li>
                    <li>Generates PDF, MD, and TXT diagnostic reports</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PAGE 2: DETECT PAGE
# =========================================================
def render_detect_page():
    st.markdown(
        """
        <div style="margin-bottom:20px;">
            <h2 style="font-weight:800; margin-bottom:4px; background:linear-gradient(135deg,#fff,#94a3b8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Deepfake Detection Workspace</h2>
            <p style="color:var(--text-muted);">Upload media files for multi-agent validation, model inference, and natural language forensic report generation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    media_kind = st.radio("Select Target Media Format:", ["Image", "Audio"], horizontal=True)

    if media_kind == "Image":
        with st.expander("🧪 Calibrate image label orientation (recommended before trusting results)"):
            st.caption(
                "Upload ONE image you know is REAL and ONE image you know is a DEEPFAKE. "
                "This runs both through the model with no inversion applied and tells you "
                "whether the raw output is correct or backwards for this deployment — "
                "instead of guessing."
            )
            cal_c1, cal_c2 = st.columns(2)
            with cal_c1:
                cal_real_file = st.file_uploader("Known REAL image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="cal_real_uploader")
            with cal_c2:
                cal_fake_file = st.file_uploader("Known DEEPFAKE image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="cal_fake_uploader")

            if st.button("Run calibration check", key="run_calibration_btn", disabled=(cal_real_file is None or cal_fake_file is None)):
                real_path = save_uploaded_file(cal_real_file)
                fake_path = save_uploaded_file(cal_fake_file)
                cal_result = calibrate_image_label_orientation(real_path, fake_path)

                if not cal_result.get("ok"):
                    st.error(f"Calibration failed: {cal_result.get('reason')}")
                else:
                    st.write(f"Known-REAL sample → model's raw prediction: **{cal_result['real_sample_raw_prediction']}**")
                    st.write(f"Known-DEEPFAKE sample → model's raw prediction: **{cal_result['fake_sample_raw_prediction']}**")

                    if cal_result["correct_as_is"]:
                        st.success("✅ Model output is correctly oriented. Leave the sidebar 'Invert' toggle OFF.")
                        st.session_state.invert_image_labels = False
                    elif cal_result["backwards"]:
                        st.warning("🔁 Model output is backwards for this deployment. Enabling the invert toggle automatically.")
                        st.session_state.invert_image_labels = True
                    else:
                        st.error(
                            "⚠️ Inconclusive — the model didn't cleanly separate your two calibration "
                            "images (e.g. it returned the same label for both, or an unmapped label). "
                            "Try clearer, unambiguous calibration samples rather than toggling inversion blindly."
                        )

    uploaded_file = None
    col_up, col_prev = st.columns([1.2, 0.8])

    with col_up:
        # Streamlit renders widgets outside HTML divs, so wrap with a container styled via CSS
        with st.container():
            st.markdown('<div class="upload-zone-label">📂 Upload Media File</div>', unsafe_allow_html=True)
            if media_kind == "Image":
                uploaded_file = st.file_uploader(
                    "Drag & drop an Image file here or click to browse",
                    type=["jpg", "jpeg", "png", "bmp", "webp"],
                    key="image_uploader",
                )
            else:
                uploaded_file = st.file_uploader(
                    "Drag & drop an Audio file here or click to browse",
                    type=["wav", "mp3", "flac", "ogg", "m4a"],
                    key="audio_uploader",
                )

    with col_prev:
        with st.container():
            if uploaded_file is not None:
                st.markdown("<h5 style='color:var(--accent-cyan); margin-bottom:10px;'>📺 Media Preview</h5>", unsafe_allow_html=True)
                if media_kind == "Image":
                    st.image(uploaded_file, caption=f"📄 {uploaded_file.name}", width="stretch")
                else:
                    st.audio(uploaded_file)
                    st.caption(f"🎵 Audio File: {uploaded_file.name}")
            else:
                st.markdown(
                    """
                    <div style="text-align:center; color:var(--text-sub); padding:40px 20px; border:1px dashed rgba(255,255,255,0.1); border-radius:14px;">
                        <div style="font-size:2.5rem; margin-bottom:8px;">📁</div>
                        <div style="font-weight:600;">No file uploaded yet</div>
                        <div style="font-size:0.8rem; margin-top:4px;">Select a file to preview before analysis</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("⚡ Run Multi-Agent Analysis", key="analyze_btn", type="primary", disabled=uploaded_file is None, use_container_width=True)

    if analyze_clicked and uploaded_file is not None:
        placeholder = st.empty()
        with placeholder.container():
            st.markdown(
                """
                <div class="loading-wrapper">
                    <div class="ai-spinner"></div>
                    <div class="loading-text">Analyzing Media with Autonomous Agents...</div>
                    <div class="loading-subtext">Executing Input Validation → Feature Extraction → ViT/Wav2Vec Inference → Groq LLM Synthesis</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        filepath = save_uploaded_file(uploaded_file)
        try:
            result = run_pipeline(filepath)
        except Exception as e:
            placeholder.empty()
            st.error(f"Pipeline Execution Error: {e}")
            st.stop()

        placeholder.empty()

        if result.get("error") and not result.get("prediction"):
            st.error(f"❌ Analysis Failed: {result['error']}")
        else:
            prediction = result.get("prediction", "UNKNOWN")
            confidence = result.get("confidence", 0.0) or 0.0
            explanation = result.get("explanation", "-")
            recommendation = result.get("recommendation", "-")
            paths = result.get("report_paths", {})
            conf_label = result.get("confidence_label", "Low")

            # Append to Session History Log
            st.session_state.history.append({
                "file_name": uploaded_file.name,
                "media_type": media_kind,
                "prediction": prediction,
                "confidence": confidence,
                "confidence_label": conf_label,
                "real_probability": result.get("real_probability", 0.0),
                "deepfake_probability": result.get("deepfake_probability", 0.0),
                "timestamp": timestamp_now(),
                "paths": paths
            })

            # --- Detection Results Card ---
            st.markdown("<h3 style='color:var(--text-main); font-weight:700; margin-top:20px;'>Detection Results</h3>", unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                v_class = "verdict-real" if prediction == "REAL" else ("verdict-fake" if prediction == "DEEPFAKE" else "verdict-unknown")
                b_class = "badge-real" if prediction == "REAL" else ("badge-fake" if prediction == "DEEPFAKE" else "badge-unknown")
                
                st.markdown(
                    f"""
                    <div class="verdict-card {v_class}">
                        <div style="font-size:0.85rem; color:var(--text-muted); font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Forensic Verdict</div>
                        <div class="verdict-badge {b_class}">{prediction}</div>
                        <div style="margin-top:15px; font-size:0.9rem; color:var(--text-main);">
                            Confidence Level: <b>{conf_label}</b> ({format_confidence(confidence)})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown("<div class='glass-card' style='display:flex; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
                render_circular_gauge(confidence, prediction)
                st.markdown("</div>", unsafe_allow_html=True)

            if media_kind == "Image":
                rp = float(result.get("real_probability", 0.0) or 0.0)
                fp = float(result.get("deepfake_probability", 0.0) or 0.0)
                q = result.get("image_quality", {}) or {}
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Real probability", format_confidence(rp))
                pc2.metric("Deepfake probability", format_confidence(fp))
                pc3.metric("Decision margin", format_confidence(result.get("decision_margin", 0.0) or 0.0))
                if prediction == "UNKNOWN":
                    st.warning("UNCERTAIN: the model did not provide enough reliable evidence for a binary decision. " + str(result.get("decision_reason", "")))
                if q.get("issues"):
                    st.info("Image quality checks: " + "; ".join(q["issues"]) + ".")

            st.markdown("<br>", unsafe_allow_html=True)

            # Explanation & Recommendation Cards
            e_col, r_col = st.columns(2)
            with e_col:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <h4 style="color:var(--accent-cyan); margin-top:0;">💡 Agent Forensic Explanation</h4>
                        <p style="color:var(--text-main); font-size:0.95rem; line-height:1.6;">{explanation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r_col:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <h4 style="color:var(--accent-purple); margin-top:0;">🛡️ Actionable Recommendation</h4>
                        <p style="color:var(--text-main); font-size:0.95rem; line-height:1.6;">{recommendation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Raw Scores Expander
            with st.expander("🔬 View Raw Model Tensors & Scores"):
                inv_state = st.session_state.get("invert_image_labels", Config.IMAGE_MODEL_LABELS_INVERTED_DEFAULT)
                st.caption(f"Label inversion currently: {'ON' if inv_state else 'OFF'}")
                if result.get("raw_model_labels_uncorrected"):
                    st.markdown("**Model output before any correction:**")
                    st.json(result.get("raw_model_labels_uncorrected", {}))
                st.markdown("**Scores used for the final decision (after correction, if any):**")
                st.json(result.get("raw_scores", {}))

            # Download Report Buttons
            st.markdown("<h4 style='color:var(--text-main); margin-top:20px;'>📥 Download Forensic Reports</h4>", unsafe_allow_html=True)
            dl_cols = st.columns(3)
            if paths.get("txt"):
                with open(paths["txt"], "rb") as f:
                    dl_cols[0].download_button("📄 TXT Report", f, file_name=os.path.basename(paths["txt"]), key="dl_txt", use_container_width=True)
            if paths.get("md"):
                with open(paths["md"], "rb") as f:
                    dl_cols[1].download_button("📝 Markdown Report", f, file_name=os.path.basename(paths["md"]), key="dl_md", use_container_width=True)
            if paths.get("pdf"):
                with open(paths["pdf"], "rb") as f:
                    dl_cols[2].download_button("📕 PDF Report", f, file_name=os.path.basename(paths["pdf"]), key="dl_pdf", use_container_width=True)


# =========================================================
# PAGE 3: REPORTS PAGE
# =========================================================
def render_reports_page():
    st.markdown(
        """
        <div style="margin-bottom:20px;">
            <h2 style="font-weight:800; margin-bottom:4px; background:linear-gradient(135deg,#fff,#94a3b8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Diagnostic Reports History</h2>
            <p style="color:var(--text-muted);">Review and export all media scans completed during the current active session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.history

    if not history:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:50px 20px;">
                <div style="font-size:3rem; margin-bottom:15px;">📊</div>
                <h3 style="color:var(--text-main); font-weight:700;">No Detection Reports Yet</h3>
                <p style="color:var(--text-muted);">Run an image or audio analysis on the Detect tab to generate downloadable reports.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔍 Go to Detect Workspace", key="reports_goto_detect_btn", type="primary"):
            st.session_state.active_page = "Detect"
            st.rerun()
        return

    # Summary Metrics Header
    total_scans = len(history)
    deepfakes = sum(1 for h in history if h["prediction"] == "DEEPFAKE")
    reals = sum(1 for h in history if h["prediction"] == "REAL")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div>
                    <div class="stat-value">{total_scans}</div>
                    <div class="stat-label">Total Files Analyzed</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="stat-card" style="border-color:rgba(239,68,68,0.3);">
                <div class="stat-icon" style="color:var(--status-fake); background:rgba(239,68,68,0.1);">⚠️</div>
                <div>
                    <div class="stat-value" style="color:var(--status-fake);">{deepfakes}</div>
                    <div class="stat-label">Deepfakes Identified</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="stat-card" style="border-color:rgba(16,185,129,0.3);">
                <div class="stat-icon" style="color:var(--status-real); background:rgba(16,185,129,0.1);">✅</div>
                <div>
                    <div class="stat-value" style="color:var(--status-real);">{reals}</div>
                    <div class="stat-label">Authentic Media Verified</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:var(--text-main); font-weight:700;'>Session History Log</h3>", unsafe_allow_html=True)

    for idx, item in enumerate(reversed(history)):
        b_style = "color:var(--status-real);" if item["prediction"] == "REAL" else ("color:var(--status-fake);" if item["prediction"] == "DEEPFAKE" else "color:var(--status-warn);")
        
        st.markdown(
            f"""
            <div class="glass-card" style="margin-bottom:15px; padding:18px 24px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h4 style="margin:0; color:var(--text-main);">{item['file_name']} <span style="font-size:0.8rem; color:var(--text-muted);">({item['media_type']})</span></h4>
                        <div style="font-size:0.85rem; color:var(--text-sub); margin-top:4px;">Scanned: {item['timestamp']}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:1.3rem; font-weight:800; {b_style}">{item['prediction']}</div>
                        <div style="font-size:0.85rem; color:var(--text-muted);">Confidence: {format_confidence(item['confidence'])}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("🗑️ Clear Session History", key="clear_history_btn"):
        st.session_state.history = []
        st.rerun()


# =========================================================
# PAGE 4: ABOUT PAGE
# =========================================================
def render_about_page():
    st.markdown(
        """
        <div style="margin-bottom:20px;">
            <h2 style="font-weight:800; margin-bottom:4px; background:linear-gradient(135deg,#fff,#94a3b8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">About Deepfake Shield AI</h2>
            <p style="color:var(--text-muted);">Architecture specifications, machine learning model details, and technical framework stack.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color:var(--accent-cyan); margin-top:0;">🧠 LangGraph Multi-Agent Architecture</h3>
            <p style="color:var(--text-muted); line-height:1.6;">
                The platform utilizes a stateful multi-agent directed graph built on <b>LangGraph</b>. Instead of relying on a single monolithic prompt or isolated classifier, specialized agent nodes process media sequentially and pass validated state:
            </p>
            <ol style="color:var(--text-main); line-height:1.9; padding-left:20px;">
                <li><b>Supervisor Agent</b>: Orchestrates analysis workflow state and initializes execution tracking.</li>
                <li><b>Input Validation Agent</b>: Validates file integrity, extension compliance, and file size metrics.</li>
                <li><b>Media Type Agent</b>: Identifies image vs. audio input to route state to specialized ML models.</li>
                <li><b>Image / Audio Detection Agents</b>: Invokes Hugging Face Transformers pipelines (ViT &amp; Wav2Vec2).</li>
                <li><b>Confidence Agent</b>: Calculates numerical confidence tiers (Very High, High, Moderate, Low).</li>
                <li><b>Explanation Agent</b>: Synthesizes plain-language explanations via Groq LLM (openai/gpt-oss-20b).</li>
                <li><b>Recommendation Agent</b>: Generates risk mitigation advice based on prediction confidence.</li>
                <li><b>Report Generator Agent</b>: Produces standalone PDF, Markdown, and TXT diagnostic reports.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:var(--text-main); margin-top:0;">📦 Neural Model Specifications</h4>
                <ul style="color:var(--text-muted); line-height:1.7; font-size:0.95rem;">
                    <li><b>Image Model:</b> <code style="color:var(--accent-cyan);">prithivMLmods/Deep-Fake-Detector-v2-Model</code> (Fallback: <code style="color:var(--text-sub);">Wvolf/ViT_Deepfake_Detection</code>)</li>
                    <li><b>Audio Model:</b> <code style="color:var(--accent-purple);">MelodyMachine/Deepfake-audio-detection-V2</code> (Fallback: <code style="color:var(--text-sub);">mo-thecreator/Deepfake-audio-detection</code>)</li>
                    <li><b>LLM Engine:</b> <code style="color:var(--accent-cyan);">openai/gpt-oss-20b</code> on Groq LPUs</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:var(--text-main); margin-top:0;">⚙️ Technical Stack</h4>
                <ul style="color:var(--text-muted); line-height:1.7; font-size:0.95rem;">
                    <li><b>Framework:</b> Streamlit 1.38+</li>
                    <li><b>Orchestration:</b> LangGraph & LangChain</li>
                    <li><b>Forensics Stack:</b> OpenCV, PIL, Librosa, PyDub</li>
                    <li><b>Report Generator:</b> FPDF2</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# MAIN APP ROUTER
# =========================================================

def main():
    render_top_nav()

    page = st.session_state.active_page

    if page == "Home":
        render_home_page()
    elif page == "Detect":
        render_detect_page()
    elif page == "Reports":
        render_reports_page()
    elif page == "About":
        render_about_page()

    st.markdown(
        """
        <div class="dashboard-footer">
            Deepfake Shield AI Dashboard &bull; Multi-Agent Digital Forensics Engine &bull; Built with Streamlit & LangGraph
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
