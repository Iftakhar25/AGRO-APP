# app.py
import os
import io
import base64
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# ---------------------------
# PAGE CONFIG & THEME
# ---------------------------
st.set_page_config(
    page_title="AgriDiag — Gemini Plant Disease Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# BACKGROUND IMAGE FUNCTION
# ---------------------------
def add_bg_from_local(image_file):
    """Applies a local background image in base64 encoding."""
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: rgba(0, 0, 0, 0);
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.95);
        }}
        .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Replace this path with your local image file
add_bg_from_local("images/farmer_bg.jpg")  # Example: place your image here

# ---------------------------
# PAGE HEADER
# ---------------------------
st.title("🌿 AgriDiag — Plant Disease Detection & Advice")
st.caption("Your friendly AI farm companion — diagnose, manage, and protect your crops with confidence.")

# ---------------------------
# GEMINI CONFIG
# ---------------------------
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
if not GEMINI_KEY:
    st.error("❌ Gemini API key not set. Please add it in environment or Streamlit secrets.")
    st.stop()

client = genai.Client()

# ---------------------------
# SIDEBAR: IMAGE UPLOAD & CONTROLS
# ---------------------------
with st.sidebar:
    st.header("📸 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a plant image (leaf, stem, fruit)",
        type=["jpg", "jpeg", "png", "webp", "heic"]
    )

    st.markdown("### 🧾 Image Tips")
    st.markdown("""
    - Focus on symptomatic areas  
    - Capture both close-up and full view  
    - Avoid blur or heavy shadow  
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Actions")
    st.markdown("""
    <style>
    /* Normal button style */
    .stButton>button {
        background-color: #4CAF50;  /* Green */
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        transition: 0.3s;  /* Smooth hover effect */
    }
    /* Hover effect */
    .stButton>button:hover {
        background-color: #45a049;  /* Darker green when hovering */
        color: yellow;
    }
    </style>
""", unsafe_allow_html=True)

    find_disease = st.button("🔬 Find Disease (Auto)")
    suggestions = st.button("🩺 Suggestions & Advice")
    custom_question = st.button("❓ Ask (Custom Prompt)")
    clear_results = st.button("🗑️ Clear All Results")

# ---------------------------
# SESSION STATE
# ---------------------------
if "results" not in st.session_state:
    st.session_state.results = []

if clear_results:
    st.session_state.results = []
    st.rerun()

# ---------------------------
# GEMINI HELPER FUNCTION
# ---------------------------
def call_gemini_with_image(image_bytes: bytes, prompt_text: str, thinking_budget: int = 0):
    """Send image + prompt to Gemini 2.5 Flash."""
    try:
        img_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        contents = [img_part, prompt_text]
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=contents, config=cfg
        )
        return response.text
    except Exception as e:
        return f"⚠️ Error calling Gemini API: {e}"

# ---------------------------
# MAIN CONTENT
# ---------------------------
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption=f"Uploaded: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Couldn't open image: {e}")
        st.stop()

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    image_bytes = buf.getvalue()

    # Check size
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > 18:
        st.warning(f"Image size {size_mb:.1f} MB — near Gemini limit (~20 MB). Consider resizing.")

    # --- PROMPTS ---
    prompt_find_disease = (
        "You are an expert plant pathologist. Analyze the image and:\n"
        "1️⃣ Identify the most likely disease(s) or disorders.\n"
        "2️⃣ Describe key visible symptoms.\n"
        "3️⃣ Suggest probable causal agents (fungus/bacteria/virus/stress).\n"
        "4️⃣ Give confidence level and further confirmation steps."
    )

    prompt_suggestions = (
        "You are an experienced crop protection specialist. Based on the supplied image and probable issue, give:\n"
        "A) Immediate actions (removal, isolation, sanitation)\n"
        "B) Cultural/non-chemical solutions\n"
        "C) Chemical options (types, active ingredients, safety notes)\n"
        "D) Monitoring plan and follow-up guidance."
    )

    custom_user_prompt = st.text_input(
        "💬 Custom Question",
        placeholder="e.g. What lab test should I run to confirm fungal infection?"
    )

    # --- HANDLE ACTION BUTTONS ---
    if find_disease:
        with st.spinner("🔍 Analyzing image for likely disease..."):
            output = call_gemini_with_image(image_bytes, prompt_find_disease, thinking_budget=500)
        st.session_state.results.append({
            "title": "🔬 Likely Disease(s) & Diagnostic Clues",
            "content": output
        })

    if suggestions:
        with st.spinner("🧪 Generating management suggestions..."):
            output = call_gemini_with_image(image_bytes, prompt_suggestions, thinking_budget=400)
        st.session_state.results.append({
            "title": "🩺 Practical Suggestions & Monitoring Plan",
            "content": output
        })

    if custom_question:
        if not custom_user_prompt.strip():
            st.warning("⚠️ Please type a custom question first.")
        else:
            combined_prompt = (
                "You are a helpful plant pathology assistant. Use the image to inform your answer.\n\n"
                f"User question: {custom_user_prompt}\n\n"
                "Provide a concise, practical answer and list any assumptions you made."
            )
            with st.spinner("🤖 Asking the model your custom question..."):
                output = call_gemini_with_image(image_bytes, combined_prompt, thinking_budget=200)
            st.session_state.results.append({
                "title": f"❓ Answer to: {custom_user_prompt}",
                "content": output
            })

    # --- DISPLAY RESULTS AS CARDS ---
    if st.session_state.results:
        st.markdown("---")
        st.markdown("### 🌾 Results Feed")
        for result in reversed(st.session_state.results):
            st.markdown(
                f"""
                <div style="
                    background-color: rgba(255,255,255,0.9);
                    border-radius: 15px;
                    padding: 1.2rem;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
                    margin-bottom: 1rem;
                    border: 1px solid #e5e5e5;
                ">
                    <h4 style="color:#2b7a0b;">{result['title']}</h4>
                    <p style="color:#333; font-size:16px; line-height:1.6;">{result['content']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.info("🌱 Tip: Re-take the photo with better focus if results seem uncertain. This app supports farmers; always confirm key actions with local experts.")

else:
    st.info("📤 Please upload an image to get started. Use the sidebar to select a clear, well-lit photo of the symptomatic plant area.")
st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)

# --- Minimalist Footer ---
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 50%;
        bottom: 10px;
        transform: translateX(-50%);
        text-align: center;
        font-family: sans-serif;
        z-index: 9999;
    }
    .footer .small-text {
        font-size: 12px;
        color: #555;
        margin-right: 5px;
    }
    .footer .big-text {
        font-size: 16px;
        font-weight: bold;
        color: #000;
        border: 1px solid #000;  /* Outline border around name */
        padding: 2px 6px;
        border-radius: 4px;
    }
    </style>
    <div class="footer">
        <span class="small-text">Made by</span>
        <span class="big-text">IFTAKHAR</span>
    </div>
    """,
    unsafe_allow_html=True
)
