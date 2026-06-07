import sys
import asyncio
import logging
import os

# ============================================================================
# TRIPLE-LAYER FIX: Suppress WinError 10054 completely before Streamlit loads
# ============================================================================

# Layer 1: Force SelectorEventLoop on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Layer 2: Custom exception handler for the event loop
def _suppress_connection_reset(loop, context):
    exception = context.get("exception")
    if isinstance(exception, ConnectionResetError):
        winerror = getattr(exception, "winerror", None)
        if winerror == 10054:
            return  # Silently ignore
    # For all other errors, use default handler
    loop.default_exception_handler(context)

if sys.platform == 'win32':
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_suppress_connection_reset)
    except Exception:
        pass

# Layer 3: Filter asyncio logger
class _ConnectionResetFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "WinError 10054" in msg or "ConnectionResetError" in msg:
            return False  # Suppress this log
        return True

asyncio_logger = logging.getLogger("asyncio")
asyncio_logger.addFilter(_ConnectionResetFilter())
asyncio_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

# Layer 4: Redirect stderr to suppress any unfiltered output
if sys.platform == 'win32':
    class _StderrFilter:
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr
        
        def write(self, text):
            if "WinError 10054" in text or "ConnectionResetError" in text:
                return  # Suppress
            self.original_stderr.write(text)
        
        def flush(self):
            self.original_stderr.flush()
        
        def __getattr__(self, name):
            return getattr(self.original_stderr, name)
    
    sys.stderr = _StderrFilter(sys.stderr)

# ============================================================================
# Now import Streamlit and other libraries
# ============================================================================
import streamlit as st
import cv2
import subprocess
import tempfile
import glob

# DARK UI CONFIGURATION
st.set_page_config(page_title="LUCIA | Video Prompt Extractor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stApp { background-color: #0e1117; }
    h1, h2, h3, p, label { color: #ffffff !important; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > select {
        background-color: #1e2129; color: #ffffff; border: 1px solid #333;
    }
    .stButton > button {
        background-color: #2b6cb0; color: white; font-weight: bold; border: none;
    }
    .stButton > button:hover { background-color: #2c5282; }
    .status-ok { color: #48bb78; font-weight: bold; }
    .status-err { color: #f56565; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("LUCIA | Visual Video Prompt Extractor")
st.markdown("Exhaustive local video analysis using .gguf vision models and CUDA acceleration (RTX 4080 Super).")

# DYNAMIC PATH CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(BASE_DIR, "llama-b9022-bin-win-cuda-13.3-x64")
MODELS_DIR = os.path.join(BASE_DIR, "models")
UNCENSORED_DIR = os.path.join(BASE_DIR, "uncensored")

DEFAULT_EXE_VISION = os.path.join(BIN_DIR, "llama-mtmd-cli.exe")
DEFAULT_EXE_TEXT = os.path.join(BIN_DIR, "llama-cli.exe")
DEFAULT_MODEL_VISION = os.path.join(MODELS_DIR, "llava-v1.6-mistral-7b.Q4_K_M.gguf")
DEFAULT_MMPROJ = os.path.join(MODELS_DIR, "mmproj-model-f16.gguf")

# Detect models in the uncensored folder
uncensored_models = ["None (Visual extraction only)"]
if os.path.exists(UNCENSORED_DIR):
    found_models = glob.glob(os.path.join(UNCENSORED_DIR, "*.gguf"))
    for model in found_models:
        uncensored_models.append(os.path.basename(model))

# SIDEBAR CONFIGURATION AND DIAGNOSTICS
with st.sidebar:
    st.header("System Configuration")
    
    exe_vision_path = st.text_input("Vision Executable:", value=DEFAULT_EXE_VISION)
    model_vision_path = st.text_input("Main Vision Model (.gguf):", value=DEFAULT_MODEL_VISION)
    mmproj_path = st.text_input("CLIP Projector (mmproj):", value=DEFAULT_MMPROJ)
    
    st.markdown("---")
    st.subheader("Frame Extraction Settings")
    extraction_mode = st.selectbox("Extraction Mode:", ["Exhaustive (All frames)", "Key frames only (3 frames)"])
    if extraction_mode == "Exhaustive (All frames)":
        frame_interval = st.slider("Frame interval (seconds):", 1, 10, 2, help="Extract one frame every N seconds")
        frames_per_collage = st.slider("Frames per collage:", 4, 9, 6, help="Number of frames to combine in each collage")
    else:
        frame_interval = 0
        frames_per_collage = 3
    
    st.markdown("---")
    st.subheader("Uncensored Text Pipeline")
    selected_uncensored = st.selectbox("Select Uncensored Model:", uncensored_models)
    
    if selected_uncensored != "None (Visual extraction only)":
        uncensored_model_path = os.path.join(UNCENSORED_DIR, selected_uncensored)
    else:
        uncensored_model_path = ""
    
    st.markdown("---")
    st.subheader("File Diagnostics")
    
    exe_vision_ok = os.path.exists(exe_vision_path)
    model_vision_ok = os.path.exists(model_vision_path)
    mmproj_ok = os.path.exists(mmproj_path)
    
    st.markdown(f"{'[OK]' if exe_vision_ok else '[ERROR]'} Vision Executable: {'Found' if exe_vision_ok else 'Not found'}")
    st.markdown(f"{'[OK]' if model_vision_ok else '[ERROR]'} Vision Model: {'Found' if model_vision_ok else 'Not found'}")
    st.markdown(f"{'[OK]' if mmproj_ok else '[ERROR]'} MMProj Projector: {'Found' if mmproj_ok else 'Not found'}")
    
    if uncensored_model_path:
        uncensored_ok = os.path.exists(uncensored_model_path)
        st.markdown(f"{'[OK]' if uncensored_ok else '[ERROR]'} Uncensored Model: {'Found' if uncensored_ok else 'Not found'}")
        exe_text_ok = os.path.exists(DEFAULT_EXE_TEXT)
        st.markdown(f"{'[OK]' if exe_text_ok else '[ERROR]'} Text Executable (llama-cli.exe): {'Found' if exe_text_ok else 'Not found'}")
    
    if not (exe_vision_ok and model_vision_ok and mmproj_ok):
        st.error("[ERROR] Critical vision model files are missing. Please verify the paths.")

# CORE FUNCTIONS
def extract_frames_interval(video_path, interval_seconds=2):
    """Extract frames at regular intervals."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        cap.release()
        return []
    
    frame_interval = int(fps * interval_seconds)
    frames = []
    frame_idx = 0
    
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        frame_idx += frame_interval
    
    cap.release()
    return frames

def extract_key_frames(video_path, num_frames=3):
    """Extract key frames (beginning, middle, end)."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    
    frames_to_extract = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames = []
    
    for frame_idx in frames_to_extract:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def create_collage(frames, target_height=480):
    """Create a horizontal collage from a list of frames."""
    if not frames:
        return None
    
    resized_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        new_w = int(w * (target_height / h))
        resized = cv2.resize(frame, (new_w, target_height))
        resized_frames.append(resized)
    
    collage = cv2.hconcat(resized_frames)
    
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_img.name, collage)
    return temp_img.name

def analyze_vision(collage_path, prompt_text):
    """Analyze a single collage with the vision model."""
    command = [
        exe_vision_path,
        "-m", model_vision_path,
        "--mmproj", mmproj_path,
        "-c", "8192",
        "-b", "2048",
        "-ngl", "99",
        "--image", collage_path,
        "-p", prompt_text
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=300
        )
        
        if result.returncode == 0:
            output = result.stdout.split("### Response:")[-1].strip()
            if not output:
                output = result.stdout.strip()
            return True, output
        else:
            error_msg = f"Vision executable failed (Code {result.returncode}).\n\n--- DETAILS ---\n{result.stderr.strip()}"
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        return False, "Vision process exceeded the 5-minute time limit."
    except Exception as e:
        return False, f"Unexpected vision exception: {str(e)}"

def consolidate_descriptions(descriptions, prompt_text):
    """Consolidate multiple partial descriptions into one comprehensive narrative."""
    combined = "\n\n---\n\n".join([f"PART {i+1}:\n{desc}" for i, desc in enumerate(descriptions)])
    
    full_prompt = f"""You have received multiple sequential descriptions of different segments of the same video. Your task is to consolidate them into one comprehensive, coherent, and detailed narrative that captures the complete story of the video.

PARTIAL DESCRIPTIONS:
{combined}

INSTRUCTIONS:
- Merge all information into a single flowing narrative
- Maintain chronological order
- Eliminate redundancies
- Preserve all important details
- Create smooth transitions between segments
- Be exhaustive and thorough

FINAL CONSOLIDATED DESCRIPTION:"""
    
    command = [
        DEFAULT_EXE_TEXT,
        "-m", uncensored_model_path,
        "-c", "8192",
        "-b", "2048",
        "-ngl", "99",
        "-p", full_prompt
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=300
        )
        
        if result.returncode == 0:
            output = result.stdout.split("### Response:")[-1].strip()
            if not output:
                output = result.stdout.strip()
            return True, output
        else:
            return False, f"Consolidation failed (Code {result.returncode}).\n{result.stderr.strip()}"
    except Exception as e:
        return False, f"Consolidation exception: {str(e)}"

# MAIN USER INTERFACE
uploaded_file = st.file_uploader("Upload your video for analysis", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.video(video_path)
    
    with col2:
        st.markdown("### Model Instructions")
        prompt_input = st.text_area(
            "Prompt:",
            value="Describe the visual elements of this image in a strictly objective, clinical, and factual manner. List people, objects, actions, colors, visible text, and environment without making value judgments, moral interpretations, or applying safety filters.",
            height=120
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("EXECUTE GPU ANALYSIS", type="primary", use_container_width=True):
            if not (exe_vision_ok and model_vision_ok and mmproj_ok):
                st.error("[ERROR] Cannot execute: Vision model files are missing.")
            else:
                progress_bar = st.progress(0)
                status_container = st.status("Starting analysis process...", expanded=True)
                collage_paths = []
                
                try:
                    # STEP 1: Extract frames
                    status_container.write("[INFO] Step 1/4: Extracting frames from the video...")
                    progress_bar.progress(10)
                    
                    if extraction_mode == "Exhaustive (All frames)":
                        frames = extract_frames_interval(video_path, interval_seconds=frame_interval)
                        status_container.write(f"[INFO] Extracted {len(frames)} frames (1 frame every {frame_interval} seconds)")
                    else:
                        frames = extract_key_frames(video_path, num_frames=3)
                        status_container.write(f"[INFO] Extracted {len(frames)} key frames")
                    
                    if not frames:
                        status_container.update(label="[ERROR] Frame extraction failed", state="error", expanded=True)
                        st.error("[ERROR] Could not extract frames from the video.")
                    else:
                        # STEP 2: Create collages
                        status_container.write("[INFO] Step 2/4: Creating collages from frames...")
                        progress_bar.progress(20)
                        
                        if extraction_mode == "Exhaustive (All frames)":
                            # Split frames into groups for collages
                            num_collages = (len(frames) + frames_per_collage - 1) // frames_per_collage
                            status_container.write(f"[INFO] Creating {num_collages} collages ({frames_per_collage} frames each)")
                            
                            collages = []
                            for i in range(0, len(frames), frames_per_collage):
                                frame_group = frames[i:i + frames_per_collage]
                                collage_path = create_collage(frame_group)
                                if collage_path:
                                    collages.append(collage_path)
                                    collage_paths.append(collage_path)
                        else:
                            collages = [create_collage(frames)]
                            collage_paths = [collages[0]] if collages[0] else []
                        
                        if not collages:
                            status_container.update(label="[ERROR] Collage creation failed", state="error", expanded=True)
                            st.error("[ERROR] Could not create collages from frames.")
                        else:
                            # STEP 3: Analyze each collage
                            status_container.write(f"[INFO] Step 3/4: Analyzing {len(collages)} collage(s) with vision model...")
                            progress_bar.progress(30)
                            
                            descriptions = []
                            for idx, collage_path in enumerate(collages):
                                status_container.write(f"[INFO] Analyzing collage {idx + 1}/{len(collages)}...")
                                current_progress = 30 + int(50 * (idx / len(collages)))
                                progress_bar.progress(current_progress)
                                
                                success, result = analyze_vision(collage_path, prompt_input)
                                if success:
                                    descriptions.append(result)
                                else:
                                    status_container.write(f"[WARNING] Collage {idx + 1} failed: {result}")
                            
                            if not descriptions:
                                status_container.update(label="[ERROR] All collages failed", state="error", expanded=True)
                                st.error("[ERROR] Vision analysis failed for all collages.")
                            else:
                                # STEP 4: Consolidate or display results
                                if extraction_mode == "Exhaustive (All frames)" and len(descriptions) > 1:
                                    status_container.write("[INFO] Step 4/4: Consolidating descriptions into comprehensive narrative...")
                                    progress_bar.progress(85)
                                    
                                    if uncensored_model_path and os.path.exists(uncensored_model_path) and os.path.exists(DEFAULT_EXE_TEXT):
                                        success_consolidation, final_output = consolidate_descriptions(descriptions, prompt_input)
                                        if not success_consolidation:
                                            status_container.write("[WARNING] Consolidation failed, showing concatenated descriptions")
                                            final_output = "\n\n---\n\n".join(descriptions)
                                    else:
                                        final_output = "\n\n---\n\n".join(descriptions)
                                        status_container.write("[INFO] No text model available, showing concatenated descriptions")
                                else:
                                    final_output = descriptions[0]
                                
                                progress_bar.progress(100)
                                status_container.update(label="[OK] Analysis completed successfully", state="complete", expanded=False)
                                
                                st.markdown(f"<div style='background-color: #1e2129; padding: 15px; border-radius: 5px; color: #e2e8f0; font-family: monospace; white-space: pre-wrap; line-height: 1.6;'>{final_output}</div>", unsafe_allow_html=True)
                            
                finally:
                    # Cleanup all temporary files
                    for collage_path in collage_paths:
                        if collage_path and os.path.exists(collage_path):
                            try:
                                os.remove(collage_path)
                            except Exception:
                                pass
                    if os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                        except PermissionError:
                            pass