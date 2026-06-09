import sys
import asyncio
import logging
import os

# TRIPLE-LAYER FIX: Suppress WinError 10054
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def _suppress_connection_reset(loop, context):
    exception = context.get("exception")
    if isinstance(exception, ConnectionResetError):
        winerror = getattr(exception, "winerror", None)
        if winerror == 10054:
            return
    loop.default_exception_handler(context)

if sys.platform == 'win32':
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_suppress_connection_reset)
    except Exception:
        pass

class _ConnectionResetFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "WinError 10054" in msg or "ConnectionResetError" in msg:
            return False
        return True

asyncio_logger = logging.getLogger("asyncio")
asyncio_logger.addFilter(_ConnectionResetFilter())
asyncio_logger.setLevel(logging.ERROR)

if sys.platform == 'win32':
    class _StderrFilter:
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr
        
        def write(self, text):
            if "WinError 10054" in text or "ConnectionResetError" in text:
                return
            self.original_stderr.write(text)
        
        def flush(self):
            self.original_stderr.flush()
        
        def __getattr__(self, name):
            return getattr(self.original_stderr, name)
    
    sys.stderr = _StderrFilter(sys.stderr)

import streamlit as st
import streamlit.components.v1 as components
import cv2
import subprocess
import tempfile
import glob

# Initialize session state
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'selected_image_model' not in st.session_state:
    st.session_state.selected_image_model = None
if 'selected_video_model' not in st.session_state:
    st.session_state.selected_video_model = None
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

# DARK UI CONFIGURATION
st.set_page_config(page_title="LUCIA | Visual Prompt Extractor", layout="wide", initial_sidebar_state="expanded")

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
    .stSelectbox:has(select:disabled) > div > div > select {
        background-color: #2a2a2a !important;
        color: #666 !important;
        cursor: not-allowed !important;
    }
    .reset-button > button {
        background-color: #c53030 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        width: 100% !important;
    }
    .reset-button > button:hover {
        background-color: #9b2c2c !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("LUCIA | Visual Prompt Extractor")
st.markdown("Advanced image and video analysis using GGUF vision models with CUDA acceleration.")

# DYNAMIC PATH CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(BASE_DIR, "llama-b9022-bin-win-cuda-13.3-x64")
IMAGES_DIR = os.path.join(BASE_DIR, "Images")
VIDEO_DIR = os.path.join(BASE_DIR, "Video")

DEFAULT_EXE_VISION = os.path.join(BIN_DIR, "llama-mtmd-cli.exe")

# File type definitions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}

def detect_file_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    return None

def get_model_folders_from_base(base_path):
    """Get list of model folders from base directory."""
    folders = []
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                folders.append(item)
    return sorted(folders)

def auto_detect_model_and_mmproj(folder_path):
    """Automatically detect model file and mmproj file in folder."""
    model_file = None
    mmproj_file = None
    
    if not os.path.exists(folder_path):
        return None, None
    
    # Find all gguf files
    gguf_files = glob.glob(os.path.join(folder_path, "*.gguf"))
    
    for gguf in gguf_files:
        basename = os.path.basename(gguf)
        if "mmproj" in basename.lower():
            mmproj_file = basename
        else:
            # First non-mmproj file is the model
            if model_file is None:
                model_file = basename
    
    return model_file, mmproj_file

def display_result_with_copy_button(result_text, container_key="result"):
    """Display the analysis result with a larger box and copy-to-clipboard button clearly outside."""
    # Escape special characters for JavaScript
    escaped_text = result_text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
    
    # HTML/JS component with button OUTSIDE and visually separated from the box
    copy_html = f"""
    <!-- CAJA DEL PROMPT GENERADO -->
    <div style="background-color: #1e2129; padding: 20px; border-radius: 8px; border: 1px solid #333; min-height: 400px; box-sizing: border-box; margin-bottom: 20px;">
        <div style="color: #e2e8f0; font-family: monospace; white-space: pre-wrap; line-height: 1.6; word-wrap: break-word;">{result_text}</div>
    </div>
    
    <!-- BOTÓN DE COPIAR FUERA DE LA CAJA, con estilo destacado -->
    <div style="margin-top: 20px; padding: 15px; background-color: #161b22; border-radius: 8px; border: 1px solid #2b6cb0; display: flex; align-items: center; gap: 15px;">
        <button id="copy-btn-{container_key}" onclick="copyToClipboard('{container_key}')" style="background-color: #2b6cb0; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 15px; transition: all 0.2s; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
            📋 Copy Prompt
        </button>
        <span id="copy-feedback-{container_key}" style="color: #48bb78; font-weight: bold; display: none; font-size: 14px;">✓ ¡Copiado al portapapeles!</span>
    </div>
    
    <script>
    function copyToClipboard(key) {{
        const text = `{escaped_text}`;
        navigator.clipboard.writeText(text).then(function() {{
            const btn = document.getElementById('copy-btn-' + key);
            const feedback = document.getElementById('copy-feedback-' + key);
            btn.textContent = '✓ ¡Copiado!';
            btn.style.backgroundColor = '#2f855a';
            feedback.style.display = 'inline';
            setTimeout(function() {{
                btn.textContent = '📋 Copy Prompt';
                btn.style.backgroundColor = '#2b6cb0';
                feedback.style.display = 'none';
            }}, 2000);
        }}).catch(function(err) {{
            console.error('Failed to copy: ', err);
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            const feedback = document.getElementById('copy-feedback-' + key);
            feedback.textContent = '✓ ¡Copiado (alternativo)!';
            feedback.style.display = 'inline';
            setTimeout(function() {{
                feedback.style.display = 'none';
                feedback.textContent = '✓ ¡Copiado al portapapeles!';
            }}, 2000);
        }});
    }}
    </script>
    """
    components.html(copy_html, height=800, scrolling=True)

# Get available model folders
image_model_folders = get_model_folders_from_base(IMAGES_DIR)
video_model_folders = get_model_folders_from_base(VIDEO_DIR)

# SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("System Configuration")
    
    exe_vision_path = st.text_input("Vision Executable:", value=DEFAULT_EXE_VISION)
    
    st.markdown("---")
    st.subheader("AI Model Selection")
    
    # Determine if dropdowns should be disabled based on current selection
    image_disabled = False
    video_disabled = False
    
    if st.session_state.file_type == "video" and st.session_state.selected_video_model:
        image_disabled = True  # Block image dropdown when video model is selected
    elif st.session_state.file_type == "image" and st.session_state.selected_image_model:
        video_disabled = True  # Block video dropdown when image model is selected
    
    # Image Model Selection
    if not image_model_folders:
        st.error("[ERROR] No image model folders found in 'Images' folder")
        selected_image_model = "None"
    else:
        selected_image_model = st.selectbox(
            "Image AI Model:",
            ["None"] + image_model_folders,
            disabled=image_disabled,
            key=f"img_model_{st.session_state.reset_counter}"
        )
    
    # Video Model Selection
    if not video_model_folders:
        st.error("[ERROR] No video model folders found in 'Video' folder")
        selected_video_model = "None"
    else:
        selected_video_model = st.selectbox(
            "Video AI Model:",
            ["None"] + video_model_folders,
            disabled=video_disabled,
            key=f"vid_model_{st.session_state.reset_counter}"
        )
    
    # RESET BUTTON - Must be placed after dropdowns to properly reset state
    st.markdown("---")
    if st.button("RESET", type="primary", use_container_width=True, key="reset_button"):
        st.session_state.file_type = None
        st.session_state.selected_image_model = None
        st.session_state.selected_video_model = None
        st.session_state.reset_counter += 1  # Force widget recreation
        st.rerun()
    
    # Update session state based on current selections
    if selected_image_model and selected_image_model != "None":
        st.session_state.file_type = "image"
        st.session_state.selected_image_model = selected_image_model
    elif selected_video_model and selected_video_model != "None":
        st.session_state.file_type = "video"
        st.session_state.selected_video_model = selected_video_model
    
    st.markdown("---")
    st.subheader("Video Processing Settings")
    frame_interval = st.slider("Frame interval (seconds):", 1, 10, 2, help="Extract one frame every N seconds")
    max_frames = st.slider("Maximum frames:", 10, 100, 50, help="Max frames to send to model")
    
    st.markdown("---")
    st.subheader("File Diagnostics")
    
    exe_vision_ok = os.path.exists(exe_vision_path)
    images_dir_ok = os.path.exists(IMAGES_DIR)
    video_dir_ok = os.path.exists(VIDEO_DIR)
    
    st.markdown(f"{'[OK]' if exe_vision_ok else '[ERROR]'} Vision Executable: {'Found' if exe_vision_ok else 'Not found'}")
    st.markdown(f"{'[OK]' if images_dir_ok else '[ERROR]'} Images Folder: {'Found' if images_dir_ok else 'Not found'}")
    st.markdown(f"{'[OK]' if video_dir_ok else '[ERROR]'} Video Folder: {'Found' if video_dir_ok else 'Not found'}")
    
    # Auto-detect and show diagnostics for selected model
    if st.session_state.file_type == "image" and st.session_state.selected_image_model:
        folder_path = os.path.join(IMAGES_DIR, st.session_state.selected_image_model)
        model_file, mmproj_file = auto_detect_model_and_mmproj(folder_path)
        
        st.markdown(f"{'[OK]' if os.path.exists(folder_path) else '[ERROR]'} Model Folder: {st.session_state.selected_image_model}")
        if model_file:
            st.markdown(f"{'[OK]'} Model File: {model_file}")
        else:
            st.markdown(f"{'[ERROR]'} No model file (.gguf) found in folder")
        
        if mmproj_file:
            st.markdown(f"{'[OK]'} MMProj File: {mmproj_file}")
        else:
            st.markdown(f"{'[INFO]'} No MMProj file found (using merged model)")
            
    elif st.session_state.file_type == "video" and st.session_state.selected_video_model:
        folder_path = os.path.join(VIDEO_DIR, st.session_state.selected_video_model)
        model_file, mmproj_file = auto_detect_model_and_mmproj(folder_path)
        
        st.markdown(f"{'[OK]' if os.path.exists(folder_path) else '[ERROR]'} Model Folder: {st.session_state.selected_video_model}")
        if model_file:
            st.markdown(f"{'[OK]'} Model File: {model_file}")
        else:
            st.markdown(f"{'[ERROR]'} No model file (.gguf) found in folder")
        
        if mmproj_file:
            st.markdown(f"{'[OK]'} MMProj File: {mmproj_file}")
        else:
            st.markdown(f"{'[INFO]'} No MMProj file found (using merged model)")
    
    if not (exe_vision_ok and images_dir_ok and video_dir_ok):
        st.error("[ERROR] Critical files or folders are missing.")

# CORE FUNCTIONS
def extract_frames_from_video(video_path, interval_seconds=2, max_frames=50, max_dimension=1024):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        cap.release()
        return []
    
    frame_interval = int(fps * interval_seconds)
    frames = []
    frame_idx = 0
    count = 0
    
    while frame_idx < total_frames and count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Smart resize: preserve aspect ratio, limit max dimension to prevent VRAM OOM
            h, w = frame.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            frames.append(frame)
            count += 1
        frame_idx += frame_interval
    
    cap.release()
    return frames

def save_image(image_array):
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_img.name, image_array)
    return temp_img.name

def build_vision_command(model_path, mmproj_path, prompt_text, image_paths):
    command = [
        exe_vision_path,
        "-m", model_path,
        "-c", "8192",
        "-b", "2048",
        "-ngl", "99",
        "-p", prompt_text
    ]
    
    # Conditionally add mmproj ONLY if it exists
    if mmproj_path and os.path.exists(mmproj_path):
        command.extend(["--mmproj", mmproj_path])
    
    # Add images
    for img_path in image_paths:
        command.extend(["--image", img_path])
        
    return command

def run_inference(command, timeout):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=timeout
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
        return False, f"Vision process exceeded the {timeout//60}-minute time limit."
    except Exception as e:
        return False, f"Unexpected exception: {str(e)}"

# MAIN USER INTERFACE
uploaded_file = st.file_uploader(
    "Upload your file for analysis (Image or Video)",
    type=["jpg", "jpeg", "png", "bmp", "webp", "gif", "mp4", "avi", "mov", "mkv", "webm", "flv"]
)

if uploaded_file is not None:
    file_type = detect_file_type(uploaded_file.name)
    
    if file_type is None:
        st.error("[ERROR] Unsupported file type. Please upload an image or video file.")
    else:
        st.session_state.file_type = file_type
        
        file_extension = uploaded_file.name.split('.')[-1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
        tfile.write(uploaded_file.read())
        tfile.close()
        file_path = tfile.name
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if file_type == "image":
                st.image(file_path, caption="Uploaded Image")
            else:
                st.video(file_path)
        
        with col2:
            st.markdown("### Model Instructions")
            
            if file_type == "image":
                default_prompt = "Describe this image comprehensively. Include all visible details: objects, people, actions, environment, colors, text, lighting, composition, and any other relevant visual elements. Be thorough and objective."
            else:
                default_prompt = "Analyze this video comprehensively. Describe the sequence of events, actions, movements, objects, people, environment, and any text visible. Provide a detailed temporal narrative of what happens from beginning to end. Focus on the progression and changes throughout the video."
            
            prompt_input = st.text_area(
                "Prompt:",
                value=default_prompt,
                height=120
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if file_type == "image":
                selected_folder = st.session_state.selected_image_model
                model_base_path = IMAGES_DIR
                st.info(f"Mode: **Image Analysis** | Model: {selected_folder if selected_folder else 'None'}")
            else:
                selected_folder = st.session_state.selected_video_model
                model_base_path = VIDEO_DIR
                st.info(f"Mode: **Video Analysis** | Model: {selected_folder if selected_folder else 'None'}")
            
            if st.button("EXECUTE GPU ANALYSIS", type="primary", use_container_width=True):
                if not selected_folder or selected_folder == "None":
                    st.error(f"[ERROR] Please select a model from the sidebar.")
                else:
                    progress_bar = st.progress(0)
                    status_container = st.status("Starting analysis process...", expanded=True)
                    temp_files = [file_path]
                    
                    try:
                        # Auto-detect model and mmproj files
                        folder_path = os.path.join(model_base_path, selected_folder)
                        model_file, mmproj_file = auto_detect_model_and_mmproj(folder_path)
                        
                        if not model_file:
                            status_container.update(label="[ERROR] No model file found", state="error", expanded=True)
                            st.error(f"[ERROR] No .gguf model file found in folder: {folder_path}")
                        else:
                            model_path = os.path.join(folder_path, model_file)
                            mmproj_path = os.path.join(folder_path, mmproj_file) if mmproj_file else None
                            
                            if file_type == "image":
                                status_container.write("[INFO] Step 1/2: Preparing image for analysis...")
                                progress_bar.progress(30)
                                
                                status_container.write("[INFO] Step 2/2: Analyzing image with vision model...")
                                progress_bar.progress(60)
                                
                                command = build_vision_command(model_path, mmproj_path, prompt_input, [file_path])
                                success, result = run_inference(command, 300)
                                
                                if success:
                                    progress_bar.progress(100)
                                    status_container.update(label="[OK] Analysis completed successfully", state="complete", expanded=False)
                                    st.success("Generated Prompt:")
                                    display_result_with_copy_button(result, "image_result")
                                else:
                                    status_container.update(label="[ERROR] Analysis failed", state="error", expanded=True)
                                    st.error("[ERROR] Analysis failed:")
                                    st.code(result, language="text")
                            
                            else:
                                status_container.write("[INFO] Step 1/3: Extracting frames from video...")
                                progress_bar.progress(20)
                                
                                frames = extract_frames_from_video(file_path, interval_seconds=frame_interval, max_frames=max_frames)
                                
                                if not frames:
                                    status_container.update(label="[ERROR] Frame extraction failed", state="error", expanded=True)
                                    st.error("[ERROR] Could not extract frames from the video.")
                                else:
                                    status_container.write(f"[INFO] Extracted {len(frames)} frames (1 frame every {frame_interval} seconds)")
                                    progress_bar.progress(40)
                                    
                                    status_container.write("[INFO] Step 2/3: Saving frames as temporary images...")
                                    image_paths = []
                                    for idx, frame in enumerate(frames):
                                        img_path = save_image(frame)
                                        image_paths.append(img_path)
                                        temp_files.append(img_path)
                                    
                                    progress_bar.progress(60)
                                    
                                    status_container.write("[INFO] Step 3/3: Analyzing video frames with vision model...")
                                    progress_bar.progress(80)
                                    
                                    command = build_vision_command(model_path, mmproj_path, prompt_input, image_paths)
                                    success, result = run_inference(command, 600)
                                    
                                    if success:
                                        progress_bar.progress(100)
                                        status_container.update(label="[OK] Analysis completed successfully", state="complete", expanded=False)
                                        st.success("Generated Prompt:")
                                        display_result_with_copy_button(result, "video_result")
                                    else:
                                        status_container.update(label="[ERROR] Analysis failed", state="error", expanded=True)
                                        st.error("[ERROR] Analysis failed:")
                                        st.code(result, language="text")
                                        
                    finally:
                        for temp_file in temp_files:
                            if temp_file and os.path.exists(temp_file):
                                try:
                                    os.remove(temp_file)
                                except Exception:
                                    pass
