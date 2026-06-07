<p align="center">
  <img src="OSO.JPG" alt="LUCIA Visual Video Prompt Extractor Interface" width="100%">
</p>


# LUCIA | Visual Video Prompt Extractor

A local, privacy-focused video analysis tool that extracts detailed visual prompts using GGUF vision models and CUDA acceleration. It features an optional two-step pipeline that expands visual descriptions using unrestricted (uncensored) text models.

## System Architecture

The application operates in two distinct phases:
1. **Visual Extraction (Phase 1):** Uses a multimodal vision model (LLaVA) to extract clinical, objective, and factual descriptions from video frames.
2. **Unrestricted Expansion (Phase 2 - Optional):** Uses a text-only uncensored LLM to expand the visual data into a comprehensive, narrative, and unrestricted description.

## Hardware Requirements

### Level 1: Minimum Absolute (Viable but slow execution)
- **CPU:** Intel Core i5 (8th Gen+) or AMD Ryzen 5 (3000 series+). Must support AVX2.
- **RAM:** 16 GB DDR4.
- **GPU:** Integrated graphics or 4 GB VRAM dedicated GPU (e.g., GTX 1650). *Note: The model will run primarily on CPU/RAM.*
- **Storage:** SATA SSD with at least 20 GB free space.
- **Performance:** 1 to 4 tokens/second. Analysis may take 1-3 minutes.

### Level 2: Minimum Recommended (Fluid and productive use)
- **CPU:** Intel Core i5 / AMD Ryzen 5 (last 4 generations).
- **RAM:** 16 GB DDR4/DDR5.
- **GPU:** NVIDIA RTX 3060 (12 GB), RTX 4060 (8 GB), or RTX 3050 (8 GB). *Minimum 8 GB VRAM required.*
- **Storage:** NVMe SSD (PCIe 3.0/4.0).
- **Performance:** 15 to 35 tokens/second. Real-time generation.

### Level 3: High Performance (Reference Configuration)
- **GPU:** NVIDIA RTX 4080 Super (16 GB VRAM) or higher.
- **Performance:** 50 to 80+ tokens/second. Supports full context (8192 tokens) and simultaneous dual-model pipelines.

## Installation Guide

**IMPORTANT:** Follow these steps in the exact order listed below. Do not skip any step.

### Step 1: Download Vision Models (models folder)
1. Create a folder named `models` in the project root directory.
2. Download a Vision-Language Model (VLM) in GGUF format and its corresponding CLIP projector (mmproj).
   - **Recommended Model:** LLaVA v1.6 Mistral 7B
   - **Main Model File:** `llava-v1.6-mistral-7b.Q4_K_M.gguf`
   - **Projector File:** `mmproj-model-f16.gguf`
3. You can find these files on Hugging Face (huggingface.co) by searching for "llava 1.6 mistral 7b gguf".
4. Place both downloaded files directly inside the `models` folder.

### Step 2: Download Uncensored AI Models (uncensored folder)
To enable the unrestricted text expansion pipeline, you must acquire uncensored or "abliterated" Large Language Models.
1. Create a folder named `uncensored` in the project root directory.
2. Visit **Hugging Face (huggingface.co)** and search for GGUF models using the following keywords:
   - `Dolphin GGUF` (e.g., `Dolphin-2.9-Llama3-8B-GGUF`)
   - `Abliterated GGUF`
   - `Uncensored GGUF`
   - `Wizard Vicuna Uncensored GGUF`
3. Download the `.gguf` file(s) you prefer (Q4_K_M or Q5_K_M quantization is recommended for 8GB+ VRAM).
4. Place the downloaded `.gguf` file(s) directly inside the `uncensored` folder.
5. The application will automatically detect and list these models in the sidebar dropdown menu.

### Step 3: Download llama.cpp Binaries
The application relies on pre-compiled C++ binaries for CUDA acceleration. These must be downloaded from the official repository.
1. Navigate to the official llama.cpp releases page: **https://github.com/ggml-org/llama.cpp/releases**
2. Find the latest release (e.g., b9022 or newer).
3. Under the "Assets" section of the release, download the following two files:
   - **Windows x64 (CUDA 13)** - This contains the main executable files (llama-mtmd-cli.exe, llama-cli.exe, etc.)
   - **CUDA 13.3 DLLs** - This contains the CUDA runtime libraries required for GPU acceleration
4. Create a folder named `llama-b9022-bin-win-cuda-13.3-x64` in the project root directory.
5. **Extract the contents of BOTH zip files directly into this folder.**
   - All files from both archives must be placed together in the same folder
   - Ensure that `llama-mtmd-cli.exe`, `llama-cli.exe`, and all the `.dll` files are in the root of this folder
   - Do not create subfolders; all files must be at the same level
   
### Step 4: Download more uncensored AI based on your PC hardware

https://huggingface.co/mradermacher/Llama-3.2-3B-Instruct-uncensored-GGUF
https://huggingface.co/BlouseJury/DolphinPod_dolphin-2.9.1-llama3.1-8b-Q6_K-GGUF/tree/main


1- Download and paste the models folder

llava-v1.6-mistral-7b.Q4_K_M.gguf
https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/blob/main/llava-v1.6-mistral-7b.Q4_K_M.gguf

mmproj-model-f16.gguf
https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf?download=true


2- Download and paste the uncensored folder

dolphin-2.9-llama3-8b-q6_K.gguf
https://huggingface.co/BlouseJury/DolphinPod_dolphin-2.9.1-llama3.1-8b-Q6_K-GGUF/resolve/main/dolphin-2.9.1-llama3.1-8b-Q6_K.gguf?download=true

Llama-3.2-3B-Instruct-uncensored.f16.gguf
https://huggingface.co/mradermacher/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored.f16.gguf?download=true


### Step 5: Run INSTALL.bat (Python and Dependencies)
1. Double-click `INSTALL.bat` in the project root.
2. The script will automatically:
   - Check if Python is installed on your system.
   - If Python is not found, it will download and install Python 3.12.4 silently from python.org.
   - Install the required Python libraries: `streamlit`, `opencv-python`, and `numpy`.
3. Wait for the message "Installation completed successfully!" before proceeding.

## Usage

1. Run `run.bat` to start the Streamlit server.
2. Open the provided local URL (usually `http://localhost:8501`) in your web browser.
3. **Sidebar Configuration:**
   - Verify that the File Diagnostics section shows `[OK]` for all required files.
   - Select your preferred **Extraction Mode** (Exhaustive or Key frames).
   - If using exhaustive mode, adjust the frame interval and frames per collage.
   - Select an **Uncensored Model** from the dropdown if you downloaded one in Step 2.
4. **Main Interface:**
   - Upload a video file (MP4, AVI, MOV, MKV).
   - Adjust the prompt if necessary (the default prompt is optimized to bypass safety filters by requesting clinical/objective descriptions).
   - Click **EXECUTE GPU ANALYSIS**.
5. Monitor the progress bar. The final output will be displayed in a formatted text box.

## Troubleshooting

- **WinError 10054 / ConnectionResetError:** This is a benign Windows socket error caused by the browser disconnecting from Streamlit. It has been suppressed in the code and does not affect functionality.
- **GGML_ASSERT failed / Memory Slot Error:** Occurs if too many images are sent at once. The application uses a "Temporal Collage" technique to mitigate this. Ensure you are using the updated `app.py`.
- **Model Rejection ("I cannot help with that"):** The vision model's safety filters are triggered by aggressive prompts. Use the default clinical/objective prompt provided in the UI, or rely on the Uncensored Text Pipeline to expand the initial factual description.
- **Path Errors:** Do not use spaces or special characters (like accents) in the project folder path. Rename `Ejecutar este código` to `Ejecutar_Este_Codigo`.
- **Missing DLL Errors:** Ensure you extracted BOTH zip files (Windows x64 CUDA 13 and CUDA 13.3 DLLs) into the `llama-b9022-bin-win-cuda-13.3-x64` folder.
- **Model Not Found:** Verify that the `.gguf` files are in the correct folders (`models` or `uncensored`) and that the filenames match exactly what the application expects.

## Technical Notes

- **Context Size (-c 8192):** Optimized for 7B/8B models on 16GB VRAM GPUs. Do not increase to 131072 unless using specialized hardware and models.
- **Batch Size (-b 2048):** Tuned to process the ~576 tokens generated by the vision collage without triggering memory fragmentation errors.
- **Encoding:** Ensure all Python scripts are saved in UTF-8 encoding to prevent `UnicodeDecodeError` on Windows.
- **Folder Structure:** The project must maintain the exact folder structure described in the Installation Guide for automatic detection to work properly.
- **CUDA Compatibility:** The binaries are compiled for CUDA 13.3. Ensure your NVIDIA drivers are up to date to support this version.

## Project Structure

```
Ejecutar_Este_Codigo/
├── app.py                          # Main application script
├── run.bat                         # Launch script
── INSTALL.bat                     # Dependencies installer
├── README.md                       # This documentation
├── VIDEO.mp4                       # Sample video for testing
├── models/                         # Vision models folder
│   ├── llava-v1.6-mistral-7b.Q4_K_M.gguf
│   └── mmproj-model-f16.gguf
├── uncensored/                     # Uncensored text models folder
│   ── [your-uncensored-model].gguf
└── llama-b9022-bin-win-cuda-13.3-x64/  # llama.cpp binaries
    ├── llama-mtmd-cli.exe
    ├── llama-cli.exe
    └── [all .dll files from both archives]
```

## License

This project is provided as-is for educational and research purposes. The underlying models (LLaVA, Mistral, etc.) are subject to their respective licenses. Please review the terms of use for each model before deployment.
