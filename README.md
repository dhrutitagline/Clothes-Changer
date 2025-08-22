# Clothes Changer

Clothes Changer is an AI-powered virtual outfit try-on tool. Upload your photo and see how different clothes fit on you. The project leverages advanced deep learning models for realistic garment fitting and pose estimation.

---

## Features

- AI-based outfit try-on from user images
- Supports both CPU and GPU
- Human parsing for precise garment placement
- Pretrained models for high-quality results
- Gradio-based web interface for easy use

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/dhrutitagline/Clothes-Changer.git
cd Clothes-Changer
```

2. Create a Python virtual environment and activate it:

```bash
python3 -m venv env
source env/bin/activate  # macOS/Linux
env\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pretrained models:
```bash
python download_models.py
```

5. Start the Gradio interface:
```bash
cd run
python gradio_ootd.py
```
The app will automatically use GPU.

Access the local web interface at http://127.0.0.1:7860.

### Folder Structure
```bash
VirtualWardrobe/
├─ checkpoints/         # Pretrained models
├─ ootd/                # Model inference code
├─ preprocess/          # OpenPose & human parsing utilities
├─ run/                 # Gradio interface scripts
├─ download_models.py   # Script to download models
├─ .gitignore
├─ requirements.txt
├─ README.md
```

