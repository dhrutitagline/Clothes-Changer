import os
import subprocess
from huggingface_hub import snapshot_download

def download_humanparsing():
    os.makedirs("checkpoints/humanparsing", exist_ok=True)

    atr_path = "checkpoints/humanparsing/parsing_atr.onnx"
    lip_path = "checkpoints/humanparsing/parsing_lip.onnx"

    # Download only if missing or file size is 0 (partial download)
    if not os.path.exists(atr_path) or os.path.getsize(atr_path) == 0:
        print("Downloading parsing_atr.onnx...")
        subprocess.run([
            "wget", "-O", atr_path,
            "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx"
        ])
    else:
        print("parsing_atr.onnx already exists, skipping download.")

    if not os.path.exists(lip_path) or os.path.getsize(lip_path) == 0:
        print("Downloading parsing_lip.onnx...")
        subprocess.run([
            "wget", "-O", lip_path,
            "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx"
        ])
    else:
        print("parsing_lip.onnx already exists, skipping download.")


def download_ootdiffusion():
    ootd_dir = "checkpoints/ootd"
    print("Checking OOTDiffusion...")

    snapshot_download(
        repo_id="levihsu/OOTDiffusion",
        repo_type="model",
        local_dir="./",
        allow_patterns=[
            "checkpoints/ootd/model_index.json",
            "checkpoints/ootd/vae/*",
            "checkpoints/ootd/text_encoder/*",
            "checkpoints/ootd/tokenizer/*",
            "checkpoints/ootd/feature_extractor/*",
            "checkpoints/ootd/scheduler/*",
            "checkpoints/ootd/ootd_hd/checkpoint-36000/*",
            "checkpoints/ootd/ootd_dc/checkpoint-36000/*"
        ],
        resume_download=True,  # ✅ only download missing or partial files
        max_workers=2
    )


def download_clip():
    clip_dir = "./checkpoints/clip-vit-large-patch14"
    print("Checking CLIP...")

    snapshot_download(
        repo_id="openai/clip-vit-large-patch14",
        local_dir=clip_dir,
        resume_download=True,  # ✅ only download missing or partial files
        max_workers=2
    )


if __name__ == "__main__":
    print("Downloading Human Parsing Models...")
    download_humanparsing()

    download_ootdiffusion()
    download_clip()

    print("All models ready!")


import fileinput
import os
import sys
import site

# --- Get the correct site-packages path ---
site_packages = site.getsitepackages()[0]  # e.g. /usr/local/lib/python3.12/dist-packages

# --- Target files ---
target_file = os.path.join(site_packages, "diffusers/utils/dynamic_modules_utils.py")
degradations_py = os.path.join(site_packages, "basicsr/data/degradations.py")

# --- Safety checks ---
if not os.path.exists(target_file):
    raise FileNotFoundError(f"File not found: {target_file}")
if not os.path.exists(degradations_py):
    raise FileNotFoundError(f"File not found: {degradations_py}")

# --- Patch diffusers (cached_download → hf_hub_download) ---
with fileinput.FileInput(target_file, inplace=True, backup='.bak') as file:
    for line in file:
        if "cached_download" in line:
            print(line.replace("cached_download", "hf_hub_download"), end='')
        else:
            print(line, end='')

# --- Patch basicsr (functional_tensor → functional) ---
with fileinput.FileInput(degradations_py, inplace=True, backup='.bak') as file:
    for line in file:
        if "functional_tensor" in line:
            print(line.replace("functional_tensor", "functional"), end='')
        else:
            print(line, end='')

print("✅ Patching completed successfully!")
