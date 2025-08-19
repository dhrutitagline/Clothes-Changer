import gradio as gr
import os
from pathlib import Path
import sys
import torch
from PIL import Image, ImageOps

from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import time
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


# Auto-detect GPU
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    gpu_id = 0
    device_info = f'cuda:{gpu_id}'
else:
    gpu_id = -1
    device_info = 'cpu'
print(f"[INFO] Using device: {device_info}")

# Initialize models safely
openpose_model_hd = OpenPose(gpu_id if gpu_id >= 0 else 0)
parsing_model_hd = Parsing(gpu_id if gpu_id >= 0 else 0)
ootd_model_hd = OOTDiffusionHD(gpu_id if gpu_id >= 0 else 0)

openpose_model_dc = OpenPose(gpu_id if gpu_id >= 0 else 0)
parsing_model_dc = Parsing(gpu_id if gpu_id >= 0 else 0)
ootd_model_dc = OOTDiffusionDC(gpu_id if gpu_id >= 0 else 0)


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']


example_path = os.path.join(os.path.dirname(__file__), 'examples')
model_hd = os.path.join(example_path, 'model/model_1.png')
garment_hd = os.path.join(example_path, 'garment/03244_00.jpg')
model_dc = os.path.join(example_path, 'model/model_8.png')
garment_dc = os.path.join(example_path, 'garment/048554_1.jpg')

def process_hd(vton_img, garm_img, n_samples, n_steps, image_scale, seed):
    model_type = 'hd'
    category = 0 # 0:upperbody; 1:lowerbody; 2:dress

    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))
        keypoints = openpose_model_hd(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_hd(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_hd(
            model_type=model_type,
            category=category_dict[category],
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

    return images

def process_dc(vton_img, garm_img, category, n_samples, n_steps, image_scale, seed):
    model_type = 'dc'
    if category == 'Upper-body':
        category = 0
    elif category == 'Lower-body':
        category = 1
    else:
        category =2

    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))
        keypoints = openpose_model_dc(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_dc(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_dc(
            model_type=model_type,
            category=category_dict[category],
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

    return images


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("***Support upper-body/lower-body/dresses; garment category must be paired!!!***")
    with gr.Row():
        with gr.Column():
            vton_img_dc = gr.Image(label="Model", sources='upload', type="filepath", height=384, value=model_dc)
        with gr.Column():
            garm_img_dc = gr.Image(label="Garment", sources='upload', type="filepath", height=384, value=garment_dc)
            category_dc = gr.Dropdown(label="Garment category (important option!!!)", choices=["Upper-body", "Lower-body", "Dress"], value="Upper-body")
        with gr.Column():
            result_gallery_dc = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True, scale=1)   
    with gr.Column():
        run_button_dc = gr.Button(value="Run")
        n_samples_dc = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
        n_steps_dc = gr.Slider(label="Steps", minimum=20, maximum=40, value=20, step=1)
        # scale_dc = gr.Slider(label="Scale", minimum=1.0, maximum=12.0, value=5.0, step=0.1)
        image_scale_dc = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
        seed_dc = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
        
    ips_dc = [vton_img_dc, garm_img_dc, category_dc, n_samples_dc, n_steps_dc, image_scale_dc, seed_dc]
    run_button_dc.click(fn=process_dc, inputs=ips_dc, outputs=[result_gallery_dc])

block.launch(debug=True,share=True)