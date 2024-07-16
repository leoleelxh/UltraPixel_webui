import gradio as gr
from einops import rearrange
import argparse
import math
import random
import numpy as np
import torch.nn.functional as F
from train import WurstCore_t2i as WurstCoreC
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight, AdaptiveLossWeight
from train import WurstCoreB
from core.utils import load_or_fail
from inference.utils import *
import os
import yaml
import torch
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath('./'))


def generate_images(height, width, seed, dtype, config_c, config_b, prompt, num_image, output_dir, stage_a_tiled, pretrained_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dtype = torch.bfloat16 if dtype == 'bf16' else torch.float

    # SETUP STAGE C
    with open(config_c, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)

    core = WurstCoreC(config_dict=loaded_config, device=device, training=False)

    # SETUP STAGE B
    with open(config_b, "r", encoding="utf-8") as file:
        config_file_b = yaml.safe_load(file)

    core_b = WurstCoreB(config_dict=config_file_b,
                        device=device, training=False)

    extras = core.setup_extras_pre()
    models = core.setup_models(extras)
    models.generator.eval().requires_grad_(False)

    extras_b = core_b.setup_extras_pre()
    models_b = core_b.setup_models(extras_b, skip_clip=True)
    models_b = WurstCoreB.Models(
        **{**models_b.to_dict(), 'tokenizer': models.tokenizer, 'text_model': models.text_model}
    )
    models_b.generator.bfloat16().eval().requires_grad_(False)

    captions = [prompt] * num_image

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pretrained_path = pretrained_path
    sdd = torch.load(pretrained_path, map_location='cpu')
    collect_sd = {}
    for k, v in sdd.items():
        collect_sd[k[7:]] = v

    models.train_norm.load_state_dict(collect_sd)

    models.generator.eval()
    models.train_norm.eval()

    batch_size = 1
    height_lr, width_lr = get_target_lr_size(height / width, std_size=32)
    stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(
        height, width, batch_size=batch_size)
    stage_c_latent_shape_lr, stage_b_latent_shape_lr = calculate_latent_sizes(
        height_lr, width_lr, batch_size=batch_size)

    # Stage C Parameters
    extras.sampling_configs['cfg'] = 4
    extras.sampling_configs['shift'] = 1
    extras.sampling_configs['timesteps'] = 20
    extras.sampling_configs['t_start'] = 1.0
    extras.sampling_configs['sampler'] = DDPMSampler(extras.gdf)

    # Stage B Parameters
    extras_b.sampling_configs['cfg'] = 1.1
    extras_b.sampling_configs['shift'] = 1
    extras_b.sampling_configs['timesteps'] = 10
    extras_b.sampling_configs['t_start'] = 1.0

    images = []
    for cnt, caption in enumerate(captions):
        batch = {'captions': [caption] * batch_size}
        conditions = core.get_conditions(
            batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
        unconditions = core.get_conditions(
            batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)

        conditions_b = core_b.get_conditions(
            batch, models_b, extras_b, is_eval=True, is_unconditional=False)
        unconditions_b = core_b.get_conditions(
            batch, models_b, extras_b, is_eval=True, is_unconditional=True)

        with torch.no_grad():
            models.generator.cuda()
            with torch.cuda.amp.autocast(dtype=dtype):
                sampled_c = generation_c(
                    batch, models, extras, core, stage_c_latent_shape, stage_c_latent_shape_lr, device)

            models.generator.cpu()
            torch.cuda.empty_cache()

            conditions_b = core_b.get_conditions(
                batch, models_b, extras_b, is_eval=True, is_unconditional=False)
            unconditions_b = core_b.get_conditions(
                batch, models_b, extras_b, is_eval=True, is_unconditional=True)
            conditions_b['effnet'] = sampled_c
            unconditions_b['effnet'] = torch.zeros_like(sampled_c)

            with torch.cuda.amp.autocast(dtype=dtype):
                sampled = decode_b(conditions_b, unconditions_b, models_b,
                                   stage_b_latent_shape, extras_b, device, stage_a_tiled=stage_a_tiled)

            torch.cuda.empty_cache()
            imgs = show_images(sampled)
            for idx, img in enumerate(imgs):
                img_path = os.path.join(
                    output_dir, prompt[:20] + '_' + str(cnt).zfill(5) + '.jpg')
                img.save(img_path)
                images.append(img_path)

    return images


iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Number(value=2560, label="Image Height"),
        gr.Number(value=5120, label="Image Width"),
        gr.Number(value=123, label="Random Seed"),
        gr.Textbox(value="bf16", label="Data Type (bf16 or float32)"),
        gr.Textbox(value="configs/training/t2i.yaml",
                   label="Config File for Stage C"),
        gr.Textbox(value="configs/inference/stage_b_1b.yaml",
                   label="Config File for Stage B"),
        gr.Textbox(value="A photo-realistic image of a west highland white terrier in the garden, high quality, detail rich, 8K", label="Text Prompt"),
        gr.Number(value=10, label="Number of Images"),
        gr.Textbox(value="figures/output_results/", label="Output Directory"),
        gr.Checkbox(label="Use Tiled Decoding for Stage A"),
        gr.Textbox(value="models/ultrapixel_t2i.safetensors",
                   label="Pretrained Path")
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="UltraPixel Image Generation",
    description="Generate photo-realistic images based on text prompts."
)

if __name__ == "__main__":
    iface.launch()
