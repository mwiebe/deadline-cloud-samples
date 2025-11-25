import argparse
import os

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from accelerate import Accelerator
from safetensors.torch import save_file

# Set cache directory for model downloads
CACHE_DIR = os.path.expanduser("~/.models/huggingface")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# VAE output scaling factor (standard for Stable Diffusion)
VAE_SCALE_FACTOR = 0.18215


class ImageDataset(Dataset):
    """Simple dataset that loads images from a directory and tokenizes a prompt."""

    def __init__(self, data_dir, tokenizer, prompt, resolution):
        self.data_dir = Path(data_dir)
        self.images = (
            list(self.data_dir.glob("*.jpg"))
            + list(self.data_dir.glob("*.png"))
            + list(self.data_dir.glob("*.jpeg"))
        )
        if not self.images:
            raise ValueError(f"No images found in {data_dir}")
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.resolution = resolution

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = image.resize((self.resolution, self.resolution))
        # Convert to tensor: [0, 255] -> [-1, 1]
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.permute(2, 0, 1)  # HWC -> CHW

        tokens = self.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {"pixel_values": image, "input_ids": tokens.input_ids[0]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--instance-prompt", required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--max-train-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_models(model_name):
    """Load all model components from HuggingFace."""
    print(f"Downloading model components from {model_name}...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    print("Model download complete")
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def train(args):
    accelerator = Accelerator(gradient_accumulation_steps=4)
    print(f"Using device: {accelerator.device}")

    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(args.model_name)

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Add LoRA adapters to UNet
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    dataset = ImageDataset(args.dataset_path, tokenizer, args.instance_prompt, args.resolution)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Training loop - cycle through dataset
    data_iter = iter(dataloader)
    for step in range(args.max_train_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with torch.no_grad():
            latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
            latents = latents * VAE_SCALE_FACTOR
            encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(model_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % 25 == 0:
            print(f"Step {step}/{args.max_train_steps}, Loss: {loss.item():.4f}")

    # Save LoRA weights with metadata
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        lora_state_dict = get_peft_model_state_dict(unet)

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")

        metadata = {
            "base_model": args.model_name,
            "lora_rank": str(args.lora_rank),
            "lora_alpha": str(args.lora_alpha),
            "instance_prompt": args.instance_prompt,
        }
        save_file(lora_state_dict, output_path, metadata=metadata)
        print(f"Training complete! LoRA weights saved to {output_path}")
        print(f"Embedded metadata: {metadata}")


if __name__ == "__main__":
    train(parse_args())
