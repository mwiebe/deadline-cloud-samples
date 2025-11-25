import argparse
import os

import torch
from diffusers import StableDiffusionPipeline
from safetensors import safe_open
from safetensors.torch import load_file
from peft import LoraConfig, PeftModel, set_peft_model_state_dict

# Set cache directory for model downloads
CACHE_DIR = os.path.expanduser("~/.models/huggingface")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--image-index", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_lora_with_metadata(filepath):
    """Load LoRA weights and extract embedded metadata from safetensors file."""
    if not os.path.isfile(filepath):
        raise ValueError(f"LoRA weights file not found: {filepath}")

    with safe_open(filepath, framework="pt") as f:
        metadata = f.metadata()

    if not metadata:
        raise ValueError(
            f"No metadata found in LoRA file '{filepath}'. "
            "This file may have been created with an older version of the training script."
        )

    required_keys = ["base_model", "lora_rank"]
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        raise ValueError(f"LoRA file missing required metadata: {missing}")

    state_dict = load_file(filepath)
    return state_dict, metadata


def main():
    args = parse_args()

    # Load LoRA weights and metadata
    lora_state_dict, metadata = load_lora_with_metadata(args.lora_path)
    base_model = metadata["base_model"]
    lora_rank = int(metadata["lora_rank"])
    lora_alpha = int(metadata.get("lora_alpha", lora_rank))

    print(f"LoRA metadata: base_model={base_model}, rank={lora_rank}, alpha={lora_alpha}")
    print(f"Loaded {len(lora_state_dict)} LoRA tensors")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipe.unet = PeftModel(pipe.unet, lora_config)
    set_peft_model_state_dict(pipe.unet, lora_state_dict)

    lora_params = sum(1 for k in pipe.unet.state_dict() if "lora" in k.lower())
    if lora_params == 0:
        raise RuntimeError("No LoRA parameters loaded into model")
    print(f"Loaded {lora_params} LoRA parameters, scale={lora_alpha / lora_rank}")

    # Generate image
    seed = (args.seed + args.image_index) if args.seed >= 0 else random.randrange(0, 0xffff_ffff_ffff_ffff)
    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"image_{args.image_index:04d}.png")
    image.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
