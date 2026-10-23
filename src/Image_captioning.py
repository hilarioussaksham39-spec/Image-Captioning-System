
import argparse

import torch
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"


def load_model():
    """Load the pretrained image-captioning model and its processors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    return model, processor, tokenizer, device


def generate_caption(image_path, model, processor, tokenizer, device):
    """Generate a caption for a single image."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        pixel_values = processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            pixel_values,
            max_length=16,
            num_beams=4
        )

    caption = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0].strip()

    return caption


def main():
    parser = argparse.ArgumentParser(
        description="Generate a caption for an image."
    )
    parser.add_argument("image", help="Path to the input image")
    args = parser.parse_args()

    model, processor, tokenizer, device = load_model()
    caption = generate_caption(
        args.image, model, processor, tokenizer, device
    )

    print(f"Image: {args.image}")
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
