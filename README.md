# Image Captioning Using ViT and GPT-2

## Overview

This project generates natural-language captions for images using a pretrained deep learning model. It combines a Vision Transformer (ViT) encoder with a GPT-2 decoder through the Hugging Face Transformers library.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Vision Transformer (ViT)
- GPT-2
- Pillow (PIL)

## How It Works

1. Load an input image.
2. Convert the image to RGB format.
3. Process the image using the pretrained ViT image processor.
4. Pass the processed image to the encoder-decoder model.
5. Generate a caption using beam search.
6. Decode the generated tokens into readable text.

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd image-captioning
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with the path to an image:

```bash
python src/caption.py "images/example.jpg"
```

The pretrained model files are downloaded from Hugging Face the first time the program runs. An internet connection is required for the initial download.

## Model

Pretrained model: [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)

## Hardware

The project supports CPU execution and CUDA-compatible GPUs. Inference speed depends on the hardware and image size.

## Limitations

Generated captions may omit details, misidentify objects, or describe an image inaccurately. The model is intended for general image captioning, not as a guarantee of factual accuracy.

## Future Improvements

- Support captioning multiple images in one run.
- Build a simple web interface.
- Save generated captions to a file.
- Evaluate caption quality on a test dataset.

## License

Check the pretrained model's license and the licenses of its dependencies before redistributing or deploying the project.
