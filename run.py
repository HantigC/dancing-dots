import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'

from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

image = Image.open("data/image-matching-challenge-2025/train/imc2023_heritage/dioscuri_archive_0069.png")
depth = pipe(image)["depth"]

image.show(title="Image")
depth.show(title="Depth")