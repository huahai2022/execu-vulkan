import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from export_smolvlm_vulkan import AutoModelForVision2Seq, SmolVLMVisionEncoder


def load_image_tensor(image_path: str, size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    array = np.transpose(array, (2, 0, 1)).copy()
    return torch.from_numpy(array).unsqueeze(0)


def load_image_bin_tensor(image_bin_path: str, size: int) -> torch.Tensor:
    array = np.fromfile(image_bin_path, dtype=np.float32)
    expected = 3 * size * size
    if array.size != expected:
      raise ValueError(
          f"unexpected image bin size: got {array.size} float32 values, expected {expected}"
      )
    array = array.reshape(1, 3, size, size)
    return torch.from_numpy(array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--image")
    parser.add_argument("--image-bin")
    args = parser.parse_args()

    if bool(args.image) == bool(args.image_bin):
        raise ValueError("exactly one of --image or --image-bin must be provided")

    if args.image:
        pixel_values = load_image_tensor(args.image, args.size)
    else:
        pixel_values = load_image_bin_tensor(args.image_bin, args.size)

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_dir,
        local_files_only=True,
        device_map="cpu",
        torch_dtype=torch.float32,
    ).eval()
    vision_encoder = SmolVLMVisionEncoder(model).eval()

    with torch.no_grad():
        outputs = vision_encoder(pixel_values)
        embeddings = outputs[0].detach().cpu().contiguous().to(torch.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    embeddings.numpy().tofile(output)
    print(f"wrote {output} shape={tuple(embeddings.shape)} dtype=float32")


if __name__ == "__main__":
    main()
