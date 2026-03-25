import argparse
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from dump_image_embeddings import load_image_bin_tensor, load_image_tensor
from export_smolvlm_vulkan import AutoModelForVision2Seq, SmolVLMVisionEncoder


PREFIX_TEXT = "<|im_start|>User:<fake_token_around_image><global-img>"
SUFFIX_TEMPLATE = "<fake_token_around_image>{prompt}<end_of_utterance>\nAssistant:"


def encode_text(tokenizer: Tokenizer, text: str) -> torch.Tensor:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    token_ids = np.asarray(encoding.ids, dtype=np.int64)
    return torch.from_numpy(token_ids).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Can you describe this image?")
    parser.add_argument("--tokenizer-path")
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

    tokenizer_path = (
        Path(args.tokenizer_path)
        if args.tokenizer_path
        else Path(args.model_dir) / "tokenizer.json"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    prefix_tokens = encode_text(tokenizer, PREFIX_TEXT)
    suffix_tokens = encode_text(tokenizer, SUFFIX_TEMPLATE.format(prompt=args.prompt))

    with torch.no_grad():
        prefix_embeddings = model.model.text_model.embed_tokens(prefix_tokens)
        image_embeddings = vision_encoder(pixel_values)[0]
        suffix_embeddings = model.model.text_model.embed_tokens(suffix_tokens)
        prompt_embeddings = torch.cat(
            [prefix_embeddings, image_embeddings, suffix_embeddings], dim=1
        ).detach().cpu().contiguous().to(torch.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_embeddings.numpy().tofile(output)
    print(
        f"wrote {output} shape={tuple(prompt_embeddings.shape)} dtype=float32 "
        f"prefix_len={prefix_embeddings.shape[1]} image_len={image_embeddings.shape[1]} "
        f"suffix_len={suffix_embeddings.shape[1]}"
    )


if __name__ == "__main__":
    main()
