import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
PY_SHIMS = THIS_DIR / "py_shims"
sys.path.insert(0, str(PY_SHIMS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    image = image.resize((args.size, args.size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    array = np.transpose(array, (2, 0, 1)).copy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(output)
    print(f"wrote {output} shape={array.shape} dtype=float32")


if __name__ == "__main__":
    main()
