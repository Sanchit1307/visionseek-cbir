"""CLIP ViT-B/32 image and text encoders via open_clip (Person B).

Public API (interface contract):
    encode_images(imgs_rgb: list[np.ndarray]) -> (N, 512) float32, L2-normalised
    encode_text(texts: list[str])             -> (N, 512) float32, L2-normalised

Image convention: RGB uint8, shape (H, W, 3). OpenCV images are BGR, so convert
at the boundary with cv2.cvtColor(img, cv2.COLOR_BGR2RGB) (or
src.dataset.bgr_to_rgb) before calling encode_images.

The model is loaded once (lazy) and runs on CPU; every encode call is wrapped in torch.no_grad().
Environment overrides (rarely needed):
    VISIONSEEK_CLIP_MODEL       default 'ViT-B-32'
    VISIONSEEK_CLIP_PRETRAINED  default 'openai'; 'none' = random weights,
                                only for offline smoke tests, results are meaningless.
"""

from __future__ import annotations

import os

import numpy as np

MODEL_NAME = os.environ.get("VISIONSEEK_CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("VISIONSEEK_CLIP_PRETRAINED", "openai")
EMBED_DIM = 512
BATCH_SIZE = 64

_state: dict = {}


def _load():
    if "model" not in _state:
        import open_clip
        import torch

        pre = None if PRETRAINED.lower() == "none" else PRETRAINED
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=pre, device="cpu")
        model.eval()
        _state.update(model=model, preprocess=preprocess,
                      tokenizer=open_clip.get_tokenizer(MODEL_NAME), torch=torch)
    return _state


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def _check_rgb(img: np.ndarray) -> None:
    if (not isinstance(img, np.ndarray) or img.dtype != np.uint8
            or img.ndim != 3 or img.shape[2] != 3):
        raise ValueError("each image must be a uint8 RGB array of shape (H, W, 3)")


def encode_images(imgs_rgb: list[np.ndarray], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """CLIP image embeddings, shape (N, 512), float32, unit L2 norm.

    Input images are RGB uint8 (NOT BGR). Empty list -> (0, 512) array.
    """
    if len(imgs_rgb) == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    from PIL import Image

    st = _load()
    torch = st["torch"]
    out = []
    for s in range(0, len(imgs_rgb), batch_size):
        batch = []
        for im in imgs_rgb[s:s + batch_size]:
            _check_rgb(im)
            batch.append(st["preprocess"](Image.fromarray(im)))
        with torch.no_grad():   # per call: grad mode is thread-local (Streamlit uses threads)
            feats = st["model"].encode_image(torch.stack(batch))
        out.append(feats.float().cpu().numpy())
    return _l2(np.concatenate(out, axis=0))


def encode_text(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """CLIP text embeddings, shape (N, 512), float32, unit L2 norm."""
    if len(texts) == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    st = _load()
    out = []
    for s in range(0, len(texts), batch_size):
        tokens = st["tokenizer"](list(texts[s:s + batch_size]))
        with st["torch"].no_grad():
            feats = st["model"].encode_text(tokens)
        out.append(feats.float().cpu().numpy())
    return _l2(np.concatenate(out, axis=0))
