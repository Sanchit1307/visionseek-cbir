"""Logic test for src/features/clip_encoder.py with fake torch/open_clip modules.

Run from the repo root:  python tests\\test_clip_encoder.py

This checks batching, ordering, L2-normalisation, input validation, the empty
case and that the model is loaded once. It does NOT test the real CLIP weights:
that is what scripts/eval_clip.py (retrieval P@10 + zero-shot accuracy) is for.
"""

import contextlib
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

loads = {"n": 0, "pretrained": None}


class FakeTensor:
    def __init__(self, a): self.a = np.asarray(a, dtype=np.float64)
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


class FakeModel:
    """Embedding = fixed linear map of the input, so results are checkable."""
    W = np.random.default_rng(0).normal(size=(3, 512))

    def eval(self): return self
    def encode_image(self, batch): return FakeTensor(np.asarray(batch).reshape(len(batch), -1)[:, :3] @ self.W)
    def encode_text(self, tokens): return FakeTensor(np.asarray(tokens, float)[:, :3] @ self.W)


def fake_create(name, pretrained=None, device="cpu"):
    loads["n"] += 1
    loads["pretrained"] = pretrained
    def preprocess(pil):
        return np.asarray(pil.resize((4, 4)), dtype=np.float64) / 255.0
    return FakeModel(), None, preprocess


fake_torch = types.SimpleNamespace(no_grad=contextlib.nullcontext, stack=lambda xs: np.stack(xs))
fake_open_clip = types.SimpleNamespace(
    create_model_and_transforms=fake_create,
    get_tokenizer=lambda name: (lambda texts: [[len(t), sum(map(ord, t)) % 97, 1] for t in texts]),
)
sys.modules["torch"] = fake_torch
sys.modules["open_clip"] = fake_open_clip

from src.features import clip_encoder as ce  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 256, (20 + i, 30, 3), dtype=np.uint8) for i in range(7)]

    a = ce.encode_images(imgs, batch_size=3)
    b = ce.encode_images(imgs, batch_size=64)
    assert a.shape == (7, 512) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(a, b, atol=1e-6), "batching must not change results or order"
    assert np.allclose(ce.encode_images(imgs[3:4]), a[3:4], atol=1e-6)
    print("ok: images, shape/dtype/norm, batch-size independent, order kept")

    t = ce.encode_text(["a red flower", "sunflower", "rose"], batch_size=2)
    assert t.shape == (3, 512) and np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-5)
    assert np.allclose(ce.encode_text(["sunflower"]), t[1:2], atol=1e-6)
    print("ok: text")

    assert ce.encode_images([]).shape == (0, 512) and ce.encode_text([]).shape == (0, 512)
    for bad in (np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32), np.zeros((10, 10, 4), np.uint8)):
        try:
            ce.encode_images([bad])
        except ValueError:
            continue
        raise AssertionError("invalid image accepted")
    print("ok: empty input and input validation")

    assert loads["n"] == 1 and loads["pretrained"] == "openai", loads
    print("ok: model loaded once, pretrained tag 'openai'")
    print("\nclip_encoder logic tests passed")


if __name__ == "__main__":
    main()
