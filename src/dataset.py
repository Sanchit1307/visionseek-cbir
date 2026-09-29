"""Gallery loaders, shared paths and the fixed query sample (Person B).

Galleries expose one small interface so the same code runs on Flowers-102 and
on the practical demo dataset later:

    len(g)              number of images
    g.get(i)            -> (BGR uint8 image, integer class label)
    g.class_names       optional list of class names (or None)

Paths can be overridden with the environment variables VISIONSEEK_DATA_DIR,
VISIONSEEK_INDEX_DIR and VISIONSEEK_RESULTS_DIR (used for tests / other machines).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("VISIONSEEK_DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.environ.get("VISIONSEEK_INDEX_DIR", ROOT / "index"))
RESULTS_DIR = Path(os.environ.get("VISIONSEEK_RESULTS_DIR", ROOT / "results"))

SEED = 42          # same seed and size as scripts/baseline_classical.py
N_QUERIES = 500

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Oxford Flowers-102 class names in label order (the list used by CLIP's
# zero-shot evaluation). Used for display and for the text-search sanity check;
# scripts/eval_clip.py reports zero-shot accuracy, which is only high if this
# order matches the labels torchvision returns.
FLOWERS102_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster",
    "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
    "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula",
    "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen",
    "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
]
assert len(FLOWERS102_CLASSES) == 102


class Flowers102Gallery:
    """Flowers-102 train + val + test as one gallery of 8,189 images.

    Global index order is train, val, test (offsets 0 / 1020 / 2040), identical
    to scripts/baseline_classical.py, so saved feature files line up.
    """

    SPLITS = ("train", "val", "test")
    class_names = FLOWERS102_CLASSES

    def __init__(self, root: Path = DATA_DIR) -> None:
        from torchvision.datasets import Flowers102  # lazy: needs torch

        Path(root).mkdir(parents=True, exist_ok=True)
        self.sets = [Flowers102(root=str(root), split=s, download=True)
                     for s in self.SPLITS]
        self.offsets = np.cumsum([0] + [len(s) for s in self.sets])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def split_range(self, split: str) -> tuple[int, int]:
        k = self.SPLITS.index(split)
        return int(self.offsets[k]), int(self.offsets[k + 1])

    def test_range(self) -> tuple[int, int]:
        return self.split_range("test")

    def labels(self) -> np.ndarray:
        """All class labels (N,) without decoding any image."""
        parts = []
        for s in self.sets:
            lab = getattr(s, "_labels", None)
            if lab is None:  # unexpected torchvision version: fall back to decoding
                lab = [s[i][1] for i in range(len(s))]
            parts.append(np.asarray(lab, dtype=np.int64))
        return np.concatenate(parts)

    def get(self, g: int) -> tuple[np.ndarray, int]:
        k = int(np.searchsorted(self.offsets, g, side="right") - 1)
        pil, label = self.sets[k][g - int(self.offsets[k])]
        bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        return bgr, int(label)


class FolderGallery:
    """Gallery from a folder laid out as root/<class_name>/<image files>.

    Used for the practical demo dataset (own photos, Caltech-101 subset, ...).
    Files are sorted so the order is stable between runs.
    """

    def __init__(self, root: Path) -> None:
        root = Path(root)
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        if not classes:
            raise FileNotFoundError(f"no class sub-folders found in {root}")
        self.class_names = classes
        self.paths: list[Path] = []
        labels = []
        for ci, cname in enumerate(classes):
            for p in sorted((root / cname).rglob("*")):
                if p.suffix.lower() in IMAGE_EXTS:
                    self.paths.append(p)
                    labels.append(ci)
        if not self.paths:
            raise FileNotFoundError(f"no images found under {root}")
        self.labels_arr = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.paths)

    def labels(self) -> np.ndarray:
        return self.labels_arr

    def get(self, g: int) -> tuple[np.ndarray, int]:
        bgr = cv2.imread(str(self.paths[g]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"cannot read image {self.paths[g]}")
        return bgr, int(self.labels_arr[g])


def open_gallery(spec: str = "flowers102") -> object:
    """'flowers102' or 'folder:<path>' -> gallery object."""
    if spec == "flowers102":
        return Flowers102Gallery(DATA_DIR)
    if spec.startswith("folder:"):
        return FolderGallery(Path(spec.split(":", 1)[1]))
    raise ValueError(f"unknown dataset '{spec}' (use 'flowers102' or 'folder:<path>')")


def gallery_query_range(gallery) -> tuple[int, int]:
    """Range queries are drawn from: the test split for Flowers-102, else everything."""
    return gallery.test_range() if hasattr(gallery, "test_range") else (0, len(gallery))


def select_query_ids(start: int, end: int, n: int = N_QUERIES, seed: int = SEED) -> np.ndarray:
    """Fixed seeded query sample from global indices [start, end).

    Uses the exact same recipe as scripts/baseline_classical.py, so the result
    equals the saved index/query_ids.npy for Flowers-102 (test range, seed 42).
    """
    rng = np.random.default_rng(seed)
    n = min(n, end - start)
    return np.sort(rng.choice(np.arange(start, end), size=n, replace=False))


def load_query_ids(index_dir: Path = INDEX_DIR) -> np.ndarray | None:
    p = Path(index_dir) / "query_ids.npy"
    return np.load(p) if p.exists() else None


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
