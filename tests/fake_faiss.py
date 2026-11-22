"""A STRICT stand-in for the faiss Python API (test helper; the real package is not installable in
every environment). It implements only what src/index.py uses, and, like the real library, it
rejects arrays that are not float32 + C-contiguous, wrong dimensions, and pads with -1 ids when
k > ntotal. Install with:  sys.modules["faiss"] = fake_faiss"""

import pickle

import numpy as np


def _need(x, d=None, name="x"):
    if not isinstance(x, np.ndarray) or x.dtype != np.float32 or not x.flags["C_CONTIGUOUS"] or x.ndim != 2:
        raise TypeError(f"faiss needs a 2-D float32 C-contiguous array for {name}, got "
                        f"{getattr(x, 'dtype', type(x))} contiguous={getattr(getattr(x, 'flags', None), 'c_contiguous', None)}")
    if d is not None and x.shape[1] != d:
        raise AssertionError(f"dimension mismatch: index d={d}, {name} d={x.shape[1]}")


class IndexFlatIP:
    def __init__(self, d: int) -> None:
        self.d = int(d)
        self._xb = np.zeros((0, self.d), np.float32)

    @property
    def ntotal(self) -> int:
        return self._xb.shape[0]

    def add(self, x: np.ndarray) -> None:
        _need(x, self.d, "add")
        self._xb = np.vstack([self._xb, x])

    def search(self, x: np.ndarray, k: int):
        _need(x, self.d, "search")
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise TypeError("k must be a positive int")
        sims = x @ self._xb.T
        order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        scores = np.take_along_axis(sims, order, axis=1).astype(np.float32)
        ids = order.astype(np.int64)
        if k > self.ntotal:                                   # real faiss pads with -1
            pad = k - self.ntotal
            scores = np.pad(scores, ((0, 0), (0, pad)), constant_values=-3.4e38)
            ids = np.pad(ids, ((0, 0), (0, pad)), constant_values=-1)
        return scores, ids


def write_index(index: IndexFlatIP, path: str) -> None:
    if not isinstance(path, str):
        raise TypeError("faiss.write_index needs a str path")
    with open(path, "wb") as f:
        pickle.dump((index.d, index._xb), f)


def read_index(path: str) -> IndexFlatIP:
    if not isinstance(path, str):
        raise TypeError("faiss.read_index needs a str path")
    with open(path, "rb") as f:
        d, xb = pickle.load(f)
    ix = IndexFlatIP(d)
    ix._xb = xb
    return ix
