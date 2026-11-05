"""Identity bank: save aggregated reference identities and reuse them later.

Each saved person is a small `.npz` under ``<models>/identities/`` holding the
unit ArcFace identity vector, the raw (un-normalised) embedding magnitude the
converter-based swappers expect, gender/age/quality metadata and a 160px
thumbnail of the best reference crop. No reference photo is stored.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

THUMB = 160


@dataclass
class SavedIdentity:
    name: str
    identity: np.ndarray
    raw_magnitude: float
    gender: int | None
    age: int | None
    quality: float
    references: int
    thumbnail: np.ndarray | None
    created: float
    path: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def gender_label(self):
        return "?" if self.gender is None else ("M" if int(self.gender) == 1 else "F")


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._ -]+", "", str(name)).strip().replace(" ", "_")
    return slug[:64] or f"identity_{int(time.time())}"


class IdentityBank:
    def __init__(self, root):
        self.root = Path(root)

    def _dir(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def path_for(self, name: str) -> Path:
        return self._dir() / f"{_slug(name)}.npz"

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.npz"))

    def save(self, name: str, refset, thumbnail_image=None) -> SavedIdentity:
        """Persist a ReferenceSet (from VisenloPipeline.analyze_references)."""
        records = refset.records
        best = refset.best_index
        raw = np.asarray(records[0].face.embedding, np.float32)
        weights = np.asarray([max(r.quality, 1e-8) for r in records])
        magnitude = float(np.average([np.linalg.norm(np.asarray(r.face.embedding, np.float32)) for r in records],
                                     weights=weights)) if len(records) > 1 else float(np.linalg.norm(raw))
        thumb = None
        source = thumbnail_image if thumbnail_image is not None else (refset.images[best] if refset.images else None)
        if source is not None:
            x1, y1, x2, y2 = records[best].bbox
            crop = source[y1:y2, x1:x2]
            if crop.size:
                thumb = cv2.resize(crop, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
        meta = {
            "name": str(name),
            "gender": None if refset.gender is None else int(refset.gender),
            "age": None if records[best].age is None else int(records[best].age),
            "quality": float(max(r.quality for r in records)),
            "references": int(len(records)),
            "created": time.time(),
        }
        path = self.path_for(name)
        np.savez_compressed(path, identity=np.asarray(refset.identity, np.float32),
                            raw_magnitude=np.float32(magnitude),
                            thumbnail=thumb if thumb is not None else np.zeros((0, 0, 3), np.uint8),
                            meta=json.dumps(meta))
        return self.load(path.stem)

    def load(self, name: str) -> SavedIdentity:
        path = self.path_for(name)
        if not path.is_file():
            raise FileNotFoundError(f"No saved identity named '{name}'")
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            thumb = data["thumbnail"]
            return SavedIdentity(meta.get("name", path.stem), np.asarray(data["identity"], np.float32),
                                 float(data["raw_magnitude"]), meta.get("gender"), meta.get("age"),
                                 float(meta.get("quality", 0)), int(meta.get("references", 1)),
                                 thumb if thumb.size else None, float(meta.get("created", 0)), str(path))

    def delete(self, name: str) -> bool:
        path = self.path_for(name)
        if path.is_file():
            path.unlink()
            return True
        return False

    def as_reference_set(self, name: str):
        """Rebuild a ReferenceSet-compatible object from a saved identity.

        Embedding-driven swappers (inswapper, reswapper, hyperswap, converters)
        work fully; image-conditioned backends (BlendSwap, UniFace, FaceDancer,
        MobileFaceSwap) need real reference pixels and will reject it.
        """
        from types import SimpleNamespace
        from .engine import FaceRecord
        from .pipeline import ReferenceSet

        saved = self.load(name)
        raw = saved.identity * saved.raw_magnitude
        face = SimpleNamespace(normed_embedding=saved.identity, embedding=raw, kps=None,
                               gender=saved.gender, age=saved.age)
        record = FaceRecord(face, (0, 0, THUMB, THUMB), np.empty((0, 2), np.float32),
                            saved.identity, 1.0, quality=max(saved.quality, .05))
        record.gender, record.age = saved.gender, saved.age
        refset = ReferenceSet(saved.identity, [record], [], label=saved.name)
        refset.saved = saved  # type: ignore[attr-defined]
        return refset


def default_bank(models_path) -> IdentityBank:
    return IdentityBank(os.path.join(models_path, "identities"))
