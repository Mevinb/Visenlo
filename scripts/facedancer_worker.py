"""Private local worker; requires scripts/setup_facedancer.py's environment."""
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "models/runtime-source/FaceDancer"))
from networks.layers import AdaIN, AdaptiveAttention


def main():
    model, request, response = map(Path, sys.argv[1:])
    generator = tf.keras.models.load_model(str(model), compile=False, custom_objects={
        "AdaIN": AdaIN, "AdaptiveAttention": AdaptiveAttention,
        "InstanceNormalization": InstanceNormalization})
    recognizer = tf.keras.models.load_model(str(model.parent / "ArcFace-Res50.h5"), compile=False)
    with np.load(request, allow_pickle=False) as data:
        source = data["source"][:, :, ::-1][None].astype(np.float32) / 255
        identity = recognizer(source, training=False)
        outputs = []
        for tile in data["tiles"]:
            target = (tile[:, :, ::-1][None].astype(np.float32) - 127.5) / 127.5
            prediction = generator([target, identity], training=False).numpy()[0]
            if not np.isfinite(prediction).all():
                raise RuntimeError("FaceDancer returned non-finite pixels")
            outputs.append((np.clip((prediction + 1) / 2, 0, 1)[:, :, ::-1] * 255).astype(np.uint8))
    np.save(response, np.stack(outputs), allow_pickle=False)


if __name__ == "__main__":
    main()
