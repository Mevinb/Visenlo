"""Private Paddle worker operating on Visenlo-selected face crops only."""
from pathlib import Path
import sys
import numpy as np
import paddle

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "models/runtime-source/MobileFaceSwap"))
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet


def tensor(image):
    return paddle.to_tensor(image[:, :, ::-1].transpose(2, 0, 1)[None].copy(), dtype="float32") / 255


def main():
    model, request, response = map(Path, sys.argv[1:])
    paddle.set_device("cpu")
    recognizer = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    recognizer.set_dict(paddle.load(str(model.parent / "arcface.pdparams")))
    recognizer.eval()
    generator = FaceSwap(False)
    with np.load(request, allow_pickle=False) as data, paddle.no_grad():
        source = tensor(data["source"])
        mean = paddle.to_tensor([.485, .456, .406]).reshape([1, 3, 1, 1])
        std = paddle.to_tensor([.229, .224, .225]).reshape([1, 3, 1, 1])
        identity, features = recognizer((source - mean) / std)
        generator.set_model_param(l2_norm(identity), features, model_weight=paddle.load(str(model)))
        generator.eval()
        output, _ = generator(tensor(data["target"]))
        output = output.numpy()[0].transpose(1, 2, 0)[:, :, ::-1]
        if not np.isfinite(output).all():
            raise RuntimeError("MobileFaceSwap returned non-finite pixels")
        np.save(response, (np.clip(output, 0, 1) * 255).astype(np.uint8), allow_pickle=False)


if __name__ == "__main__":
    main()
