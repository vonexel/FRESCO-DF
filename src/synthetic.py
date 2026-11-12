from pathlib import Path
import numpy as np
import pandas as pd
from .utils import atomic_json, digest_file


def make_cache(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng, rows, features, covariates = np.random.default_rng(42), [], [], []
    for split, number in [("train", 64), ("val", 24), ("test", 24), ("external", 24)]:
        for v in range(number):
            label, dataset = v % 2, "celebdfpp" if split == "external" else "ffpp"
            method = "real" if label == 0 else (["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"][v//2 % 4]
                                                if dataset == "ffpp" else "NovelGenerator")
            base = rng.normal(size=32)
            base[0] += 1.5 * label
            for frame in range(4):
                z = base+rng.normal(0, .15, size=32)
                c = rng.normal(size=4)
                conditions = ["clean", "jpeg90", "jpeg30"] if split in {"test", "external"} else ["clean"]
                for condition in conditions:
                    strength = {"clean": 0, "jpeg90": .2, "jpeg30": 1.5}[condition]
                    changed, quality = z.copy(), c.copy()
                    changed[0] -= strength * (2*label-1)*.4
                    changed[1] += strength
                    quality[0] += strength
                    rows.append({"row": len(rows), "dataset": dataset, "video_id": f"{dataset}/{split}/{v}",
                                 "label": label, "split": "test" if split == "external" else split,
                                 "method": method, "scenario": "real" if not label else "FaceSwap",
                                 "group_id": f"{dataset}/{split}/source{v//2}", "frame_idx": frame,
                                 "condition": condition})
                    features.append(changed)
                    covariates.append(quality)
    np.save(directory / "features.npy", np.asarray(features, np.float32))
    np.save(directory / "covariates.npy", np.asarray(covariates, np.float32))
    pd.DataFrame(rows).to_csv(directory / "metadata.csv", index=False)
    atomic_json(directory / "manifest.json", {
        "synthetic": True, "warning": "Integration fixture only",
        **{name + "_sha256": digest_file(directory / filename) for name, filename in
           [("features", "features.npy"), ("covariates", "covariates.npy"), ("metadata", "metadata.csv")]},
    })
    return directory