import re
import csv
import time
import json
import hashlib
from collections import Counter
from __future__ import annotations
from pathlib import Path, PurePosixPath


INDEX_COLUMNS = ["dataset", "video_id", "path", 
                 "label", "split", "method",
                 "scenario", "compression", "group_id"]
CORE_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
SPLIT_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{split}.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_file(source: Path, target: Path) -> None:
    for attempt in range(8):
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.05 * (attempt + 1), .2))


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    replace_file(temporary, path)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    replace_file(temporary, path)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def safe_relative_path(value: str) -> str:
    if not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in raw_parts):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path.as_posix()


def resolve_under(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    value = (root / safe_relative_path(relative)).resolve()
    if not value.is_relative_to(root):
        raise ValueError(f"Path escapes dataset root: {relative!r}")
    return value


def load_official_splits(split_dir: Path) -> tuple[dict[str, str], dict[str, str], dict]:
    mapping, groups, provenance = {}, {}, {}
    for split in ("train", "val", "test"):
        path = Path(split_dir) / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing official FF++ split {path}. Download {SPLIT_URL.format(split=split)}")
        pairs = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(pairs, list):
            raise ValueError(f"Invalid official split {path}")
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2 or any(not re.fullmatch(r"\d{3}", str(x)) for x in pair):
                raise ValueError(f"Invalid source pair {pair!r} in {path}")
            left, right = map(str, pair)
            a, b = find(left), find(right)
            parent[max(a, b)] = min(a, b)
        for source in parent:
            if source in mapping:
                raise ValueError(f"FF++ source {source} appears more than once across official splits")
            mapping[source] = split
            groups[source] = f"ffpp:{find(source)}"
        provenance[split] = {"url": SPLIT_URL.format(split=split), "sha256": sha256_file(path),
                             "pairs": len(pairs), "source_videos": len(parent)}
    return mapping, groups, provenance


def parse_celeb_test_list(path: Path, root: Path) -> list[dict]:
    rows, seen, missing = [], set(), []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in {"0", "1"}:
            raise ValueError(f"Invalid Celeb-DF++ list line {number}: {line!r}")
        official_label, relative = parts[0], safe_relative_path(parts[1])
        if PurePosixPath(relative).suffix.lower() != ".mp4":
            raise ValueError(f"Unsupported official video extension on line {number}: {relative}")
        if relative.casefold() in seen:
            raise ValueError(f"Duplicate Celeb-DF++ path on line {number}: {relative}")
        seen.add(relative.casefold())
        resolved = resolve_under(root, relative)
        if not resolved.is_file():
            missing.append(relative)
        segments = PurePosixPath(relative).parts
        is_real = segments[0] in {"Celeb-real", "YouTube-real"}
        is_fake = segments[0] == "Celeb-synthesis" and len(segments) == 4
        if not (is_real or is_fake) or (official_label == "1") != is_real:
            raise ValueError(f"Official label/path disagreement on line {number}: {line}")
        identities = re.findall(r"id\d+", PurePosixPath(relative).stem)
        group = "celeb:" + (identities[0] if identities else "/".join(segments[:1] + (PurePosixPath(relative).stem,)))
        rows.append({"dataset": "celebdfpp", "video_id": f"celebdfpp/{relative[:-4]}",
                     "path": relative, "label": int(not is_real), "split": "test",
                     "method": "real" if is_real else segments[2],
                     "scenario": "real" if is_real else segments[1],
                     "compression": "native", "group_id": group})
    if missing:
        raise FileNotFoundError(f"{len(missing)} official Celeb-DF++ videos missing; examples: {missing[:10]}")
    if not rows:
        raise ValueError("Official Celeb-DF++ testing list is empty")
    return rows


def index_datasets(ffpp_root: Path, celeb_root: Path, split_dir: Path, out_dir: Path) -> Path:
    ffpp_root, celeb_root, out_dir = Path(ffpp_root).resolve(), Path(celeb_root).resolve(), Path(out_dir)
    split_map, groups, provenance = load_official_splits(Path(split_dir))
    ff_rows, counts, exclusions = [], {}, {}
    for method in ("original", *CORE_METHODS, "FaceShifter"):
        directory = ffpp_root / method
        if not directory.is_dir():
            directory = ffpp_root / ("original_sequences/youtube" if method == "original" else f"manipulated_sequences/{method}") / "c23/videos"
        if not directory.is_dir():
            if method == "FaceShifter":
                exclusions["FaceShifter"] = {"videos": 0, "reason": "Optional unseen-method directory unavailable; no FaceShifter evaluation possible."}
                continue
            raise FileNotFoundError(f"Required FF++ method directory missing: {method}")
        files = sorted(directory.glob("*.mp4"))
        counts[method] = len(files)
        expected_names = set(split_map) if method == "original" else None
        for video in files:
            stem = video.stem
            pattern = r"\d{3}" if method == "original" else r"\d{3}_\d{3}"
            if not re.fullmatch(pattern, stem):
                raise ValueError(f"Unexpected FF++ video name: {video}")
            sources = stem.split("_")
            if any(source not in split_map for source in sources):
                raise ValueError(f"FF++ video has no official split: {video}")
            if len({split_map[s] for s in sources}) != 1 or len({groups[s] for s in sources}) != 1:
                raise ValueError(f"FF++ target/source identity leakage: {video}")
            split = split_map[sources[0]]
            if method == "FaceShifter" and split != "test":
                exclusions["FaceShifter_non_test"] = exclusions.get("FaceShifter_non_test", 0) + 1
                continue
            relative = video.relative_to(ffpp_root).as_posix()
            ff_rows.append({"dataset": "ffpp", "video_id": f"ffpp/{method}/{stem}", "path": relative,
                            "label": int(method != "original"), "split": split,
                            "method": "real" if method == "original" else method,
                            "scenario": "unseen_method" if method == "FaceShifter" else "in_domain",
                            "compression": "c23", "group_id": groups[sources[0]]})
        if method == "original" and {p.stem for p in files} != expected_names:
            raise ValueError("FF++ original files do not exactly match official source IDs")
        if method != "original":
            targets = [p.stem.split("_")[0] for p in files]
            if set(targets) != set(split_map) or len(targets) != len(set(targets)):
                raise ValueError(f"FF++ {method}: expected one manipulated target per official source; got {len(files)}")
    dfd_count = sum(1 for _ in (ffpp_root / "DeepFakeDetection").glob("*.mp4"))
    exclusions["DeepFakeDetection"] = {"videos": dfd_count, "reason": "Separate DFD dataset; FF++ splits do not apply; no matched genuine DFD controls in supplied corpus."}
    test_list = celeb_root / "List_of_testing_videos.txt"
    celeb_rows = parse_celeb_test_list(test_list, celeb_root)
    celeb_counts = Counter(r["method"] for r in celeb_rows)
    nominal = {"real": 178, **{r["method"]: (300 if r["scenario"] == "TalkingFace" else 200)
                             for r in celeb_rows if r["method"] != "real"}}
    count_discrepancies = {method: {"nominal_readme": expected, "official_list_observed": celeb_counts[method]}
                           for method, expected in nominal.items() if celeb_counts[method] != expected}
    rows = ff_rows + celeb_rows
    if len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate globally unique video IDs")
    split_groups: dict[str, set] = {}
    for row in ff_rows:
        split_groups.setdefault(row["group_id"], set()).add(row["split"])
    if any(len(splits) > 1 for splits in split_groups.values()):
        raise ValueError("FF++ source groups span multiple splits")
    index_path = out_dir / "index.csv"
    write_csv(index_path, rows, INDEX_COLUMNS)
    available_celeb = sum(1 for _ in celeb_root.rglob("*.mp4"))
    audit = {"schema_version": 1, "label_convention": "0=real,1=fake; official Celeb list inverted",
             "roots": {"ffpp": str(ffpp_root), "celebdfpp": str(celeb_root)},
             "official_ffpp_splits": provenance,
             "celeb_test_list": {"sha256": sha256_file(test_list), "listed": len(celeb_rows),
                                 "available_mp4": available_celeb, "unlisted_excluded": available_celeb - len(celeb_rows),
                                 "nominal_readme_total": sum(nominal.values()),
                                 "count_discrepancies": count_discrepancies,
                                 "discrepancy_policy": "Retain exactly the official list; never downsample to a nominal README count.",
                                 "nominal_source": "https://github.com/OUC-VAS/Celeb-DF-PP#training-and-testing-split"},
             "ffpp_inventory": counts, "excluded": exclusions,
             "counts": dict(sorted(Counter(f"{r['dataset']}/{r['split']}/{r['method']}" for r in rows).items())),
             "index_sha256": sha256_file(index_path),
             "limitations": ["FF++ original IDs are source-video groups, not verified person identities.",
                             "Celeb-DF++ is evaluation-only; unlisted videos are never added.",
                             "Celeb first identity token defines bootstrap groups, not verified biometric identity labels.",
                             "Celeb donor/driving identities may recur across first-token groups; multiway/per-identity sensitivity remains advisable."]}
    write_json(out_dir / "index.meta.json", audit)
    return index_path