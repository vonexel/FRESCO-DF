from __future__ import annotations

import re
import open_clip
from pathlib import Path
from .data import sha256_file
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class EncoderCheckpoint:
    identity: dict
    path: Path | None
    metadata: dict


def resolve_encoder_checkpoint(encoder_name: str, pretrained: str, cache_dir: Path) -> EncoderCheckpoint:
    if encoder_name.startswith(("hf-hub:", "local-dir:")):
        raise ValueError("Use a registered OpenCLIP encoder_name and a pretrained tag or .safetensors path")

    metadata = dict(open_clip.get_pretrained_cfg(encoder_name, pretrained) or {})
    if pretrained == "openai":
        url = metadata.get("url", "")
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        if (parsed.scheme != "https" or parsed.hostname != "openaipublic.azureedge.net"
                or len(parts) < 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[-2])):
            raise ValueError("OpenAI encoder requires its official HTTPS URL with a full SHA256")
        return EncoderCheckpoint({"kind": "official_openai", "encoder": encoder_name,
                                  "url": url, "sha256": parts[-2]}, None, metadata)

    candidate = Path(pretrained).expanduser()
    if candidate.is_file():
        path, kind = candidate.resolve(), "local_safetensors"
    else:
        if not metadata:
            raise ValueError(f"Unknown pretrained tag or missing checkpoint: {pretrained!r}")
        from open_clip.pretrained import download_pretrained
        path = Path(download_pretrained(metadata, cache_dir=str(cache_dir))).resolve()
        kind = "resolved_tag_safetensors"
    # OpenCLIP selects its safe loader with a case-sensitive endswith check
    if path.suffix != ".safetensors":
        raise ValueError("Custom/tag encoder checkpoints must use .safetensors. Pickle/.pt/.pth/.bin "
                         "loading is disabled, including weights_only = True on the pinned PyTorch 2.6. "
                         "The SHA-verified official pretrained = openai checkpoint remains supported.")
    if not path.is_file():
        raise FileNotFoundError(f"Resolved encoder checkpoint is missing: {path}")
    identity = {"kind": kind, "encoder": encoder_name, "sha256": sha256_file(path),
                "format": "safetensors", "pretrained_metadata": metadata}
    return EncoderCheckpoint(identity, path, metadata)


def verify_checkpoint_file(checkpoint: EncoderCheckpoint) -> None:
    if checkpoint.path is not None:
        if checkpoint.path.suffix != ".safetensors":
            raise ValueError("Only custom .safetensors checkpoints may be deserialized")
        if sha256_file(checkpoint.path) != checkpoint.identity["sha256"]:
            raise ValueError("Encoder checkpoint changed after identity resolution; restart with a new cache")