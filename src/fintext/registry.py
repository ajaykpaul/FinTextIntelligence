"""Immutable model registry (PRD 6.6, 9)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .progress import ByteBar


class RegistryError(Exception):
    pass


@dataclass(frozen=True)
class Entry:
    version: str
    artifact_path: str
    artifact_digest: str
    created_at: float
    backend: str = "onnx"
    tokenizer_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    latency_p95_ms: float | None = None
    promoted: bool = False
    promoted_at: float | None = None
    parent_version: str | None = None
    notes: str = ""
    sentiment_offsets: dict[str, dict[str, float]] = field(default_factory=dict)

    def load_tokenizer(self) -> Any:
        """Tokenizer and weights are versioned together: a mismatch is silent at load time and
        wrong at score time.
        """
        if not self.tokenizer_path:
            raise RegistryError(f"{self.version}: no tokenizer recorded with the artefact")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_path)


def file_digest(path: str | Path) -> str:
    """Streaming, so a large artefact is not read into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class Registry:
    def __init__(self, root: str | Path, *, p95_target_ms: float = 50.0) -> None:
        self.root = Path(root)
        self.p95_target_ms = p95_target_ms
        self.entries: dict[str, Entry] = {}
        self.active_version: str | None = None
        self.shadow_version: str | None = None
        self.load()

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def load(self) -> None:
        if not self.index_path.exists():
            return
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        known = set(Entry.__dataclass_fields__)
        self.entries = {
            version: Entry(**{k: v for k, v in entry.items() if k in known})
            for version, entry in payload.get("entries", {}).items()
        }
        self.active_version = payload.get("active")
        self.shadow_version = payload.get("shadow")

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(
                {
                    "active": self.active_version,
                    "shadow": self.shadow_version,
                    "entries": {v: asdict(e) for v, e in self.entries.items()},
                    "updated_at": time.time(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def register(self, entry: Entry, *, verify_digest: bool = True) -> Entry:
        existing = self.entries.get(entry.version)
        if existing:
            if existing.artifact_digest != entry.artifact_digest:
                raise RegistryError(
                    f"version {entry.version!r} is already registered with different content; "
                    "versions are immutable"
                )
            return existing
        if verify_digest and Path(entry.artifact_path).exists():
            actual = file_digest(entry.artifact_path)
            if actual != entry.artifact_digest:
                raise RegistryError(
                    f"artefact digest mismatch for {entry.version!r}: "
                    f"recorded {entry.artifact_digest}, found {actual}"
                )
        self.entries[entry.version] = entry
        self.save()
        return entry

    def promote(self, version: str, *, force: bool = False) -> Entry:
        """Make a version servable."""
        entry = self.resolve(version)
        if not force:
            if entry.latency_p95_ms is None:
                raise RegistryError(
                    f"{version}: no latency benchmark recorded; run the CI benchmark first"
                )
            if entry.latency_p95_ms > self.p95_target_ms:
                raise RegistryError(
                    f"{version}: p95 {entry.latency_p95_ms:.1f} ms exceeds the "
                    f"{self.p95_target_ms:.0f} ms objective; promotion refused"
                )
        promoted = replace(
            entry,
            promoted=True,
            promoted_at=time.time(),
            notes=f"{entry.notes} [forced promotion]".strip() if force else entry.notes,
        )
        self.entries[version] = promoted
        self.active_version = version
        self.save()
        return promoted

    def set_shadow(self, version: str | None) -> None:
        """Shadow needs no latency gate: its responses are discarded, so a slow candidate costs the
        live path nothing.
        """
        if version is not None:
            self.resolve(version)
        self.shadow_version = version
        self.save()

    def resolve(self, version: str) -> Entry:
        if version not in self.entries:
            known = ", ".join(sorted(self.entries)) or "<empty registry>"
            raise RegistryError(f"unknown model version {version!r}; known: {known}")
        return self.entries[version]

    def active(self) -> Entry | None:
        return self.entries.get(self.active_version) if self.active_version else None

    def shadow(self) -> Entry | None:
        return self.entries.get(self.shadow_version) if self.shadow_version else None

    def versions(self) -> list[str]:
        return sorted(self.entries)

    def history(self, version: str) -> list[Entry]:
        """The parent chain, newest first."""
        chain: list[Entry] = []
        seen: set[str] = set()
        current: str | None = version
        while current and current not in seen and current in self.entries:
            seen.add(current)
            chain.append(self.entries[current])
            current = self.entries[current].parent_version
        return chain


# --------------------------------------------------------------------------
# Bringing a run back from Colab
# --------------------------------------------------------------------------

RUN_FILES = ("manifest.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json")
"""What a trained run has to contain to be scored or exported."""


def import_run(bundle: str | Path, models_dir: str | Path) -> Path:
    """Unpack a run archive from the training notebook into `models_dir`."""
    import zipfile

    with zipfile.ZipFile(bundle) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        unsafe = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
        if unsafe:
            raise RegistryError(f"archive has unsafe paths: {unsafe[:3]}")

        manifests = [n for n in names if Path(n).name == "manifest.json"]
        if len(manifests) != 1:
            raise RegistryError(
                f"expected one manifest.json in the archive, found {len(manifests)}"
            )
        prefix = str(Path(manifests[0]).parent)
        prefix = "" if prefix == "." else prefix + "/"

        present = {n[len(prefix) :] for n in names if n.startswith(prefix)}
        missing = [f for f in RUN_FILES if f not in present]
        if missing:
            raise RegistryError(f"archive is missing {', '.join(missing)}")

        manifest = json.loads(archive.read(manifests[0]))
        run_name = str(manifest.get("run_name") or "")
        if not run_name or Path(run_name).name != run_name:
            raise RegistryError(f"manifest has no usable run_name: {run_name!r}")

        destination = Path(models_dir) / run_name
        if destination.exists():
            raise RegistryError(f"{destination} already exists; a run is never overwritten")
        destination.mkdir(parents=True)
        wanted = [n for n in present if n != "checkpoint.pt"]  # resume state; Colab only
        total = sum(archive.getinfo(prefix + n).file_size for n in wanted)
        with ByteBar(total, desc=f"importing {run_name}") as bar:
            for name in wanted:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(prefix + name) as source, target.open("wb") as out:
                    while chunk := source.read(1 << 20):
                        out.write(chunk)
                        bar.update(len(chunk))
    return destination
