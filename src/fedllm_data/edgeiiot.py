from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any

from fedllm_data.labels import EDGEIIOT_LABEL_ALIASES, normalize_edgeiiot_label


def build_file_manifest(root: Path | str, count_rows: bool = False) -> dict[str, Any]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Edge-IIoTset root does not exist: {root_path}")

    files = [
        _describe_csv(root_path, path, count_rows=count_rows)
        for path in sorted(root_path.rglob("*.csv"), key=lambda item: item.as_posix())
    ]
    sources = sorted({item["source"] for item in files if item["source"]})
    groups = {group: 0 for group in ["attack", "normal", "selected", "other"]}
    for item in files:
        groups[item["group"]] = groups.get(item["group"], 0) + 1

    return {
        "dataset": "edgeiiotset",
        "root": str(root_path.resolve()),
        "file_count": len(files),
        "source_count": len(sources),
        "groups": groups,
        "sources": sources,
        "files": files,
    }


def make_source_split_plan(
    manifest: dict[str, Any],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> dict[str, Any]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    source_files: dict[str, list[str]] = {}
    excluded_sources = set()
    for item in manifest.get("files", []):
        source = item.get("source")
        if not source:
            continue
        if item.get("group") == "selected":
            excluded_sources.add(source)
            continue
        source_files.setdefault(source, []).append(item["relative_path"])

    sources = sorted(source_files)
    random.Random(seed).shuffle(sources)

    train_count = int(len(sources) * train_ratio)
    val_count = int(len(sources) * val_ratio)
    if sources and train_count == 0:
        train_count = 1
    if len(sources) - train_count - val_count <= 0 and val_count > 0:
        val_count -= 1

    train = sources[:train_count]
    val = sources[train_count : train_count + val_count]
    test = sources[train_count + val_count :]

    return {
        "strategy": "source-aware",
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "excluded_sources": sorted(excluded_sources),
        "source_files": {source: sorted(paths) for source, paths in sorted(source_files.items())},
    }


def make_stratified_row_split(
    csv_path: Path | str,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    label_key: str = "Attack_type",
) -> dict[str, Any]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    path = Path(csv_path)
    label_indices: dict[str, list[int]] = {}
    row_labels: dict[int, str] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or label_key not in reader.fieldnames:
            raise ValueError(f"CSV is missing label column {label_key!r}: {path}")
        for index, row in enumerate(reader):
            label = normalize_edgeiiot_label(row.get(label_key))
            if not label:
                continue
            label_indices.setdefault(label, []).append(index)
            row_labels[index] = label

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    split_label_counts: dict[str, dict[str, int]] = {}

    for label in sorted(label_indices):
        indices = list(label_indices[label])
        rng.shuffle(indices)
        train_count = int(len(indices) * train_ratio)
        val_count = int(len(indices) * val_ratio)
        if len(indices) >= 3:
            train_count = max(1, train_count)
            val_count = max(1, val_count)
            if train_count + val_count >= len(indices):
                val_count = max(0, len(indices) - train_count - 1)
        train_part = indices[:train_count]
        val_part = indices[train_count : train_count + val_count]
        test_part = indices[train_count + val_count :]
        train_indices.extend(train_part)
        val_indices.extend(val_part)
        test_indices.extend(test_part)
        split_label_counts[label] = {
            "train": len(train_part),
            "val": len(val_part),
            "test": len(test_part),
            "total": len(indices),
        }

    return {
        "strategy": "stratified-row-level",
        "csv_path": str(path.resolve()),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "label_key": label_key,
        "labels": sorted(label_indices),
        "label_counts": {label: len(indices) for label, indices in sorted(label_indices.items())},
        "split_label_counts": split_label_counts,
        "train_indices": sorted(train_indices),
        "val_indices": sorted(val_indices),
        "test_indices": sorted(test_indices),
        "row_labels": {str(index): label for index, label in sorted(row_labels.items())},
    }


def make_dirichlet_client_partition(
    train_indices: list[int],
    row_labels: dict[int, str] | dict[str, str],
    num_clients: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    labels_by_index = {int(index): label for index, label in row_labels.items()}
    by_label: dict[str, list[int]] = {}
    for index in train_indices:
        label = labels_by_index[int(index)]
        by_label.setdefault(label, []).append(int(index))

    rng = random.Random(seed)
    clients = [{"client_id": f"client_{i:03d}", "indices": []} for i in range(num_clients)]
    for label in sorted(by_label):
        indices = list(by_label[label])
        rng.shuffle(indices)
        proportions = _sample_dirichlet(num_clients, alpha, rng)
        counts = _proportions_to_counts(len(indices), proportions)
        cursor = 0
        for client, count in zip(clients, counts, strict=True):
            client["indices"].extend(indices[cursor : cursor + count])
            cursor += count

    label_space = sorted(by_label)
    for client in clients:
        client["indices"] = sorted(client["indices"])
        counts = Counter(labels_by_index[index] for index in client["indices"])
        client["label_counts"] = {label: counts.get(label, 0) for label in label_space}
        client["sample_count"] = len(client["indices"])

    return {
        "strategy": "dirichlet-label-skew",
        "seed": seed,
        "num_clients": num_clients,
        "alpha": alpha,
        "clients": clients,
    }


def build_label_inventory(
    manifest: dict[str, Any],
    label_key: str = "Attack_type",
    scan_selected: bool = False,
) -> dict[str, Any]:
    raw_counts: Counter[str] = Counter()
    raw_by_source: dict[str, dict[str, int]] = {}
    selected_counts: dict[str, dict[str, int]] = {}

    for item in manifest.get("files", []):
        group = item.get("group")
        source = item.get("source")
        if group in {"attack", "normal"}:
            label = item.get("label_from_first_row") or item.get("label_hint")
            rows = item.get("rows")
            if not label or rows is None:
                continue
            normalized = normalize_edgeiiot_label(str(label))
            raw_counts[normalized] += int(rows)
            raw_by_source[str(source)] = {normalized: int(rows)}
        elif group == "selected" and scan_selected:
            selected_counts[Path(item["relative_path"]).name] = _count_label_values(Path(item["path"]), label_key)

    return {
        "dataset": manifest.get("dataset", "edgeiiotset"),
        "label_key": label_key,
        "normalization": EDGEIIOT_LABEL_ALIASES,
        "raw_source_label_counts": dict(sorted(raw_counts.items())),
        "raw_source_labels": {source: labels for source, labels in sorted(raw_by_source.items())},
        "selected_label_counts": selected_counts,
    }


def read_csv_rows(path: Path | str, limit: int) -> list[dict[str, str]]:
    if limit < 1:
        return []
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({key: value for key, value in row.items() if key is not None})
            if len(rows) >= limit:
                break
        return rows


def _describe_csv(root: Path, path: Path, count_rows: bool) -> dict[str, Any]:
    relative_path = path.relative_to(root).as_posix()
    parts = relative_path.split("/")
    group = _infer_group(parts)
    source = _infer_source(parts, path, group)
    label_hint = _infer_label_hint(path, group)
    selected_kind = _infer_selected_kind(path, group)
    header = _read_header(path)

    return {
        "path": str(path.resolve()),
        "relative_path": relative_path,
        "group": group,
        "source": source,
        "label_hint": label_hint,
        "label_from_first_row": _read_first_label(path, header),
        "selected_kind": selected_kind,
        "bytes": path.stat().st_size,
        "columns": len(header),
        "header": header,
        "rows": _count_csv_rows(path) if count_rows else None,
    }


def _infer_group(parts: list[str]) -> str:
    if not parts:
        return "other"
    if parts[0] == "Attack traffic":
        return "attack"
    if parts[0] == "Normal traffic":
        return "normal"
    if parts[0] == "Selected dataset for ML and DL":
        return "selected"
    return "other"


def _infer_source(parts: list[str], path: Path, group: str) -> str:
    if group == "attack":
        return _strip_attack_suffix(path.stem)
    if group == "normal":
        return parts[1] if len(parts) > 1 else path.stem
    if group == "selected":
        return _infer_selected_kind(path, group) or path.stem
    return path.stem


def _infer_label_hint(path: Path, group: str) -> str | None:
    if group == "attack":
        return _strip_attack_suffix(path.stem)
    if group == "normal":
        return "Normal"
    return None


def _infer_selected_kind(path: Path, group: str) -> str | None:
    if group != "selected":
        return None
    stem = path.stem
    if stem.startswith("ML-"):
        return "ML"
    if stem.startswith("DNN-"):
        return "DNN"
    return stem


def _strip_attack_suffix(stem: str) -> str:
    return stem.removesuffix("_attack")


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration:
            return []


def _read_first_label(path: Path, header: list[str]) -> str | None:
    if not header:
        return None
    label_keys = ["Attack_type", "label", "Label"]
    label_index = next((header.index(key) for key in label_keys if key in header), None)
    if label_index is None:
        return None
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if label_index < len(row):
                value = row[label_index].strip()
                if value:
                    return value
    return None


def _count_label_values(path: Path, label_key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get(label_key) or "").strip()
            if label:
                counts[normalize_edgeiiot_label(label)] += 1
    return dict(sorted(counts.items()))


def _sample_dirichlet(size: int, alpha: float, rng: random.Random) -> list[float]:
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(size)]
    total = sum(draws)
    if total == 0:
        return [1.0 / size] * size
    return [draw / total for draw in draws]


def _proportions_to_counts(total: int, proportions: list[float]) -> list[int]:
    raw = [total * proportion for proportion in proportions]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(proportions)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def _count_csv_rows(path: Path) -> int:
    line_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            line_count += chunk.count(b"\n")
    if path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                line_count += 1
    return max(0, line_count - 1)
