#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from fedllm_data.edgeiiot import build_file_manifest, build_label_inventory, make_source_split_plan, read_csv_rows
from fedllm_data.edgeiiot import make_dirichlet_client_partition, make_stratified_row_split
from fedllm_data.prompts import render_edgeiiot_prompt
from fedllm_data.snli import build_snli_manifest


DEFAULT_PROMPT_FEATURES = [
    "tcp.flags",
    "tcp.len",
    "tcp.dstport",
    "tcp.srcport",
    "udp.port",
    "udp.time_delta",
    "dns.qry.name.len",
    "dns.qry.type",
    "mqtt.len",
    "mqtt.msgtype",
    "mbtcp.len",
    "mbtcp.unit_id",
]

EXCLUDED_PROMPT_COLUMNS = {
    "Attack_label",
    "Attack_type",
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
}


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    edge_out = out_dir / "edgeiiot"
    snli_out = out_dir / "snli"
    edge_out.mkdir(parents=True, exist_ok=True)
    snli_out.mkdir(parents=True, exist_ok=True)

    repo_root = Path.cwd().resolve()

    edge_manifest = build_file_manifest(Path(args.edge_root), count_rows=args.count_rows)
    edge_manifest_out = make_portable_manifest(edge_manifest, repo_root=repo_root) if args.relative_paths else edge_manifest
    write_json(edge_out / "file_manifest.json", add_generation_metadata(edge_manifest_out, args, path_mode=edge_manifest_out.get("path_mode", "absolute")))

    split_plan = make_source_split_plan(
        edge_manifest,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_json(edge_out / f"source_split_seed{args.seed}.json", add_generation_metadata(split_plan, args, path_mode="relative"))

    label_inventory = build_label_inventory(edge_manifest, scan_selected=not args.skip_selected_label_scan)
    write_json(edge_out / "label_inventory.json", add_generation_metadata(label_inventory, args, path_mode="relative"))

    sample_count = write_prompt_smoke_samples(edge_manifest, edge_out / "prompt_smoke_samples.jsonl", args.sample_count)
    selected_split = None
    client_partition = None
    selected_path = _selected_ml_path(edge_manifest)
    if selected_path is not None:
        selected_split = make_stratified_row_split(
            selected_path,
            seed=args.seed,
            train_ratio=args.selected_train_ratio,
            val_ratio=args.selected_val_ratio,
        )
        selected_split_out = make_portable_manifest(selected_split, repo_root=repo_root) if args.relative_paths else selected_split
        write_json(
            edge_out / f"selected_ml_stratified_split_seed{args.seed}.json",
            add_generation_metadata(
                selected_split_out,
                args,
                path_mode=selected_split_out.get("path_mode", "absolute"),
            ),
        )
        client_partition = make_dirichlet_client_partition(
            selected_split["train_indices"],
            selected_split["row_labels"],
            num_clients=args.num_clients,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
        )
        write_json(
            edge_out / f"selected_ml_clients_seed{args.seed}_K{args.num_clients}_alpha{args.dirichlet_alpha:g}.json",
            add_generation_metadata(client_partition, args, path_mode="relative"),
        )

    snli_manifest = build_snli_manifest(Path(args.snli_root))
    snli_manifest_out = make_portable_manifest(snli_manifest, repo_root=repo_root) if args.relative_paths else snli_manifest
    write_json(snli_out / "manifest.json", add_generation_metadata(snli_manifest_out, args, path_mode=snli_manifest_out.get("path_mode", "absolute")))

    print(
        json.dumps(
            {
                "edge_files": edge_manifest["file_count"],
                "edge_sources": edge_manifest["source_count"],
                "path_mode": "relative" if args.relative_paths else "absolute",
                "source_split": {
                    "train": len(split_plan["train"]),
                    "val": len(split_plan["val"]),
                    "test": len(split_plan["test"]),
                    "excluded": len(split_plan["excluded_sources"]),
                },
                "prompt_smoke_samples": sample_count,
                "raw_label_count": len(label_inventory["raw_source_label_counts"]),
                "selected_label_files": len(label_inventory["selected_label_counts"]),
                "selected_split": None
                if selected_split is None
                else {
                    "train": len(selected_split["train_indices"]),
                    "val": len(selected_split["val_indices"]),
                    "test": len(selected_split["test_indices"]),
                },
                "snli_rows": {split: item["rows"] for split, item in snli_manifest["splits"].items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reproducible dataset artifacts for SPECTRA-FedCore.")
    parser.add_argument("--edge-root", default="data/raw/edgeiiotset/full_dataset")
    parser.add_argument("--snli-root", default="data/raw/snli/current")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--count-rows", action="store_true", help="Count CSV rows in every Edge-IIoTset CSV file.")
    parser.add_argument("--selected-train-ratio", type=float, default=0.8)
    parser.add_argument("--selected-val-ratio", type=float, default=0.1)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument(
        "--skip-selected-label-scan",
        action="store_true",
        help="Skip label distribution scans over selected merged Edge-IIoTset CSV files.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write repository-relative paths into generated manifests for portable server reproduction.",
    )
    return parser.parse_args()


def add_generation_metadata(payload: dict[str, Any], args: argparse.Namespace, *, path_mode: str) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result.setdefault("schema_version", "2026-06-10")
    result["path_mode"] = path_mode
    result["generation"] = {
        "tool": "scripts/prepare_datasets.py",
        "seed": args.seed,
        "count_rows": bool(args.count_rows),
        "relative_paths": bool(args.relative_paths),
    }
    return result


def make_portable_manifest(payload: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result["path_mode"] = "relative"

    if "root" in result:
        result["root"] = _to_relative_or_original(result["root"], repo_root)
    if "csv_path" in result:
        result["csv_path"] = _to_relative_or_original(result["csv_path"], repo_root)

    for item in result.get("files", []):
        if "path" in item:
            item["path"] = _to_relative_or_original(item["path"], repo_root)

    for split_item in result.get("splits", {}).values():
        if "path" in split_item:
            split_item["path"] = _to_relative_or_original(split_item["path"], repo_root)
    return result


def _to_relative_or_original(path_value: str, repo_root: Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        # Preserve only the dataset-internal suffix when the original manifest was
        # generated on another machine. This avoids committing local absolute paths.
        parts = path.parts
        for marker in ("data", "raw"):
            if marker in parts:
                index = parts.index(marker)
                return Path(*parts[index:]).as_posix()
        return path.name


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_prompt_smoke_samples(manifest: dict[str, Any], out_path: Path, sample_count: int) -> int:
    selected = _choose_prompt_source(manifest)
    if selected is None or sample_count < 1:
        out_path.write_text("", encoding="utf-8")
        return 0

    feature_names = _choose_prompt_features(selected["header"])
    rows = read_csv_rows(selected["path"], sample_count)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = render_edgeiiot_prompt(row, feature_names=feature_names)
            prompt["source_file"] = selected["relative_path"]
            handle.write(json.dumps(prompt, sort_keys=True) + "\n")
    return len(rows)


def _choose_prompt_source(manifest: dict[str, Any]) -> dict[str, Any] | None:
    files = manifest.get("files", [])
    for item in files:
        if item.get("group") == "selected" and item.get("selected_kind") == "ML":
            return item
    for item in files:
        if item.get("group") == "selected":
            return item
    return files[0] if files else None


def _selected_ml_path(manifest: dict[str, Any]) -> Path | None:
    for item in manifest.get("files", []):
        if item.get("group") == "selected" and item.get("selected_kind") == "ML":
            return Path(item["path"])
    return None


def _choose_prompt_features(header: list[str], max_features: int = 12) -> list[str]:
    features = [name for name in DEFAULT_PROMPT_FEATURES if name in header]
    if len(features) >= max_features:
        return features[:max_features]

    for name in header:
        if name in EXCLUDED_PROMPT_COLUMNS or name in features:
            continue
        features.append(name)
        if len(features) >= max_features:
            break
    return features


if __name__ == "__main__":
    raise SystemExit(main())
