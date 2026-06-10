#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fedllm_data.edgeiiot import build_file_manifest, build_label_inventory, make_source_split_plan, read_csv_rows
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

    edge_manifest = build_file_manifest(Path(args.edge_root), count_rows=args.count_rows)
    write_json(edge_out / "file_manifest.json", edge_manifest)

    split_plan = make_source_split_plan(
        edge_manifest,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_json(edge_out / f"source_split_seed{args.seed}.json", split_plan)

    label_inventory = build_label_inventory(edge_manifest, scan_selected=not args.skip_selected_label_scan)
    write_json(edge_out / "label_inventory.json", label_inventory)

    sample_count = write_prompt_smoke_samples(edge_manifest, edge_out / "prompt_smoke_samples.jsonl", args.sample_count)

    snli_manifest = build_snli_manifest(Path(args.snli_root))
    write_json(snli_out / "manifest.json", snli_manifest)

    print(
        json.dumps(
            {
                "edge_files": edge_manifest["file_count"],
                "edge_sources": edge_manifest["source_count"],
                "source_split": {
                    "train": len(split_plan["train"]),
                    "val": len(split_plan["val"]),
                    "test": len(split_plan["test"]),
                    "excluded": len(split_plan["excluded_sources"]),
                },
                "prompt_smoke_samples": sample_count,
                "raw_label_count": len(label_inventory["raw_source_label_counts"]),
                "selected_label_files": len(label_inventory["selected_label_counts"]),
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
    parser.add_argument(
        "--skip-selected-label-scan",
        action="store_true",
        help="Skip label distribution scans over selected merged Edge-IIoTset CSV files.",
    )
    return parser.parse_args()


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
