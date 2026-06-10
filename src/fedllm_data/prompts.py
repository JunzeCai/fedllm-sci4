from __future__ import annotations

from typing import Mapping, Sequence

from fedllm_data.labels import EDGEIIOT_LABELS, normalize_edgeiiot_label


def render_edgeiiot_prompt(
    row: Mapping[str, str],
    feature_names: Sequence[str],
    label_key: str = "Attack_type",
) -> dict[str, str]:
    label = normalize_edgeiiot_label(str(row.get(label_key, "")).strip())
    features = []
    for name in feature_names:
        value = str(row.get(name, "")).strip()
        features.append(f"{name}={value}")

    return {
        "instruction": (
            "Classify this Industrial IoT intrusion-detection record. "
            f"Valid labels: {', '.join(EDGEIIOT_LABELS)}."
        ),
        "input": "; ".join(features),
        "output": label,
    }
