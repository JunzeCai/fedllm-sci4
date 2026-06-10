from fedllm_data.edgeiiot import make_source_split_plan


def test_make_source_split_plan_is_deterministic_and_keeps_sources_intact():
    manifest = {
        "files": [
            {"relative_path": "Attack traffic/A.csv", "group": "attack", "source": "A"},
            {"relative_path": "Attack traffic/B.csv", "group": "attack", "source": "B"},
            {"relative_path": "Normal traffic/N1/N1.csv", "group": "normal", "source": "N1"},
            {"relative_path": "Normal traffic/N2/N2.csv", "group": "normal", "source": "N2"},
            {
                "relative_path": "Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv",
                "group": "selected",
                "source": "ML",
            },
        ]
    }

    split_a = make_source_split_plan(manifest, seed=7, train_ratio=0.5, val_ratio=0.25)
    split_b = make_source_split_plan(manifest, seed=7, train_ratio=0.5, val_ratio=0.25)

    assert split_a == split_b
    assigned = split_a["train"] + split_a["val"] + split_a["test"]
    assert sorted(assigned) == sorted(["A", "B", "N1", "N2"])
    assert "ML" not in assigned
    assert split_a["excluded_sources"] == ["ML"]
