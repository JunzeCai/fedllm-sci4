from pathlib import Path

from fedllm_data.edgeiiot import make_dirichlet_client_partition, make_stratified_row_split


def write_selected_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "1,Normal",
        "2,Normal",
        "3,Normal",
        "4,Normal",
        "5,MITM",
        "6,MITM",
        "7,MITM",
        "8,MITM",
        "9,OS_Fingerprinting",
        "10,OS_Fingerprinting",
    ]
    path.write_text("feature,Attack_type\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_make_stratified_row_split_normalizes_labels_and_is_deterministic(tmp_path: Path):
    csv_path = tmp_path / "ML-EdgeIIoT-dataset.csv"
    write_selected_csv(csv_path)

    split_a = make_stratified_row_split(csv_path, seed=11, train_ratio=0.5, val_ratio=0.25)
    split_b = make_stratified_row_split(csv_path, seed=11, train_ratio=0.5, val_ratio=0.25)

    assert split_a == split_b
    assert sorted(split_a["labels"]) == ["Fingerprinting", "MITM", "Normal"]
    assert split_a["label_counts"]["Fingerprinting"] == 2
    assigned = split_a["train_indices"] + split_a["val_indices"] + split_a["test_indices"]
    assert sorted(assigned) == list(range(10))


def test_make_dirichlet_client_partition_preserves_train_indices(tmp_path: Path):
    csv_path = tmp_path / "ML-EdgeIIoT-dataset.csv"
    write_selected_csv(csv_path)
    split = make_stratified_row_split(csv_path, seed=13, train_ratio=0.6, val_ratio=0.2)

    partition = make_dirichlet_client_partition(
        split["train_indices"],
        split["row_labels"],
        num_clients=3,
        alpha=0.5,
        seed=17,
    )

    assigned = []
    for client in partition["clients"]:
        assigned.extend(client["indices"])
        assert client["sample_count"] == len(client["indices"])
        assert sum(client["label_counts"].values()) == client["sample_count"]
    assert sorted(assigned) == sorted(split["train_indices"])
    assert partition["num_clients"] == 3
