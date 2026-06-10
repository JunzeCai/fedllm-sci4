from fedllm_data.labels import EDGEIIOT_LABELS, normalize_edgeiiot_label


def test_normalize_edgeiiot_label_maps_raw_os_fingerprinting_to_selected_label():
    assert normalize_edgeiiot_label("OS_Fingerprinting") == "Fingerprinting"
    assert "Fingerprinting" in EDGEIIOT_LABELS
    assert "OS_Fingerprinting" not in EDGEIIOT_LABELS
