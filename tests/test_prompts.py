from fedllm_data.prompts import render_edgeiiot_prompt


def test_render_edgeiiot_prompt_is_deterministic_and_label_bounded():
    row = {
        "frame.len": "74",
        "tcp.flags": "0x00000012",
        "http.request.method": "GET",
        "Attack_type": "MITM",
    }

    prompt = render_edgeiiot_prompt(row, feature_names=["frame.len", "tcp.flags", "http.request.method"])

    assert "Industrial IoT intrusion-detection record" in prompt["instruction"]
    assert "frame.len=74" in prompt["input"]
    assert prompt["output"] == "MITM"
    assert "Valid labels:" in prompt["instruction"]


def test_render_edgeiiot_prompt_normalizes_label_aliases():
    row = {"Attack_type": "OS_Fingerprinting"}

    prompt = render_edgeiiot_prompt(row, feature_names=[])

    assert prompt["output"] == "Fingerprinting"
