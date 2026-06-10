from __future__ import annotations


EDGEIIOT_LABEL_ALIASES = {
    "OS_Fingerprinting": "Fingerprinting",
}

EDGEIIOT_LABELS = [
    "Normal",
    "Backdoor",
    "DDoS_HTTP",
    "DDoS_ICMP",
    "DDoS_TCP",
    "DDoS_UDP",
    "Fingerprinting",
    "MITM",
    "Password",
    "Port_Scanning",
    "Ransomware",
    "SQL_injection",
    "Uploading",
    "Vulnerability_scanner",
    "XSS",
]


def normalize_edgeiiot_label(label: str | None) -> str:
    value = (label or "").strip()
    return EDGEIIOT_LABEL_ALIASES.get(value, value)
