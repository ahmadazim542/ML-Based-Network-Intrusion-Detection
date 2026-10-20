"""Fake day files in the corrected-CICFlowMeter CSV format, for smoke-testing the pipeline
without the real dataset. The numbers mean nothing; only the schema and quirks are realistic
(identifier columns, Attempted labels, correlated and constant features, inf values, duplicates).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = (
    "Flow Duration|Total Fwd Packet|Total Bwd packets|Total Length of Fwd Packet|Total Length of Bwd Packet|"
    "Fwd Packet Length Max|Fwd Packet Length Min|Fwd Packet Length Mean|Fwd Packet Length Std|"
    "Bwd Packet Length Max|Bwd Packet Length Min|Bwd Packet Length Mean|Bwd Packet Length Std|Flow Bytes/s|"
    "Flow Packets/s|Flow IAT Mean|Flow IAT Std|Flow IAT Max|Flow IAT Min|Fwd IAT Total|Fwd IAT Mean|"
    "Bwd IAT Total|Bwd IAT Mean|Fwd PSH Flags|Fwd Header Length|Bwd Header Length|Fwd Packets/s|Bwd Packets/s|"
    "Packet Length Min|Packet Length Max|Packet Length Mean|Packet Length Std|FIN Flag Count|SYN Flag Count|"
    "RST Flag Count|PSH Flag Count|ACK Flag Count|Down/Up Ratio|Average Packet Size|Fwd Bytes/Bulk Avg|"
    "Subflow Fwd Packets|Subflow Fwd Bytes|FWD Init Win Bytes|Bwd Init Win Bytes|Fwd Act Data Pkts|"
    "Active Mean|Idle Mean|ICMP Code|ICMP Type|Total TCP Flow Time"
).split("|")

DAYS = {
    "monday": {"BENIGN": 1.0},
    "tuesday": {"BENIGN": 0.7, "FTP-Patator": 0.12, "SSH-Patator": 0.1, "FTP-Patator - Attempted": 0.08},
    "wednesday": {"BENIGN": 0.6, "DoS Hulk": 0.25, "DoS Slowloris": 0.15, "Heartbleed": 0.0},
    "thursday": {"BENIGN": 0.8, "Web Attack - XSS": 0.12, "Infiltration": 0.08},
    "friday": {"BENIGN": 0.55, "Portscan": 0.2, "DDoS": 0.15, "Botnet": 0.1},
}


def _rows(label: str, n: int, rng: np.random.Generator, shift: float) -> pd.DataFrame:
    k = len(FEATURES)
    # class-specific log-mean per feature; benign is centred, attacks are offset on a subset
    h = zlib.crc32(label.split(" - ")[0].encode())
    crng = np.random.default_rng(h)
    mu = np.zeros(k) + 3
    if not label.startswith("BENIGN") and "Attempted" not in label:
        idx = crng.choice(k, size=8, replace=False)
        mu[idx] += crng.normal(0, 2.0, size=8)
    X = np.exp(rng.normal(mu + shift, 1.0, size=(n, k)))
    df = pd.DataFrame(X, columns=FEATURES)
    df["Subflow Fwd Packets"] = df["Total Fwd Packet"]               # perfectly correlated pair
    df["Fwd Bytes/Bulk Avg"] = 0.0                                     # constant column
    df[["ICMP Code", "ICMP Type"]] = -1
    df["Label"] = label
    return df


def write_days(directory: str | Path, rows_per_day: int = 3000, seed: int = 0, shift: float = 0.0,
               heartbleed_rows: int = 6) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for d, (day, mix) in enumerate(DAYS.items()):
        parts = [_rows(lab, int(rows_per_day * p), rng, shift) for lab, p in mix.items() if p > 0]
        if day == "wednesday":
            parts.append(_rows("Heartbleed", heartbleed_rows, rng, shift))
        df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
        n = len(df)
        start = pd.Timestamp("2017-07-03 08:00") + pd.Timedelta(days=d)
        df.insert(0, "Timestamp", start + pd.to_timedelta(np.sort(rng.uniform(0, 8 * 3600, n)), unit="s"))
        df.insert(0, "Protocol", rng.choice([6, 17], n))
        df.insert(0, "Dst Port", rng.integers(1, 65535, n))
        df.insert(0, "Dst IP", "192.168.10.50")
        df.insert(0, "Src Port", rng.integers(1024, 65535, n))
        df.insert(0, "Src IP", "172.16.0.1")
        df.insert(0, "Flow ID", "x")
        df["Attempted Category"] = np.where(df["Label"].str.contains("Attempted"), 0, -1)
        df.iloc[:5, df.columns.get_loc("Flow Bytes/s")] = np.inf          # CICFlowMeter quirk
        df = pd.concat([df, df.iloc[:20]], ignore_index=True)               # duplicates
        df.index.name = "id"
        df.to_csv(directory / f"{day}.csv")
    return directory


if __name__ == "__main__":
    from .config import PROJECT_ROOT

    base = PROJECT_ROOT / "data/raw/synthetic"
    write_days(base / "cicids2017", rows_per_day=1500, seed=1)
    write_days(base / "cicids2018", rows_per_day=800, seed=2, shift=0.3)
    print("synthetic data written to", base)
