"""Dataset loaders.

Every loader returns one DataFrame with:
  * numeric flow features (float32, snake_case names)
  * `label`     raw label string from the dataset
  * `family`    grouped attack family (Benign, DoS, DDoS, PortScan, BruteForce, WebAttack,
                Botnet, Infiltration, Heartbleed)
  * `is_attack` 0/1
  * `day`, `timestamp` metadata (never used as model inputs)

Supported datasets
  cicids2017_fixed  corrected CICIDS2017 (Liu, Engelen et al., IEEE CNS 2022)
  cicids2018_fixed  corrected CSE-CIC-IDS2018, same fixed CICFlowMeter -> identical feature schema
  nf_v2             UQ NetFlow-v2 datasets (NF-CSE-CIC-IDS2018-v2, NF-UNSW-NB15-v2, ...). These
                    use a different feature set, so they can only be compared with each other.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = ["label", "family", "is_attack", "day", "timestamp"]

# Identifiers and endpoint fields. They identify the testbed hosts rather than describe
# traffic behaviour, so a model that uses them learns "this IP/port was the attacker"
# and will not transfer to another network.
CIC_DROP = {
    "id", "flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "timestamp",
    "label", "attempted_category",
}
NF_DROP = {
    "ipv4_src_addr", "ipv4_dst_addr", "l4_src_port", "l4_dst_port",
    "label", "attack", "dataset",
}

FAMILIES = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Botnet",
            "Infiltration", "Heartbleed"]

# Order matters: the first matching pattern wins ("Web Attack - Brute Force" is a web
# attack, "DDoS" must win over "DoS"). Families describe what the *flow* looks like:
# "Infiltration - Portscan" (~72k flows in corrected CICIDS2017) is the port scan run from
# the compromised host, so it is a PortScan flow; folding it into Infiltration would let 72k
# scan flows stand in for the 36 genuine infiltration flows and inflate Infiltration recall.
_FAMILY_PATTERNS = [
    ("WebAttack", r"web|xss|sql"),
    ("DDoS", r"ddos"),
    ("Heartbleed", r"heartbleed"),
    ("DoS", r"\bdos\b|dos[ -]|hulk|goldeneye|slowloris|slowhttp"),
    ("PortScan", r"port ?scan|reconnaissance|scanning"),
    ("Infiltration", r"infil"),
    ("Botnet", r"\bbot"),
    ("BruteForce", r"patator|brute|ftp|ssh|password"),
]


def snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def label_to_family(label: str, attempted_as_benign: bool = True) -> str:
    s = str(label).strip().lower()
    if s in {"benign", "normal", "0"}:
        return "Benign"
    if attempted_as_benign and "attempted" in s:
        return "Benign"
    for fam, pat in _FAMILY_PATTERNS:
        if re.search(pat, s):
            return fam
    return "Other"


def _finish(df: pd.DataFrame, drop: set[str], label_col: str, attempted_as_benign: bool,
            day: str) -> pd.DataFrame:
    df.columns = [snake(c) for c in df.columns]
    label_col = snake(label_col)
    labels = df[label_col].astype(str)
    out = pd.DataFrame(index=df.index)
    feats = [c for c in df.columns if c not in drop]
    num = df[feats].apply(pd.to_numeric, errors="coerce")
    # float32 halves memory (2M x 80 features ~ 650 MB instead of 1.3 GB)
    num = num.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    out = pd.concat([out, num], axis=1)
    fam_map = {l: label_to_family(l, attempted_as_benign) for l in labels.unique()}
    out["label"] = labels.values
    out["family"] = labels.map(fam_map).values
    out["is_attack"] = (out["family"] != "Benign").astype(np.int8)
    out["day"] = day
    ts = df["timestamp"] if "timestamp" in df.columns else None
    out["timestamp"] = pd.to_datetime(ts, errors="coerce") if ts is not None else pd.NaT
    return out


def read_cic_fixed_csv(path: Path, attempted_as_benign: bool = True,
                       usecols=None, nrows=None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, usecols=usecols, nrows=nrows)
    return _finish(df, CIC_DROP, "Label", attempted_as_benign, day=Path(path).stem.lower())


def load_cic_fixed(directory: str | Path, attempted_as_benign: bool = True) -> pd.DataFrame:
    """Load every CSV/parquet day file in `directory` (corrected CICIDS2017 or CSE-CIC-IDS2018).

    Parquet files are what scripts/fetch_data.py writes for the streamed 2018 subset; they are
    already in the processed format.
    """
    directory = Path(directory)
    parts = []
    for p in sorted(directory.glob("*.parquet")):
        parts.append(pd.read_parquet(p))
    for p in sorted(directory.glob("*.csv")):
        parts.append(read_cic_fixed_csv(p, attempted_as_benign))
    if not parts:
        raise FileNotFoundError(
            f"No .csv or .parquet files in {directory}. See README.md -> 'Getting the data'.")
    return pd.concat(parts, ignore_index=True)


def load_nf_v2(directory: str | Path) -> pd.DataFrame:
    """UQ NetFlow-v2 CSVs: binary `Label` (0/1) and string `Attack` columns."""
    parts = []
    for p in sorted(Path(directory).glob("*.csv")):
        df = pd.read_csv(p, low_memory=False)
        df.columns = [snake(c) for c in df.columns]
        out = _finish(df, NF_DROP, "attack", attempted_as_benign=False, day=p.stem.lower())
        out["is_attack"] = df["label"].astype(np.int8).values
        parts.append(out)
    if not parts:
        raise FileNotFoundError(f"No .csv files in {directory}")
    return pd.concat(parts, ignore_index=True)


LOADERS = {
    "cicids2017_fixed": load_cic_fixed,
    "cicids2018_fixed": load_cic_fixed,
    "nf_v2": lambda d, **_: load_nf_v2(d),
}


def load(name: str, directory: str | Path, attempted_as_benign: bool = True) -> pd.DataFrame:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    return LOADERS[name](directory, attempted_as_benign=attempted_as_benign)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]
