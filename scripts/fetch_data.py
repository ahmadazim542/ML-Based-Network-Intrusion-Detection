"""Download the corrected CICIDS2017 / CSE-CIC-IDS2018 datasets (Liu, Engelen et al., CNS 2022).

    python scripts/fetch_data.py cicids2017
        -> data/raw/cicids2017_fixed/{monday..friday}.csv        (328 MB download, ~1.15 GB on disk)

    python scripts/fetch_data.py cicids2018 --days Wednesday-14-02-2018 Friday-16-02-2018 --benign-frac 0.05
        -> data/raw/cicids2018_fixed/<day>.parquet

The 2018 archive is 9.7 GB (each day ~3-4 GB of CSV), more than this laptop can hold. Instead of
downloading it, this script reads single files out of the remote zip with HTTP range requests,
streams them through pandas in chunks, keeps every attack flow plus a random fraction of benign
flows, and writes a small parquet file. Only ~1 GB per day crosses the network; almost nothing
is stored.

Run `python scripts/fetch_data.py list` to see the files and sizes in both archives.
"""
from __future__ import annotations

import argparse
import http.client
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ids.datasets import CIC_DROP, _finish  # noqa: E402

BASE = "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/"
URLS = {"cicids2017": BASE + "CICIDS2017_improved.zip", "cicids2018": BASE + "CSECICIDS2018_improved.zip"}

# Days with the attack families that overlap CICIDS2017 (brute force, DoS, DDoS, botnet, web, infiltration)
DEFAULT_2018_DAYS = ["Wednesday-14-02-2018", "Thursday-15-02-2018", "Friday-16-02-2018",
                     "Wednesday-21-02-2018", "Friday-02-03-2018"]


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            if r.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError("server does not support range requests")
            self.size = int(r.headers["Content-Length"])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b, retries: int = 12):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        # A multi-GB stream over a slow intercontinental link will see dropped connections (and a
        # laptop that sleeps kills every open socket). Each request is a byte range, so retrying
        # it resumes exactly where the stream stopped.
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == n:
                    break
                raise http.client.IncompleteRead(data, n - len(data))
            except (http.client.HTTPException, urllib.error.URLError, OSError) as e:
                if attempt == retries - 1:
                    raise
                wait = min(2 ** attempt, 120)
                print(f"\n  range {self.pos}: {type(e).__name__}; retry {attempt + 1} in {wait}s", flush=True)
                time.sleep(wait)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=8 << 20))


def cmd_list(_):
    for name, url in URLS.items():
        print(f"{name}: {url}")
        for i in remote_zip(url).infolist():
            print(f"   {i.filename:32s} {i.compress_size / 1e6:8.0f} MB zipped  {i.file_size / 1e6:8.0f} MB csv")


def cmd_2017(a):
    out = ROOT / "data/raw/cicids2017_fixed"
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / "CICIDS2017_improved.zip"
    if not zpath.exists():
        print(f"downloading {URLS['cicids2017']} (328 MB) ...")
        urllib.request.urlretrieve(URLS["cicids2017"], zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    if not a.keep_zip:
        zpath.unlink()
    print("done:", sorted(p.name for p in out.glob("*.csv")))


def cmd_2018(a):
    out = ROOT / "data/raw/cicids2018_fixed"
    out.mkdir(parents=True, exist_ok=True)
    z = remote_zip(URLS["cicids2018"])
    rng = np.random.default_rng(a.seed)
    for day in a.days:
        target = out / f"{day.lower()}.parquet"
        if target.exists():
            print("exists:", target.name)
            continue
        member = f"{day}.csv"
        print(f"streaming {member} ({z.getinfo(member).compress_size / 1e6:.0f} MB) ...")
        parts, seen = [], 0
        with z.open(member) as f:
            for chunk in pd.read_csv(f, chunksize=200_000, low_memory=False):
                seen += len(chunk)
                is_benign = chunk["Label"].astype(str).str.upper().eq("BENIGN") | \
                    chunk["Label"].astype(str).str.contains("Attempted", case=False)
                keep = ~is_benign | (rng.random(len(chunk)) < a.benign_frac)
                parts.append(_finish(chunk.loc[keep].copy(), CIC_DROP, "Label", True, day=day.lower()))
                print(f"\r  {seen:,} rows read", end="", flush=True)
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(target)
        print(f"\n  kept {len(df):,} rows -> {target.name}; families: {df['family'].value_counts().to_dict()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("cicids2017")
    p.add_argument("--keep-zip", action="store_true")
    p.set_defaults(fn=cmd_2017)
    p = sub.add_parser("cicids2018")
    p.add_argument("--days", nargs="+", default=DEFAULT_2018_DAYS)
    p.add_argument("--benign-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_2018)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
