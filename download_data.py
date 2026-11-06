"""Download the exact public exploratory CSV; refuse a changed source file."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = "https://osf.io/download/3vqmp/"
EXPECTED = "8368313b060f4015a0c6fb34e6d788163cee29554144aba9390b14922eb9d8ce"


def main():
    destination = ROOT / "data/raw/upworthy.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != EXPECTED:
            raise RuntimeError("Existing raw file has a different checksum; nothing overwritten.")
        print("Existing raw data verified:", destination)
        return
    request = urllib.request.Request(URL, headers={"User-Agent": "pie-headlines-reproducibility/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise RuntimeError("Source checksum changed. Review provenance before using these bytes.")
    destination.write_bytes(data)
    print("Downloaded and verified:", destination)


if __name__ == "__main__":
    main()
