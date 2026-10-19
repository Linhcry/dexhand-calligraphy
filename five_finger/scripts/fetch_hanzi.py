"""Download requested stroke data from its original public repository, with attribution."""
import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "assets" / "hanzi"
BASE = "https://raw.githubusercontent.com/chanind/hanzi-writer-data/master/"


def fetch(item):
    remote, filename = item
    url = BASE + urllib.parse.quote(remote)
    target = OUT / filename
    error = None
    for attempt in range(3):
        try:
            if target.exists():
                content = target.read_bytes()
            else:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "five-finger-research"}), timeout=15) as response:
                    content = response.read()
                if filename.endswith(".json") and not json.loads(content)["medians"]:
                    raise ValueError("Missing stroke medians")
                target.write_bytes(content)
            return {"file": filename, "url": url, "sha256": hashlib.sha256(content).hexdigest()}
        except Exception as exc:
            error = str(exc)
            time.sleep(attempt + 1)
    return {"file": filename, "url": url, "error": error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="永静夜思床前明月光疑是地上霜举头望低故乡")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = [(f"data/{c}.json", f"{c}.json") for c in sorted(set(args.text))]
    items += [("ARPHICPL.TXT", "ARPHICPL.TXT"), ("README.md", "UPSTREAM_README.md")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, items))
    manifest_path = OUT / "sources.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", []) if manifest_path.exists() else []
    indexed = {entry["file"]: entry for entry in previous}
    indexed.update({entry["file"]: entry for entry in results})
    manifest = {"repository": "https://github.com/chanind/hanzi-writer-data", "files": list(indexed.values())}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    errors = [r for r in results if "error" in r]
    print(json.dumps({"downloaded_or_verified": len(results)-len(errors), "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
