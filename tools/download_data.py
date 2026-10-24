#!/usr/bin/env python3
"""Download and verify the prepared ETA datasets using the standard library."""
import argparse
import hashlib
import json
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "docs" / "data_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(root, relative):
    # Archives and manifests always use POSIX paths, on Windows as well.
    name = PurePosixPath(relative)
    if "\\" in relative or name.is_absolute() or ".." in name.parts:
        raise ValueError("Unsafe data path: {}".format(relative))
    if not name.parts or name.parts[0] != "data" or ":" in relative:
        raise ValueError("Unexpected data path: {}".format(relative))
    destination = root.joinpath(*name.parts).resolve()
    if root.resolve() not in destination.parents:
        raise ValueError("Data path escapes destination: {}".format(relative))
    return destination


def verify_files(root, manifest):
    failures = []
    for entry in manifest["files"]:
        path = checked_path(root, entry["path"])
        if not path.is_file():
            failures.append("Missing: {}".format(entry["path"]))
        elif path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            failures.append("Checksum mismatch: {}".format(entry["path"]))
    return failures


def download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "ETA-data-downloader/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0"))
        received = 0
        with destination.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                received += len(block)
                suffix = "/{:.1f} MiB".format(total / 2 ** 20) if total else " MiB"
                print("\rDownloading: {:.1f}{}".format(received / 2 ** 20, suffix),
                      end="", flush=True)
        print()


def extract_verified(archive, root, manifest, temporary):
    expected = {entry["path"] for entry in manifest["files"]}
    unpacked = temporary / "unpacked"
    unpacked.mkdir()
    with zipfile.ZipFile(archive) as source:
        actual = set()
        for info in source.infolist():
            checked_path(unpacked, info.filename)
            if info.is_dir():
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Symlinks are not allowed in the data archive")
            if info.filename in actual:
                raise ValueError("Duplicate archive member: {}".format(info.filename))
            actual.add(info.filename)
        if actual != expected:
            raise ValueError("Archive contents do not match the data manifest")
        source.extractall(unpacked)
    failures = verify_files(unpacked, manifest)
    if failures:
        raise ValueError("\n".join(failures))
    # All entries are checked before any destination file is changed.
    for entry in manifest["files"]:
        destination = checked_path(root, entry["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checked_path(unpacked, entry["path"]), destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="use an already downloaded ZIP")
    parser.add_argument("--check", action="store_true", help="verify existing data only")
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="repository root receiving the data/ directory")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    root = args.root.resolve()
    failures = verify_files(root, manifest)
    if not failures:
        print("Verified all {} prepared data files.".format(len(manifest["files"])))
        return 0
    if args.check:
        print("\n".join(failures), file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="eta-data-") as temporary_name:
        temporary = Path(temporary_name)
        archive = args.archive.resolve() if args.archive else temporary / "eta-data-v1.zip"
        if not args.archive:
            print("Source: {}".format(manifest["url"]))
            download(manifest["url"], archive)
        if archive.stat().st_size != manifest["archive_size"]:
            raise ValueError("Archive size mismatch; download the complete ETA data archive")
        if sha256(archive) != manifest["archive_sha256"]:
            raise ValueError("Archive SHA-256 mismatch")
        extract_verified(archive, root, manifest, temporary)
    failures = verify_files(root, manifest)
    if failures:
        raise ValueError("\n".join(failures))
    print("Verified and installed {} data files under {}.".format(
        len(manifest["files"]), root / "data"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError) as error:
        print("Data preparation failed: {}".format(error), file=sys.stderr)
        sys.exit(1)
