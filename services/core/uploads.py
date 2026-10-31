"""Large files cross the 30-second local proxy as checksummed parts, then assemble on disk."""

import hashlib
import json
import re
import shutil
import time
import uuid

from services.core import store

MAX_BYTES = 256 * 1024 * 1024
MAX_PART = 4 * 1024 * 1024
STALE_SECONDS = 24 * 3600
UPLOAD_ID = re.compile(r"^up-[0-9a-f]{12}$")


def folder(upload_id=None):
    root = store.DATA / "uploads"
    if upload_id is None:
        return root
    if not UPLOAD_ID.match(upload_id):
        raise KeyError("Unknown upload")
    return root / upload_id


def clean():
    for path in folder().glob("up-*") if folder().exists() else []:
        if time.time() - path.stat().st_mtime > STALE_SECONDS:
            shutil.rmtree(path, ignore_errors=True)


def start(name, size, sha256):
    if size > MAX_BYTES:
        raise ValueError(f"Uploads are limited to {MAX_BYTES // 2**20} MB")
    clean()
    upload_id = f"up-{uuid.uuid4().hex[:12]}"
    target = folder(upload_id)
    target.mkdir(parents=True)
    (target / "meta.json").write_text(json.dumps({"name": name[:200], "size": size, "sha256": sha256}))
    return {"id": upload_id, "part_bytes": MAX_PART}


def part(upload_id, index, text, sha256):
    target = folder(upload_id)
    if not (target / "meta.json").exists():
        raise KeyError("Unknown upload")
    data = text.encode()
    if len(data) > MAX_PART:
        raise ValueError("Part too large")
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Part checksum mismatch")
    (target / f"part-{index:05d}").write_bytes(data)
    return {"index": index, "bytes": len(data)}


def complete(upload_id, parts):
    target = folder(upload_id)
    meta = json.loads((target / "meta.json").read_text())
    digest, total = hashlib.sha256(), 0
    with (target / "data").open("wb") as out:
        for index in range(parts):
            path = target / f"part-{index:05d}"
            if not path.exists():
                raise ValueError(f"Missing part {index}")
            data = path.read_bytes()
            digest.update(data)
            total += len(data)
            out.write(data)
            path.unlink()
    if digest.hexdigest() != meta["sha256"] or total != meta["size"]:
        (target / "data").unlink()
        raise ValueError("Assembled upload does not match its declared size and checksum")
    return {"id": upload_id, "bytes": total, "sha256": meta["sha256"]}


def text(upload_id):
    path = folder(upload_id) / "data"
    if not path.exists():
        raise KeyError("Upload is not complete")
    return path.read_text()
