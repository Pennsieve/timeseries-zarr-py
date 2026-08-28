"""De-identification: the two operations the format makes trivial.

Identity can exist in exactly two places in a conformant bundle: the
``meta/`` group (removable JSON) and event bodies (greppable NDJSON
text). ``deidentify`` deletes ``meta/`` and reports the remaining
body surfaces for human review; ``dateshift`` edits the bundle's one
wall-clock value in place.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import zarr


@dataclass(frozen=True)
class BodySurface:
    """One human-readable annotation surface left after de-id."""

    channel: str
    name: str
    n_bodies: int
    media_type: str
    path: str


def deidentify(path: str | Path) -> list[BodySurface]:
    """Delete ``meta/``; return the body surfaces still to review.

    Per the spec this is the whole structural step: nothing outside
    ``meta/`` may carry identity or wall-clock time, and the bundle
    degrades gracefully to a pure onset-relative timeline. The
    returned surfaces are the newline-delimited text files a reviewer
    should still grep for free-text identifiers.
    """
    path = Path(path)
    meta = path / "meta"
    if meta.exists():
        shutil.rmtree(meta)
        _prune_consolidated(path)
    return body_surfaces(path)


def body_surfaces(path: str | Path) -> list[BodySurface]:
    """List every event channel with stored bodies."""
    path = Path(path)
    root = zarr.open_group(str(path), mode="r")
    out = []
    for name in sorted(root.group_keys()):
        if name == "meta":
            continue
        grp = root[name]
        if grp.attrs.get("kind") != "event":
            continue
        if "bodies" not in grp.array_keys():
            continue
        offs = grp["body_offsets"]
        out.append(
            BodySurface(
                channel=name,
                name=str(grp.attrs.get("name", "")),
                n_bodies=int(offs.shape[0]) - 1,
                media_type=str(
                    grp.attrs.get("body_media_type", "text/plain")
                ),
                path=str(path / name / "bodies" / "c" / "0"),
            )
        )
    return out


def dateshift(path: str | Path, shift_us: int) -> int:
    """Shift ``meta/session.start_us``; return the new value.

    This is the entire date-shift: every other timestamp in the
    bundle is onset-relative.
    """
    path = Path(path)
    zj = path / "meta" / "zarr.json"
    if not zj.exists():
        raise FileNotFoundError(
            "bundle has no meta/ (already de-identified?)"
        )
    doc = json.loads(zj.read_text())
    session = doc.setdefault("attributes", {}).setdefault("session", {})
    if "start_us" not in session:
        raise KeyError("meta/session.start_us is not set")
    session["start_us"] = int(session["start_us"]) + int(shift_us)
    zj.write_text(json.dumps(doc, indent=1))
    return int(session["start_us"])


def _prune_consolidated(path: Path) -> None:
    """Drop stale meta/ entries from consolidated metadata, if any."""
    zj = path / "zarr.json"
    doc = json.loads(zj.read_text())
    cm = doc.get("consolidated_metadata", {}).get("metadata")
    if not cm:
        return
    stale = [k for k in cm if k == "meta" or k.startswith("meta/")]
    if stale:
        for k in stale:
            del cm[k]
        zj.write_text(json.dumps(doc, indent=1))
