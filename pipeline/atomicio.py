"""Publish a file only once it is complete.

Every place in this repo that streamed a download (or dumped JSON) straight
into its final path had the same failure: a dropped connection, a full disk or
a killed process left a TRUNCATED file sitting where a complete one belongs.
That is worse than no file at all, because the readers downstream cannot tell
the difference:

  * `studio/assets.py` guards with `if not os.path.exists(dest)`, so a partial
    stock clip was re-used as the background on every later render, with its
    partial size recorded as the real one (QM-023);
  * `pipeline/kidsong/comfy.py` writes the RENDERED TAKE — a truncated mp4 there
    is handed straight to the contact sheet, the reviewer and the editor;
  * `studio/setup.py` rewrites `config.json` in place, so a crash mid-write
    truncates the app's entire configuration;
  * `pipeline/kidsong/runstate.py` and `director.py` write the shot list a
    resume reads back.

`pipeline/gpu_lock.py` already got this right the hard way (see its module
docstring); this is the same discipline factored out so the rest of the repo
does not have to rediscover it. Write to a sibling `.part`, fsync, then
`os.replace` — which is atomic on both POSIX and Windows for a same-directory
rename, so a reader sees either the old file or the new one, never a half one.
"""
import contextlib
import json
import os


def _quiet_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


@contextlib.contextmanager
def atomic_path(dest, suffix=".part"):
    """Yield a temp path to write to; move it onto `dest` only on success.

    The temp file is a sibling of `dest` so the rename stays within one
    filesystem (`os.replace` across devices raises). Anything raised inside the
    block removes the temp file and propagates — including `KeyboardInterrupt`
    and `SystemExit`, which is the interruption that produced the stump in the
    first place and which a plain `except Exception` would miss.

        with atomic_path(out) as tmp:
            with open(tmp, "wb") as f:
                f.write(data)
    """
    dest = os.path.abspath(dest)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + suffix
    _quiet_remove(tmp)  # a stale part from an earlier crash must not be reused
    try:
        yield tmp
    except BaseException:
        _quiet_remove(tmp)
        raise
    if not os.path.exists(tmp):
        _quiet_remove(tmp)
        raise FileNotFoundError(
            f"atomic_path({dest!r}): nothing was written to {tmp!r}"
        )
    os.replace(tmp, dest)


def atomic_write_bytes(dest, chunks):
    """Stream an iterable of byte chunks to `dest`, atomically."""
    with atomic_path(dest) as tmp:
        with open(tmp, "wb") as f:
            for chunk in chunks:
                if chunk:
                    f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
    return dest


def atomic_write_json(dest, payload, **dump_kwargs):
    """Serialise `payload` to `dest`, atomically.

    Serialising INSIDE the temp file is deliberate: a payload that fails to
    encode (a set, a numpy scalar) raises before anything is renamed, so the
    previous file survives instead of being replaced by a half-written one.
    """
    with atomic_path(dest) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
    return dest
