Data files on disk are checked against what they are supposed to be, so a
stale copy is no longer used silently.

- A release package is checked against its manifest entry whenever a run
  resolves it (size, and the digest of the database it was cut from); a
  mismatch raises ``StaleDataError`` naming the refresh command.
  ``config.verify_data_packages = False`` accepts a local build under a release
  name. ``python -m MCEq.data.download --verify`` compares full sha256 digests.
- A manifest in a separate ``config.data_dir`` that is older than the one
  installed with MCEq gives ``StaleManifestWarning``.
- The CI databases are pinned by sha256 in ``tests/data/ci_assets.py``. Each
  ``pytest`` session replaces files that differ from the pins before it starts
  (``MCEQ_SKIP_CI_DATA_CHECK=1`` skips this), and both CI workflows fetch
  through it with the file's hash as cache key.
