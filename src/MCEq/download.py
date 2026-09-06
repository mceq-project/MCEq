"""Compat shim: the download code moved to :mod:`MCEq.data.download`.

Phase 4 (plan section 1, §13.2 item 5) moved the fetch/verify code under the
``data`` package. ``MCEq.download`` stays as the public entry point user code
and ``MCEq.core`` already reach through; it re-exports the five names the
surface test pins and nothing else. Phase 7 deletes this shim (decision D6).
"""

from MCEq.data.download import (  # noqa: F401
    FileIntegrityCheck,
    base_url,
    ensure_db_available,
    file_checksum,
    release_tag,
)
