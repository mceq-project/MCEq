"""Compatibility exports of database download and integrity-check helpers
from :mod:`MCEq.data.download`.

``MCEq.download`` re-exports the five names the surface test pins and nothing
else.
"""

from MCEq.data.download import (  # noqa: F401
    FileIntegrityCheck,
    base_url,
    ensure_db_available,
    file_checksum,
    release_tag,
)
