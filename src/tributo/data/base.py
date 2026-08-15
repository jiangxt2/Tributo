"""Compatibility re-exports for shared data contracts.

New code should import these contracts from ``tributo.data.contracts``.  This
module remains only so existing S3 and write-mode imports do not need to move
at the same time as the DataConnector removal.
"""

from tributo.data.contracts.modes import WriteMode
from tributo.data.contracts.storage import S3Config

__all__ = ["S3Config", "WriteMode"]
