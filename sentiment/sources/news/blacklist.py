import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ATTEMPTS_FIELDS = ["source_name", "success"]

WINDOW = 10
THRESHOLD = 5


class SourceBlacklist:
    """Tracks per-source fetch failures and maintains a persistent blacklist.

    Two CSVs are kept under *data_dir*:

    - ``blacklist.csv`` — blacklisted source names.
    - ``source_attempts.csv`` — rolling window of the last :data:`WINDOW`
      fetch attempts per source (``success`` is ``1`` or ``0``).

    A source is blacklisted automatically when it accumulates at least
    :data:`THRESHOLD` failures in its last :data:`WINDOW` attempts.

    Layout::

        <data_dir>/
            blacklist.csv
            source_attempts.csv
    """

    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            data_dir = Path(__file__).parents[3] / "data" / "news"
        self._data_dir = data_dir
        self._blacklist_path = data_dir / "blacklist.csv"
        self._attempts_path = data_dir / "source_attempts.csv"
        self._dirty = False
        self._blacklist = self._load_blacklist()
        self._attempts = self._load_attempts()

    def is_blacklisted(self, source_name: str) -> bool:
        """Return True if *source_name* is on the blacklist."""
        return source_name in self._blacklist

    def record_attempt(self, source_name: str, success: bool) -> bool:
        """Record a fetch attempt for *source_name*.

        Trims the per-source history to the last :data:`WINDOW` entries and
        auto-blacklists the source if it crosses the failure threshold.

        Returns:
            True if the source was newly blacklisted as a result of this call.
        """
        row = pd.DataFrame([{"source_name": source_name, "success": int(success)}])
        self._attempts = pd.concat([self._attempts, row], ignore_index=True)

        # Trim to last WINDOW rows for this source
        mask = self._attempts["source_name"] == source_name
        source_idx = self._attempts.index[mask].tolist()
        if len(source_idx) > WINDOW:
            drop_idx = source_idx[: len(source_idx) - WINDOW]
            self._attempts = self._attempts.drop(index=drop_idx).reset_index(drop=True)

        self._dirty = True

        if source_name in self._blacklist:
            return False

        recent = self._attempts[self._attempts["source_name"] == source_name]
        failures = (recent["success"] == 0).sum()
        if failures >= THRESHOLD:
            self._blacklist.add(source_name)
            logger.info(
                "blacklisted source: %s (%d/%d failures)",
                source_name,
                failures,
                len(recent),
            )
            return True

        return False

    def flush(self) -> None:
        """Persist both CSVs to disk if modified."""
        if not self._dirty:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"source_name": sorted(self._blacklist)}).to_csv(
            self._blacklist_path, index=False
        )
        self._attempts.to_csv(self._attempts_path, index=False)
        self._dirty = False

    def _load_blacklist(self) -> set[str]:
        if not self._blacklist_path.exists():
            return set()
        df = pd.read_csv(self._blacklist_path, dtype=str)
        return set(df["source_name"].dropna().tolist())

    def _load_attempts(self) -> pd.DataFrame:
        if not self._attempts_path.exists():
            return pd.DataFrame(columns=_ATTEMPTS_FIELDS)
        return pd.read_csv(
            self._attempts_path, dtype={"source_name": str, "success": int}
        )
