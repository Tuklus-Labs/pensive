"""Forward-only schema migrations for the canonical store.

This module owns the schema-version ladder. `CURRENT_SCHEMA_VERSION` is the
highest version this build knows how to produce; `store.py` imports it from
here so there is a single source of truth. Migrations never run backwards: a
database recorded at a version newer than this build is refused rather than
downgraded, because canonical data must never become unrecoverable.
"""

CURRENT_SCHEMA_VERSION = 2


def migrate(store):
    """Advance ``store`` from its recorded schema version up to current.

    A fresh database (no recorded version) is treated as version 0. The v0->v1
    step is a no-op beyond stamping the version, because ``schema.sql`` already
    installed the v1 tables. Later versions add their upgrade step here.
    """
    version = store._readVersion()
    if version is None:
        version = 0

    if version > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema_version {version} is newer than this build "
            f"supports (max {CURRENT_SCHEMA_VERSION}); refusing to open"
        )

    dirty = False
    if version < 1:
        # v0 -> v1: baseline. Tables already exist; record that we are at v1.
        store._setVersion(1)
        version = 1
        dirty = True

    if version < 2:
        _upgrade_1_to_2(store)
        store._setVersion(2)
        version = 2
        dirty = True

    if dirty:  # a no-op re-open of an up-to-date db writes nothing
        store._commit()


def _upgrade_1_to_2(store):
    """Add lifecycle job tables and the provenance source_ref hot-path index."""
    store._conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_prov_source_ref ON provenance(source_ref);

        CREATE TABLE IF NOT EXISTS recall_log (
          id           TEXT PRIMARY KEY,
          atom_id      TEXT NOT NULL REFERENCES atoms(id),
          query        TEXT,
          source_ref   TEXT,
          weight       REAL NOT NULL DEFAULT 1.0,
          recorded_at  INTEGER NOT NULL,
          processed_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_recall_log_unprocessed
          ON recall_log(processed_at, atom_id);

        CREATE TABLE IF NOT EXISTS supersession_proposals (
          id          TEXT PRIMARY KEY,
          old_atom_id TEXT NOT NULL REFERENCES atoms(id),
          new_atom_id TEXT NOT NULL REFERENCES atoms(id),
          similarity  REAL NOT NULL,
          reason      TEXT NOT NULL,
          status      TEXT NOT NULL DEFAULT 'proposed',
          created_at  INTEGER NOT NULL,
          UNIQUE(old_atom_id, new_atom_id)
        );
        CREATE INDEX IF NOT EXISTS idx_supersession_proposals_status
          ON supersession_proposals(status);
        """
    )
