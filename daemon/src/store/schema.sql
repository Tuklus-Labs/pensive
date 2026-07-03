PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY, value TEXT NOT NULL
);  -- holds schema_version

CREATE TABLE IF NOT EXISTS atoms (
  id           TEXT PRIMARY KEY,          -- ULID
  text         TEXT NOT NULL,
  kind         TEXT NOT NULL,             -- atom|narrative|snapshot|document_chunk
  project      TEXT,
  created_at   INTEGER NOT NULL,          -- unix seconds, when recorded
  occurred_at  INTEGER,                   -- when the remembered thing happened
  importance   REAL NOT NULL DEFAULT 0.0,
  status       TEXT NOT NULL DEFAULT 'live', -- live|superseded|tombstone
  schema_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_atoms_project ON atoms(project);
CREATE INDEX IF NOT EXISTS idx_atoms_created ON atoms(created_at);
CREATE INDEX IF NOT EXISTS idx_atoms_status  ON atoms(status);

CREATE TABLE IF NOT EXISTS provenance (
  id          TEXT PRIMARY KEY,           -- ULID
  atom_id     TEXT NOT NULL REFERENCES atoms(id),
  source      TEXT NOT NULL,              -- claude-code|codex|explicit-emit|bulk-import|distiller; person-* reserved for person imports and supersession exclusion
  session_id  TEXT,
  agent       TEXT,
  source_ref  TEXT,                       -- transcript span/file/message id
  recorded_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prov_atom ON provenance(atom_id);
CREATE INDEX IF NOT EXISTS idx_prov_source_ref ON provenance(source_ref);

CREATE TABLE IF NOT EXISTS edges (
  id            TEXT PRIMARY KEY,         -- ULID
  src_atom      TEXT NOT NULL REFERENCES atoms(id),
  dst_atom      TEXT NOT NULL REFERENCES atoms(id),
  type          TEXT NOT NULL,            -- supersedes|contradicts|causes|relates|same_thread
  weight        REAL NOT NULL DEFAULT 1.0,
  created_at    INTEGER NOT NULL,
  provenance_id TEXT REFERENCES provenance(id)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_atom, type);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_atom, type);

CREATE TABLE IF NOT EXISTS facets (
  atom_id TEXT NOT NULL REFERENCES atoms(id),
  key     TEXT NOT NULL,                  -- project|entity|tag|era
  value   TEXT NOT NULL,
  PRIMARY KEY (atom_id, key, value)
);
CREATE INDEX IF NOT EXISTS idx_facets_kv ON facets(key, value);

CREATE TABLE IF NOT EXISTS embeddings (
  atom_id    TEXT NOT NULL REFERENCES atoms(id),
  model_id   TEXT NOT NULL,
  vector     BLOB NOT NULL,               -- float32 little-endian
  embedded_at INTEGER NOT NULL,
  PRIMARY KEY (atom_id, model_id)
);

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

CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
  text, content='atoms', content_rowid='rowid'
);
-- triggers keep fts in sync
CREATE TRIGGER IF NOT EXISTS atoms_ai AFTER INSERT ON atoms BEGIN
  INSERT INTO fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS atoms_ad AFTER DELETE ON atoms BEGIN
  INSERT INTO fts(fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS atoms_au AFTER UPDATE ON atoms BEGIN
  INSERT INTO fts(fts, rowid, text) VALUES ('delete', old.rowid, old.text);
  INSERT INTO fts(rowid, text) VALUES (new.rowid, new.text);
END;
