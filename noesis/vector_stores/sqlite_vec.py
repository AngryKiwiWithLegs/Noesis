"""
noesis/vector_stores/sqlite_vec.py

Local vector store backed by sqlite-vec.
No server, no network — everything in a single file on disk,
loaded into RAM for <10ms retrieval.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import sqlite_vec
    _HAS_VEC = True
except ImportError:
    _HAS_VEC = False
    logger.warning(
        "sqlite-vec not installed. Vector search disabled; "
        "falling back to recency-ordered retrieval. "
        "Run: pip install sqlite-vec"
    )


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class SqliteVecStore:
    """
    Dual-table design:
      items      — all metadata (hash_id, text, type, status, …)
      vec_items  — sqlite-vec virtual table, rowid = items.id
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS items (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            hash_id         TEXT    UNIQUE NOT NULL,
            text            TEXT    NOT NULL DEFAULT '',
            type            TEXT    NOT NULL DEFAULT 'position',
            status          TEXT    NOT NULL DEFAULT 'tentative',
            confidence      REAL    NOT NULL DEFAULT 0.0,
            user_id         TEXT    NOT NULL DEFAULT '',
            source_tool     TEXT    NOT NULL DEFAULT '',
            source_session  TEXT    NOT NULL DEFAULT '',
            topic_cluster   TEXT    NOT NULL DEFAULT '',
            created_at      REAL    NOT NULL,
            fact_ref        TEXT,
            evolved_from    TEXT,
            superseded_by   TEXT,
            extra           TEXT    NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_user_status
            ON items (user_id, status);
        CREATE INDEX IF NOT EXISTS idx_type
            ON items (type);
        CREATE INDEX IF NOT EXISTS idx_created
            ON items (created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_cluster
            ON items (topic_cluster);
    """

    def __init__(self, db_path: str, dim: int = 384):
        self.dim = dim
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        self._con = sqlite3.connect(str(path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")

        if _HAS_VEC:
            self._con.enable_load_extension(True)
            sqlite_vec.load(self._con)
            self._con.enable_load_extension(False)

        self._con.executescript(self.SCHEMA)

        if _HAS_VEC:
            self._con.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items "
                f"USING vec0(embedding float[{dim}])"
            )

        self._con.commit()

    # ── Write ────────────────────────────────────────────────────────────────

    def insert(self, hash_id: str, vector: list[float], payload: dict) -> int:
        cur = self._con.execute(
            """
            INSERT OR IGNORE INTO items
                (hash_id, text, type, status, confidence, user_id,
                 source_tool, source_session, topic_cluster,
                 created_at, fact_ref, evolved_from, extra)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                hash_id,
                payload.get("text", ""),
                payload.get("type", "position"),
                payload.get("status", "tentative"),
                float(payload.get("confidence", 0.0)),
                payload.get("user_id", ""),
                payload.get("source_tool", ""),
                payload.get("source_session", ""),
                payload.get("topic_cluster", ""),
                payload.get("created_at", time.time()),
                payload.get("fact_ref"),
                payload.get("evolved_from"),
                json.dumps(payload.get("extra", {})),
            ),
        )
        # cur.rowcount == 1 means a row was actually inserted
        # (INSERT OR IGNORE succeeds only when the row is new)
        inserted = cur.rowcount > 0
        rowid = cur.lastrowid if inserted else 0

        if _HAS_VEC and inserted:
            self._con.execute(
                "INSERT OR IGNORE INTO vec_items(rowid, embedding) VALUES (?,?)",
                [rowid, _pack(vector)],
            )

        self._con.commit()
        return rowid or 0

    def update(self, hash_id: str, payload: dict):
        allowed = {
            "text", "type", "status", "confidence",
            "topic_cluster", "fact_ref", "evolved_from",
            "superseded_by", "source_tool",
        }
        cols = [k for k in payload if k in allowed]
        if not cols:
            return
        sql = f"UPDATE items SET {', '.join(f'{c}=?' for c in cols)} WHERE hash_id=?"
        self._con.execute(sql, [payload[c] for c in cols] + [hash_id])
        self._con.commit()

    def soft_delete(self, hash_id: str):
        self.update(hash_id, {"status": "superseded"})

    def delete_all(self, user_id: str):
        self._con.execute("DELETE FROM items WHERE user_id=?", [user_id])
        self._con.commit()

    # ── Read ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        filter: Optional[dict] = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        if not _HAS_VEC:
            return self._fallback_search(top_k, filter)

        where, params = self._build_where(filter, base="i")
        # sqlite-vec returns distance (lower = better); we convert to score
        fetch = top_k * 4

        try:
            rows = self._con.execute(
                f"""
                SELECT i.*, v.distance
                FROM vec_items v
                JOIN items i ON i.id = v.rowid
                WHERE v.embedding MATCH ?
                  AND k = {fetch}
                  AND {where}
                ORDER BY v.distance
                LIMIT ?
                """,
                [_pack(query_vec)] + params + [top_k],
            ).fetchall()
        except Exception as e:
            logger.warning(f"Vector search failed ({e}), using fallback")
            return self._fallback_search(top_k, filter)

        results = []
        for r in rows:
            d = dict(r)
            dist  = float(d.pop("distance", 1.0))
            score = max(0.0, 1.0 - dist)
            if score < min_score:
                continue
            d["id"]    = d["hash_id"]
            d["score"] = score
            results.append(d)

        return results

    def get_recent(
        self,
        user_id: str,
        n: int = 3,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        where = ["user_id=?", "status NOT IN ('superseded','tentative')"]
        params: list[Any] = [user_id]

        if filter and "status" in filter:
            sv = filter["status"]
            if isinstance(sv, dict) and "$in" in sv:
                ph = ",".join("?" * len(sv["$in"]))
                where.append(f"status IN ({ph})")
                params.extend(sv["$in"])

        rows = self._con.execute(
            f"SELECT * FROM items WHERE {' AND '.join(where)} "
            f"ORDER BY created_at DESC LIMIT ?",
            params + [n],
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def get_by_type(
        self, user_id: str, types: list[str], top_k: int = 3
    ) -> list[dict]:
        ph = ",".join("?" * len(types))
        rows = self._con.execute(
            f"SELECT * FROM items "
            f"WHERE user_id=? AND type IN ({ph}) "
            f"  AND status IN ('provisional','settled') "
            f"ORDER BY confidence DESC, created_at DESC LIMIT ?",
            [user_id] + types + [top_k],
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def get_many(self, hash_ids: list[str]) -> list[dict]:
        if not hash_ids:
            return []
        ph = ",".join("?" * len(hash_ids))
        rows = self._con.execute(
            f"SELECT * FROM items WHERE hash_id IN ({ph})", hash_ids
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def get(self, hash_id: str) -> Optional[dict]:
        row = self._con.execute(
            "SELECT * FROM items WHERE hash_id=?", [hash_id]
        ).fetchone()
        return self._to_dict(row) if row else None

    def get_all(self, user_id: str) -> list[dict]:
        rows = self._con.execute(
            "SELECT * FROM items WHERE user_id=? AND status!='superseded'",
            [user_id],
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def get_vector(self, hash_id: str) -> Optional[list[float]]:
        if not _HAS_VEC:
            return None
        row = self._con.execute(
            "SELECT id FROM items WHERE hash_id=?", [hash_id]
        ).fetchone()
        if not row:
            return None
        vrow = self._con.execute(
            "SELECT embedding FROM vec_items WHERE rowid=?", [row["id"]]
        ).fetchone()
        return _unpack(vrow[0]) if vrow else None

    def exists(self, hash_id: str) -> bool:
        return bool(
            self._con.execute(
                "SELECT 1 FROM items WHERE hash_id=?", [hash_id]
            ).fetchone()
        )

    def count(self, user_id: str, filter: Optional[dict] = None) -> int:
        where = ["user_id=?", "status!='superseded'"]
        params: list[Any] = [user_id]
        if filter and "status" in filter:
            where.append("status=?")
            params.append(filter["status"])
        row = self._con.execute(
            f"SELECT COUNT(*) FROM items WHERE {' AND '.join(where)}", params
        ).fetchone()
        return int(row[0]) if row else 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_where(
        self, filter: Optional[dict], base: str = ""
    ) -> tuple[str, list]:
        prefix = f"{base}." if base else ""
        clauses = [f"{prefix}status != 'superseded'"]
        params: list[Any] = []

        if not filter:
            return clauses[0], params

        for key, val in filter.items():
            col = f"{prefix}{key}"
            if isinstance(val, dict) and "$in" in val:
                ph = ",".join("?" * len(val["$in"]))
                clauses.append(f"{col} IN ({ph})")
                params.extend(val["$in"])
            else:
                clauses.append(f"{col}=?")
                params.append(val)

        return " AND ".join(clauses), params

    def _fallback_search(
        self, top_k: int, filter: Optional[dict]
    ) -> list[dict]:
        where, params = self._build_where(filter)
        rows = self._con.execute(
            f"SELECT * FROM items WHERE {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            params + [top_k],
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def _to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["id"]    = d.get("hash_id", "")
        d["score"] = d.pop("score", 1.0) if "score" in d else 1.0
        d.pop("distance", None)
        return d

    def close(self):
        self._con.close()
