#!/usr/bin/env python3
"""
Noesis 记忆星图查看器 — 服务端
读取 ~/.noesis/hot.db, 通过 HTTP 提供记忆数据 (JSON) 和可视化页面 (HTML)。

用法:
    python3 serve.py [--port 8787] [--db ~/.noesis/hot.db]
然后浏览器打开 http://localhost:8787
"""
import argparse
import json
import os
import sqlite3
import textwrap
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))

def read_memories(db_path: str) -> list[dict]:
    """读取所有记忆节点 (跨所有用户)。"""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, hash_id, text, type, status, confidence, "
            "       user_id, source_tool, topic_cluster, created_at "
            "FROM items ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def read_stats(db_path: str) -> dict:
    """统计信息。"""
    mems = read_memories(db_path)
    if not mems:
        return {"total": 0, "by_type": {}, "by_status": {}, "users": []}

    by_type, by_status = {}, {}
    for m in mems:
        by_type[m["type"]] = by_type.get(m["type"], 0) + 1
        by_status[m["status"]] = by_status.get(m["status"], 0) + 1

    return {
        "total": len(mems),
        "by_type": by_type,
        "by_status": by_status,
        "users": sorted({m["user_id"] for m in mems}),
    }


class Handler(BaseHTTPRequestHandler):
    DB_PATH = os.path.expanduser("~/.noesis/hot.db")

    def log_message(self, *args):
        pass  # 静默

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/memories":
            user = urlparse(self.path).query.replace("user=", "")
            mems = read_memories(self.DB_PATH)
            if user and user != "all":
                mems = [m for m in mems if m["user_id"] == user]
            self._json({"memories": mems})
        elif path == "/api/stats":
            self._json(read_stats(self.DB_PATH))
        elif path == "/" or path == "/index.html":
            with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
                self._html(f.read())
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser(description="Noesis 记忆星图查看器")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--db", default=os.path.expanduser("~/.noesis/hot.db"))
    args = ap.parse_args()

    Handler.DB_PATH = args.db

    stats = read_stats(args.db)
    print("=" * 56)
    print("  Noesis 记忆星图查看器")
    print("=" * 56)
    print(f"  数据库: {args.db}")
    print(f"  记忆数: {stats['total']}")
    if stats["total"]:
        print(f"  类型:   {stats['by_type']}")
        print(f"  状态:   {stats['by_status']}")
    print()
    print(f"  → 浏览器打开: http://localhost:{args.port}")
    print()
    print("  Ctrl+C 退出")
    print("-" * 56)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
        server.server_close()


if __name__ == "__main__":
    main()
