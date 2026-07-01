#!/usr/bin/env python3
import argparse, json, os, sqlite3, time, math, random
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = {"memories": [], "loaded_path": None, "mtime": 0}

STATUS_WEIGHT = {"settled": 1.0, "provisional": 0.7, "tentative": 0.4, "superseded": 0.2}
TYPE_COLORS = {
    "identity": "#22d3ee", "preference": "#f472b6", "position": "#a78bfa",
    "event": "#fbbf24", "fact": "#34d399", "question": "#fb923c", "default": "#94a3b8"
}

def generate_demo_data():
    now = time.time()
    demos = [
        "我偏好用 Python 做数据分析，觉得 pandas 比 Excel 强得多",
        "我是 Linux 重度用户，日常用 macOS 开发",
        "我在团队里负责基础设施和数据库架构",
        "我是计算机科学硕士毕业，擅长分布式系统",
        "我的技术栈主要是 Python 和 Go，偶尔写点 Rust",
        "我目前住在杭州，在一家电商公司工作",
        "我叫张伟，是一名有 8 年经验的后端工程师",
        "最近在学 Kubernetes，感觉很有意思",
        "周末喜欢爬山和摄影",
        "对大模型应用开发很感兴趣"
    ]
    types = ["preference", "identity", "position", "identity", "position", "position", "identity", "event", "preference", "preference"]
    clusters = ["tech-stack", "general", "work", "education", "tech-stack", "general", "identity", "learning", "lifestyle", "ai"]
    mems = []
    for i, text in enumerate(demos):
        mems.append({
            "id": i, "hash_id": f"demo_{i}", "text": text,
            "type": types[i], "status": random.choice(["settled", "provisional", "tentative"]),
            "confidence": round(random.uniform(0.3, 0.9), 3),
            "user_id": "demo", "source_tool": "demo-gen",
            "topic_cluster": clusters[i], "created_at": now - i * 86400 * random.uniform(1, 30)
        })
    return mems

def load_memories(path):
    if path == "__demo__":
        return generate_demo_data()
    if not os.path.exists(path):
        return []
    if path.endswith('.jsonl'):
        mems = [json.loads(l) for l in open(path) if l.strip()]
    elif path.endswith('.json'):
        mems = json.load(open(path))
    else:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, hash_id, text, type, status, confidence, "
                "user_id, source_tool, topic_cluster, created_at FROM items ORDER BY created_at DESC"
            ).fetchall()
            mems = [dict(r) for r in rows]
        finally:
            conn.close()
    now = time.time()
    for i, m in enumerate(mems):
        m["id"] = m.get("id", m.get("hash_id", i))
        m["created_at"] = float(m.get("created_at", now - i*3600))
        m["confidence"] = float(m.get("confidence", 0.5))
        m["status"] = m.get("status", "provisional")
        m["type"] = m.get("type", "fact")
        age_days = (now - m["created_at"]) / 86400
        recency_boost = math.exp(-age_days / 60)
        m["importance"] = m["confidence"] * STATUS_WEIGHT.get(m["status"], 0.5) * (0.3 + 0.7*recency_boost)
        m["topic_cluster"] = m.get("topic_cluster") or m.get("topic") or "general"
    return mems

def get_memories(path):
    if path == "__demo__":
        if not CACHE["memories"]:
            CACHE["memories"] = load_memories(path)
        return CACHE["memories"]
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0
    if CACHE["loaded_path"]!= path or CACHE["mtime"]!= mtime:
        CACHE["memories"] = load_memories(path)
        CACHE["loaded_path"] = path
        CACHE["mtime"] = mtime
    return CACHE["memories"]

def search_memories(mems, q, limit=100):
    q = q.lower()
    scored = []
    for m in mems:
        text = (m.get("text","") + " " + m.get("topic_cluster","")).lower()
        score = SequenceMatcher(None, q, text).ratio()
        if q in text:
            score += 0.5
        if score > 0.2:
            scored.append((score, m))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [m for _, m in scored[:limit]]

def related_memories(mems, target_id, limit=8):
    target = next((m for m in mems if str(m["id"]) == str(target_id)), None)
    if not target:
        return []
    t_text = target.get("text","").lower()
    t_topic = target.get("topic_cluster","")
    scored = []
    for m in mems:
        if str(m["id"]) == str(target_id):
            continue
        score = 0
        if m.get("topic_cluster") == t_topic:
            score += 0.6
        score += SequenceMatcher(None, t_text, m.get("text","").lower()).ratio() * 0.4
        scored.append((score, m))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [m for _, m in scored[:limit]]

class Handler(BaseHTTPRequestHandler):
    DATA_PATH = os.path.expanduser("~/.noesis/hot.db")
    def log_message(self, *args): pass
    def _send(self, data, code=200, ctype="application/json"):
        if ctype == "application/json":
            body = json.dumps(data, ensure_ascii=False, default=str).encode()
        else:
            body = data.encode()
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        mems = get_memories(self.DATA_PATH)
        if path == "/api/memories":
            user = qs.get("user", ["all"])[0]
            if user!= "all":
                mems = [m for m in mems if m.get("user_id") == user]
            self._send({"memories": mems, "colors": TYPE_COLORS})
        elif path == "/api/search":
            q = qs.get("q", [""])[0]
            self._send({"results": search_memories(mems, q)})
        elif path == "/api/related":
            tid = qs.get("id", [""])[0]
            self._send({"results": related_memories(mems, tid)})
        elif path == "/api/stats":
            by_type, by_status = {}, {}
            for m in mems:
                by_type[m["type"]] = by_type.get(m["type"], 0) + 1
                by_status[m["status"]] = by_status.get(m["status"], 0) + 1
            self._send({
                "total": len(mems), "by_type": by_type, "by_status": by_status,
                "users": sorted({m.get("user_id","unknown") for m in mems}),
                "time_range": [min(m["created_at"] for m in mems), max(m["created_at"] for m in mems)] if mems else [0,0]
            })
        elif path == "/api/health":
            self._send({"status": "ok", "memories": len(mems), "path": self.DATA_PATH})
        elif path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index_v2_fixed.html"), encoding="utf-8") as f:
                self._send(f.read(), ctype="text/html")
        else:
            self.send_error(404)

def main():
    ap = argparse.ArgumentParser(description="Noesis Memory Constellations v2")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--db", default=None, help="Path to sqlite.db")
    ap.add_argument("--jsonl", default=None, help="Path to.jsonl file")
    ap.add_argument("--json", default=None, help="Path to.json file")
    ap.add_argument("--demo", action="store_true", help="Use demo data")
    args = ap.parse_args()
    if args.demo:
        Handler.DATA_PATH = "__demo__"
    else:
        Handler.DATA_PATH = args.db or args.jsonl or args.json or os.path.expanduser("~/.noesis/hot.db")
    mems = get_memories(Handler.DATA_PATH)
    print("="*60)
    print(" Noesis Memory Constellations v2 - Fixed")
    print("="*60)
    print(f" Data: {Handler.DATA_PATH}")
    print(f" Memories: {len(mems)}")
    print(f" → http://localhost:{args.port}")
    print("-"*60)
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nExited")
        server.server_close()

if __name__ == "__main__":
    main()
