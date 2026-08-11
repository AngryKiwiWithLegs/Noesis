#!/usr/bin/env python3
"""
Noesis Star Map v2 — Four-Force Physics Visualization
======================================================
Implements the design doc §7 (Module 1): a 3D force-directed layout where
constellations *emerge* from semantic similarity, not manual topic banding.

Physics model (per-frame, four forces):
  A. Semantic attraction — pulls similar thoughts together on XY
  B. Global repulsion    — prevents overlap (spatial-hash optimized)
  C. Time spring (Z)     — newer nodes float higher (stratigraphic)
  D. Semantic Z adhesion — keeps same-topic nodes at similar heights

Server-side: decodes embedding vectors from the sqlite-vec shadow table
(no extension needed) and precomputes each node's top-K semantic neighbors.

Usage:
    python3 serve_galaxy_v2.py [--port 8791] [--db ~/.noesis/hot.db]
"""
import argparse
import json
import math
import os
import re
import sqlite3
import struct
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DIM = 384  # all-MiniLM-L6-v2 dimensionality

# ── Topic inference (fallback for legacy nodes without stored cluster) ────────

_TOPIC_RULES = [
    ("vector-db",     [r"sqlite-vec", r"faiss", r"milvus", r"qdrant",
                       r"向量", r"vector", r"embedding", r"嵌入"]),
    ("database",      [r"postgres", r"mysql", r"redis", r"mongo", r"rdb",
                       r"aof", r"数据库", r"关系型"]),
    ("languages",     [r"python", r"\brust\b", r"\bgo\b", r"golang", r"java\b",
                       r"swift", r"kotlin", r"编程", r"语言"]),
    ("frontend",      [r"react", r"vue", r"svelte", r"tailwind", r"css",
                       r"typescript", r"javascript", r"前端", r"\bui\b"]),
    ("api-design",    [r"graphql", r"\brest\b", r"grpc", r"\bapi\b", r"endpoint"]),
    ("cloud-infra",   [r"kubernetes", r"docker", r"serverless", r"lambda",
                       r"aws", r"microservice", r"monorepo", r"ci/cd", r"argocd",
                       r"架构", r"分布式"]),
    ("messaging",     [r"kafka", r"rabbitmq", r"队列", r"queue"]),
    ("tools",         [r"\bgit\b", r"rebase", r"vim", r"neovim", r"linux", r"macos",
                       r"hugo", r"wordpress", r"gitbook", r"obsidian"]),
    ("data-science",  [r"pandas", r"excel", r"scikit", r"machine", r"数据科学",
                       r"scientist"]),
    ("engineering",   [r"code review", r"unit test", r"\bci\b", r"optimi",
                       r"refactor", r"测试", r"优化", r"bug", r"code review"]),
    ("identity",      [r"我叫", r"我是", r"engineer", r"scientist", r"developer",
                       r"硕士", r"经验", r"住在", r"i am", r"my name"]),
]

_TOPIC_COLORS = {
    "vector-store": "#22d3ee", "vector-db": "#22d3ee",
    "database": "#34d399", "languages": "#a78bfa", "frontend": "#f472b6",
    "api-design": "#fbbf24", "cloud-infra": "#3b82f6", "messaging": "#fb923c",
    "devtools": "#c4b5fd", "tools": "#c4b5fd", "data-science": "#2dd4bf",
    "engineering": "#818cf8", "llm-choice": "#facc15", "career": "#ff8a8a",
    "identity": "#ff6b6b", "general": "#9aa0b4",
}

_PALETTE = [
    "#22d3ee", "#34d399", "#a78bfa", "#f472b6", "#fbbf24", "#3b82f6",
    "#fb923c", "#c4b5fd", "#2dd4bf", "#818cf8", "#facc15", "#f87171",
    "#4ade80", "#60a5fa", "#e879f9", "#fde047", "#67e8f9", "#a3e635",
    "#fca5a5", "#bef264", "#7dd3fc", "#ddd6fe", "#fdba74", "#bbf7d0",
]


def infer_topic(text: str) -> str:
    t = (text or "").lower()
    for topic, patterns in _TOPIC_RULES:
        for pat in patterns:
            if re.search(pat, t):
                return topic
    return "general"


def _topic_color(topic: str) -> str:
    if topic in _TOPIC_COLORS:
        return _TOPIC_COLORS[topic]
    return _PALETTE[abs(hash(topic)) % len(_PALETTE)]


# ── Vector decoding from sqlite-vec shadow table (no extension needed) ────────

def _load_vectors(db_path: str) -> dict[int, list[float]]:
    """Decode embedding vectors from the shadow table without loading sqlite-vec.

    sqlite-vec stores vectors in chunked shadow tables:
      vec_items_chunks         — chunk_id, validity bitmap, rowids blob
      vec_items_vector_chunks00 — rowid=chunk_id, vectors=contiguous blob

    Each vector is DIM float32 = DIM*4 bytes, little-endian.
    Returns dict: {item_rowid: [float, ...]}.
    """
    try:
        db_path = os.path.expanduser(db_path)
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row

        # Check if shadow tables exist
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "vec_items_chunks" not in tables:
            con.close()
            return {}

        vectors_by_rowid: dict[int, list[float]] = {}

        for chunk in con.execute(
            "SELECT chunk_id, validity, rowids FROM vec_items_chunks"
        ).fetchall():
            chunk_id = chunk["chunk_id"]
            validity = chunk["validity"]
            rowids_blob = chunk["rowids"]

            # Read the vector blob for this chunk
            vrow = con.execute(
                "SELECT vectors FROM vec_items_vector_chunks00 WHERE rowid=?",
                [chunk_id]
            ).fetchone()
            if not vrow:
                continue
            vectors_blob = vrow[0]

            n_slots = len(rowids_blob) // 8
            for off in range(n_slots):
                # Check validity bit
                if len(validity) > off // 8:
                    if not (validity[off // 8] >> (off % 8)) & 1:
                        continue
                # Decode rowid (8-byte little-endian int64)
                rid = struct.unpack_from("<q", rowids_blob, off * 8)[0]
                # Decode vector (DIM float32s)
                start = off * DIM * 4
                end = start + DIM * 4
                if end > len(vectors_blob):
                    break
                vec = list(struct.unpack_from(f"<{DIM}f", vectors_blob, start))
                vectors_by_rowid[rid] = vec

        con.close()
        return vectors_by_rowid
    except Exception:
        return {}


def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product (vectors are pre-normalized, so this is cosine sim)."""
    return sum(x * y for x, y in zip(a, b))


def _compute_neighbors(vectors: dict[int, list[float]],
                       top_k: int = 12,
                       min_sim: float = 0.45) -> dict[int, list[list]]:
    """For each node, find its top-k most similar neighbors above min_sim.

    Returns {rowid: [[target_rowid, similarity], ...]}.
    O(N²) but N is typically <500 and dot products are fast.
    """
    if not vectors:
        return {}

    ids = list(vectors.keys())
    vecs = [vectors[i] for i in ids]
    n = len(ids)
    neighbors: dict[int, list[list]] = {}

    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            s = _cosine(vecs[i], vecs[j])
            if s >= min_sim:
                sims.append([ids[j], round(s, 4)])
        sims.sort(key=lambda x: -x[1])
        neighbors[ids[i]] = sims[:top_k]

    return neighbors


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(db_path: str) -> dict:
    db_path = os.path.expanduser(db_path)
    if not Path(db_path).exists():
        return {"nodes": [], "topics": {}, "stats": {}, "range": {}, "params": {}}

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, hash_id, text, type, status, confidence, "
        "       user_id, source_tool, topic_cluster, created_at "
        "FROM items ORDER BY created_at ASC"
    ).fetchall()
    con.close()

    if not rows:
        return {"nodes": [], "topics": {}, "stats": {}, "range": {}, "params": {}}

    # Load + decode vectors, compute semantic neighbors
    vectors = _load_vectors(db_path)
    neighbors = _compute_neighbors(vectors)

    # Time range
    times = [r["created_at"] for r in rows if r["created_at"]]
    t_min, t_max = min(times), max(times)
    t_span = (t_max - t_min) or 1.0

    nodes = []
    for r in rows:
        topic = (r["topic_cluster"] or "").strip() or infer_topic(r["text"] or "")
        if not topic:
            topic = "general"
        nodes.append({
            "id": r["id"],
            "hash": (r["hash_id"] or "")[:12],
            "text": r["text"] or "",
            "type": r["type"] or "unknown",
            "status": r["status"] or "tentative",
            "confidence": round(r["confidence"] or 0.0, 3),
            "user": r["user_id"] or "?",
            "source_tool": r["source_tool"] or "",
            "topic": topic,
            "color": _topic_color(topic),
            "created_at": r["created_at"],
            "time_norm": round(((r["created_at"] or t_min) - t_min) / t_span, 4),
            "sim": neighbors.get(r["id"], []),  # semantic neighbors
        })

    # Topic metadata
    topic_counts: dict[str, int] = {}
    for n in nodes:
        topic_counts[n["topic"]] = topic_counts.get(n["topic"], 0) + 1
    topics_meta = {}
    for i, t in enumerate(sorted(topic_counts, key=lambda k: -topic_counts[k])):
        topics_meta[t] = {"color": _topic_color(t), "count": topic_counts[t]}

    by_status, by_type = {}, {}
    for n in nodes:
        by_status[n["status"]] = by_status.get(n["status"], 0) + 1
        by_type[n["type"]] = by_type.get(n["type"], 0) + 1

    # Physics parameters (from design doc §7.7)
    params = {
        "Ka": 1.0,          # semantic attraction
        "Kr": 80,           # repulsion
        "Kz": 0.18,         # time spring
        "Ks": 0.06,         # semantic Z adhesion
        "Ht": 18,           # time height coefficient
        "H0": 0,            # time height offset
        "alpha": 1.0,       # time scaling
        "sim_threshold": 0.45,  # semantic attraction threshold
        "system_start": t_min,  # for t_abs calculation
        "damping": 0.85,    # velocity damping per frame
    }

    return {
        "nodes": nodes,
        "topics": topics_meta,
        "stats": {"total": len(nodes),
                  "by_status": by_status, "by_type": by_type},
        "range": {"t_min": t_min, "t_max": t_max},
        "params": params,
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Noesis · Star Map</title>
<style>
  :root { --bg:#000; --panel:rgba(12,12,22,0.92); --border:rgba(255,255,255,0.12); --text:#fff; --muted:#9aa0b4; }
  * { box-sizing:border-box; margin:0; padding:0; -webkit-tap-highlight-color:transparent; }
  html, body { height:100%; overflow:hidden; background:#000; color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Helvetica Neue",sans-serif; }
  #c { position:absolute; inset:0; }
  .panel { position:fixed; top:12px; left:12px; width:300px; z-index:10;
    background:var(--panel); backdrop-filter:blur(24px); -webkit-backdrop-filter:blur(24px);
    border:1px solid var(--border); border-radius:20px; padding:16px;
    max-height:calc(100vh - 24px); overflow-y:auto; }
  .panel h1 { font-size:17px; font-weight:700; }
  .sub { font-size:12px; color:var(--muted); margin:5px 0 12px; }
  .axes { font-size:11px; color:var(--muted); margin-bottom:12px; line-height:1.7;
    background:rgba(255,255,255,0.04); border-radius:10px; padding:9px 11px; }
  .axes b { color:#fff; }
  .stats { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:12px; font-size:12px; }
  .stat b { color:#fff; font-size:16px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:10px; }
  .chip { display:flex; align-items:center; gap:6px; font-size:11px; padding:6px 8px;
    background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1);
    border-radius:10px; cursor:pointer; transition:background .15s; }
  .chip.off { opacity:0.3; }
  .chip:hover { background:rgba(255,255,255,0.12); }
  .dot { width:9px; height:9px; border-radius:50%; box-shadow:0 0 8px currentColor; flex:none; }
  .ct { margin-left:auto; color:var(--muted); font-size:10px; }
  .btns { display:flex; gap:8px; flex-wrap:wrap; }
  .btn { flex:1; min-width:80px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.14);
    color:#fff; padding:10px; font-size:13px; border-radius:12px; font-weight:500; cursor:pointer;
    text-align:center; transition:background .15s; }
  .btn:active { transform:scale(0.97); opacity:0.8; }
  .btn.active { background:rgba(34,211,238,0.2); border-color:rgba(34,211,238,0.4); }
  #tip { position:fixed; z-index:20; pointer-events:none; background:rgba(20,20,35,0.95);
    backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.15); border-radius:14px;
    padding:10px 14px; font-size:13px; max-width:320px; display:none; transform:translate(12px,12px); }
  #tip .topic { font-size:11px; margin-bottom:4px; }
  #tip .meta { font-size:11px; color:var(--muted); margin-top:6px; display:flex; gap:6px; flex-wrap:wrap; }
  #tip .tag { padding:1px 6px; border-radius:6px; background:rgba(255,255,255,0.1); }
  .axis-label { position:fixed; color:var(--muted); font-size:11px; z-index:5; pointer-events:none;
    background:rgba(0,0,0,0.5); padding:2px 8px; border-radius:6px; }
  #loading { position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
    color:#22d3ee; font-size:18px; z-index:30; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="loading">\u26a1 Simulating physics...</div>
<div class="panel">
  <h1>Star Map \u00b7 Thought Galaxy</h1>
  <div class="sub">Drag to orbit \u00b7 Shift/right-drag to pan \u00b7 Scroll to zoom \u00b7 Hover for details</div>
  <div class="axes">
    <div><b>X axis</b>: Time (old \u2192 new)</div>
    <div><b>Y axis</b>: Confidence (low \u2192 high)</div>
    <div><b>Z axis</b>: Topic depth (semantic clustering)</div>
    <div>Star <b>size</b> = type + activity \u00b7 <b>brightness</b> = activation weight</div>
    <div>Hover for content \u00b7 Click topic to filter</div>
  </div>
  <div class="stats" id="stats"></div>
  <div class="grid" id="g"></div>
  <div class="btns">
    <button class="btn" id="toggleLines">Hide Lines</button>
    <button class="btn" id="toggleColor">Color by Age</button>
    <button class="btn" id="replay">Resimulate</button>
    <button class="btn" id="reset">Reset View</button>
  </div>
</div>
<div id="tip"></div>

<script type="module">
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

const data = await fetch('/api').then(r => r.json());
const nodes = data.nodes;
const topics = data.topics;
const P = data.params;

// ── Scene ──
const canvas = document.getElementById('c');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 5000);
const renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);

// ── Type half-life table (design doc §7.4) — days, null = never decays ──
const HALF_LIFE = {
  event: 7.0, question: 30.0, position: 90.0, preference: 90.0,
  synthesis: 180.0, contradiction: null, identity: null
};

// ── Activation weight: w = exp(-λ·Δt), identity/contradiction never decay ──
function activationWeight(node) {
  const hl = HALF_LIFE[node.type] ?? 90;
  if (hl === null) return 1.0;
  const ageDays = (Date.now() / 1000 - node.created_at) / 86400;
  return Math.exp(-0.693 * ageDays / hl);
}

// ── Age color: blue-white (young) → white → yellow → red (old) ──
function ageColor(node) {
  const ageDays = (Date.now() / 1000 - node.created_at) / 86400;
  if (ageDays <= 7)  return '#a4d8ff';  // blue-white (young hot star)
  if (ageDays <= 30) return '#ffffff';  // white (active stable)
  if (ageDays <= 90) return '#fde047';  // yellow (cooling)
  return '#ef4444';                      // red (red dwarf)
}

// ── All stars are glowing spheres — type differentiated by size + brightness ──
// (Design doc §7.4 specifies shapes, but visually different geometries look like
//  debris, not stars. We keep the star aesthetic and encode type via size/brightness.)
function starSize(type, weight) {
  const base = Math.max(1.0, Math.sqrt(weight) * 1.8);
  switch (type) {
    case 'identity':      return base * 1.6;   // largest (never decays)
    case 'contradiction': return base * 1.3;   // prominent
    case 'synthesis':     return base * 1.2;
    case 'position':      return base * 1.0;
    case 'preference':    return base * 0.9;
    case 'question':      return base * 0.85;
    case 'event':         return base * 0.75;  // smallest
    default:              return base;
  }
}

// ── Build star meshes with initial positions ──
// Initial: tight random cluster, physics will settle them into galaxies
const stars = [];
const starData = [];

nodes.forEach((n, i) => {
  // Initial: WIDE random spread in a large sphere — gives physics room to
  // settle into volumetric 3D constellations, not flat sheets.
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  const r = 60 + Math.random() * 120;
  const x = r * Math.sin(phi) * Math.cos(theta);
  const y = r * Math.sin(phi) * Math.sin(theta);
  const z = r * Math.cos(phi);

  const w = activationWeight(n);
  const color = n.color;
  const size = starSize(n.type, w);

  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(size, 12, 12),
    new THREE.MeshBasicMaterial({ color, transparent:true, opacity: 0.85 + w * 0.15 })
  );
  mesh.position.set(x, y, z);
  mesh.userData = n;
  scene.add(mesh);
  stars.push(mesh);

  // Glow halo — larger and brighter for the luminous star feel
  const glow = new THREE.Mesh(
    new THREE.SphereGeometry(size * 2.8, 12, 12),
    new THREE.MeshBasicMaterial({ color, transparent:true, opacity: 0.10 + w * 0.15 })
  );
  glow.position.copy(mesh.position);
  scene.add(glow);
  mesh.userData.glow = glow;

  starData.push({
    pos: new THREE.Vector3(x, y, z),
    vel: new THREE.Vector3(0, 0, 0),
    weight: w,
    node: n,
  });
});

// ── Map node id → star index for neighbor lookups ──
const idToIndex = {};
nodes.forEach((n, i) => { idToIndex[n.id] = i; });

// ── Physics: Full 3D Four-Force Model ──
// All forces operate in full XYZ space so constellations are volumetric,
// not collapsed into 2D sheets or 1D chains.
const attractPairs = [];  // [[i, j, sim], ...]
nodes.forEach((n, i) => {
  if (!n.sim) return;
  n.sim.forEach(([targetId, s]) => {
    const j = idToIndex[targetId];
    if (j !== undefined && j > i) {
      attractPairs.push([i, j, s]);
    } else if (j !== undefined) {
      attractPairs.push([j, i, s]);
    }
  });
});

// Axis layout constants — define the spatial scale of the galaxy
// X = time (old→new), Y = confidence (low→high), Z = topic depth (emerges from physics)
const X_SPAN = 400;   // time axis extent
const Y_SPAN = 200;   // confidence axis extent

// Precompute axis target positions for each star
starData.forEach((sd) => {
  // X target from time_norm (0=oldest, 1=newest)
  sd.xTarget = (sd.node.time_norm - 0.5) * X_SPAN;
  // Y target from confidence (0.0=bottom, 1.0=top)
  sd.yTarget = sd.node.confidence * Y_SPAN;
  // Z target from time — newer floats slightly higher (gentle, from §7.2 C)
  const tAbs = (sd.node.created_at - P.system_start) / 86400;
  sd.zTarget = P.H0 + P.Ht * 0.5 * Math.log(1 + P.alpha * Math.max(0, tAbs));
});

function simulate() {
  const EPS = 0.01;

  // ── Force 0: Axis bias — gentle springs toward meaningful coordinates ──
  // Weak enough that semantic attraction can override locally (clusters form),
  // strong enough that you can read time→right, confidence→up at a glance.
  for (const sd of starData) {
    sd.vel.x += (sd.xTarget - sd.pos.x) * 0.008;
    sd.vel.y += (sd.yTarget - sd.pos.y) * 0.008;
    sd.vel.z += (sd.zTarget - sd.pos.z) * 0.003;  // weakest on Z so physics dominates
  }

  // ── Force A: Semantic attraction — FULL 3D ──
  // Pulls similar thoughts together in XYZ, forming volumetric clusters.
  for (const [i, j, sim] of attractPairs) {
    if (!stars[i].visible || !stars[j].visible) continue;
    const si = starData[i], sj = starData[j];
    const dx = sj.pos.x - si.pos.x;
    const dy = sj.pos.y - si.pos.y;
    const dz = sj.pos.z - si.pos.z;
    const d = Math.sqrt(dx*dx + dy*dy + dz*dz + EPS);
    if (d > 350) continue;
    // Attraction: stronger when far, scaled by similarity
    const F = sim * d * 0.0025;
    const fx = (dx / d) * F;
    const fy = (dy / d) * F;
    const fz = (dz / d) * F;
    si.vel.x += fx; si.vel.y += fy; si.vel.z += fz;
    sj.vel.x -= fx; sj.vel.y -= fy; sj.vel.z -= fz;
  }

  // ── Force B: Repulsion — FULL 3D spatial hash ──
  // Two regimes:
  //   Same-topic:  short-range anti-overlap (keeps cluster packed but not degenerate)
  //   Diff-topic:  long-range cluster separation (pushes galaxies apart with space)
  const CELL = 18;
  const grid = {};
  starData.forEach((sd, i) => {
    if (!stars[i].visible) return;
    const cx = Math.floor(sd.pos.x / CELL);
    const cy = Math.floor(sd.pos.y / CELL);
    const cz = Math.floor(sd.pos.z / CELL);
    const key = `${cx},${cy},${cz}`;
    if (!grid[key]) grid[key] = [];
    grid[key].push(i);
  });

  for (const [key, bucket] of Object.entries(grid)) {
    const [cx, cy, cz] = key.split(',').map(Number);
    const nearby = [];
    // Check wider neighborhood (5×5×5 = 125 cells) to catch inter-cluster pairs
    for (let dx = -2; dx <= 2; dx++) {
      for (let dy = -2; dy <= 2; dy++) {
        for (let dz = -2; dz <= 2; dz++) {
          const nkey = `${cx+dx},${cy+dy},${cz+dz}`;
          if (grid[nkey]) nearby.push(...grid[nkey]);
        }
      }
    }
    for (let a = 0; a < bucket.length; a++) {
      for (let b = 0; b < nearby.length; b++) {
        const i = bucket[a], j = nearby[b];
        if (i >= j) continue;
        if (!stars[i].visible || !stars[j].visible) continue;
        const si = starData[i], sj = starData[j];
        const dx = sj.pos.x - si.pos.x;
        const dy = sj.pos.y - si.pos.y;
        const dz = sj.pos.z - si.pos.z;
        const d2 = dx*dx + dy*dy + dz*dz + EPS;
        const sameTopic = si.node.topic === sj.node.topic;

        if (sameTopic) {
          // Intra-cluster: only anti-overlap at very short range
          if (d2 > 200) continue;
          const F = 12 / d2;
          const d = Math.sqrt(d2);
          si.vel.x -= (dx / d) * F;
          si.vel.y -= (dy / d) * F;
          si.vel.z -= (dz / d) * F;
          sj.vel.x += (dx / d) * F;
          sj.vel.y += (dy / d) * F;
          sj.vel.z += (dz / d) * F;
        } else {
          // Inter-cluster: strong long-range separation to create space
          // between different-topic galaxies
          if (d2 > 5000) continue;  // act up to ~70 units
          const d = Math.sqrt(d2);
          const F = 120 / (d2 + 5);
          si.vel.x -= (dx / d) * F;
          si.vel.y -= (dy / d) * F;
          si.vel.z -= (dz / d) * F;
          sj.vel.x += (dx / d) * F;
          sj.vel.y += (dy / d) * F;
          sj.vel.z += (dz / d) * F;
        }
      }
    }
  }

  // ── Integrate positions with damping ──
  for (const sd of starData) {
    sd.vel.multiplyScalar(P.damping);
    const maxV = 4;
    if (sd.vel.length() > maxV) sd.vel.setLength(maxV);
    sd.pos.add(sd.vel);
  }
}

// ── Run physics for ~3 seconds to converge, then freeze ──
let physicsActive = true;
let physicsFrames = 0;
const CONVERGE_FRAMES = 180;  // ~3s at 60fps

function runPhysics() {
  if (!physicsActive) return;
  simulate();
  physicsFrames++;
  if (physicsFrames >= CONVERGE_FRAMES) {
    physicsActive = false;
    document.getElementById('loading').style.display = 'none';
    updateMeshPositions();
    buildConnections();
  } else {
    updateMeshPositions();
    requestAnimationFrame(runPhysics);
  }
}

function updateMeshPositions() {
  stars.forEach((mesh, i) => {
    mesh.position.copy(starData[i].pos);
    if (mesh.userData.glow) mesh.userData.glow.position.copy(mesh.position);
  });
}

// ── Connection lines (design doc §7.5) ──
// Within-galaxy: time-sorted trajectory within same semantic cluster
// Between-galaxy: high similarity (>0.7) + time proximity
const lineGroup = new THREE.Group();
scene.add(lineGroup);

function buildConnections() {
  lineGroup.clear();
  const seen = new Set();

  // Within-galaxy: connect each node to its top-2 semantic neighbors
  const lineMat = new THREE.LineBasicMaterial({
    color: 0x334466, transparent: true, opacity: 0.2
  });
  for (const [i, j, sim] of attractPairs) {
    if (sim < 0.5) continue;  // only strong connections
    if (!stars[i].visible || !stars[j].visible) continue;
    const key = i < j ? `${i}-${j}` : `${j}-${i}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const geo = new THREE.BufferGeometry().setFromPoints([
      starData[i].pos, starData[j].pos
    ]);
    // Color lines by similarity strength
    const mat = new THREE.LineBasicMaterial({
      color: new THREE.Color().lerpColors(
        new THREE.Color(0x224466), new THREE.Color(0x22d3ee), Math.min(1, (sim - 0.5) * 3)
      ),
      transparent: true,
      opacity: 0.15 + (sim - 0.5) * 0.3
    });
    lineGroup.add(new THREE.Line(geo, mat));
  }
}

// ── Background stars — dense field for depth ──
const bgGeo = new THREE.BufferGeometry();
const bgPos = [];
for (let i = 0; i < 1200; i++)
  bgPos.push((Math.random()-0.5)*1800, (Math.random()-0.5)*1000, (Math.random()-0.5)*1800);
bgGeo.setAttribute('position', new THREE.Float32BufferAttribute(bgPos, 3));
scene.add(new THREE.Points(bgGeo, new THREE.PointsMaterial({ color:0x445577, size:0.6 })));

// ── 3D Axis guides — Time (X), Confidence (Y), Topic Depth (Z) ──
const axisGroup = new THREE.Group();
scene.add(axisGroup);
const AX = { x:X_SPAN/2 + 30, y:Y_SPAN + 30, z:80 };
const axisColors = { x:0xff6b6b, y:0x34d399, z:0x60a5fa };
['x','y','z'].forEach(ax => {
  const mat = new THREE.LineBasicMaterial({ color:axisColors[ax], transparent:true, opacity:0.3 });
  const neg = new THREE.Vector3(ax==='x'?-AX.x:0, ax==='y'?-10:0, ax==='z'?-AX.z:0);
  const pos = new THREE.Vector3(ax==='x'?AX.x:0, ax==='y'?AX.y:0, ax==='z'?AX.z:0);
  axisGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([neg, pos]), mat));
});

// HTML labels for axis tips (projected each frame)
const labelLayer = document.createElement('div');
labelLayer.style.cssText = 'position:fixed;inset:0;z-index:6;pointer-events:none;';
document.body.appendChild(labelLayer);
function makeLabel(text, color) {
  const el = document.createElement('div');
  el.style.cssText = `position:absolute;color:${color};font-size:11px;font-weight:600;
    background:rgba(0,0,0,0.5);padding:2px 7px;border-radius:6px;transform:translate(-50%,-50%);`;
  el.textContent = text;
  labelLayer.appendChild(el);
  return el;
}
const lblX = makeLabel('TIME (old \u2192 new) \u2192', '#ff8a8a');
const lblY = makeLabel('\u2191 CONFIDENCE', '#5eead4');
const lblZ = makeLabel('\u2299 TOPIC DEPTH', '#7dd3fc');

// ── Orbit controls with inertia ──
let rotX = 0.42, rotY = -0.55, dist = 350, panX = 0, panY = 0;
const ROT_MIN = -1.45, ROT_MAX = 1.45;
const DIST_MIN = 150, DIST_MAX = 2000;
let velRotX = 0, velRotY = 0, velPanX = 0, velPanY = 0;
const DAMP = 0.92, ROT_GAIN = 0.005, PAN_GAIN = 0.5, ZOOM_GAIN = 0.0014;

function updateCam() {
  const cx = Math.sin(rotY) * Math.cos(rotX) * dist + panX;
  const cy = Math.sin(rotX) * dist + panY;
  const cz = Math.cos(rotY) * Math.cos(rotX) * dist;
  camera.position.set(cx, cy, cz);
  camera.lookAt(panX, panY, 0);
}
updateCam();

let isDown = false, mode = 'orbit', lastX = 0, lastY = 0;
canvas.addEventListener('pointerdown', e => {
  isDown = true;
  mode = (e.button === 2 || e.button === 1 || e.shiftKey) ? 'pan' : 'orbit';
  lastX = e.clientX; lastY = e.clientY;
  velRotX = velRotY = velPanX = velPanY = 0;
  try { canvas.setPointerCapture(e.pointerId); } catch(_) {}
  e.preventDefault();
});
canvas.addEventListener('pointermove', e => {
  if (!isDown) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (mode === 'pan') {
    panX += dx * PAN_GAIN; panY -= dy * PAN_GAIN;
    velPanX = dx * PAN_GAIN; velPanY = -dy * PAN_GAIN;
    updateCam();
  } else {
    rotY += dx * ROT_GAIN;
    rotX = Math.max(ROT_MIN, Math.min(ROT_MAX, rotX - dy * ROT_GAIN));
    velRotY = dx * ROT_GAIN; velRotX = -dy * ROT_GAIN;
    updateCam();
  }
});
function endDrag(e) {
  if (!isDown) return;
  isDown = false;
  try { canvas.releasePointerCapture(e.pointerId); } catch(_) {}
}
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = Math.exp(e.deltaY * ZOOM_GAIN);
  dist = Math.max(DIST_MIN, Math.min(DIST_MAX, dist * factor));
  updateCam();
}, { passive: false });

// Touch pinch
let pinchDist0 = 0, distAtPinch0 = dist;
canvas.addEventListener('touchstart', e => {
  if (e.touches.length === 2) {
    const [a, b] = e.touches;
    pinchDist0 = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    distAtPinch0 = dist;
  }
}, { passive: true });
canvas.addEventListener('touchmove', e => {
  if (e.touches.length === 2) {
    e.preventDefault();
    const [a, b] = e.touches;
    const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    if (pinchDist0 > 0)
      dist = Math.max(DIST_MIN, Math.min(DIST_MAX, distAtPinch0 * (pinchDist0 / d)));
  }
}, { passive: false });

// ── Panel UI ──
const st = data.stats;
  document.getElementById('stats').innerHTML = `
  <div class="stat"><b>${st.total}</b> thoughts</div>
  <div class="stat"><b>${Object.keys(topics).length}</b> galaxies</div>
  <div class="stat"><b>${st.by_status.settled||0}</b> settled</div>
`;

const grid = document.getElementById('g');
const activeTopics = new Set(Object.keys(topics));
Object.entries(topics).sort((a,b) => b[1].count - a[1].count).forEach(([name, meta]) => {
  const div = document.createElement('div');
  div.className = 'chip';
  div.innerHTML = `<span class="dot" style="background:${meta.color};color:${meta.color}"></span>${name}<span class="ct">${meta.count}</span>`;
  div.onclick = () => {
    if (activeTopics.has(name)) { activeTopics.delete(name); div.classList.add('off'); }
    else { activeTopics.add(name); div.classList.remove('off'); }
    stars.forEach((s, i) => {
      const visible = activeTopics.has(starData[i].node.topic);
      s.visible = visible;
      if (s.userData.glow) s.userData.glow.visible = visible;
    });
    buildConnections();
  };
  grid.appendChild(div);
});

// ── Button handlers ──
let linesVisible = true;
document.getElementById('toggleLines').onclick = function() {
  linesVisible = !linesVisible;
  lineGroup.visible = linesVisible;
  this.textContent = linesVisible ? 'Hide Lines' : 'Show Lines';
};

let colorMode = 'topic';
document.getElementById('toggleColor').onclick = function() {
  colorMode = colorMode === 'topic' ? 'age' : 'topic';
  this.textContent = colorMode === 'topic' ? 'Color by Age' : 'Color by Topic';
  this.classList.toggle('active', colorMode === 'age');
  applyColorMode();
};

function applyColorMode() {
  stars.forEach((mesh, i) => {
    const n = starData[i].node;
    const color = colorMode === 'topic' ? n.color : ageColor(n);
    mesh.material.color.set(color);
    if (mesh.userData.glow) mesh.userData.glow.material.color.set(color);
  });
}

document.getElementById('replay').onclick = () => {
  // Reset positions and replay physics
  starData.forEach((sd, i) => {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = 50 + Math.random() * 100;
    sd.pos.set(
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta),
      r * Math.cos(phi)
    );
    sd.vel.set(0, 0, 0);
  });
  physicsActive = true;
  physicsFrames = 0;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('loading').textContent = '\u26a1 Resimulating...';
  runPhysics();
};

document.getElementById('reset').onclick = () => {
  rotX = 0.42; rotY = -0.55; dist = 350; panX = 0; panY = 0;
  velRotX = velRotY = velPanX = velPanY = 0;
  updateCam();
};

// ── Tooltip ──
const tip = document.getElementById('tip');
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
canvas.addEventListener('pointermove', e => {
  mouse.x = (e.clientX / innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / innerHeight) * 2 + 1;
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObjects(stars.filter(s => s.visible));
  const hit = hits[0];
  if (hit) {
    const d = hit.object.userData;
    tip.style.display = 'block';
    tip.style.left = e.clientX + 'px';
    tip.style.top = e.clientY + 'px';
    const w = activationWeight(d).toFixed(2);
    tip.innerHTML =
      `<div class="topic" style="color:${d.color}">${d.topic}</div>` +
      `<div>${d.text.length > 200 ? d.text.slice(0,200) + '…' : d.text}</div>` +
      `<div class="meta">` +
      `<span class="tag">${d.type}</span>` +
      `<span class="tag">${d.status}</span>` +
      `<span class="tag">weight ${w}</span>` +
      `<span class="tag">conf ${d.confidence}</span>` +
      `</div>`;
  } else {
    tip.style.display = 'none';
  }
});

// ── Animation loop ──
function animate() {
  requestAnimationFrame(animate);

  // Pulsing glow during physics simulation
  if (physicsActive) {
    stars.forEach((s, i) => {
      if (s.visible) {
        s.material.opacity = 0.7 + Math.sin(Date.now() * 0.003 + i) * 0.15;
      }
    });
  }

  // Camera inertia
  if (!isDown) {
    if (Math.abs(velRotX) > 1e-5 || Math.abs(velRotY) > 1e-5 ||
        Math.abs(velPanX) > 1e-5 || Math.abs(velPanY) > 1e-5) {
      rotY += velRotY;
      rotX = Math.max(ROT_MIN, Math.min(ROT_MAX, rotX + velRotX));
      panX += velPanX; panY += velPanY;
      velRotX *= DAMP; velRotY *= DAMP;
      velPanX *= DAMP; velPanY *= DAMP;
      updateCam();
    }
  }

  // Project axis labels to screen
  const _p = new THREE.Vector3();
  function projLabel(v, el) {
    _p.copy(v).project(camera);
    el.style.display = (_p.z > 1) ? 'none' : 'block';
    el.style.left = ((_p.x * 0.5 + 0.5) * innerWidth) + 'px';
    el.style.top = ((-_p.y * 0.5 + 0.5) * innerHeight) + 'px';
  }
  projLabel(new THREE.Vector3(AX.x, 0, 0), lblX);
  projLabel(new THREE.Vector3(0, AX.y, 0), lblY);
  projLabel(new THREE.Vector3(0, 0, AX.z), lblZ);

  renderer.render(scene, camera);
}
animate();

// Start physics
runPhysics();

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    DB_PATH = "~/.noesis/hot.db"
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/api"):
            body = json.dumps(load_data(self.DB_PATH), ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description="Noesis Star Map v2 — Four-Force Physics")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--db", default="~/.noesis/hot.db")
    args = ap.parse_args()
    Handler.DB_PATH = args.db
    d = load_data(args.db)
    vec_count = sum(1 for n in d["nodes"] if n.get("sim"))
    print("=" * 56)
    print("  Noesis · 思想星图 v2 (Four-Force Physics)")
    print("=" * 56)
    print(f"  DB: {args.db}")
    print(f"  {d['stats'].get('total',0)} memories · {len(d['topics'])} topics")
    print(f"  {vec_count} nodes have semantic neighbors")
    print(f"  Physics: Ka={d['params']['Ka']} Kr={d['params']['Kr']} "
          f"Kz={d['params']['Kz']} Ks={d['params']['Ks']}")
    print(f"\n  → http://localhost:{args.port}\n  Ctrl+C 退出\n" + "-" * 56)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
