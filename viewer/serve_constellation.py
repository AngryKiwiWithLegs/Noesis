#!/usr/bin/env python3
"""
Noesis Topic Constellation Viewer
=================================
A true 3D visualization of experiment memories organized by *shared ideas*,
not by persona.

Axes (every dimension is meaningful):
  X — time           (created_at → left=oldest, right=newest)
  Y — confidence     (0.0=bottom → 1.0=top; settled stars float highest)
  Z — topic axis     (each inferred topic gets a fixed Z-band, so stars about
                       the same idea cluster into a depth-slice / constellation)

Connections: stars that share the same inferred topic get a visible link,
so connected ideas form constellations that bridge personas.

Usage:
    python3 serve_constellation.py [--port 8790] [--db ~/.noesis/hot.db]
"""
import argparse
import json
import os
import re
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Topic inference ───────────────────────────────────────────────────────────
# Bilingual keyword → topic. Order matters: the first matching topic wins,
# so put the most specific domains first. This is the same philosophy as
# noesis/wiki/extractor.py's _DOMAIN_KEYWORDS but expanded to cover the
# experiment's full bilingual vocabulary.
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


def infer_topic(text: str) -> str:
    """Return the first topic whose keyword regex matches the (lowercased) text."""
    t = (text or "").lower()
    for topic, patterns in _TOPIC_RULES:
        for pat in patterns:
            if re.search(pat, t):
                return topic
    return "general"


# Distinct color per topic — reused by the panel legend + the stars.
_TOPIC_COLORS = {
    "vector-store": "#22d3ee",
    "vector-db":    "#22d3ee",
    "database":     "#34d399",
    "languages":    "#a78bfa",
    "frontend":     "#f472b6",
    "api-design":   "#fbbf24",
    "cloud-infra":  "#3b82f6",
    "messaging":    "#fb923c",
    "devtools":     "#c4b5fd",
    "tools":        "#c4b5fd",
    "data-science": "#2dd4bf",
    "engineering":  "#818cf8",
    "llm-choice":   "#facc15",
    "career":       "#ff8a8a",
    "identity":     "#ff6b6b",
    "general":      "#9aa0b4",
}

# Extended palette for clusters not in the explicit map above. Used by
# _topic_color() so any new cluster from `recluster` gets a distinct color
# without needing a code change.
_PALETTE = [
    "#22d3ee", "#34d399", "#a78bfa", "#f472b6", "#fbbf24", "#3b82f6",
    "#fb923c", "#c4b5fd", "#2dd4bf", "#818cf8", "#facc15", "#f87171",
    "#4ade80", "#60a5fa", "#e879f9", "#fde047", "#67e8f9", "#a3e635",
    "#fca5a5", "#bef264", "#7dd3fc", "#ddd6fe", "#fdba74", "#bbf7d0",
]


def _topic_color(topic: str) -> str:
    """Deterministic color for a cluster name. Known clusters use the explicit
    map; unknown ones get a stable palette slot by hash so they're always
    consistent across reloads."""
    if topic in _TOPIC_COLORS:
        return _TOPIC_COLORS[topic]
    return _PALETTE[abs(hash(topic)) % len(_PALETTE)]


def load_data(db_path: str) -> dict:
    db_path = os.path.expanduser(db_path)
    if not Path(db_path).exists():
        return {"nodes": [], "topics": {}, "stats": {}, "range": {}}

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, hash_id, text, type, status, confidence, "
        "       user_id, source_tool, topic_cluster, created_at "
        "FROM items ORDER BY created_at ASC"
    ).fetchall()
    con.close()

    if not rows:
        return {"nodes": [], "topics": {}, "stats": {}, "range": {}}

    # Time range → normalized 0..1
    times = [r["created_at"] for r in rows if r["created_at"]]
    t_min, t_max = min(times), max(times)
    t_span = (t_max - t_min) or 1.0

    nodes = []
    for r in rows:
        # Read the DB's stored topic_cluster (the source of truth — updated by
        # `noesis recluster`). Fall back to inference only for legacy nodes
        # that have no cluster stored.
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
            "topic": topic,
            "color": _topic_color(topic),
            "created_at": r["created_at"],
            "time_norm": round(((r["created_at"] or t_min) - t_min) / t_span, 4),
        })

    # Topic metadata: count + a stable Z-band index. Colors are assigned
    # dynamically so any cluster the recluster command creates gets a color.
    topic_counts: dict[str, int] = {}
    for n in nodes:
        topic_counts[n["topic"]] = topic_counts.get(n["topic"], 0) + 1
    topics_meta = {}
    for i, t in enumerate(sorted(topic_counts, key=lambda k: -topic_counts[k])):
        topics_meta[t] = {"color": _topic_color(t), "count": topic_counts[t],
                          "band": i}

    by_status, by_type = {}, {}
    for n in nodes:
        by_status[n["status"]] = by_status.get(n["status"], 0) + 1
        by_type[n["type"]] = by_type.get(n["type"], 0) + 1

    return {
        "nodes": nodes,
        "topics": topics_meta,
        "stats": {"total": len(nodes),
                  "by_status": by_status, "by_type": by_type},
        "range": {"t_min": t_min, "t_max": t_max},
    }


HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Noesis · 思想星座</title>
<style>
  :root { --bg:#000; --panel:rgba(12,12,22,0.92); --border:rgba(255,255,255,0.12); --text:#fff; --muted:#9aa0b4; }
  * { box-sizing:border-box; margin:0; padding:0; -webkit-tap-highlight-color:transparent; }
  html, body { height:100%; overflow:hidden; background:#000; color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Helvetica Neue",sans-serif; }
  #c { position:absolute; inset:0; }
  .panel { position:fixed; top:12px; left:12px; width:280px; z-index:10;
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
  .btns { display:flex; gap:8px; }
  .btn { flex:1; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.14);
    color:#fff; padding:10px; font-size:13px; border-radius:12px; font-weight:500; cursor:pointer; }
  .btn:active { transform:scale(0.97); opacity:0.8; }
  #tip { position:fixed; z-index:20; pointer-events:none; background:rgba(20,20,35,0.95);
    backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.15); border-radius:14px;
    padding:10px 14px; font-size:13px; max-width:300px; display:none; transform:translate(12px,12px); }
  #tip .topic { font-size:11px; margin-bottom:4px; }
  #tip .meta { font-size:11px; color:var(--muted); margin-top:6px; display:flex; gap:6px; flex-wrap:wrap; }
  #tip .tag { padding:1px 6px; border-radius:6px; background:rgba(255,255,255,0.1); }
  .axis-label { position:fixed; color:var(--muted); font-size:11px; z-index:5; pointer-events:none;
    background:rgba(0,0,0,0.5); padding:2px 8px; border-radius:6px; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div class="panel">
  <h1>思想星座 · Topic Constellation</h1>
  <div class="sub">左键拖拽旋转 · Shift/右键平移 · 滚轮缩放 · 悬停查看</div>
  <div class="axes">
    <div>每个<b>主题</b>是一团星系，颜色相同</div>
    <div>同主题的星点之间有<b>连接线</b></div>
    <div>星点大小 = 状态 (settled &gt; provisional &gt; tentative)</div>
    <div>悬停查看记忆内容 · 点击主题筛选</div>
  </div>
  <div class="stats" id="stats"></div>
  <div class="grid" id="g"></div>
  <div class="btns">
    <button class="btn" id="toggle">隐藏连接线</button>
    <button class="btn" id="reset">重置视角</button>
  </div>
</div>
<div class="axis-label" style="bottom:14px; left:14px;">← 时间旧 | 时间新 →</div>
<div id="tip"></div>

<script type="module">
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

const data = await fetch('/api').then(r => r.json());
const nodes = data.nodes;
const topics = data.topics;

// ── Scene ──
const canvas = document.getElementById('c');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 3000);
const renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

// ── Axes define the field. Each star's BASE position is determined by its
//    real values, then organic scatter is layered on so clusters still read
//    as galaxies, not plot points.
//    X = time (oldest→newest, left→right)
//    Y = confidence (low→high, bottom→top)
//    Z = topic band (each topic = a depth slice)
const nTopics = Object.keys(topics).length;
const X_SPAN = 260;                // time extent
const Y_SPAN = 150;                // confidence extent (0→1 maps to this)
const Z_SPACING = 26;              // distance between adjacent topic bands
const HALF_Z = (nTopics - 1) * Z_SPACING / 2;

// Topic → Z band center (depth slice)
const topicZ = {};
Object.entries(topics).forEach(([name, meta]) => {
  topicZ[name] = meta.band * Z_SPACING - HALF_Z;
});

// ── Orbit controls with inertia ──────────────────────────────────────────────
// Conventions (match Google Earth / Three.js OrbitControls — what users expect):
//   • left-drag        → orbit (grab the world and turn it)
//   • shift+left-drag  → pan   (also right-drag, also one-finger on trackpad w/ shift)
//   • wheel / pinch    → zoom
// Motion carries inertia: release while moving and it glides to a stop.
let rotX = 0.42, rotY = -0.55, dist = 360, panX = 0, panY = 0;
const ROT_MIN = -1.45, ROT_MAX = 1.45;   // clamp just shy of ±90° → no pole flip
const DIST_MIN = 100, DIST_MAX = 900;

// Velocity state for inertia
let velRotX = 0, velRotY = 0, velPanX = 0, velPanY = 0;
const DAMP = 0.92;          // per-frame velocity retention (lower = stops sooner)
const ROT_GAIN = 0.005;     // px → radians
const PAN_GAIN = 0.5;       // px → world units
const ZOOM_GAIN = 0.0014;   // wheel-delta → dist (logarithmic feel)

function updateCam(){
  const cx = Math.sin(rotY) * Math.cos(rotX) * dist + panX;
  const cy = Math.sin(rotX) * dist + panY;
  const cz = Math.cos(rotY) * Math.cos(rotX) * dist;
  camera.position.set(cx, cy, cz);
  camera.lookAt(panX, panY, 0);
}
updateCam();

// Pointer-based orbit/pan with inertia.
// All listeners live on the canvas; setPointerCapture guarantees we keep
// receiving moves even if the pointer leaves the canvas mid-drag.
let isDown = false, mode = 'orbit', lastX = 0, lastY = 0;
canvas.addEventListener('pointerdown', e => {
  isDown = true;
  // Pan on: right button, middle button, or shift+left
  mode = (e.button === 2 || e.button === 1 || e.shiftKey) ? 'pan' : 'orbit';
  lastX = e.clientX; lastY = e.clientY;
  velRotX = velRotY = velPanX = velPanY = 0;   // stop any gliding momentum
  try { canvas.setPointerCapture(e.pointerId); } catch(_) {}  // keep drag alive off-canvas
  e.preventDefault();
});
canvas.addEventListener('pointermove', e => {
  if (!isDown) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (mode === 'pan'){
    panX += dx * PAN_GAIN;
    panY -= dy * PAN_GAIN;
    velPanX = dx * PAN_GAIN; velPanY = -dy * PAN_GAIN;
    updateCam();
  } else {
    rotY += dx * ROT_GAIN;
    rotX = Math.max(ROT_MIN, Math.min(ROT_MAX, rotX - dy * ROT_GAIN));
    velRotY = dx * ROT_GAIN;
    velRotX = -dy * ROT_GAIN;
  }
  updateCam();
});
function endDrag(e){
  if (!isDown) return;
  isDown = false;
  try { canvas.releasePointerCapture(e.pointerId); } catch(_) {}
}
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);
canvas.addEventListener('contextmenu', e => e.preventDefault());

// Wheel zoom — logarithmic so each notch feels equal regardless of distance
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = Math.exp(e.deltaY * ZOOM_GAIN);   // >1 zooms out, <1 in
  dist = Math.max(DIST_MIN, Math.min(DIST_MAX, dist * factor));
  updateCam();
}, { passive: false });

// Touch: pinch to zoom (two fingers), one finger to orbit
let pinchDist0 = 0, distAtPinch0 = dist;
canvas.addEventListener('touchstart', e => {
  if (e.touches.length === 2){
    const [a, b] = e.touches;
    pinchDist0 = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    distAtPinch0 = dist;
  }
}, { passive: true });
canvas.addEventListener('touchmove', e => {
  if (e.touches.length === 2){
    e.preventDefault();
    const [a, b] = e.touches;
    const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    if (pinchDist0 > 0){
      dist = Math.max(DIST_MIN, Math.min(DIST_MAX, distAtPinch0 * (pinchDist0 / d)));
    }
  }
}, { passive: false });

// ── Place stars: base from axes + organic scatter so each topic's stars form
//    a nebula rather than collapsing to a single Z-plane point. ──
const stars = [];
function randScatter(r){ return (Math.random() - 0.5) * 2 * r; }  // uniform in [-r, r]

nodes.forEach(n => {
  // BASE position — the meaningful dimensions
  const baseX = (n.time_norm - 0.5) * X_SPAN;
  const baseY = n.confidence * Y_SPAN;
  const baseZ = topicZ[n.topic];
  // ORGANIC scatter — keeps the galaxy feel; smaller than the axis extent so
  // the dimensional gradient remains visible through the scatter.
  const SCATTER = 20;
  const x = baseX + randScatter(SCATTER);
  const y = baseY + randScatter(SCATTER);
  const z = baseZ + randScatter(SCATTER);

  const size = n.status==='settled'?1.7 : n.status==='provisional'?1.1 : 0.75;

  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(size, 12),
    new THREE.MeshBasicMaterial({ color:n.color, transparent:true, opacity:0.92 })
  );
  mesh.position.set(x, y, z);
  mesh.userData = n;
  scene.add(mesh);
  stars.push(mesh);

  const glow = new THREE.Mesh(
    new THREE.SphereGeometry(size*2.4, 12, 12),
    new THREE.MeshBasicMaterial({ color:n.color, transparent:true, opacity:0.13 })
  );
  glow.position.copy(mesh.position);
  scene.add(glow);
});

// ── Connections: link stars sharing the same topic (visible idea bridges) ──
// To keep it readable we link each star to its 2 nearest same-topic neighbors.
const lineGroup = new THREE.Group();
scene.add(lineGroup);
function makeLines(){
  lineGroup.clear();
  const byTopic = {};
  stars.forEach(s => { (byTopic[s.userData.topic] ||= []).push(s); });
  Object.entries(byTopic).forEach(([topic, arr]) => {
    const col = new THREE.Color(topics[topic].color);
    const mat = new THREE.LineBasicMaterial({ color:col, transparent:true, opacity:0.18 });
    for (let i=0;i<arr.length;i++){
      const a = arr[i];
      const near = arr.filter(b=>b!==a)
        .map(b=>({b, d:a.position.distanceTo(b.position)}))
        .sort((x,y)=>x.d-y.d).slice(0,2);
      near.forEach(({b})=>{
        const g = new THREE.BufferGeometry().setFromPoints([a.position, b.position]);
        lineGroup.add(new THREE.Line(g, mat));
      });
    }
  });
}
makeLines();

// Background stars
const bgGeo = new THREE.BufferGeometry();
const bgPos = [];
for (let i=0;i<700;i++) bgPos.push((Math.random()-0.5)*900,(Math.random()-0.5)*450,(Math.random()-0.5)*900);
bgGeo.setAttribute('position', new THREE.Float32BufferAttribute(bgPos, 3));
scene.add(new THREE.Points(bgGeo, new THREE.PointsMaterial({ color:0x444466, size:0.6 })));

// ── Axes: faint labeled lines through the origin + tick marks ──
// Built as 3D geometry so they orbit with the scene. Labels are HTML overlays
// projected onto the axis tips each frame.
const axisGroup = new THREE.Group();
scene.add(axisGroup);
const AXIS = {
  xMax: X_SPAN/2 + 20, yMax: Y_SPAN + 20, zMax: HALF_Z + 20,
};
const axisMat = { x:0xff6b6b, y:0x34d399, z:0x60a5fa };   // time=red, conf=green, topic=blue
['x','y','z'].forEach(ax => {
  const c = axisMat[ax];
  const mat = new THREE.LineBasicMaterial({ color:c, transparent:true, opacity:0.35 });
  const neg = new THREE.Vector3(ax==='x'?-AXIS.xMax:0, ax==='y'?0:0, ax==='z'?-AXIS.zMax:0);
  const pos = new THREE.Vector3(ax==='x'?AXIS.xMax:0, ax==='y'?AXIS.yMax:0, ax==='z'?AXIS.zMax:0);
  axisGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([neg, pos]), mat));
});
// Tick marks on X (time) — 4 ticks
const tickMat = new THREE.LineBasicMaterial({ color:0x666688, transparent:true, opacity:0.4 });
for (let i=0;i<=4;i++){
  const tx = -X_SPAN/2 + (i/4)*X_SPAN;
  axisGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
    [new THREE.Vector3(tx,-4,0), new THREE.Vector3(tx,4,0)]), tickMat));
}
// Tick marks on Y (confidence) — 0,0.5,1.0
[0, 0.5, 1.0].forEach(cv => {
  const ty = cv*Y_SPAN;
  axisGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
    [new THREE.Vector3(-4,ty,0), new THREE.Vector3(4,ty,0)]), tickMat));
});

// HTML labels for axis tips, repositioned each frame via projection.
const labelLayer = document.createElement('div');
labelLayer.style.cssText = 'position:fixed;inset:0;z-index:6;pointer-events:none;';
document.body.appendChild(labelLayer);
function makeLabel(text, color){
  const el = document.createElement('div');
  el.style.cssText = `position:absolute;color:${color};font-size:11px;font-weight:600;
    background:rgba(0,0,0,0.5);padding:2px 7px;border-radius:6px;transform:translate(-50%,-50%);`;
  el.textContent = text;
  labelLayer.appendChild(el);
  return el;
}
const lblX = makeLabel('时间 (旧→新) →', '#ff8a8a');
const lblY = makeLabel('↑ 置信度', '#5eead4');
const lblZ = makeLabel('主题 (深度) ⊙', '#7dd3fc');

// ── Panel UI ──
const st = data.stats;
document.getElementById('stats').innerHTML = `
  <div class="stat"><b>${st.total}</b> 记忆</div>
  <div class="stat"><b>${Object.keys(topics).length}</b> 主题</div>
  <div class="stat"><b>${st.by_status.settled||0}</b> settled</div>
`;
const grid = document.getElementById('g');
const activeTopics = new Set(Object.keys(topics));
Object.entries(topics).sort((a,b)=>b[1].count-a[1].count).forEach(([name, meta])=>{
  const div = document.createElement('div');
  div.className = 'chip';
  div.innerHTML = `<span class="dot" style="background:${meta.color};color:${meta.color}"></span>${name}<span class="ct">${meta.count}</span>`;
  div.onclick = ()=>{
    if (activeTopics.has(name)){ activeTopics.delete(name); div.classList.add('off'); }
    else { activeTopics.add(name); div.classList.remove('off'); }
    stars.forEach(s => { s.visible = activeTopics.has(s.userData.topic); });
  };
  grid.appendChild(div);
});
document.getElementById('toggle').onclick = function(){
  lineGroup.visible = !lineGroup.visible;
  this.textContent = lineGroup.visible ? '隐藏连接线' : '显示连接线';
};
document.getElementById('reset').onclick = ()=>{
  rotX=0.42; rotY=-0.55; dist=360; panX=0; panY=0;
  velRotX=velRotY=velPanX=velPanY=0; updateCam();
};

// Tooltip
const tip = document.getElementById('tip');
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
canvas.addEventListener('pointermove', e=>{
  mouse.x=(e.clientX/innerWidth)*2-1; mouse.y=-(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObjects(stars.filter(s=>s.visible));
  const hit = hits[0];
  if (hit){
    const d = hit.object.userData;
    tip.style.display='block'; tip.style.left=e.clientX+'px'; tip.style.top=e.clientY+'px';
    tip.innerHTML =
      `<div class="topic" style="color:${d.color}">${d.topic}</div>` +
      `<div>${d.text.length>150?d.text.slice(0,150)+'…':d.text}</div>` +
      `<div class="meta"><span class="tag">${d.type}</span><span class="tag">${d.status}</span>` +
      `<span class="tag">conf ${d.confidence}</span><span class="tag">${d.user}</span></div>`;
  } else tip.style.display='none';
});

// Animate
const _proj = new THREE.Vector3();
function projectToScreen(v, el){
  _proj.copy(v).project(camera);
  el.style.display = (_proj.z > 1) ? 'none' : 'block';
  el.style.left = ((_proj.x * 0.5 + 0.5) * innerWidth) + 'px';
  el.style.top = ((-_proj.y * 0.5 + 0.5) * innerHeight) + 'px';
}
function animate(){
  requestAnimationFrame(animate);
  stars.forEach((s,i)=>{ if(s.visible) s.material.opacity = 0.82+Math.sin(Date.now()*0.001+i)*0.12; });

  // Inertia: when not dragging, apply leftover velocity and decay it.
  if (!isDown){
    if (Math.abs(velRotX) > 1e-5 || Math.abs(velRotY) > 1e-5 ||
        Math.abs(velPanX) > 1e-5 || Math.abs(velPanY) > 1e-5){
      rotY += velRotY;
      rotX = Math.max(ROT_MIN, Math.min(ROT_MAX, rotX + velRotX));
      panX += velPanX; panY += velPanY;
      velRotX *= DAMP; velRotY *= DAMP;
      velPanX *= DAMP; velPanY *= DAMP;
      updateCam();
    }
  }

  // Reposition axis labels to their 3D tips each frame
  projectToScreen(new THREE.Vector3(AXIS.xMax, 0, 0), lblX);
  projectToScreen(new THREE.Vector3(0, AXIS.yMax, 0), lblY);
  projectToScreen(new THREE.Vector3(0, 0, AXIS.zMax), lblZ);
  renderer.render(scene, camera);
}
animate();
window.addEventListener('resize', ()=>{ camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight); });
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
    ap = argparse.ArgumentParser(description="Noesis Topic Constellation Viewer")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--db", default="~/.noesis/hot.db")
    args = ap.parse_args()
    Handler.DB_PATH = args.db
    d = load_data(args.db)
    print("="*56)
    print("  Noesis · 思想星座 (Topic Constellation)")
    print("="*56)
    print(f"  DB: {args.db}")
    print(f"  {d['stats'].get('total',0)} memories · {len(d['topics'])} topics")
    print(f"  topics: { {t:m['count'] for t,m in d['topics'].items()} }")
    print(f"\n  → http://localhost:{args.port}\n  Ctrl+C 退出\n"+"-"*56)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
