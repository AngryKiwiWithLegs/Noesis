#!/usr/bin/env python3
"""
Noesis Experiment Galaxy Viewer
================================
Same organic-galaxy aesthetic as the demo (Three.js, glassmorphism panel,
orbit controls, hover tooltips), but populated with REAL experiment data
from ~/.noesis/hot.db.

Each experiment persona (prof_1, zhang_wei, han_meimei, …) becomes a galaxy.
Each captured thought (position / preference / identity / event) becomes a star,
sized by status: settled > provisional > tentative.

Usage:
    python3 serve_galaxy.py [--port 8789] [--db ~/.noesis/hot.db]
Then open http://localhost:8789
"""
import argparse
import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).parent

# ── Distinct galaxy colors per persona. Generated deterministically from a
#    fixed palette + golden-angle hue rotation so every persona gets a
#    perceptually-distinct color, regardless of count.
_PALETTE = [
    "#ff6b6b", "#a78bfa", "#22d3ee", "#34d399", "#3b82f6", "#fbbf24",
    "#f472b6", "#c4b5fd", "#fb923c", "#2dd4bf", "#818cf8", "#facc15",
    "#f87171", "#4ade80", "#60a5fa", "#e879f9", "#fde047", "#67e8f9",
    "#a3e635", "#fca5a5", "#bef264", "#7dd3fc", "#ddd6fe", "#fdba74",
    "#bbf7d0", "#fbcfe8", "#fef08a", "#c7d2fe",
]


def _assign_colors(personas: list[str]) -> dict[str, str]:
    """Deterministic persona → color. Golden-angle rotation for even spread."""
    colors = {}
    for i, p in enumerate(sorted(personas)):
        colors[p] = _PALETTE[i % len(_PALETTE)]
    return colors


def load_experiment_data(db_path: str) -> dict:
    """Read all experiment personas + memories from the hot store."""
    db_path = os.path.expanduser(db_path)
    if not Path(db_path).exists():
        return {"memories": [], "clusters": {}, "stats": {}}

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, hash_id, text, type, status, confidence, "
        "       user_id, source_tool, topic_cluster, created_at "
        "FROM items ORDER BY created_at ASC"
    ).fetchall()
    con.close()

    # Each persona is a galaxy. Exclude the bare "default" user if it's
    # mixed noise; keep personas that look like experiment subjects.
    personas = sorted({r["user_id"] for r in rows if r["user_id"]})
    colors = _assign_colors(personas)

    # Star size is driven by status (the core Noesis lifecycle concept):
    # settled gl largest, tentative smallest.
    _SIZE = {"settled": 1.0, "provisional": 0.78, "tentative": 0.5}

    memories = []
    for r in rows:
        uid = r["user_id"]
        memories.append({
            "id": r["id"],
            "hash": (r["hash_id"] or "")[:12],
            "cluster": uid,                # persona = galaxy
            "text": r["text"] or "",
            "type": r["type"] or "unknown",
            "status": r["status"] or "tentative",
            "confidence": round(r["confidence"] or 0.0, 3),
            "topic": r["topic_cluster"] or "general",
            "source_tool": r["source_tool"] or "",
            "color": colors.get(uid, "#888"),
            "importance": _SIZE.get(r["status"], 0.5),
            "created_at": r["created_at"],
        })

    # Friendly persona labels for the panel
    _LABEL = {
        "zhang_wei": "张伟 (后端)", "han_meimei": "韩梅梅 (前端)",
        "li_lei": "李雷 (全栈)",
    }
    labeled = {}
    for p in personas:
        labeled[_LABEL.get(p, p)] = colors[p]

    # Headline stats for the panel
    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for m in memories:
        by_status[m["status"]] = by_status.get(m["status"], 0) + 1
        by_type[m["type"]] = by_type.get(m["type"], 0) + 1

    return {
        "memories": memories,
        "clusters": labeled,
        "stats": {
            "total": len(memories),
            "personas": len(personas),
            "by_status": by_status,
            "by_type": by_type,
        },
    }


HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Noesis · 实验星系</title>
<style>
  :root { --bg:#000; --panel:rgba(12,12,22,0.92); --border:rgba(255,255,255,0.12); --text:#fff; --muted:#9aa0b4; }
  * { box-sizing:border-box; margin:0; padding:0; -webkit-tap-highlight-color:transparent; }
  html, body { height:100%; overflow:hidden; background:#000; color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Helvetica Neue",sans-serif; }
  #c { position:absolute; inset:0; }
  .panel { position:fixed; top:12px; left:12px; right:12px; z-index:10;
    background:var(--panel); backdrop-filter:blur(24px); -webkit-backdrop-filter:blur(24px);
    border:1px solid var(--border); border-radius:24px; padding:18px 18px 14px;
    max-height:calc(100vh - 24px); overflow-y:auto; }
  .panel h1 { font-size:20px; font-weight:700; letter-spacing:0.2px; }
  .sub { font-size:13px; color:var(--muted); margin:6px 0 10px; }
  .stats { display:flex; gap:18px; flex-wrap:wrap; margin-bottom:12px; font-size:13px; }
  .stat b { color:#fff; font-size:18px; font-weight:700; }
  .stat span { color:var(--muted); display:block; font-size:11px; }
  .legend { display:flex; gap:6px; flex-wrap:wrap; font-size:12px; margin-bottom:10px; }
  .lg { padding:3px 9px; border-radius:999px; background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.1); }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(110px,1fr)); gap:7px; margin-bottom:12px; }
  .chip { display:flex; align-items:center; gap:6px; font-size:12px; padding:6px 8px;
    background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:10px; }
  .dot { width:9px; height:9px; border-radius:50%; box-shadow:0 0 8px currentColor; flex:none; }
  .btns { display:flex; gap:10px; }
  .btn { flex:1; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.14);
    color:#fff; padding:11px; font-size:14px; border-radius:14px; font-weight:500; cursor:pointer; }
  .btn:active { transform:scale(0.97); opacity:0.8; }
  #tip { position:fixed; z-index:20; pointer-events:none; background:rgba(20,20,35,0.95);
    backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.15); border-radius:14px;
    padding:10px 14px; font-size:13px; max-width:300px; display:none; transform:translate(12px,12px); }
  #tip .meta { font-size:11px; color:var(--muted); margin-top:6px; display:flex; gap:8px; flex-wrap:wrap; }
  #tip .tag { padding:1px 6px; border-radius:6px; background:rgba(255,255,255,0.1); }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div class="panel">
  <h1 id="title">实验星系</h1>
  <div class="sub">左拖旋转 · 右拖平移 · 滚轮缩放 · 悬停查看记忆</div>
  <div class="stats" id="stats"></div>
  <div class="legend" id="legend"></div>
  <div class="grid" id="g"></div>
  <div class="btns">
    <button class="btn" id="toggle">隐藏星系间连线</button>
    <button class="btn" id="reset">收起</button>
  </div>
</div>
<div id="tip"></div>
<script type="module">
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

const data = await fetch('/api').then(r => r.json());
const memories = data.memories;
const clusters = data.clusters;
const ckeys = Object.keys(clusters);
const N = ckeys.length;

const canvas = document.getElementById('c');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 3000);
camera.position.set(0, 90, 210);
const renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

// Orbit controls (left-drag rotate, wheel zoom)
let isDown=false, lastX=0, lastY=0, rotX=0.32, rotY=0, dist=210;
function updateCam(){
  camera.position.x = Math.sin(rotY)*Math.cos(rotX)*dist;
  camera.position.y = Math.sin(rotX)*dist;
  camera.position.z = Math.cos(rotY)*Math.cos(rotX)*dist;
  camera.lookAt(0,0,0);
}
updateCam();
canvas.addEventListener('pointerdown', e=>{ isDown=true; lastX=e.clientX; lastY=e.clientY; });
window.addEventListener('pointerup', ()=> isDown=false);
canvas.addEventListener('pointermove', e=>{
  if(!isDown) return;
  const dx=e.clientX-lastX, dy=e.clientY-lastY;
  if(e.buttons===1){ rotY-=dx*0.005; rotX=Math.max(-1.2,Math.min(1.2,rotX-dy*0.005)); }
  lastX=e.clientX; lastY=e.clientY; updateCam();
});
canvas.addEventListener('wheel', e=>{ dist=Math.max(50,Math.min(600,dist+e.deltaY*0.2)); updateCam(); });

// Place each persona as a galaxy center, around a big ring.
// Radius scales with count so more personas spread out.
const ringR = 55 + N * 3.2;
const centers = {};
ckeys.forEach((name, i) => {
  const a = (i / N) * Math.PI * 2;
  centers[name] = {
    x: Math.cos(a) * ringR,
    z: Math.sin(a) * ringR,
    y: (Math.random()-0.5) * 14,
    color: clusters[name],
  };
});

// Stars
const stars = [];
memories.forEach(m => {
  const c = centers[m.cluster] || centers[ckeys[0]];
  const r = 9 + Math.random() * 16;
  const a = Math.random() * Math.PI * 2;
  const y = (Math.random()-0.5) * 10;
  const x = c.x + Math.cos(a)*r;
  const z = c.z + Math.sin(a)*r;
  const size = 0.7 + m.importance * 2.1;
  const geo = new THREE.SphereGeometry(size, 12);
  const mat = new THREE.MeshBasicMaterial({ color:m.color, transparent:true, opacity:0.9 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(x, c.y + y, z);
  mesh.userData = m;
  scene.add(mesh);
  stars.push(mesh);
  // glow
  const glow = new THREE.Mesh(
    new THREE.SphereGeometry(size*2.2, 12, 12),
    new THREE.MeshBasicMaterial({ color:m.color, transparent:true, opacity:0.14 })
  );
  glow.position.copy(mesh.position);
  scene.add(glow);
});

// Intra-galaxy lines (connect nearby stars within the same persona)
const lineGroup = new THREE.Group();
scene.add(lineGroup);
function makeLines(){
  lineGroup.clear();
  const lineMat = new THREE.LineBasicMaterial({ color:0x334466, transparent:true, opacity:0.22 });
  Object.keys(centers).forEach(cn => {
    const cs = stars.filter(s => s.userData.cluster === cn);
    for (let i=0;i<cs.length;i++){
      const a = cs[i];
      const near = cs.filter(b=>b!==a)
        .map(b=>({b, d:a.position.distanceTo(b.position)}))
        .sort((x,y)=>x.d-y.d).slice(0,3);
      near.forEach(({b})=>{
        if(Math.random()>0.6) return;
        const g = new THREE.BufferGeometry().setFromPoints([a.position, b.position]);
        lineGroup.add(new THREE.Line(g, lineMat));
      });
    }
  });
}
makeLines();

// Background stars
const bgGeo = new THREE.BufferGeometry();
const bgPos = [];
for (let i=0;i<900;i++) bgPos.push((Math.random()-0.5)*900,(Math.random()-0.5)*450,(Math.random()-0.5)*900);
bgGeo.setAttribute('position', new THREE.Float32BufferAttribute(bgPos, 3));
scene.add(new THREE.Points(bgGeo, new THREE.PointsMaterial({ color:0x555577, size:0.7 })));

// ── Panel UI ──
document.getElementById('title').textContent =
  `${data.stats.personas}个实验人格 · ${data.stats.total}条记忆`;
const st = data.stats;
document.getElementById('stats').innerHTML = `
  <div class="stat"><b>${st.total}</b><span>总记忆</span></div>
  <div class="stat"><b>${st.personas}</b><span>人格</span></div>
  <div class="stat"><b>${st.by_status.settled||0}</b><span>settled</span></div>
  <div class="stat"><b>${st.by_status.provisional||0}</b><span>provisional</span></div>
  <div class="stat"><b>${st.by_status.tentative||0}</b><span>tentative</span></div>
`;
document.getElementById('legend').innerHTML = `
  <span class="lg">● settled (大)</span>
  <span class="lg">● provisional (中)</span>
  <span class="lg">● tentative (小)</span>
`;
const grid = document.getElementById('g');
Object.entries(clusters).forEach(([name, color]) => {
  const div = document.createElement('div');
  div.className = 'chip';
  div.innerHTML = `<span class="dot" style="background:${color};color:${color}"></span>${name}`;
  grid.appendChild(div);
});
document.getElementById('toggle').onclick = function(){
  lineGroup.visible = !lineGroup.visible;
  this.textContent = lineGroup.visible ? '隐藏星系间连线' : '显示星系间连线';
};
document.getElementById('reset').onclick = ()=>{ rotX=0.32; rotY=0; dist=210; updateCam(); };

// Tooltip
const tip = document.getElementById('tip');
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
canvas.addEventListener('pointermove', e=>{
  mouse.x = (e.clientX/innerWidth)*2-1;
  mouse.y = -(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(stars)[0];
  if (hit){
    const d = hit.object.userData;
    tip.style.display='block';
    tip.style.left=e.clientX+'px'; tip.style.top=e.clientY+'px';
    tip.innerHTML =
      `<div style="color:${d.color};font-size:12px;margin-bottom:4px">${d.cluster}</div>` +
      `<div>${d.text.length>140 ? d.text.slice(0,140)+'…' : d.text}</div>` +
      `<div class="meta">` +
        `<span class="tag">${d.type}</span>` +
        `<span class="tag">${d.status}</span>` +
        `<span class="tag">conf ${d.confidence}</span>` +
      `</div>`;
  } else { tip.style.display='none'; }
});

// Animate (gentle pulse)
function animate(){
  requestAnimationFrame(animate);
  stars.forEach((s,i)=>{ s.material.opacity = 0.8 + Math.sin(Date.now()*0.001+i)*0.13; });
  renderer.render(scene, camera);
}
animate();
window.addEventListener('resize', ()=>{
  camera.aspect = innerWidth/innerHeight;
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
            payload = load_experiment_data(self.DB_PATH)
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
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
    ap = argparse.ArgumentParser(description="Noesis Experiment Galaxy Viewer")
    ap.add_argument("--port", type=int, default=8789)
    ap.add_argument("--db", default="~/.noesis/hot.db")
    args = ap.parse_args()
    Handler.DB_PATH = args.db

    data = load_experiment_data(args.db)
    s = data["stats"]
    print("=" * 56)
    print("  Noesis · 实验星系 (Experiment Galaxy Viewer)")
    print("=" * 56)
    print(f"  DB: {args.db}")
    print(f"  {s.get('total',0)} memories · {s.get('personas',0)} personas")
    if s.get("by_status"):
        print(f"  status: {s['by_status']}")
    print(f"\n  → http://localhost:{args.port}\n  Ctrl+C 退出")
    print("-" * 56)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
