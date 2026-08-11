#!/usr/bin/env python3
import json, math, random, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# --- Demo data matching your screenshot ---
CLUSTERS = {
    "太空探索": "#ff6b6b",
    "人工智能": "#a78bfa",
    "量子物理": "#22d3ee",
    "生命科学": "#34d399",
    "深海": "#3b82f6",
    "气候": "#fbbf24",
    "艺术": "#f472b6",
    "哲学": "#c4b5fd"
}

def make_demo():
    demos = [
        ("太空探索", "SpaceX星舰第三次试飞成功", 0.9),
        ("太空探索", "詹姆斯韦伯望远镜发现早期星系", 0.85),
        ("人工智能", "大模型多模态能力突破", 0.92),
        ("人工智能", "AI辅助蛋白质折叠预测", 0.88),
        ("人工智能", "开源模型性能接近GPT-4", 0.8),
        ("量子物理", "量子纠缠距离新纪录", 0.87),
        ("量子物理", "拓扑量子计算进展", 0.82),
        ("生命科学", "CRISPR基因编辑临床试验", 0.9),
        ("生命科学", "脑机接口帮助瘫痪患者", 0.86),
        ("深海", "马里亚纳海沟新物种", 0.78),
        ("深海", "深海热液生态系统", 0.75),
        ("气候", "全球碳排放峰值预测", 0.84),
        ("气候", "极地冰盖融化加速", 0.81),
        ("艺术", "生成式AI艺术展览", 0.77),
        ("艺术", "数字艺术NFT市场回暖", 0.72),
        ("哲学", "意识本质的最新讨论", 0.79),
        ("哲学", "技术伦理学新框架", 0.74),
    ]
    mems = []
    now = time.time()
    for i, (cluster, text, imp) in enumerate(demos):
        mems.append({
            "id": i,
            "cluster": cluster,
            "text": text,
            "importance": imp,
            "color": CLUSTERS[cluster],
            "created_at": now - i * 86400
        })
    # Add more random stars to fill out each cluster
    for cluster in CLUSTERS:
        for j in range(12):
            mems.append({
                "id": len(mems),
                "cluster": cluster,
                "text": f"{cluster}研究笔记 {j+1}",
                "importance": random.uniform(0.3, 0.7),
                "color": CLUSTERS[cluster],
                "created_at": now - random.randint(1, 300) * 86400
            })
    return mems

MEMORIES = make_demo()

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Noesis · 有机星系</title>
<style>
  :root { --bg: #000000; --panel: rgba(12,12,22,0.92); --border: rgba(255,255,255,0.12); --text: #fff; --muted: #9aa0b4; }
    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  html, body { height: 100%; overflow: hidden; background: #000; color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Helvetica Neue", sans-serif; }
  #c { position: absolute; inset: 0; }
 .panel { position: fixed; top: 12px; left: 12px; right: 12px; z-index: 10; background: var(--panel); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid var(--border); border-radius: 24px; padding: 18px 18px 14px; }
 .panel h1 { font-size: 20px; font-weight: 700; letter-spacing: 0.2px; }
 .panel.sub { font-size: 13px; color: var(--muted); margin: 6px 0 14px; }
 .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
 .chip { display: flex; align-items: center; gap: 7px; font-size: 14px; padding: 9px 10px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; }
 .dot { width: 11px; height: 11px; border-radius: 50%; box-shadow: 0 0 10px currentColor; }
 .btns { display: flex; gap: 10px; }
 .btn { flex: 1; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14); color: #fff; padding: 12px; font-size: 15px; border-radius: 14px; font-weight: 500; }
 .btn:active { transform: scale(0.97); opacity: 0.8; }
  #tip { position: fixed; z-index: 20; pointer-events: none; background: rgba(20,20,35,0.95); backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.15); border-radius: 14px; padding: 10px 14px; font-size: 14px; max-width: 260px; display: none; transform: translate(12px, 12px); }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div class="panel">
  <h1>8个有机星系 · 可平移缩放</h1>
  <div class="sub">左拖旋转 · 右拖/双指拖动平移 · 滚轮/捏合缩放</div>
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

const canvas = document.getElementById('c');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 2000);
camera.position.set(0, 80, 180);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

// Controls - simple orbit
let isDown = false, lastX = 0, lastY = 0, rotX = 0.3, rotY = 0, dist = 180;
function updateCam() {
  camera.position.x = Math.sin(rotY) * Math.cos(rotX) * dist;
  camera.position.y = Math.sin(rotX) * dist;
  camera.position.z = Math.cos(rotY) * Math.cos(rotX) * dist;
  camera.lookAt(0, 0, 0);
}
updateCam();

canvas.addEventListener('pointerdown', e => { isDown = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('pointerup', () => isDown = false);
canvas.addEventListener('pointermove', e => {
  if (!isDown) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  if (e.buttons === 1) { rotY -= dx * 0.005; rotX = Math.max(-1.2, Math.min(1.2, rotX - dy * 0.005)); }
  lastX = e.clientX; lastY = e.clientY; updateCam();
});
canvas.addEventListener('wheel', e => { dist = Math.max(40, Math.min(400, dist + e.deltaY * 0.2)); updateCam(); });

// Build galaxy
const clusters = {};
Object.entries(data.clusters).forEach(([name, color], i) => {
  const angle = (i / 8) * Math.PI * 2;
  clusters[name] = { x: Math.cos(angle) * 55, z: Math.sin(angle) * 55, y: (Math.random()-0.5)*15, color };
});

// Stars
const stars = [];
const starMat = new THREE.MeshBasicMaterial();
memories.forEach(m => {
  const c = clusters[m.cluster];
  const r = 12 + Math.random() * 18;
  const a = Math.random() * Math.PI * 2;
  const y = (Math.random() - 0.5) * 12;
  const x = c.x + Math.cos(a) * r;
  const z = c.z + Math.sin(a) * r;

  const size = 0.8 + m.importance * 2.2;
  const geo = new THREE.SphereGeometry(size, 12);
  const mat = new THREE.MeshBasicMaterial({
    color: m.color,
    transparent: true,
    opacity: 0.9
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(x, c.y + y, z);
  mesh.userData = m;
  scene.add(mesh);
  stars.push(mesh);

  // Glow
  const glowGeo = new THREE.SphereGeometry(size * 2.2, 12, 12);
  const glowMat = new THREE.MeshBasicMaterial({ color: m.color, transparent: true, opacity: 0.15 });
  const glow = new THREE.Mesh(glowGeo, glowMat);
  glow.position.copy(mesh.position);
  scene.add(glow);
});

// Lines
const lineGroup = new THREE.Group();
scene.add(lineGroup);
function makeLines() {
  lineGroup.clear();
  const lineMat = new THREE.LineBasicMaterial({ color: 0x334466, transparent: true, opacity: 0.25 });
  Object.keys(clusters).forEach(clusterName => {
    const clusterStars = stars.filter(s => s.userData.cluster === clusterName);
    for (let i = 0; i < clusterStars.length; i++) {
      const a = clusterStars[i];
      // Connect to 2-3 nearest in same cluster
      const nearest = clusterStars
       .filter(b => b!== a)
       .map(b => ({ b, d: a.position.distanceTo(b.position) }))
       .sort((x, y) => x.d - y.d)
       .slice(0, 3);
      nearest.forEach(({ b }) => {
        if (Math.random() > 0.6) return;
        const geo = new THREE.BufferGeometry().setFromPoints([a.position, b.position]);
        lineGroup.add(new THREE.Line(geo, lineMat));
      });
    }
  });
}
makeLines();

// Background stars
const bgGeo = new THREE.BufferGeometry();
const bgPos = [];
for (let i = 0; i < 800; i++) {
  bgPos.push((Math.random()-0.5)*800, (Math.random()-0.5)*400, (Math.random()-0.5)*800);
}
bgGeo.setAttribute('position', new THREE.Float32BufferAttribute(bgPos, 3));
scene.add(new THREE.Points(bgGeo, new THREE.PointsMaterial({ color: 0x555577, size: 0.8 })));

// UI
const grid = document.getElementById('g');
Object.entries(data.clusters).forEach(([name, color]) => {
  const div = document.createElement('div');
  div.className = 'chip';
  div.innerHTML = `<span class="dot" style="background:${color};color:${color}"></span>${name}`;
  grid.appendChild(div);
});

document.getElementById('toggle').onclick = function() {
  lineGroup.visible =!lineGroup.visible;
  this.textContent = lineGroup.visible? '隐藏星系间连线' : '显示星系间连线';
};
document.getElementById('reset').onclick = () => { rotX = 0.3; rotY = 0; dist = 180; updateCam(); };

// Tooltip
const tip = document.getElementById('tip');
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
canvas.addEventListener('pointermove', e => {
  mouse.x = (e.clientX / innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / innerHeight) * 2 + 1;
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(stars)[0];
  if (hit) {
    tip.style.display = 'block';
    tip.style.left = e.clientX + 'px';
    tip.style.top = e.clientY + 'px';
    tip.innerHTML = `<div style="color:${hit.object.userData.color};font-size:12px;margin-bottom:4px">${hit.object.userData.cluster}</div>${hit.object.userData.text}`;
  } else {
    tip.style.display = 'none';
  }
});

// Animate
function animate() {
  requestAnimationFrame(animate);
  stars.forEach((s, i) => {
    s.material.opacity = 0.8 + Math.sin(Date.now() * 0.001 + i) * 0.15;
  });
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>
"""

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "memories": MEMORIES,
                "clusters": CLUSTERS
            }, ensure_ascii=False).encode())
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode())

    def log_message(self, *args): pass

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8788
    print(f"→ http://localhost:{port}")
    print(f" {len(MEMORIES)} memories, 8 clusters")
    HTTPServer(('127.0.0.1', port), H).serve_forever()
