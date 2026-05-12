#!/usr/bin/env bash
# setup.sh — Noesis 安装脚本
# 用法: bash setup.sh
set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }

echo ""
echo "  ███╗  ██╗ ██████╗ ███████╗███████╗██╗███████╗"
echo "  ████╗ ██║██╔═══██╗██╔════╝██╔════╝██║██╔════╝"
echo "  ██╔██╗██║██║   ██║█████╗  ███████╗██║███████╗"
echo "  ██║╚████║██║   ██║██╔══╝  ╚════██║██║╚════██║"
echo "  ██║ ╚███║╚██████╔╝███████╗███████║██║███████║"
echo "  ╚═╝  ╚══╝ ╚═════╝ ╚══════╝╚══════╝╚═╝╚══════╝"
echo ""
echo "  Your thinking layer. Yours forever."
echo "  ─────────────────────────────────────────────"
echo ""

# ── Python version ────────────────────────────────────────────────────────────
python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" \
  && ok "Python $(python3 --version | cut -d' ' -f2)" \
  || fail "Python 3.11+ required. Current: $(python3 --version)"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "→ Creating virtual environment …"
  python3 -m venv .venv
fi
source .venv/bin/activate
ok "Virtual environment"

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "→ Installing dependencies (first run downloads ~250MB of models) …"
pip install --quiet --upgrade pip

pip install --quiet \
  sqlite-vec \
  sentence-transformers \
  rank-bm25 \
  anthropic \
  fastapi \
  "uvicorn[standard]" \
  httpx \
  pyyaml \
  click \
  pytest \
  pytest-asyncio

pip install --quiet -e .

# ── Verify imports ────────────────────────────────────────────────────────────
echo ""
echo "→ Verifying installation …"
python3 - << 'PYEOF'
import sys

checks = [
    ("sqlite_vec",             "sqlite-vec"),
    ("sentence_transformers",  "sentence-transformers"),
    ("rank_bm25",              "rank-bm25"),
    ("anthropic",              "anthropic"),
    ("fastapi",                "fastapi"),
    ("yaml",                   "pyyaml"),
    ("click",                  "click"),
]

all_ok = True
for module, pkg in checks:
    try:
        __import__(module)
        print(f"  \033[0;32m✓\033[0m {pkg}")
    except ImportError:
        print(f"  \033[0;31m✗\033[0m {pkg}  ← pip install {pkg}")
        all_ok = False

if not all_ok:
    sys.exit(1)
PYEOF

ok "All dependencies installed"

# ── Week 1 tests ──────────────────────────────────────────────────────────────
echo ""
echo "→ Running Week 1 acceptance tests …"
echo "  (First run downloads the embedding model ~80MB)"
echo ""
python3 -m pytest tests/test_latency.py -v --tb=short

echo ""
echo "─────────────────────────────────────────────"
echo ""
ok "Noesis is ready."
echo ""
echo "  Quick start:"
echo "    source .venv/bin/activate"
echo ""
echo "  Python:"
echo "    from noesis.memory.main import Memory"
echo "    m = Memory.from_config({'vector_store': {'config': {'db_path': '~/.noesis/hot.db'}},"
echo "                            'embedder': {'config': {'model': 'all-MiniLM-L6-v2'}},"
echo "                            'cold_store': {'config': {'vault_path': '~/NoesisVault'}}})"
echo "    m.add('你好，我是张三', user_id='me')"
echo "    print(m.build_context('我是谁', user_id='me'))"
echo ""
echo "  Tests:"
echo "    pytest tests/ -v"
echo ""
