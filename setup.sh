#!/usr/bin/env bash
# setup.sh — Unified System first-time setup
set -e

echo ""
echo "═══════════════════════════════════════════════════"
echo "   Unified AI Face + Cinematic Hand FX — Setup"
echo "═══════════════════════════════════════════════════"
echo ""

# 1. Install Python deps
pip install -r requirements.txt --quiet

# 2. Build face embeddings from data/faces/
echo "[Setup] Building face embeddings…"
python scripts/build_dataset.py

echo ""
echo "[Setup] Done!  Run with:  python main.py"
echo "  Options:"
echo "    --camera N      Camera index (default 0)"
echo "    --threshold F   Match threshold 0–1 (default 0.70)"
echo "    --debug         Show debug overlay"
echo ""
