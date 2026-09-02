#!/usr/bin/env python3
"""Review/correct the arm-grab-first classifications of siemens_simple_d405.

Stdlib-only web viewer (same spirit as icrrt's webui): per-episode composite video
(left cam | top | right cam, pre-sliced under WEB_ROOT), gripper + arm-speed traces
with the detected events, and one-click label corrections persisted to
corrections.json (atomic writes) next to results.json.

Run:
    python scripts/siemens_armgrab_viewer.py --port 8021 --host 0.0.0.0
"""

import argparse
import json
import logging
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("armgrab.viewer")

DATA_DIR = Path("/nfs_old/karim/webviewer_data/siemens_simple_d405_armgrab")
VIDEO_ROOT = Path("/nfs_old/karim/webviewer_data/siemens_simple_d405")

RESULTS = json.loads((DATA_DIR / "results.json").read_text())
TRACES = json.loads((DATA_DIR / "traces.json").read_text())
CORR_PATH = DATA_DIR / "corrections.json"
CORR_LOCK = threading.Lock()
CORRECTIONS = json.loads(CORR_PATH.read_text()) if CORR_PATH.exists() else {}


def save_corrections() -> None:
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, prefix=".corr")
    with os.fdopen(fd, "w") as f:
        json.dump(CORRECTIONS, f, indent=1)
    os.replace(tmp, CORR_PATH)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>arm-grab review — siemens_simple_d405</title>
<style>
:root { --bg:#16181d; --panel:#1e2128; --line:#2c303a; --fg:#d6d9e0; --dim:#8b90a0;
        --left:#5aa9e6; --right:#e6a55a; --ok:#6cc38f; --warn:#e0656f; --acc:#9d8cff; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; height:100vh; display:flex; flex-direction:column; }
header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }
header h1 { font-size:15px; font-weight:600; letter-spacing:.02em; }
.chip { color:var(--dim); font-size:12.5px; } .chip b { color:var(--fg); font-variant-numeric:tabular-nums; }
.chip b.l { color:var(--left); } .chip b.r { color:var(--right); } .chip b.c { color:var(--acc); }
main { flex:1; display:flex; min-height:0; }
#side { width:300px; border-right:1px solid var(--line); display:flex; flex-direction:column; min-height:0; }
#filters { padding:8px; display:flex; gap:6px; flex-wrap:wrap; border-bottom:1px solid var(--line); }
#filters button { background:var(--panel); color:var(--dim); border:1px solid var(--line); border-radius:4px;
                  padding:3px 9px; font-size:12px; cursor:pointer; }
#filters button.on { color:var(--fg); border-color:var(--acc); }
#list { flex:1; overflow-y:auto; font-variant-numeric:tabular-nums; }
.row { padding:5px 12px; display:flex; gap:10px; cursor:pointer; border-bottom:1px solid #1a1d23; font-size:12.5px; }
.row:hover { background:var(--panel); } .row.sel { background:#262a34; }
.row .ep { width:70px; color:var(--dim); } .row .lab { width:44px; font-weight:600; }
.lab.left { color:var(--left); } .lab.right { color:var(--right); } .lab.unsure,.lab.unknown { color:var(--warn); }
.row .conf { color:var(--dim); width:40px; } .row .corr { color:var(--acc); }
#panel { flex:1; display:flex; flex-direction:column; padding:14px 18px; gap:10px; min-width:0; overflow-y:auto; }
video { width:100%; max-width:1010px; background:#000; border-radius:6px; }
canvas { width:100%; max-width:1010px; height:170px; background:var(--panel); border-radius:6px; }
#ctl { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
#ctl button { border:1px solid var(--line); background:var(--panel); color:var(--fg); border-radius:5px;
              padding:6px 14px; font-size:13px; cursor:pointer; }
#ctl button.L { border-color:var(--left); } #ctl button.R { border-color:var(--right); }
#ctl button.U { border-color:var(--warn); }
#ctl button.set { outline:2px solid var(--acc); }
#meta { color:var(--dim); font-size:12.5px; } #meta b { color:var(--fg); }
kbd { background:var(--panel); border:1px solid var(--line); border-radius:3px; padding:0 5px; font-size:11px; color:var(--dim); }
.hint { color:var(--dim); font-size:12px; }
</style></head><body>
<header>
  <h1>arm-grab-first review</h1>
  <span class="chip">auto: <b class="r" id="nR">0</b> right · <b class="l" id="nL">0</b> left · <b id="nU">0</b> unknown</span>
  <span class="chip">corrected: <b class="c" id="nC">0</b></span>
  <span class="chip hint"><kbd>L</kbd>/<kbd>R</kbd> set label · <kbd>U</kbd> unsure · <kbd>X</kbd> clear · <kbd>↑</kbd><kbd>↓</kbd> episode · <kbd>space</kbd> play</span>
</header>
<main>
 <div id="side">
  <div id="filters">
    <button data-f="all" class="on">all</button>
    <button data-f="lowconf">low conf</button>
    <button data-f="left">auto left</button>
    <button data-f="corrected">corrected</button>
    <button data-f="disagree">disagree</button>
  </div>
  <div id="list"></div>
 </div>
 <div id="panel">
  <video id="vid" controls muted playsinline></video>
  <canvas id="chart" width="1010" height="170"></canvas>
  <div id="ctl">
    <button class="L" data-set="left">◀ Left grabs first</button>
    <button class="R" data-set="right">Right grabs first ▶</button>
    <button class="U" data-set="unsure">Unsure</button>
    <button data-set="">Clear correction</button>
    <span id="meta"></span>
  </div>
 </div>
</main>
<script>
let EPS = [], CORR = {}, view = [], sel = -1, filt = 'all';
const $ = id => document.getElementById(id);

async function load() {
  const d = await (await fetch('api/episodes')).json();
  EPS = d.episodes; CORR = d.corrections;
  applyFilter();
  if (view.length) select(0);
}
function counts() {
  const c = {left:0, right:0, unknown:0};
  EPS.forEach(e => c[e.label] = (c[e.label]||0) + 1);
  $('nR').textContent = c.right; $('nL').textContent = c.left; $('nU').textContent = c.unknown;
  $('nC').textContent = Object.keys(CORR).length;
}
function applyFilter() {
  counts();
  view = EPS.filter(e => {
    const c = CORR[e.episode];
    if (filt === 'lowconf') return e.confidence < 0.9;
    if (filt === 'left') return e.label === 'left';
    if (filt === 'corrected') return !!c;
    if (filt === 'disagree') return c && c.label && c.label !== e.label;
    return true;
  });
  if (filt === 'lowconf') view.sort((a,b) => a.confidence - b.confidence);
  renderList();
}
function renderList() {
  $('list').innerHTML = view.map((e,i) => {
    const c = CORR[e.episode];
    return `<div class="row ${i===sel?'sel':''}" data-i="${i}">
      <span class="ep">${String(e.episode).padStart(6,'0')}</span>
      <span class="lab ${e.label}">${e.label[0].toUpperCase()}</span>
      <span class="conf">${e.confidence.toFixed(2)}</span>
      <span class="corr">${c ? '→ ' + (c.label || '·') : ''}</span></div>`;
  }).join('');
}
function select(i) {
  sel = i; renderList();
  const e = view[i]; if (!e) return;
  const r = $('list').querySelector(`[data-i="${i}"]`); if (r) r.scrollIntoView({block:'nearest'});
  $('vid').src = `api/video?ep=${e.episode}`;
  drawChart(null);
  fetch(`api/traces?ep=${e.episode}`).then(r => r.json()).then(t => { e._tr = t; drawChart(e); });
  const c = CORR[e.episode];
  $('meta').innerHTML = `ep <b>${e.episode}</b> · auto <b>${e.label}</b> (conf ${e.confidence.toFixed(2)})`
    + ` · R closes <b>${e.close_right!=null ? (e.close_right/30).toFixed(1)+'s' : '—'}</b>`
    + ` · L closes <b>${e.close_left!=null ? (e.close_left/30).toFixed(1)+'s' : '—'}</b>`
    + ` · staging @ <b>${e.onset!=null ? (e.onset/30).toFixed(1)+'s' : '—'}</b>`
    + (c ? ` · corrected → <b>${c.label || 'cleared'}</b>` : '');
  document.querySelectorAll('#ctl button').forEach(b =>
    b.classList.toggle('set', !!c && b.dataset.set === c.label));
}
function drawChart(e) {
  const cv = $('chart'), g = cv.getContext('2d');
  g.clearRect(0,0,cv.width,cv.height);
  if (!e || !e._tr) return;
  const t = e._tr, n = t.grip_l.length, W = cv.width, H = cv.height, hz = t.hz;
  const x = i => i / (n-1) * (W-20) + 10;
  const spdMax = Math.max(1e-6, ...t.spd_l, ...t.spd_r);
  const yG = v => 12 + (1-v) * (H*0.5 - 18);          // grippers: top half
  const yS = v => H*0.55 + (1 - v/spdMax) * (H*0.4);  // speeds: bottom half
  const line = (arr, y, col, w) => { g.strokeStyle = col; g.lineWidth = w; g.beginPath();
    arr.forEach((v,i) => i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))); g.stroke(); };
  g.strokeStyle = '#2c303a'; g.strokeRect(.5,.5,W-1,H-1);
  g.fillStyle = '#8b90a0'; g.font = '11px system-ui';
  g.fillText('grippers (1=open)', 14, 22); g.fillText('arm speed rad/s', 14, H*0.55 + 14);
  line(t.grip_l, yG, '#5aa9e6', 1.6); line(t.grip_r, yG, '#e6a55a', 1.6);
  line(t.spd_l, yS, '#5aa9e666', 1.2); line(t.spd_r, yS, '#e6a55a66', 1.2);
  const mark = (fr, col, lbl) => { if (fr == null) return; const X = x(fr/30*hz);
    g.strokeStyle = col; g.setLineDash([4,3]); g.beginPath(); g.moveTo(X,8); g.lineTo(X,H-8); g.stroke();
    g.setLineDash([]); g.fillStyle = col; g.fillText(lbl, X+3, H-12); };
  mark(e.onset, '#9d8cff', 'staging'); mark(e.close_right, '#e6a55a', 'R close'); mark(e.close_left, '#5aa9e6', 'L close');
  const v = $('vid');
  if (!v.paused || v.currentTime > 0) {
    const X = x(v.currentTime * hz); g.strokeStyle = '#d6d9e0'; g.beginPath(); g.moveTo(X,0); g.lineTo(X,H); g.stroke();
  }
}
$('vid').addEventListener('timeupdate', () => drawChart(view[sel]));
$('chart').addEventListener('click', ev => {
  const e = view[sel]; if (!e || !e._tr) return;
  const frac = (ev.offsetX - 10) / ($('chart').width - 20);
  $('vid').currentTime = Math.max(0, frac * (e._tr.grip_l.length-1) / e._tr.hz);
});
$('list').addEventListener('click', ev => { const r = ev.target.closest('.row'); if (r) select(+r.dataset.i); });
$('filters').addEventListener('click', ev => { const b = ev.target.closest('button'); if (!b) return;
  document.querySelectorAll('#filters button').forEach(o => o.classList.remove('on'));
  b.classList.add('on'); filt = b.dataset.f; sel = -1; applyFilter(); if (view.length) select(0); });
async function correct(label) {
  const e = view[sel]; if (!e) return;
  await fetch('api/correct', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({episode: e.episode, label}) });
  if (label) CORR[e.episode] = {label}; else delete CORR[e.episode];
  applyFilter(); select(Math.min(sel, view.length-1));
}
document.querySelectorAll('#ctl button').forEach(b => b.onclick = () => correct(b.dataset.set));
document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT') return;
  if (ev.key === 'ArrowDown') { ev.preventDefault(); if (sel < view.length-1) select(sel+1); }
  else if (ev.key === 'ArrowUp') { ev.preventDefault(); if (sel > 0) select(sel-1); }
  else if (ev.key === ' ') { ev.preventDefault(); const v = $('vid'); v.paused ? v.play() : v.pause(); }
  else if (ev.key === 'l' || ev.key === 'L') correct('left');
  else if (ev.key === 'r' || ev.key === 'R') correct('right');
  else if (ev.key === 'u' || ev.key === 'U') correct('unsure');
  else if (ev.key === 'x' || ev.key === 'X') correct('');
});
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "armgrab-viewer/1.0"

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _video(self, path: Path):
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b), size - 1) if (a and b) else end
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/episodes":
                slim = [{k: r[k] for k in ("episode", "label", "confidence", "close_left", "close_right", "onset")}
                        for r in RESULTS]
                self._json({"episodes": slim, "corrections": CORRECTIONS})
            elif u.path == "/api/traces":
                self._json(TRACES[q["ep"]])
            elif u.path == "/api/video":
                self._video(VIDEO_ROOT / f"episode_{int(q['ep']):06d}" / "top_camera-images-rgb.mp4")
            else:
                self._json({"error": "no route"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            logger.exception("GET %s", self.path)
            try:
                self._json({"error": str(e)}, 500)
            except Exception:  # noqa: BLE001
                pass

    def do_POST(self):
        if urlparse(self.path).path != "/api/correct":
            return self._json({"error": "no route"}, 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        ep, label = str(int(body["episode"])), body.get("label") or ""
        with CORR_LOCK:
            if label:
                CORRECTIONS[ep] = {"label": label}
            else:
                CORRECTIONS.pop(ep, None)
            save_corrections()
        self._json({"ok": True, "n_corrections": len(CORRECTIONS)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8021)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("episodes: %d, corrections: %d", len(RESULTS), len(CORRECTIONS))
    logger.info("serving on http://%s:%d", args.host, args.port)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
