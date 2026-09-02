#!/usr/bin/env python3
"""Review the stop-button tail-trim detection (lerobot_tail_trim.py) per episode.

Stdlib-only web viewer (same spirit as siemens_armgrab_viewer): composite tail
clip (left|top|right, pre-sliced by siemens_tail_viewer_prep.py), full-episode
gripper / arm-speed / distance-to-park traces with the detected trim point, and
one-click good/bad marks persisted to corrections.json.

Run:
    python scripts/siemens_tail_viewer.py --port 8022 --host 0.0.0.0
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

import numpy as np

logger = logging.getLogger("tail.viewer")

DATA_DIR = Path("/nfs_old/karim/webviewer_data/siemens_simple_d405_v2_tail")

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


def summary() -> dict:
    eps = list(RESULTS.values())
    tails = np.array([e["tail_s"] for e in eps if e["flag"] == "ok"])
    total_s = sum(e["T"] for e in eps) / 30
    hist, edges = np.histogram(tails, bins=24, range=(0, 12))
    return dict(
        n=len(eps),
        ok=int(sum(e["flag"] == "ok" for e in eps)),
        cut_pct=round(100 * tails.sum() / total_s, 1),
        cut_h=round(tails.sum() / 3600, 2),
        total_h=round(total_s / 3600, 2),
        p50=round(float(np.percentile(tails, 50)), 2),
        p95=round(float(np.percentile(tails, 95)), 2),
        hist=hist.tolist(),
        edges=edges.tolist(),
    )


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>tail-trim review — siemens_simple_d405_v2</title>
<style>
:root { --bg:#16181d; --panel:#1e2128; --line:#2c303a; --fg:#d6d9e0; --dim:#8b90a0;
        --left:#5aa9e6; --right:#e6a55a; --ok:#6cc38f; --warn:#e0656f; --acc:#9d8cff; --cut:#e0656f22; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; height:100vh; display:flex; flex-direction:column; }
header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:15px; font-weight:600; }
.chip { color:var(--dim); font-size:12.5px; } .chip b { color:var(--fg); font-variant-numeric:tabular-nums; }
main { flex:1; display:flex; min-height:0; }
#side { width:270px; border-right:1px solid var(--line); display:flex; flex-direction:column; min-height:0; }
#filters { padding:8px; display:flex; gap:6px; flex-wrap:wrap; border-bottom:1px solid var(--line); }
#filters button, #view .btn { background:var(--panel); color:var(--dim); border:1px solid var(--line); border-radius:4px; padding:3px 9px; font-size:12px; cursor:pointer; }
#filters button.on { color:var(--fg); border-color:var(--acc); }
#list { flex:1; overflow-y:auto; font-variant-numeric:tabular-nums; }
.row { padding:5px 12px; display:flex; gap:10px; cursor:pointer; border-bottom:1px solid #1a1d23; font-size:12.5px; }
.row:hover { background:#20242c; } .row.sel { background:#262b36; }
.row .ep { width:64px; color:var(--dim); } .row.sel .ep { color:var(--fg); }
.row .tail { width:52px; text-align:right; }
.row .flag { flex:1; text-align:right; color:var(--warn); font-size:11px; }
.row .flag.ok { color:var(--dim); } .row .flag.good { color:var(--ok); } .row .flag.bad { color:var(--warn); font-weight:700; }
#view { flex:1; overflow-y:auto; padding:14px 18px; display:flex; flex-direction:column; gap:10px; }
video { width:100%; max-width:960px; background:#000; border:1px solid var(--line); border-radius:4px; }
canvas.chart { width:100%; max-width:960px; background:var(--panel); border:1px solid var(--line); border-radius:4px; display:block; }
#bar { display:flex; gap:8px; align-items:center; max-width:960px; flex-wrap:wrap; }
#bar .btn.good.on { color:var(--ok); border-color:var(--ok); } #bar .btn.bad.on { color:var(--warn); border-color:var(--warn); }
#epinfo { color:var(--dim); font-size:12.5px; } #epinfo b { color:var(--fg); }
.legend { color:var(--dim); font-size:11.5px; max-width:960px; }
.legend i { font-style:normal; padding:0 3px; }
kbd { background:var(--panel); border:1px solid var(--line); border-radius:3px; padding:0 4px; font-size:11px; }
</style></head><body>
<header>
  <h1>tail-trim review — siemens_simple_d405_v2</h1>
  <span class="chip" id="sum"></span>
  <canvas id="hist" width="240" height="36" title="tail_s histogram 0–12s"></canvas>
  <span class="chip">keys: <kbd>j</kbd>/<kbd>k</kbd> next/prev · <kbd>t</kbd> to trim · <kbd>g</kbd>/<kbd>b</kbd> mark · <kbd>space</kbd> play</span>
</header>
<main>
  <div id="side">
    <div id="filters">
      <button data-f="all" class="on">all</button><button data-f="ok">ok</button>
      <button data-f="flagged">flagged</button><button data-f="bad">marked bad</button>
      <select id="sort" style="margin-left:auto;background:var(--panel);color:var(--dim);border:1px solid var(--line);border-radius:4px;font-size:12px;">
        <option value="ep">by episode</option><option value="td">tail ↓</option><option value="ta">tail ↑</option>
      </select>
    </div>
    <div id="list"></div>
  </div>
  <div id="view">
    <div id="bar">
      <span id="epinfo"></span>
      <button class="btn" id="totrim">▶ from trim −1.5s</button>
      <button class="btn good" id="mgood">✓ good</button>
      <button class="btn bad" id="mbad">✗ bad trim</button>
    </div>
    <video id="vid" controls muted playsinline></video>
    <canvas id="grip" class="chart" width="960" height="130"></canvas>
    <canvas id="mot" class="chart" width="960" height="130"></canvas>
    <div class="legend">
      grippers: <i style="color:var(--left)">left</i><i style="color:var(--right)">right</i> (1=open) ·
      motion: <i style="color:var(--left)">L speed</i><i style="color:var(--right)">R speed</i>
      <i style="color:var(--ok)">dist→park (scaled)</i> ·
      markers: <i style="color:var(--right)">last close</i> <i style="color:var(--acc)">grippers open</i>
      <i style="color:var(--warn)">trim → cut region</i> · grey band = video clip window · click charts to seek
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let EPS = [], CORR = {}, SUM = null, sel = null, filt = 'all', trace = null;

const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

async function boot() {
  const d = await (await fetch('api/episodes')).json();
  EPS = d.episodes; CORR = d.corrections; SUM = d.summary;
  $('sum').innerHTML = `<b>${SUM.n}</b> eps · <b>${SUM.ok}</b> trimmed · cut <b>${SUM.cut_h}h</b>/${SUM.total_h}h (<b>${SUM.cut_pct}%</b>) · tail p50 <b>${SUM.p50}s</b> p95 <b>${SUM.p95}s</b>`;
  drawHist();
  render();
  if (EPS.length) select(EPS[0].episode);
}

function drawHist() {
  const c = $('hist'), x = c.getContext('2d'), m = Math.max(...SUM.hist);
  x.clearRect(0,0,c.width,c.height);
  SUM.hist.forEach((v,i) => {
    x.fillStyle = css('--acc');
    const w = c.width / SUM.hist.length;
    x.fillRect(i*w, c.height - v/m*(c.height-2), w-1, v/m*(c.height-2));
  });
}

function visible() {
  let l = EPS.slice();
  if (filt === 'ok') l = l.filter(e => e.flag === 'ok');
  if (filt === 'flagged') l = l.filter(e => e.flag !== 'ok');
  if (filt === 'bad') l = l.filter(e => CORR[e.episode] === 'bad');
  const s = $('sort').value;
  if (s === 'td') l.sort((a,b) => b.tail_s - a.tail_s);
  if (s === 'ta') l.sort((a,b) => a.tail_s - b.tail_s);
  if (s === 'ep') l.sort((a,b) => a.episode - b.episode);
  return l;
}

function render() {
  const l = visible();
  $('list').innerHTML = l.map(e => {
    const mark = CORR[e.episode];
    const cls = mark || (e.flag === 'ok' ? 'ok' : '');
    const note = mark || (e.flag === 'ok' ? '' : e.flag.replaceAll('_',' '));
    return `<div class="row ${e.episode===sel?'sel':''}" data-ep="${e.episode}">
      <span class="ep">${String(e.episode).padStart(6,'0')}</span>
      <span class="tail">${e.tail_s.toFixed(1)}s</span>
      <span class="flag ${cls}">${note}</span></div>`;
  }).join('');
  for (const r of $('list').children) r.onclick = () => select(+r.dataset.ep);
}

async function select(ep) {
  sel = ep; render();
  const e = EPS.find(x => x.episode === ep);
  trace = await (await fetch(`api/traces?ep=${ep}`)).json();
  $('epinfo').innerHTML = `<b>ep ${ep}</b> · ${(e.T/e.fps).toFixed(1)}s · tail <b>${e.tail_s}s</b> · ${e.flag}`;
  setMarks();
  const v = $('vid');
  v.src = `api/video?ep=${ep}`;
  v.onloadedmetadata = () => { v.currentTime = Math.max(0, (e.trim - e.clip_start)/e.fps - 1.5); v.play(); };
  drawCharts();
  const r = [...$('list').children].find(x => +x.dataset.ep === ep);
  if (r) r.scrollIntoView({block:'nearest'});
}

function setMarks() {
  const m = CORR[sel];
  $('mgood').classList.toggle('on', m === 'good');
  $('mbad').classList.toggle('on', m === 'bad');
}

function ep() { return EPS.find(x => x.episode === sel); }

function drawCharts(cursorFrame) {
  if (!trace || sel === null) return;
  const e = ep(), st = trace.stride, T = e.T;
  const X = f => f / T * 960;
  for (const [cid, series] of [['grip', ['lg','rg']], ['mot', ['lspd','rspd','dist']]]) {
    const c = $(cid), x = c.getContext('2d');
    x.clearRect(0,0,960,130);
    // clip window band + cut region
    x.fillStyle = '#ffffff0a';
    x.fillRect(X(e.clip_start), 0, X(T) - X(e.clip_start), 130);
    x.fillStyle = css('--cut');
    x.fillRect(X(e.trim), 0, X(T) - X(e.trim), 130);
    const cols = {lg:css('--left'), rg:css('--right'), lspd:css('--left'), rspd:css('--right'), dist:css('--ok')};
    for (const k of series) {
      const d = trace[k];
      const mx = (k === 'lg' || k === 'rg') ? 1.05 : Math.max(...d) * 1.1 + 1e-6;
      x.strokeStyle = cols[k]; x.lineWidth = 1.3; x.beginPath();
      d.forEach((v,i) => { const px = X(i*st), py = 124 - v/mx*118; i ? x.lineTo(px,py) : x.moveTo(px,py); });
      x.stroke();
    }
    for (const [f, col] of [[e.last_close, css('--right')], [e.open_done, css('--acc')], [e.trim, css('--warn')]]) {
      x.strokeStyle = col; x.lineWidth = 1.5;
      x.beginPath(); x.moveTo(X(f), 0); x.lineTo(X(f), 130); x.stroke();
    }
    if (cursorFrame !== undefined) {
      x.strokeStyle = '#fff'; x.lineWidth = 1;
      x.beginPath(); x.moveTo(X(cursorFrame), 0); x.lineTo(X(cursorFrame), 130); x.stroke();
    }
    // time axis ticks every 5 s
    x.fillStyle = css('--dim'); x.font = '10px system-ui';
    for (let t = 0; t < T/30; t += 5) x.fillText(`${t}s`, X(t*30) + 2, 11);
  }
}

$('vid').addEventListener('timeupdate', () => {
  if (sel === null) return;
  drawCharts(ep().clip_start + $('vid').currentTime * ep().fps);
});
for (const cid of ['grip','mot']) $(cid).onclick = evt => {
  const e = ep(); if (!e) return;
  const f = (evt.offsetX / $(cid).clientWidth) * e.T;
  $('vid').currentTime = Math.min(Math.max(0, (f - e.clip_start)/e.fps), $('vid').duration || 1e9);
};
$('totrim').onclick = () => { const e = ep(); $('vid').currentTime = Math.max(0,(e.trim - e.clip_start)/e.fps - 1.5); $('vid').play(); };

async function mark(m) {
  if (sel === null) return;
  const cur = CORR[sel] === m ? null : m;
  await fetch('api/mark', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ep:sel, mark:cur})});
  if (cur) CORR[sel] = cur; else delete CORR[sel];
  setMarks(); render();
}
$('mgood').onclick = () => mark('good');
$('mbad').onclick = () => mark('bad');

document.addEventListener('keydown', evt => {
  if (evt.target.tagName === 'SELECT') return;
  const l = visible(), i = l.findIndex(x => x.episode === sel);
  if (evt.key === 'j' && i < l.length-1) select(l[i+1].episode);
  else if (evt.key === 'k' && i > 0) select(l[i-1].episode);
  else if (evt.key === 't') $('totrim').onclick();
  else if (evt.key === 'g') mark('good');
  else if (evt.key === 'b') mark('bad');
  else if (evt.key === ' ') { evt.preventDefault(); const v = $('vid'); v.paused ? v.play() : v.pause(); }
});
for (const b of $('filters').querySelectorAll('button')) b.onclick = () => {
  filt = b.dataset.f;
  for (const o of $('filters').querySelectorAll('button')) o.classList.toggle('on', o === b);
  render();
};
$('sort').onchange = render;
boot();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "tail-viewer/1.0"

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _video(self, path: Path):
        if not path.exists():
            return self._json({"error": "no clip"}, 404)
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else 0
            end = int(b) if b else size - 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(end - start + 1))

    def do_GET(self):
        try:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/episodes":
                self._json(
                    {"episodes": sorted(RESULTS.values(), key=lambda e: e["episode"]),
                     "corrections": CORRECTIONS, "summary": summary()}
                )
            elif u.path == "/api/traces":
                self._json(TRACES[q["ep"]])
            elif u.path == "/api/video":
                self._video(DATA_DIR / "clips" / f"episode_{int(q['ep']):06d}.mp4")
            else:
                self._json({"error": "no route"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if urlparse(self.path).path != "/api/mark":
            return self._json({"error": "no route"}, 404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with CORR_LOCK:
            if body.get("mark"):
                CORRECTIONS[str(body["ep"])] = body["mark"]
            else:
                CORRECTIONS.pop(str(body["ep"]), None)
            save_corrections()
        self._json({"ok": True})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8022)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    logger.info("tail viewer on http://%s:%d (%d episodes)", args.host, args.port, len(RESULTS))
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
