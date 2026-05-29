import cv2
import yaml
import numpy as np
import threading
import time
import json
import os
from collections import deque
from enum import Enum
from urllib.parse import quote
from shapely.geometry import Polygon, box as shapely_box
from ultralytics import YOLO
from flask import Flask, Response, request, jsonify, render_template_string

try:
    import paho.mqtt.client as mqtt_lib
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# ── Resolution constants ───────────────────────────────────────────────────
D1_W,  D1_H  = 704, 576
CIF_W, CIF_H = 352, 288
CONFIG_PATH   = "config/cameras.yaml"
CLIPS_DIR     = "clips"
os.makedirs(CLIPS_DIR, exist_ok=True)

# ── Shared globals (populated by SmartNVR.run) ────────────────────────────
frame_buffers:  dict = {}   # cam_key → latest annotated JPEG bytes
frame_locks:    dict = {}   # cam_key → threading.Lock
zone_poly_refs: dict = {}   # cam_key → [Polygon|None]  (mutable container)
det_states:     dict = {}   # cam_key → DetectionState


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION STATE — thread-safe bridge between inference and display threads
# ─────────────────────────────────────────────────────────────────────────────
class DetectionState:
    def __init__(self):
        self._lock = threading.Lock()
        self.boxes:     list = []
        self.track_ids: list = []
        self.alert:     bool = False

    def update(self, boxes, track_ids, alert):
        with self._lock:
            self.boxes, self.track_ids, self.alert = boxes, track_ids, alert

    def read(self):
        with self._lock:
            return self.boxes.copy(), self.track_ids.copy(), self.alert


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA STREAM — always-latest frame reader
# ─────────────────────────────────────────────────────────────────────────────
class CameraStream:
    def __init__(self, url, out_w, out_h):
        self.out_w, self.out_h = out_w, out_h
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame   = None
        self.stopped = False
        self._lock   = threading.Lock()

    def start(self, max_fps: int = 0):
        """max_fps > 0 caps how often self.frame is updated (RPi: use 15)."""
        self._min_interval = (1.0 / max_fps) if max_fps > 0 else 0.0
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self):
        last_kept = 0.0
        while not self.stopped:
            # always drain the RTSP buffer — only store at capped rate
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            now = time.time()
            if now - last_kept >= self._min_interval:
                with self._lock:
                    self.frame = cv2.resize(frame, (self.out_w, self.out_h))
                last_kept = now

    def get(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# ALERT STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────
class AlertState(Enum):
    CLEAR    = 0
    DEBOUNCE = 1
    ALERT    = 2
    COOLDOWN = 3

class ZoneAlert:
    DEBOUNCE_FRAMES = 3
    COOLDOWN_SECS   = 15

    def __init__(self):
        self.state          = AlertState.CLEAR
        self.hits           = 0
        self.cooldown_until = 0.0

    def update(self, intruder_ids: list, now: float) -> bool:
        if self.state == AlertState.COOLDOWN:
            if now > self.cooldown_until:
                self.state, self.hits = AlertState.CLEAR, 0
            return False

        if intruder_ids:
            self.hits += 1
            if self.hits >= self.DEBOUNCE_FRAMES:
                self.state = AlertState.ALERT
        else:
            if self.state == AlertState.ALERT:
                self.state          = AlertState.COOLDOWN
                self.cooldown_until = now + self.COOLDOWN_SECS
            else:
                self.hits = 0

        return self.state == AlertState.ALERT


# ─────────────────────────────────────────────────────────────────────────────
# CLIP RECORDER — pre-buffer + post-event recording
# ─────────────────────────────────────────────────────────────────────────────
class ClipRecorder:
    PRE_SECS  = 10
    POST_SECS = 15
    FPS       = 15

    def __init__(self, cam_key):
        self.cam_key   = cam_key
        self.pre       = deque(maxlen=self.PRE_SECS * self.FPS)
        self.writer    = None
        self.rec_until = 0.0

    def feed(self, frame: np.ndarray, ts: float):
        self.pre.append(frame.copy())
        if self.writer:
            self.writer.write(frame)
            if ts >= self.rec_until:
                self.writer.release()
                self.writer = None

    def trigger(self, ts: float):
        if self.writer:
            self.rec_until = ts + self.POST_SECS
            return
        path = os.path.join(CLIPS_DIR, f"{self.cam_key}_{int(ts)}.mp4")
        self.writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*'mp4v'),
            self.FPS, (D1_W, D1_H)
        )
        for f in self.pre:
            self.writer.write(f)
        self.pre.clear()
        self.rec_until = ts + self.POST_SECS


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart NVR</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0a0a0c; color: #e8e8f0; font-family: 'IBM Plex Mono', monospace, monospace; font-size: 13px; }
  header { padding: 14px 24px; border-bottom: 1px solid #1e1e24; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 14px; letter-spacing: .12em; color: #7eb8f7; font-weight: 500; }
  #status-bar { margin-left: auto; font-size: 11px; color: #44445a; }
  .grid { display: flex; flex-wrap: wrap; gap: 16px; padding: 20px; }
  .cam-card { background: #0f0f12; border: 1px solid #1e1e24; border-radius: 6px; overflow: hidden; }
  .cam-card.alert { border-color: #e05c5c; box-shadow: 0 0 12px rgba(224,92,92,.15); }
  .cam-header { padding: 10px 14px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid #1e1e24; }
  .cam-name { font-size: 11px; letter-spacing: .1em; color: #8888a0; flex: 1; }
  .badge { font-size: 10px; letter-spacing: .08em; padding: 3px 8px; border-radius: 3px; border: 1px solid; }
  .badge.clear { color: #6bcb8b; border-color: rgba(107,203,139,.3); background: rgba(107,203,139,.06); }
  .badge.alert { color: #e05c5c; border-color: rgba(224,92,92,.3); background: rgba(224,92,92,.06); }
  .feed-wrap { position: relative; display: block; line-height: 0; }
  .feed-wrap img { display: block; width: 100%; height: auto; }
  .feed-wrap canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; }
  .cam-controls { padding: 10px 14px; display: flex; gap: 8px; background: #0a0a0c; }
  button { font-family: inherit; font-size: 11px; letter-spacing: .06em; padding: 5px 12px;
           border-radius: 3px; cursor: pointer; border: 1px solid #2a2a33;
           background: #141418; color: #8888a0; transition: border-color .15s, color .15s; }
  button:hover { border-color: #7eb8f7; color: #7eb8f7; }
  button.danger:hover { border-color: #e05c5c; color: #e05c5c; }
  .hint { font-size: 10px; color: #44445a; padding: 0 14px 10px; }
</style>
</head>
<body>
<header>
  <h1>◈ SMART NVR</h1>
  <div id="status-bar">connecting…</div>
</header>
<div class="grid" id="grid">
{% for cam in cameras %}
  <div class="cam-card" id="card-{{ cam }}">
    <div class="cam-header">
      <span class="cam-name">{{ cam }}</span>
      <span class="badge clear" id="badge-{{ cam }}">CLEAR</span>
    </div>
    <div class="feed-wrap" id="wrap-{{ cam }}">
      <img src="/feed/{{ cam }}" id="img-{{ cam }}" alt="{{ cam }}">
      <canvas id="cvs-{{ cam }}"
              onclick="addPt('{{ cam }}', event)"
              oncontextmenu="undoPt('{{ cam }}', event)">
      </canvas>
    </div>
    <div class="cam-controls">
      <button onclick="commitZone('{{ cam }}')">Set Zone</button>
      <button class="danger" onclick="clearZone('{{ cam }}')">Clear Zone</button>
    </div>
    <div class="hint">Left-click to add points &nbsp;·&nbsp; Right-click to undo &nbsp;·&nbsp; "Set Zone" to save</div>
  </div>
{% endfor %}
</div>

<script>
const pts  = {};
const D1W  = 704, D1H = 576;

function canvasToD1(cam, cx, cy) {
  const cvs = document.getElementById('cvs-' + cam);
  const r   = cvs.getBoundingClientRect();
  return [cx * D1W / r.width, cy * D1H / r.height];
}

function addPt(cam, e) {
  e.preventDefault();
  if (!pts[cam]) pts[cam] = [];
  const [x, y] = canvasToD1(cam, e.offsetX, e.offsetY);
  pts[cam].push([x, y]);
  drawOverlay(cam);
}

function undoPt(cam, e) {
  e.preventDefault();
  if (pts[cam] && pts[cam].length) { pts[cam].pop(); drawOverlay(cam); }
}

function drawOverlay(cam) {
  const cvs = document.getElementById('cvs-' + cam);
  const ctx = cvs.getContext('2d');
  // match canvas pixel size to displayed size
  cvs.width  = cvs.offsetWidth;
  cvs.height = cvs.offsetHeight;
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  const p = pts[cam] || [];
  if (p.length < 1) return;
  const sx = cvs.width / D1W, sy = cvs.height / D1H;
  ctx.strokeStyle = 'rgba(255,165,0,.9)';
  ctx.fillStyle   = 'rgba(255,165,0,.1)';
  ctx.lineWidth   = 2;
  ctx.beginPath();
  ctx.moveTo(p[0][0]*sx, p[0][1]*sy);
  for (let i = 1; i < p.length; i++) ctx.lineTo(p[i][0]*sx, p[i][1]*sy);
  ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.fillStyle = '#ffa500';
  p.forEach(([x,y]) => {
    ctx.beginPath(); ctx.arc(x*sx, y*sy, 3.5, 0, 2*Math.PI); ctx.fill();
  });
}

function commitZone(cam) {
  if (!pts[cam] || pts[cam].length < 3) { alert('Need at least 3 points.'); return; }
  fetch('/zone/' + cam, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({points: pts[cam]})
  }).then(() => console.log('Zone saved for ' + cam));
}

function clearZone(cam) {
  pts[cam] = [];
  drawOverlay(cam);
  fetch('/zone/' + cam, {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({points:[]})
  });
}

// ── alert status polling ──────────────────────────────────────────────────
function pollStatus() {
  fetch('/status').then(r => r.json()).then(data => {
    const active = Object.entries(data).filter(([,v]) => v).map(([k]) => k);
    document.getElementById('status-bar').textContent =
      active.length ? '⚠ ALERT: ' + active.join(', ') : '● All clear';
    for (const [cam, alert] of Object.entries(data)) {
      const card  = document.getElementById('card-' + cam);
      const badge = document.getElementById('badge-' + cam);
      if (!card) continue;
      card.classList.toggle('alert', alert);
      badge.className   = 'badge ' + (alert ? 'alert' : 'clear');
      badge.textContent = alert ? 'ALERT' : 'CLEAR';
    }
  }).catch(() => {});
}
setInterval(pollStatus, 1500);
pollStatus();
</script>
</body>
</html>
"""

def _generate(cam_key):
    """MJPEG stream generator for Flask."""
    while True:
        lock = frame_locks.get(cam_key)
        buf  = b''
        if lock:
            with lock:
                buf = frame_buffers.get(cam_key, b'')
        if buf:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf + b'\r\n')
        time.sleep(1 / 25)

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML, cameras=list(frame_buffers.keys()))

@app.route('/feed/<cam_key>')
def feed(cam_key):
    return Response(_generate(cam_key),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/zone/<cam_key>', methods=['POST'])
def set_zone(cam_key):
    pts = request.json.get('points', [])
    poly = Polygon(pts) if len(pts) >= 3 else None
    if cam_key in zone_poly_refs:
        zone_poly_refs[cam_key][0] = poly
    _save_zone(cam_key, pts)
    return jsonify(ok=True)

@app.route('/status')
def status():
    return jsonify({k: v.read()[2] for k, v in det_states.items()})

def _save_zone(cam_key: str, points: list):
    """Update only the zone field for cam_key — all other fields preserved."""
    with open(CONFIG_PATH, 'r') as f:
        conf = yaml.safe_load(f)
    conf['cameras'][cam_key]['zone'] = points
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(conf, f, default_flow_style=False, allow_unicode=True)


# ─────────────────────────────────────────────────────────────────────────────
# SMART NVR
# ─────────────────────────────────────────────────────────────────────────────
class SmartNVR:
    MOTION_THRESHOLD = 500   # foreground pixels required to trigger inference
    MODEL_PATH       = "yolov8n_openvino_model/"

    def __init__(self, mqtt_broker: str = "localhost", rpi: bool = False):
        self.rpi         = rpi
        self.mqtt_broker = mqtt_broker
        self.mq          = None

        if rpi:
            self._patch_openvino_for_rpi()

        if MQTT_AVAILABLE:
            try:
                self.mq = mqtt_lib.Client()
                self.mq.connect(mqtt_broker, 1883, keepalive=10)
                self.mq.loop_start()
                print(f"MQTT connected → {mqtt_broker}:1883")
            except Exception as e:
                print(f"MQTT unavailable ({e}), running without alerts.")
                self.mq = None

    @staticmethod
    def _patch_openvino_for_rpi():
        """Monkey-patch ov.Core.compile_model to set ARM-optimised CPU properties.
        Must be called before any YOLO model is instantiated.
        INFERENCE_NUM_THREADS=3 leaves one core free for camera I/O threads.
        PERFORMANCE_HINT=LATENCY beats THROUGHPUT for single-stream surveillance."""
        import openvino as ov
        _orig = ov.Core.compile_model

        def _patched(self, model, device_name="AUTO", config=None, *args, **kwargs):
            if str(device_name).upper() in ("CPU", "AUTO"):
                rpi_cfg = {
                    "INFERENCE_NUM_THREADS": "3",
                    "PERFORMANCE_HINT":      "LATENCY",
                }
                config = {**rpi_cfg, **(config or {})}
            return _orig(self, model, device_name, config, *args, **kwargs)

        ov.Core.compile_model = _patched

    @staticmethod
    def _build_url(cfg: dict, use_mjpeg: bool) -> str:
        """Construct RTSP URL. MJPEG sub-stream is subtype=2 on Dahua cameras."""
        safe_pass = quote(str(cfg['pass']))
        if use_mjpeg:
            # derive MJPEG path: subtype=1 → subtype=2
            path = cfg['path_sub'].replace('subtype=1', 'subtype=2')
        else:
            path = cfg['path_sub']
        return f"rtsp://{cfg['user']}:{safe_pass}@{cfg['ip']}:{cfg['port']}{path}&transport=tcp"

    def _inference_loop(self, cam_key: str, mjpeg: CameraStream,
                        det_state: DetectionState, zone_poly_ref: list,
                        zone_alert: ZoneAlert, recorder: ClipRecorder):
        """Reads MJPEG CIF stream, runs MOG2-gated ByteTrack inference."""
        # one model instance per camera — keeps ByteTrack state isolated
        model = YOLO(self.MODEL_PATH, task="detect")
        model.overrides['imgsz'] = [CIF_H, CIF_W]

        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=24, detectShadows=False
        )

        while True:
            frame = mjpeg.get()
            if frame is None:
                time.sleep(0.01)
                continue

            # ── motion gate ──────────────────────────────────────────────
            if cv2.countNonZero(fgbg.apply(frame)) < self.MOTION_THRESHOLD:
                det_state.update([], [], False)
                continue

            # ── ByteTrack inference ──────────────────────────────────────
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                device='cpu',
                verbose=False
            )

            boxes, track_ids, intruder_ids = [], [], []
            r = results[0]
            if r.boxes.id is not None:
                zone_poly = zone_poly_ref[0]
                for raw_box, tid in zip(
                    r.boxes.xyxy.cpu().tolist(),
                    r.boxes.id.int().cpu().tolist()
                ):
                    x1, y1, x2, y2 = (int(v) for v in raw_box)
                    boxes.append((x1, y1, x2, y2))
                    track_ids.append(tid)
                    if zone_poly and shapely_box(x1, y1, x2, y2).intersects(zone_poly):
                        intruder_ids.append(tid)

            # ── alert state machine ──────────────────────────────────────
            now       = time.time()
            was_alert = zone_alert.state == AlertState.ALERT
            is_alert  = zone_alert.update(intruder_ids, now)

            if self.mq:
                if is_alert and not was_alert:
                    self.mq.publish(f"nvr/{cam_key}/alert",
                        json.dumps({"state": "ALERT", "camera": cam_key, "ts": now}))
                elif not is_alert and was_alert:
                    self.mq.publish(f"nvr/{cam_key}/alert",
                        json.dumps({"state": "CLEAR", "camera": cam_key, "ts": now}))

            if is_alert:
                recorder.trigger(now)

            det_state.update(boxes, track_ids, is_alert)

    def _display_loop(self, cam_key: str, d1: CameraStream,
                      det_state: DetectionState, zone_poly_ref: list,
                      recorder: ClipRecorder):
        """Reads D1 H.264 stream, annotates, encodes to JPEG for Flask."""
        while True:
            frame = d1.get()
            if frame is None:
                time.sleep(0.01)
                continue

            ts = time.time()
            recorder.feed(frame, ts)

            # ── draw ROI zone ────────────────────────────────────────────
            zone_poly = zone_poly_ref[0]
            if zone_poly:
                pts = np.array(
                    [(int(x), int(y)) for x, y in zone_poly.exterior.coords],
                    dtype=np.int32
                )
                cv2.polylines(frame, [pts], True, (0, 165, 255), 2)

            # ── overlay detections (CIF ×2 → D1) ────────────────────────
            boxes, track_ids, alert = det_state.read()
            color = (0, 0, 220) if alert else (0, 200, 80)
            for (x1, y1, x2, y2), tid in zip(boxes, track_ids):
                dx1, dy1, dx2, dy2 = x1*2, y1*2, x2*2, y2*2
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
                cv2.putText(frame, f"ID {tid}", (dx1, dy1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

            if alert:
                cv2.putText(frame, "INTRUDER!", (20, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 220), 3)

            # ── encode to JPEG and push to shared buffer ─────────────────
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            with frame_locks[cam_key]:
                frame_buffers[cam_key] = buf.tobytes()

    def run(self, cam_keys: list = None, flask_port: int = 5000):
        conf    = self._load_config()
        cameras = conf['cameras']
        keys    = cam_keys or list(cameras.keys())

        for cam_key in keys:
            cfg = cameras[cam_key]
            if not cfg.get('zone'):
                print(f"[{cam_key}] No zone defined — draw one in the web UI.")

            zone_pts = cfg.get('zone') or []
            zone_poly_refs[cam_key] = [Polygon(zone_pts) if len(zone_pts) >= 3 else None]
            det_states[cam_key]     = DetectionState()
            frame_buffers[cam_key]  = b''
            frame_locks[cam_key]    = threading.Lock()

            mjpeg_url = self._build_url(cfg, use_mjpeg=True)
            d1_url    = self._build_url(cfg, use_mjpeg=False)

            fps_cap      = 15 if self.rpi else 0   # throttle on RPi, free-run on x86
            mjpeg_stream = CameraStream(mjpeg_url, CIF_W, CIF_H).start(fps_cap)
            d1_stream    = CameraStream(d1_url,    D1_W,  D1_H).start(fps_cap)

            zone_alert = ZoneAlert()
            recorder   = ClipRecorder(cam_key)

            threading.Thread(
                target=self._inference_loop,
                args=(cam_key, mjpeg_stream, det_states[cam_key],
                      zone_poly_refs[cam_key], zone_alert, recorder),
                daemon=True
            ).start()

            threading.Thread(
                target=self._display_loop,
                args=(cam_key, d1_stream, det_states[cam_key],
                      zone_poly_refs[cam_key], recorder),
                daemon=True
            ).start()

            print(f"[{cam_key}] started — D1: {d1_url.split('@')[-1]}")

        print(f"\nWeb UI → http://0.0.0.0:{flask_port}")
        print("Draw ROI zones in the browser, then click 'Set Zone' to save.\n")

        # Flask in daemon thread — main thread blocks below
        threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=flask_port,
                                   threaded=True, use_reloader=False),
            daemon=True
        ).start()

        threading.Event().wait()

    @staticmethod
    def _load_config() -> dict:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import platform
    is_rpi = platform.machine() == "aarch64"   # auto-detect RPi4/5

    nvr = SmartNVR(mqtt_broker="localhost", rpi=is_rpi)

    # Start all cameras, or pass a list: ["cam_252", "cam_254"]
    nvr.run()
