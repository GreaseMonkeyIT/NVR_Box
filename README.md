# Smart NVR — YOLOv8n + OpenVINO + ROI Alerting

A self-hosted network video recorder with edge AI inference. Runs on x86 (tested on Intel Sandy Bridge) and ARM (RPi4/5). No cloud dependency.

---

## How it works

```
MJPEG sub-stream (CIF 352×288)
  └─ CameraStream thread
       └─ MOG2 motion gate ──► skip if scene is idle
            └─ YOLOv8n OpenVINO (INT8/FP16)
                 └─ ByteTrack (persistent IDs)
                      └─ Shapely ROI intersection
                           └─ ZoneAlert state machine
                                ├─ MQTT publish (state transitions only)
                                └─ ClipRecorder trigger

H.264 sub-stream (D1 704×576)
  └─ CameraStream thread
       └─ Overlay detections (CIF coords ×2 → D1)
            └─ JPEG encode → Flask MJPEG stream → browser
                 └─ ClipRecorder feed (pre-buffer + post-event)
```

**Key design decisions:**
- MJPEG sub-stream for inference avoids costly H.264 software decode on ARM
- MOG2 gates YOLOv8n — idle scenes cost near-zero CPU
- ByteTrack gives persistent track IDs across frames, enabling per-ID alert deduplication
- Debounce (3 frames) + cooldown (15s) state machine eliminates alert flooding
- 10-second pre-buffer ensures the lead-up to an event is always captured

---

## Requirements

```
Python        3.10+
OpenVINO      2024.6.0
ultralytics   8.1.0
opencv-python 4.11+
shapely       2.1+
flask
paho-mqtt     (optional — NVR runs without it if broker unavailable)
```

Install:
```bash
pip install openvino==2024.6.0 ultralytics flask shapely opencv-python paho-mqtt
```

On RPi (system Python):
```bash
pip install openvino==2024.6.0 ultralytics flask shapely opencv-python paho-mqtt --break-system-packages
```

---

## Camera configuration

On a fresh clone, copy the example config and fill in your details:
```bash
cp config/cameras.example.yaml config/cameras.yaml
```

`cameras.yaml` is gitignored — your credentials and zone data never leave the machine. Edit it:

```yaml
cameras:
  cam_1:
    ip: 192.168.1.XXX       # camera IP on your LAN
    user: admin
    pass: YOUR_PASSWORD
    port: 554
    path_sub: /cam/realmonitor?channel=1&subtype=1   # D1 H.264 sub-stream
    threshold_pct: 30
    zone: []                # drawn via web UI — do not edit manually
```

**MJPEG sub-stream:** The NVR automatically derives the MJPEG path by replacing `subtype=1` with `subtype=2`. This is the Dahua/compatible convention. If your camera uses a different path for MJPEG, verify it in the camera's web interface under Stream Settings and update `_build_url()` in `nvr.py` accordingly.

**Supported cameras:** Any IP camera that exposes RTSP MJPEG + H.264 sub-streams. Tested on Dahua-protocol cameras. The RTSP URL format is:
```
rtsp://user:password@ip:port/path&transport=tcp
```

---

## Model export

### x86 (Intel — FP16)
```bash
python convert_model.py
# exports yolov8n_openvino_model/ with FP16 precision
```

### RPi4/5 (ARM — INT8)
```bash
python convert_model.py --rpi
# exports yolov8n_openvino_model/ with INT8 precision
# uses coco128.yaml for calibration by default
```

**Better INT8 accuracy:** Record 200–300 CIF frames (352×288) from your actual cameras under typical lighting conditions. Pass the folder path to `data=` in `convert_model.py`:
```python
model.export(format="openvino", imgsz=(288,352), half=False, int8=True, data="path/to/your/frames")
```
Calibrating on your own scene consistently outperforms COCO128 for surveillance use.

Copy the exported `yolov8n_openvino_model/` folder to the RPi alongside `nvr.py`.

---

## Running

```bash
python nvr.py
```

Platform is auto-detected: `aarch64` (RPi) enables INT8 CPU config + 15fps frame cap. x86 uses FP16 and free-runs. To force RPi mode manually, edit the `__main__` block:
```python
nvr = SmartNVR(rpi=True)
```

Open the web UI in a browser on the same network:
```
http://<host-ip>:5000
```

---

## Drawing ROI zones

ROI zones are drawn entirely in the browser — no OpenCV windows required.

1. Open `http://<host-ip>:5000`
2. Left-click on a camera feed to add polygon vertices
3. Right-click to undo the last point
4. Click **Set Zone** when done — the polygon is saved to `cameras.yaml` and takes effect immediately (no restart needed)
5. Click **Clear Zone** to remove an existing zone

The INTRUDER alert only fires when a detected person's bounding box intersects the drawn zone. Detections outside the zone are shown in green with no alert.

---

## RPi4/5 deployment

The `rpi=True` mode applies three changes automatically:

| Change | Effect |
|--------|--------|
| `INFERENCE_NUM_THREADS=3` | Leaves 1 core free for camera I/O |
| `PERFORMANCE_HINT=LATENCY` | Optimises for single-stream latency over batch throughput |
| 15fps frame cap on streams | Prevents RTSP decode from saturating all cores |

**RPi5 H.264 hardware decode (optional):** RPi5 has hardware H.264/H.265 decode via the RP1 chip. For high-res main streams, piping RTSP through ffmpeg with `-hwaccel v4l2m2m` before handing frames to OpenCV removes the software decode bottleneck. This is not wired in by default.

---

## MQTT alerting

The NVR publishes to `nvr/<cam_key>/alert` on state transitions only (not every frame).

Payload on alert:
```json
{"state": "ALERT", "camera": "cam_1", "ts": 1748123456.789}
```
Payload on clear:
```json
{"state": "CLEAR", "camera": "cam_1", "ts": 1748123460.123}
```

Default broker: `localhost:1883`. Change via:
```python
nvr = SmartNVR(mqtt_broker="192.168.1.10")
```

If the broker is unreachable at startup, MQTT is silently disabled — the NVR runs normally.

---

## Clip recording

Intrusion events are automatically recorded to `clips/`. Each clip includes:
- **10-second pre-buffer** — footage before the alert triggered
- **15 seconds post-event** — footage after the last detection in the zone

Clips are named `<cam_key>_<unix_timestamp>.mp4` at D1 resolution (704×576).

---

## Tuning

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| Motion threshold | `SmartNVR.MOTION_THRESHOLD` | 500px | Lower = more sensitive, higher = ignore small motion |
| Debounce frames | `ZoneAlert.DEBOUNCE_FRAMES` | 3 | Consecutive detections before alert fires |
| Cooldown | `ZoneAlert.COOLDOWN_SECS` | 15s | Silence period after last detection |
| Pre-buffer | `ClipRecorder.PRE_SECS` | 10s | Seconds of footage saved before alert |
| Post-buffer | `ClipRecorder.POST_SECS` | 15s | Seconds recorded after last detection |
| JPEG quality | `_display_loop` | 72 | Lower = less bandwidth, faster streaming |

---

## File structure

```
NVR_Box/
├── nvr.py                          ← main application
├── convert_model.py                ← model export (FP16 or INT8)
├── test_connection.py              ← RTSP stream tester
├── config/
│   └── cameras.yaml                ← camera credentials + zone data
├── yolov8n_openvino_model/         ← exported model (generated, not in repo)
│   ├── yolov8n.xml
│   ├── yolov8n.bin
│   └── metadata.yaml
├── clips/                          ← recorded intrusion clips (generated)
└── environment.yml                 ← conda environment spec
```

---

## Testing a camera connection

```bash
python test_connection.py cam_1 sub    # test sub-stream
python test_connection.py cam_1 main   # test main stream
```
