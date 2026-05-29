import torch
import functools
import argparse
from ultralytics import YOLO

# ── Bypass PyTorch 2.6 weights_only restriction ────────────────────────────
torch.load = functools.partial(torch.load, weights_only=False)

parser = argparse.ArgumentParser()
parser.add_argument("--rpi", action="store_true",
                    help="Export INT8 for RPi4/5 (ARM). Default: FP16 for x86.")
args = parser.parse_args()

print("Loading yolov8n.pt...")
model = YOLO("yolov8n.pt")

if args.rpi:
    # ── INT8 for RPi4/5 (ARM) ──────────────────────────────────────────────
    # FP16 has no hardware acceleration on ARM NEON — INT8 is 1.5-2× faster.
    # Requires a calibration dataset. Replace data= with a folder of your own
    # camera frames (~100-300 images) for better accuracy on your scene.
    print("Exporting INT8 for RPi4/5 (CIF 352×288)...")
    model.export(
        format="openvino",
        imgsz=(288, 352),   # (H, W) — CIF
        half=False,
        int8=True,
        data="coco128.yaml"  # swap for your camera frames if you have them
    )
    print("INT8 model → yolov8n_openvino_model/")
    print("Note: to calibrate on your own footage, record ~200 CIF frames,")
    print("save them to a folder, and point data= at that path.")
else:
    # ── FP16 for x86 (Sandy Bridge i5-2320) ──────────────────────────────
    print("Exporting FP16 for x86 (CIF 352×288)...")
    model.export(
        format="openvino",
        imgsz=(288, 352),
        half=True
    )
    print("FP16 model → yolov8n_openvino_model/")
