import cv2
import yaml
import sys
import time
from urllib.parse import quote

# D1 resolution (matches your NVR display logic)
D1_W, D1_H = 704, 576

def test_feed(cam_key, stream_type="sub"):
    # 1. Load camera config
    try:
        with open("config/cameras.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config/cameras.yaml not found.")
        return

    if cam_key not in config["cameras"]:
        print(f"Camera '{cam_key}' not found in config")
        return

    cam = config["cameras"][cam_key]
    path = cam["path_main"] if stream_type == "main" else cam["path_sub"]

    # 2. ENCODE PASSWORD: Fix for the '@' symbol in your password
    safe_pass = quote(cam['pass'])
    
    # 3. CONSTRUCT URL: Ensure transport=tcp is handled correctly for FFMPEG backend
    # We use '&transport=tcp' which is the standard argument for these RTSP paths
    url = f"rtsp://{cam['user']}:{safe_pass}@{cam['ip']}:{cam['port']}{path}&transport=tcp"

    print(f"Connecting to {cam_key} ({stream_type})")
    print(f"URL: {url}")

    # 4. INITIALIZE CAPTURE: Use FFMPEG backend
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    
    # Buffer size 1 is better for testing "Real Time" latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("FAILED: Unable to open RTSP stream. Check credentials or IP reachability.")
        return

    print("Connected. Warming up stream...")

    # Warm-up frames to clear the buffer
    for _ in range(10):
        cap.grab() # grab() is lighter than read() for warming up

    print("Streaming. Press Q or ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠ Frame not received, retrying...")
            time.sleep(0.5)
            continue

        # Resize to D1 (matches your NVR display resolution)
        frame_display = cv2.resize(frame, (D1_W, D1_H))

        # Add an overlay to the test window for confirmation
        cv2.putText(frame_display, f"Source: {cam_key} | {stream_type}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(f"TEST FEED - {cam_key}", frame_display)

        # Exit on Q or ESC
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed.")

if __name__ == "__main__":
    # Usage: python3 test_connection.py [cam_key] [sub/main]
    cam = sys.argv[1] if len(sys.argv) > 1 else "cam_251"
    stream = sys.argv[2] if len(sys.argv) > 2 else "sub"

    test_feed(cam, stream)