import time
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
import vgamepad as vg

# State
@dataclass
class ControllerState:
    enabled: bool = True
    recenter: bool = False

    # movement holds (voice)
    move_w: bool = False
    move_a: bool = False
    move_s: bool = False
    move_d: bool = False
    crouch: bool = False

    jump_tap: bool = False

    # firing (voice)
    firing_toggle: bool = False
    tap_shot: bool = False
    reload_tap: bool = False
    scope_toggle: bool = False

    # aiming (head tracking output, normalized -1..1)
    aim_x: float = 0.0
    aim_y: float = 0.0

state = ControllerState()
state_lock = threading.Lock()
stop_event = threading.Event()

# Virtual xbox360 gamepad
gamepad = vg.VX360Gamepad()

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def to_i16(x: float) -> int:
    # Map [-1,1] -> [-32768,32767]
    x = clamp(x, -1.0, 1.0)
    return int(x * 32767)

def neutralize_gamepad():
    # sticks to 0, triggers to 0, release buttons
    gamepad.left_joystick(0, 0)
    gamepad.right_joystick(0, 0)
    gamepad.left_trigger(0)
    gamepad.right_trigger(0)
    # release crouch if we use a button
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
    gamepad.update()

# Mediapipe aliases
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = str(Path(__file__).with_name("blaze_face_short_range.tflite"))

## Head tracking
def head_tracking_tasks(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # tracking smoothing
    alpha = 0.25
    deadzone = 0.03

    # stick tuning (start here, then adjust)
    stick_sens_x = 4      # higher = faster turn left/right
    stick_sens_y = 1.5      # higher = faster up/down
    stick_deadzone = 0.03   # deadzone on stick output
    invert_y = True        # set True if you want inverted look

    neutral_yaw = 0.0
    neutral_pitch = 0.0
    yaw_s = 0.0
    pitch_s = 0.0

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
    )

    with FaceDetector.create_from_options(options) as detector:
        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(time.time() * 1000)

                result = detector.detect_for_video(mp_image, timestamp_ms)

                with state_lock:
                    enabled = state.enabled
                    do_recenter = state.recenter
                    if do_recenter:
                        state.recenter = False

                yaw = 0.0
                pitch = 0.0
                have_face = False

                if result.detections:
                    det = result.detections[0]
                    have_face = True

                    bbox = det.bounding_box
                    x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

                    kps = getattr(det, "keypoints", None)

                    if kps and len(kps) >= 3:
                        pts = []
                        for kp in kps:
                            px, py = kp.x, kp.y
                            # scale if normalized
                            if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0:
                                px *= w
                                py *= h
                            pts.append((float(px), float(py)))

                        for (px, py) in pts:
                            cv2.circle(frame, (int(px), int(py)), 4, (255, 255, 0), -1)

                        pts_np = np.array(pts, dtype=np.float32)
                        left_eye = pts_np[np.argmin(pts_np[:, 0])]
                        right_eye = pts_np[np.argmax(pts_np[:, 0])]
                        eye_mid = (left_eye + right_eye) * 0.5
                        nose = pts_np[np.argmin(np.abs(pts_np[:, 0] - eye_mid[0]))]

                        eye_dist = np.linalg.norm(right_eye - left_eye) + 1e-6
                        yaw = (nose[0] - eye_mid[0]) / eye_dist
                        pitch = (nose[1] - eye_mid[1]) / (bh + 1e-6)

                        cv2.line(frame, (int(left_eye[0]), int(left_eye[1])),
                                       (int(right_eye[0]), int(right_eye[1])), (0, 0, 255), 2)
                        cv2.circle(frame, (int(eye_mid[0]), int(eye_mid[1])), 6, (0, 0, 255), -1)
                        cv2.circle(frame, (int(nose[0]), int(nose[1])), 6, (0, 255, 255), -1)
                    else:
                        cx = x + bw / 2.0
                        cy = y + bh / 2.0
                        yaw = (cx - (w / 2.0)) / (w / 2.0)
                        pitch = (cy - (h / 2.0)) / (h / 2.0)

                if do_recenter:
                    neutral_yaw = yaw
                    neutral_pitch = pitch

                yaw -= neutral_yaw
                pitch -= neutral_pitch

                yaw_s = (1 - alpha) * yaw_s + alpha * yaw
                pitch_s = (1 - alpha) * pitch_s + alpha * pitch

                yaw_used = 0.0 if abs(yaw_s) < deadzone else yaw_s
                pitch_used = 0.0 if abs(pitch_s) < deadzone else pitch_s

                # Map to stick [-1..1]
                rx = clamp(yaw_used * stick_sens_x, -1.0, 1.0)
                ry = clamp(pitch_used * stick_sens_y, -1.0, 1.0)
                if invert_y:
                    ry = -ry

                # Stick deadzone
                if abs(rx) < stick_deadzone:
                    rx = 0.0
                if abs(ry) < stick_deadzone:
                    ry = 0.0

                gain_left = 0.7
                gain_right = 1.4

                if yaw_used < 0:
                    yaw_used *= gain_left
                else:
                    yaw_used *= gain_right

                with state_lock:
                    # If disabled or no face, aim should go neutral
                    if enabled and have_face:
                        state.aim_x = rx
                        state.aim_y = ry
                    else:
                        state.aim_x = 0.0
                        state.aim_y = 0.0

                # HUD
                cv2.putText(frame, f"enabled: {enabled}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"yaw: {yaw_s:+.3f}  pitch: {pitch_s:+.3f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"stick rx: {rx:+.2f}  ry: {ry:+.2f}", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "ESC quit | C recenter | P pause", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if not have_face:
                    cv2.putText(frame, "No face detected", (10, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Face Tracker", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    stop_event.set()
                    break
                elif key in (ord("c"), ord("C")):
                    with state_lock:
                        state.recenter = True
                elif key in (ord("p"), ord("P")):
                    with state_lock:
                        state.enabled = not state.enabled
                        if not state.enabled:
                            state.firing_toggle = False
                            state.tap_shot = False
                            state.move_w = state.move_a = state.move_s = state.move_d = False
                            state.crouch = False

        finally:
            cap.release()
            cv2.destroyAllWindows()

# Gamepad output loop
def output_loop_gamepad():
    """
    Applies state continuously to the virtual controller:
    - Left stick = movement (voice)
    - Right stick = aim (head tracking)
    - RT = shoot (toggle)
    - Tap shot = short RT pulse
    - Crouch = LB (bind crouch to LB in CS2 controller binds)
    - A = jump
    """
    tap_pulse_s = 0.06  # tap shot duration
    last_fire = False

    while not stop_event.is_set():
        with state_lock:
            enabled = state.enabled

            w, a, s, d = state.move_w, state.move_a, state.move_s, state.move_d
            crouch = state.crouch

            firing = state.firing_toggle
            do_tap = state.tap_shot
            if do_tap:
                state.tap_shot = False

            do_jump = state.jump_tap
            if do_jump:
                state.jump_tap = False
            
            scope = state.scope_toggle
            do_reload = state.reload_tap
            if do_reload:
                state.reload_tap = False

            aim_x = state.aim_x
            aim_y = state.aim_y

        if not enabled:
            neutralize_gamepad()
            time.sleep(0.02)
            continue

        # Movement vector from voice (left stick)
        lx = (-1.0 if a else 0.0) + (1.0 if d else 0.0)
        ly = (1.0 if w else 0.0) + (-1.0 if s else 0.0)

        mag = (lx * lx + ly * ly) ** 0.5
        if mag > 1.0:
            lx /= mag
            ly /= mag

        gamepad.left_joystick(x_value=to_i16(lx), y_value=to_i16(ly))
        gamepad.right_joystick(x_value=to_i16(aim_x), y_value=to_i16(aim_y))
        gamepad.left_trigger(value=255 if scope else 0)

        # Crouch as a button (pick one and bind in CS2)
        # Using LB here:
        if crouch:
            gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        else:
            gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)

        # Shoot toggle: RT held
        if firing:
            gamepad.right_trigger(value=255)
        else:
            gamepad.right_trigger(value=0)

        # Tap shot: short RT pulse (even if toggle is off)
        if do_tap:
            gamepad.right_trigger(value=255)
            gamepad.update()
            time.sleep(tap_pulse_s)
            # restore RT to whatever toggle wants
            gamepad.right_trigger(value=255 if firing else 0)
        
        if do_jump:
            gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            gamepad.update()
            time.sleep(0.05)
            gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        
        if do_reload:
            gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            gamepad.update()
            time.sleep(0.06)
            gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

        gamepad.update()
        time.sleep(0.01)

    neutralize_gamepad()

# Voice loop
def voice_loop_vosk(model_dir: str):
    import json
    import queue
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(bytes(indata))

    model = Model(model_dir)

    grammar = json.dumps([
        "forward", "back", "left", "right", "stop",
        "crouch", "stand",
        "shoot", "stop shooting", "tap", "tap fire",
        "jump", "scope", "unscope", "reload",
        "pause", "resume",
        "center",
        "terminate program"
    ])

    rec = KaldiRecognizer(model, 16000, grammar)
    rec.SetWords(False)

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                           channels=1, callback=callback):
        while not stop_event.is_set():
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                cmd = (result.get("text") or "").strip().lower()
                if not cmd:
                    continue

                with state_lock:
                    if cmd == "pause":
                        state.enabled = False
                        state.firing_toggle = False
                        state.tap_shot = False
                        state.move_w = state.move_a = state.move_s = state.move_d = False
                        state.crouch = False
                    elif cmd == "resume":
                        state.enabled = True
                    elif cmd == "center":
                        state.recenter = True

                    elif cmd == "forward":
                        state.move_w = True; state.move_s = False
                    elif cmd == "back":
                        state.move_s = True; state.move_w = False
                    elif cmd == "left":
                        state.move_a = True; state.move_d = False
                    elif cmd == "right":
                        state.move_d = True; state.move_a = False
                    elif cmd == "stop":
                        state.move_w = state.move_a = state.move_s = state.move_d = False

                    elif cmd == "crouch":
                        state.crouch = True
                    elif cmd == "stand":
                        state.crouch = False
                    elif cmd == "jump":
                        state.jump_tap = True

                    elif cmd == "shoot":
                        state.firing_toggle = True
                    elif cmd in ("tap", "tap fire"):
                        state.tap_shot = True

                    elif cmd == "stop shooting":
                        state.firing_toggle = False
                    elif cmd == "reload":
                        state.reload_tap = True
                    elif cmd == "scope":
                        state.scope_toggle = not state.scope_toggle
                    elif cmd == "unscope":
                        state.scope_toggle = False

                    elif cmd == "terminate program":
                        stop_event.set()
                        break

                print(f"[voice] {cmd}")

if __name__ == "__main__":
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Put blaze_face_short_range.tflite next to this script."
        )

    try:
        threading.Thread(target=output_loop_gamepad, daemon=True).start()
        threading.Thread(target=voice_loop_vosk, args=("vosk-model-small-en-us-0.15",), daemon=True).start()
        head_tracking_tasks(0)
    finally:
        stop_event.set()
        neutralize_gamepad()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
