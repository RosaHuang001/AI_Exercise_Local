import os
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from PIL import Image, ImageDraw, ImageFont

# =========================
# 路徑與設定
# =========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(_SCRIPT_DIR, "front", "Noto_Sans_TC", "static", "NotoSansTC-Regular.ttf")
_font_cache = {}

MODEL_PATH = os.path.join(_SCRIPT_DIR, "yolo11n-pose.pt")
DISPLAY_W, DISPLAY_H = 1280, 720
INNER_HEIGHT = 480
BACKGROUND_COLOR = (25, 25, 25)
FLIP_USER_VIEW = True
SMOOTH_FACTOR = 0.08
END_HOLD_SECONDS = 3.0
REST_DURATION = 5.0

MAX_SERIES_LEN = 300
PEAK_DISTANCE, PEAK_PROMINENCE = 20, 5
DTW_MIN_LEN = 30
DTW_SCORE_SCALE, DTW_SCORE_BIAS = 5.0, 100.0
DTW_SMOOTH_MINLEN = 10

# =========================
# 工具函式
# =========================
def _get_font(size: int):
    if size not in _font_cache:
        if os.path.isfile(FONT_PATH):
            _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def put_chinese_text(img_bgr, text, pos_bottom_left, font_size, color_bgr, thickness=2):
    if not text: return
    font = _get_font(font_size)
    bbox = font.getbbox(text)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0: return

    pil_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    t = max(0, int(thickness or 0))
    if t > 0:
        for dy in range(-t, t + 1):
            for dx in range(-t, t + 1):
                if dx == 0 and dy == 0: continue
                draw.text((-bbox[0] + dx, -bbox[1] + dy), text, font=font, fill=(0, 0, 0, 180))
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0], 255))
    
    arr = np.array(pil_img)
    x, y = pos_bottom_left
    y_top = y - h
    x0, y0 = max(0, x), max(0, y_top)
    x1, y1 = min(img_bgr.shape[1], x + w), min(img_bgr.shape[0], y_top + h)
    if x0 >= x1 or y0 >= y1: return

    sy0, sx0 = y0 - y_top, x0 - x
    sh, sw = y1 - y0, x1 - x0
    roi = img_bgr[y0:y1, x0:x1]
    alpha = (arr[sy0:sy0 + sh, sx0:sx0 + sw, 3:4] / 255.0)
    rgb = arr[sy0:sy0 + sh, sx0:sx0 + sw, :3][:, :, ::-1]
    roi[:] = (roi * (1 - alpha) + rgb * alpha).astype(np.uint8)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0))))

def draw_score_circle(img, center, radius, score):
    score = float(np.clip(score, 0, 100))
    color = (0, 200, 0) if score >= 80 else (0, 180, 255) if score >= 60 else (0, 0, 255)
    cv2.circle(img, center, radius, (60, 60, 60), 8)
    cv2.ellipse(img, center, (radius, radius), -90, 0, int(360 * score / 100), color, 8)
    text = f"{int(score)}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(img, text, (int(center[0] - tw / 2), int(center[1] + th / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

def draw_progress_bar(img, progress):
    progress = float(np.clip(progress, 0, 1))
    bar_w, bar_x, bar_y, bar_h = int(DISPLAY_W * 0.7), int(DISPLAY_W * 0.15), DISPLAY_H - 60, 12
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (255, 255, 255), -1)

def safe_resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h <= 0: return img
    return cv2.resize(img, (max(1, int(w * (target_h / h))), target_h))

def extract_right_knee_angle_from_results(results_obj):
    try:
        kpts = results_obj.keypoints
        if not kpts or kpts.xy is None or len(kpts.xy) == 0: return None
        pts = kpts.xy[0].cpu().numpy()
        if pts.shape[0] < 17: return None
        return calculate_angle(pts[12], pts[14], pts[16])
    except: return None

def compute_dtw_similarity(r_ref, r_user):
    if len(r_ref) == 0 or len(r_user) == 0: return 0.0
    L = min(len(r_ref), len(r_user))
    if L <= 1: return 0.0
    distance, _ = fastdtw(r_ref, r_user, dist=euclidean)
    score = max(0.0, DTW_SCORE_BIAS - (float(distance) / (L + 1e-6)) * DTW_SCORE_SCALE)
    return float(np.clip(score, 0, 100))

# =========================
# 核心串流產生器 (Yield Generator)
# =========================
def generate_frames(playlist):
    if not playlist: return
    
    # 確保路徑絕對正確
    playlist = [p if os.path.isabs(p) else os.path.join(_SCRIPT_DIR, p) for p in playlist]

    model = YOLO(MODEL_PATH)
    cap_user = cv2.VideoCapture(0)
    if not cap_user.isOpened():
        print("無法開啟攝影機")
        return

    current_vid_idx = 0
    cap_ref = cv2.VideoCapture(playlist[current_vid_idx])
    
    is_resting = False
    rest_start_time = 0
    reference_series, user_series = [], []
    similarity_score = display_similarity = display_score = 0.0
    total_reps, current_reps = 0, 0
    sim_sum, sim_count = 0.0, 0

    session_end_flag = False
    session_end_time = None

    try:
        while True:
            ret_user, frame_user = cap_user.read()
            if not ret_user: break
            if FLIP_USER_VIEW: frame_user = cv2.flip(frame_user, 1)

            user_results = model(frame_user, verbose=False)
            frame_user_pose = user_results[0].plot()
            user_angle = extract_right_knee_angle_from_results(user_results[0])

            real_score = 0.0
            feedback_text = "請開始動作"

            if is_resting:
                rest_elapsed = time.time() - rest_start_time
                remain = max(0, REST_DURATION - rest_elapsed)
                
                frame_ref_pose = np.zeros((INNER_HEIGHT, int(INNER_HEIGHT*16/9), 3), dtype=np.uint8)
                put_chinese_text(frame_ref_pose, f"休息時間: {int(remain)} 秒", (150, 200), 40, (0, 255, 255))
                
                if current_vid_idx + 1 < len(playlist):
                    next_vid_name = os.path.basename(playlist[current_vid_idx + 1])
                    put_chinese_text(frame_ref_pose, f"下一個動作準備：", (150, 280), 26, (200, 200, 200))
                    put_chinese_text(frame_ref_pose, next_vid_name, (150, 330), 22, (255, 255, 255))

                feedback_text = "中場休息，請調節呼吸"
                display_score += (0 - display_score) * SMOOTH_FACTOR

                if remain <= 0:
                    is_resting = False
                    current_vid_idx += 1
                    cap_ref = cv2.VideoCapture(playlist[current_vid_idx])
                    reference_series.clear()
                    user_series.clear()
                    total_reps += current_reps
                    current_reps = 0
                    continue
            else:
                ret_ref, frame_ref = cap_ref.read()
                if not ret_ref:
                    if current_vid_idx < len(playlist) - 1:
                        is_resting = True
                        rest_start_time = time.time()
                        cap_ref.release()
                        continue
                    else:
                        if not session_end_flag:
                            total_reps += current_reps
                            session_end_flag = True
                            session_end_time = time.time()
                        frame_ref_pose = np.zeros((INNER_HEIGHT, int(INNER_HEIGHT*16/9), 3), dtype=np.uint8)
                else:
                    ref_results = model(frame_ref, verbose=False)
                    frame_ref_pose = ref_results[0].plot()
                    ref_angle = extract_right_knee_angle_from_results(ref_results[0])
                    if ref_angle is not None:
                        reference_series.append(float(ref_angle))
                        if len(reference_series) > MAX_SERIES_LEN: reference_series.pop(0)

                    if user_angle is not None:
                        user_series.append(float(user_angle))
                        if len(user_series) > MAX_SERIES_LEN: user_series.pop(0)
                        if len(user_series) > 50:
                            arr = np.array(user_series, dtype=np.float32)
                            peaks, _ = find_peaks(-arr, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
                            current_reps = int(len(peaks))

                    if (len(reference_series) >= DTW_SMOOTH_MINLEN) and (len(user_series) >= DTW_MIN_LEN):
                        L = min(len(reference_series), len(user_series), MAX_SERIES_LEN)
                        similarity_score = compute_dtw_similarity(reference_series[-L:], user_series[-L:])
                        sim_sum += float(similarity_score)
                        sim_count += 1
                        real_score = similarity_score

                        if similarity_score >= 80: feedback_text = "動作符合示範"
                        elif similarity_score >= 60: feedback_text = "動作不錯，再穩定一點"
                        else: feedback_text = "動作差距較大，請調整"
                    else:
                        feedback_text = "示範模板建立中..." if len(reference_series) < DTW_SMOOTH_MINLEN else "請跟著動作..."

                    display_score += (real_score - display_score) * SMOOTH_FACTOR
                    display_similarity += (similarity_score - display_similarity) * SMOOTH_FACTOR

            frame_ref_pose = safe_resize_to_height(frame_ref_pose, INNER_HEIGHT)
            frame_user_pose = safe_resize_to_height(frame_user_pose, INNER_HEIGHT)
            combined = np.hstack([frame_ref_pose, frame_user_pose])

            canvas = np.full((DISPLAY_H, DISPLAY_W, 3), BACKGROUND_COLOR, dtype=np.uint8)
            h_c, w_c = combined.shape[:2]
            offset_x, offset_y = max(0, (DISPLAY_W - w_c) // 2), 80
            x1, y1 = min(DISPLAY_W, offset_x + w_c), min(DISPLAY_H, offset_y + h_c)
            canvas[offset_y:y1, offset_x:x1] = combined[:(y1 - offset_y), :(x1 - offset_x)]

            curr_vid_name = os.path.basename(playlist[current_vid_idx]) if not session_end_flag else "完成"
            put_chinese_text(canvas, f"AI 智慧跟練 (動作 {current_vid_idx+1}/{len(playlist)})", (40, 50), 26, (255, 255, 255))
            put_chinese_text(canvas, f"當前動作：{curr_vid_name}", (40, 80), 18, (180, 180, 180), thickness=1)

            draw_score_circle(canvas, (1100, 200), 60, display_score)
            put_chinese_text(canvas, "動作回饋：", (40, 600), 22, (200, 200, 200))
            put_chinese_text(canvas, feedback_text, (180, 600), 22, (255, 255, 255))
            put_chinese_text(canvas, f"動作相似度: {int(display_similarity)}%", (900, 350), 22, (255, 255, 255))
            put_chinese_text(canvas, f"總完成次數: {total_reps + current_reps}", (900, 400), 22, (0, 200, 255))

            progress = (current_vid_idx + (0.5 if is_resting else 0)) / len(playlist)
            if session_end_flag: progress = 1.0
            draw_progress_bar(canvas, progress)

            if session_end_flag:
                cv2.rectangle(canvas, (300, 200), (980, 520), (20, 20, 20), -1)
                put_chinese_text(canvas, "今日完整課表訓練完成！", (450, 260), 34, (255, 255, 255))
                avg_sim = (sim_sum / (sim_count + 1e-6)) if sim_count > 0 else 0.0
                put_chinese_text(canvas, f"整體平均相似度: {int(avg_sim)}%", (480, 340), 26, (0, 255, 0))
                put_chinese_text(canvas, f"總累計完成次數: {total_reps}", (500, 390), 26, (0, 200, 255))

            # === 關鍵：將 OpenCV 畫布轉換為 JPG 串流發送給網頁 ===
            ret, buffer = cv2.imencode('.jpg', canvas)
            if not ret: continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # 如果訓練結束並停留了 3 秒，就自動關閉串流
            if session_end_flag and (time.time() - session_end_time) >= END_HOLD_SECONDS:
                break

    except GeneratorExit:
        # 當網頁被關閉或使用者按下「結束訓練」時，觸發此處安全關閉攝影機
        print("串流已中斷，安全釋放攝影機")
    finally:
        cap_user.release()
        if cap_ref: cap_ref.release()