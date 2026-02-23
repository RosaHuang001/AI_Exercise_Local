#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python 3.x | opencv-python, numpy, ultralytics, scipy, fastdtw, Pillow
#
# ✅ 最終版重點：
# 1) 不靠 JSON：示範影片即時 YOLO → 即時 reference_series
# 2) 左邊示範影片顯示骨架（frame_ref_pose）
# 3) 右邊使用者顯示骨架（frame_user_pose）
# 4) reference_series vs user_series 做 DTW → similarity_score
# 5) 仍保留：rep 計數、進度條、中文 UI、結束彈窗、結果存檔
#
# 執行：
#   python demo_guided_training_ui_final.py <示範影片路徑(可選)>
# 例：
#   python demo_guided_training_ui_final.py exercise_videos/仰臥蹲.mp4

import os
import sys
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
# 路徑與字型
# =========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(_SCRIPT_DIR, "front", "Noto_Sans_TC", "static", "NotoSansTC-Regular.ttf")
_font_cache = {}


def _get_font(size: int):
    if size not in _font_cache:
        if os.path.isfile(FONT_PATH):
            _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def put_chinese_text(img_bgr, text, pos_bottom_left, font_size, color_bgr, thickness=2):
    """在 OpenCV 畫布上繪製中文（Pillow）。pos_bottom_left = (x, y) 與 cv2.putText 一致。"""
    if not text:
        return
    font = _get_font(font_size)
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return

    pil_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)

    t = max(0, int(thickness or 0))
    if t > 0:
        for dy in range(-t, t + 1):
            for dx in range(-t, t + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((-bbox[0] + dx, -bbox[1] + dy), text, font=font, fill=(0, 0, 0, 180))

    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0], 255))

    arr = np.array(pil_img)
    x, y = pos_bottom_left
    y_top = y - h

    x0 = max(0, x)
    y0 = max(0, y_top)
    x1 = min(img_bgr.shape[1], x + w)
    y1 = min(img_bgr.shape[0], y_top + h)
    if x0 >= x1 or y0 >= y1:
        return

    sy0 = y0 - y_top
    sx0 = x0 - x
    sh = y1 - y0
    sw = x1 - x0

    roi = img_bgr[y0:y1, x0:x1]
    alpha = (arr[sy0:sy0 + sh, sx0:sx0 + sw, 3:4] / 255.0)
    rgb = arr[sy0:sy0 + sh, sx0:sx0 + sw, :3][:, :, ::-1]
    roi[:] = (roi * (1 - alpha) + rgb * alpha).astype(np.uint8)


# =========================
# 基本設定（可調）
# =========================
MODEL_PATH = os.path.join(_SCRIPT_DIR, "yolo11n-pose.pt")  # 你也可改 yolov8n-pose.pt
EXERCISE_DIR = os.path.join(_SCRIPT_DIR, "exercise_videos")

DISPLAY_W = 1280
DISPLAY_H = 720
INNER_HEIGHT = 480
BACKGROUND_COLOR = (25, 25, 25)

WINDOW_NAME = "AI 智慧復健跟練系統（Final - No JSON）"

FLIP_USER_VIEW = True

# 分數動畫平滑係數
SMOOTH_FACTOR = 0.08

# 跟練時長
SESSION_DURATION = 30  # 秒
END_HOLD_SECONDS = 3.0

# series
MAX_SERIES_LEN = 300

# peaks
PEAK_DISTANCE = 20
PEAK_PROMINENCE = 5

# DTW
DTW_MIN_LEN = 30               # 至少累積多少點才開始 DTW
DTW_SCORE_SCALE = 5.0          # normalized distance * scale
DTW_SCORE_BIAS = 100.0         # 100 - normalized*scale
DTW_SMOOTH_MINLEN = 10         # reference_series 還太短時不做 DTW


# =========================
# 工具函式
# =========================
def calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    dot = float(np.dot(ba, bc))
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))

    if norm_ba == 0 or norm_bc == 0:
        return None

    cos_angle = dot / (norm_ba * norm_bc)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def load_videos():
    if not os.path.isdir(EXERCISE_DIR):
        return []
    files = [f for f in os.listdir(EXERCISE_DIR) if f.lower().endswith(".mp4")]
    files.sort()
    return [os.path.join(EXERCISE_DIR, f) for f in files]


def draw_score_circle(img, center, radius, score):
    score = float(np.clip(score, 0, 100))
    angle = int(360 * score / 100)

    if score >= 80:
        color = (0, 200, 0)
    elif score >= 60:
        color = (0, 180, 255)
    else:
        color = (0, 0, 255)

    cv2.circle(img, center, radius, (60, 60, 60), 8)
    cv2.ellipse(img, center, (radius, radius), -90, 0, angle, color, 8)

    text = f"{int(score)}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    tx = int(center[0] - tw / 2)
    ty = int(center[1] + th / 2)
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


def draw_progress_bar(img, progress):
    progress = float(np.clip(progress, 0, 1))
    bar_w = int(DISPLAY_W * 0.7)
    bar_x = int(DISPLAY_W * 0.15)
    bar_y = DISPLAY_H - 60
    bar_h = 12

    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * progress)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (255, 255, 255), -1)


def safe_resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h <= 0:
        return img
    scale = target_h / h
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_h))


def extract_right_knee_angle_from_results(results_obj):
    """
    Ultralytics Pose keypoints index assumed 17 points:
      12 hip, 14 knee, 16 ankle (right side in COCO format)
    """
    try:
        kpts = results_obj.keypoints
        if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
            return None
        pts = kpts.xy[0].cpu().numpy()
        if pts.shape[0] < 17:
            return None
        hip = pts[12]
        knee = pts[14]
        ankle = pts[16]
        return calculate_angle(hip, knee, ankle)
    except Exception:
        return None


def compute_dtw_similarity(r_ref, r_user):
    """
    DTW distance -> similarity score (0~100)
    """
    if len(r_ref) == 0 or len(r_user) == 0:
        return 0.0

    L = min(len(r_ref), len(r_user))
    if L <= 1:
        return 0.0

    # fastdtw expects sequences
    distance, _ = fastdtw(r_ref, r_user, dist=euclidean)
    normalized = float(distance) / (L + 1e-6)
    score = max(0.0, DTW_SCORE_BIAS - normalized * DTW_SCORE_SCALE)
    return float(np.clip(score, 0, 100))


# =========================
# 主程式
# =========================
def main():
    # 0) 影片選擇
    if len(sys.argv) >= 2:
        video_path = sys.argv[1].strip()
        if not os.path.isabs(video_path):
            video_path = os.path.normpath(os.path.join(_SCRIPT_DIR, video_path))
        if not os.path.isfile(video_path):
            print("找不到指定影片:", sys.argv[1])
            return
    else:
        video_list = load_videos()
        if not video_list:
            print("找不到示範影片，請確認資料夾：", EXERCISE_DIR)
            return
        video_path = video_list[0]

    # 1) 載入模型（GPU 會自動用 CUDA；你看到 pynvml warning 可忽略）
    if not os.path.isfile(MODEL_PATH):
        print("找不到 YOLO 模型:", MODEL_PATH)
        return

    print("載入 YOLO 模型中...")
    model = YOLO(MODEL_PATH)

    # 2) 開啟示範影片 + 攝影機
    cap_ref = cv2.VideoCapture(video_path)
    if not cap_ref.isOpened():
        print("無法開啟示範影片:", video_path)
        return

    cap_user = cv2.VideoCapture(0)
    if not cap_user.isOpened():
        print("無法開啟攝影機")
        cap_ref.release()
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # 3) series & state
    reference_series = []  # 即時模板（示範影片膝角）
    user_series = []       # 使用者膝角

    similarity_score = 0.0
    display_similarity = 0.0

    real_score = 0.0
    display_score = 0.0

    rep_counter = 0

    sim_sum = 0.0
    sim_count = 0

    start_time = time.time()
    session_end_flag = False
    session_end_time = None
    saved_summary = False

    ref_fail_count = 0

    # 4) loop
    while True:
        # --- 讀示範影片 ---
        ret_ref, frame_ref = cap_ref.read()
        if not ret_ref:
            ref_fail_count += 1
            cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if ref_fail_count > 30:
                print("示範影片連續讀取失敗，請檢查檔案/codec：", video_path)
                break
            continue
        ref_fail_count = 0

        # --- 讀使用者 ---
        ret_user, frame_user = cap_user.read()
        if not ret_user:
            break
        if FLIP_USER_VIEW:
            frame_user = cv2.flip(frame_user, 1)

        # 示範影片 YOLO → 骨架畫面 + reference_angle
        ref_results = model(frame_ref, verbose=False)
        frame_ref_pose = ref_results[0].plot()
        ref_angle = extract_right_knee_angle_from_results(ref_results[0])
        if ref_angle is not None:
            reference_series.append(float(ref_angle))
            if len(reference_series) > MAX_SERIES_LEN:
                reference_series.pop(0)

        # 使用者 YOLO → 骨架畫面 + user_angle
        user_results = model(frame_user, verbose=False)
        frame_user_pose = user_results[0].plot()
        user_angle = extract_right_knee_angle_from_results(user_results[0])

        feedback_text = "請開始動作"
        real_score = 0.0

        # --- 更新 user_series / rep ---
        if user_angle is not None:
            user_series.append(float(user_angle))
            if len(user_series) > MAX_SERIES_LEN:
                user_series.pop(0)

            if len(user_series) > 50:
                arr = np.array(user_series, dtype=np.float32)
                peaks, _ = find_peaks(-arr, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
                rep_counter = int(len(peaks))

        # --- DTW：reference vs user（同長度尾端對齊）---
        if (len(reference_series) >= DTW_SMOOTH_MINLEN) and (len(user_series) >= DTW_MIN_LEN):
            L = min(len(reference_series), len(user_series), MAX_SERIES_LEN)
            r_ref = reference_series[-L:]
            r_user = user_series[-L:]

            similarity_score = compute_dtw_similarity(r_ref, r_user)
            sim_sum += float(similarity_score)
            sim_count += 1

            # 以相似度當作主要分數（你也可混合角度 diff）
            real_score = similarity_score

            # 回饋文字（用相似度）
            if similarity_score >= 80:
                feedback_text = "動作符合示範影片"
            elif similarity_score >= 60:
                feedback_text = "動作不錯，可以再穩定一點"
            else:
                feedback_text = "動作差距較大，請跟著節奏調整"
        else:
            # 還在暖機：模板/使用者角度不夠
            if len(reference_series) < DTW_SMOOTH_MINLEN:
                feedback_text = "示範模板建立中..."
            elif len(user_series) < DTW_MIN_LEN:
                feedback_text = "請先做幾下動作，建立使用者序列..."

        # --- 平滑動畫 ---
        display_score += (real_score - display_score) * SMOOTH_FACTOR
        display_similarity += (similarity_score - display_similarity) * SMOOTH_FACTOR

        # --- 縮放左右畫面 ---
        frame_ref_pose = safe_resize_to_height(frame_ref_pose, INNER_HEIGHT)
        frame_user_pose = safe_resize_to_height(frame_user_pose, INNER_HEIGHT)
        combined = np.hstack([frame_ref_pose, frame_user_pose])

        # --- Canvas 合成 ---
        canvas = np.full((DISPLAY_H, DISPLAY_W, 3), BACKGROUND_COLOR, dtype=np.uint8)

        h_c, w_c = combined.shape[:2]
        offset_x = max(0, (DISPLAY_W - w_c) // 2)
        offset_y = 80
        x1 = min(DISPLAY_W, offset_x + w_c)
        y1 = min(DISPLAY_H, offset_y + h_c)

        canvas[offset_y:y1, offset_x:x1] = combined[: (y1 - offset_y), : (x1 - offset_x)]

        # --- UI ---
        put_chinese_text(canvas, "AI 智慧復健跟練系統（示範=即時模板）", (40, 50), 26, (255, 255, 255))
        put_chinese_text(canvas, f"示範影片：{os.path.basename(video_path)}", (40, 80), 18, (180, 180, 180), thickness=1)

        draw_score_circle(canvas, (1100, 200), 60, display_score)

        put_chinese_text(canvas, "動作回饋：", (40, 600), 22, (200, 200, 200))
        put_chinese_text(canvas, feedback_text, (180, 600), 22, (255, 255, 255))

        put_chinese_text(canvas, f"動作相似度: {int(display_similarity)}%", (900, 350), 22, (255, 255, 255))
        put_chinese_text(canvas, f"完成次數: {rep_counter}", (900, 400), 22, (0, 200, 255))

        # --- 進度條 ---
        elapsed = time.time() - start_time
        progress = min(elapsed / SESSION_DURATION, 1.0)
        draw_progress_bar(canvas, progress)

        # --- 結束狀態判斷 ---
        if elapsed >= SESSION_DURATION and not session_end_flag:
            session_end_flag = True
            session_end_time = time.time()

        if session_end_flag:
            cv2.rectangle(canvas, (300, 200), (980, 520), (20, 20, 20), -1)
            put_chinese_text(canvas, "訓練完成", (500, 260), 34, (255, 255, 255))

            avg_sim = (sim_sum / (sim_count + 1e-6)) if sim_count > 0 else 0.0
            put_chinese_text(canvas, f"平均相似度: {int(avg_sim)}%", (480, 340), 26, (0, 255, 0))
            put_chinese_text(canvas, f"總完成次數: {rep_counter}", (500, 390), 26, (0, 200, 255))
            put_chinese_text(canvas, "3 秒後自動結束（或按 Q）", (420, 460), 20, (200, 200, 200), thickness=1)

            if not saved_summary:
                summary = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "video": os.path.basename(video_path),
                    "average_similarity": float(avg_sim),
                    "repetition_count": int(rep_counter),
                    "duration_seconds": float(elapsed),
                    "note": "reference_series comes from reference video YOLO in real-time (no JSON)."
                }
                os.makedirs(os.path.join(_SCRIPT_DIR, "results"), exist_ok=True)
                filename = os.path.join(_SCRIPT_DIR, time.strftime("results/session_%Y%m%d_%H%M%S.json"))
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print("成績已儲存:", filename)
                saved_summary = True

            if session_end_time is not None and (time.time() - session_end_time) >= END_HOLD_SECONDS:
                cv2.imshow(WINDOW_NAME, canvas)
                cv2.waitKey(1)
                break

        # --- 顯示 ---
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap_ref.release()
    cap_user.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()