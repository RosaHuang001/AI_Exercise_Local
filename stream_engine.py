import os
import time
import json
import threading
from queue import Queue, Empty
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks
from PIL import Image, ImageDraw, ImageFont

# =========================
# 路徑與設定
# =========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 專案根：若本檔在專案根則用 _SCRIPT_DIR，若在 modules/ 則用上一層，避免變成 D:\
_PROJECT_ROOT = _SCRIPT_DIR if os.path.isdir(os.path.join(_SCRIPT_DIR, "exercise_videos")) else os.path.dirname(_SCRIPT_DIR)
FONT_PATH = os.path.join(_SCRIPT_DIR, "front", "Noto_Sans_TC", "static", "NotoSansTC-Regular.ttf")
_font_cache = {}

MODEL_PATH = os.path.join(_SCRIPT_DIR, "yolo11n-pose.pt")
DISPLAY_W, DISPLAY_H = 1280, 720
INNER_HEIGHT = 480
HALF_W = DISPLAY_W // 2  # 1:1 畫面分配，左右各 640px
BACKGROUND_COLOR = (25, 25, 25)
FLIP_USER_VIEW = True
END_HOLD_SECONDS = 3.0
REST_DURATION = 5.0

PEAK_DISTANCE, PEAK_PROMINENCE = 20, 5

# 🔥 [效能優化參數] 小圖推論減少 LAG
YOLO_IMGSZ = 320
YOLO_INPUT_W, YOLO_INPUT_H = 320, 240  # 縮小輸入尺寸以加速推論
# 跳幀：每 N 幀才跑一次 YOLO，中間幀沿用上次結果，運算量約降 50%
YOLO_SKIP_FRAMES = 2
# 骨架繪製信心門檻（光線不足或距離遠時可調低以顯示骨架）
POSE_CONF_THRESHOLD = 0.3
# 是否使用背景執行緒跑 YOLO（編碼不等待 AI，減少卡頓）
USE_YOLO_THREAD = True

# 跟練 session 統計（供前端成就報告使用）
LAST_SESSION_STATS = {
    "valid_reps": 0,
    "total_reps": 0,
    "failed_reps": 0,
    "ended": False,
    "updated_at": 0.0,
}


def _update_session_stats(valid: int, total: int, ended: bool = False):
    """更新全域 session 統計，讓 API 可以提供 /session_stats 查詢。"""
    valid = int(max(0, valid or 0))
    total = int(max(valid, total or 0))
    failed = max(0, total - valid)
    LAST_SESSION_STATS.update(
        {
            "valid_reps": valid,
            "total_reps": total,
            "failed_reps": failed,
            "ended": bool(ended),
            "updated_at": time.time(),
        }
    )


def get_last_session_stats():
    """回傳最新一筆 session 統計（淺複製避免被外部修改）。"""
    return dict(LAST_SESSION_STATS)


def _yolo_worker(model, in_queue, result_holder, lock, stop_event):
    """背景執行緒：從 queue 取幀、跑 YOLO、寫入 result_holder。"""
    while not stop_event.is_set():
        try:
            frame = in_queue.get(timeout=0.1)
        except Empty:
            continue
        if frame is None:
            break
        small = cv2.resize(frame, (YOLO_INPUT_W, YOLO_INPUT_H))
        try:
            res = model(small, verbose=False, imgsz=YOLO_IMGSZ)
            angle = extract_right_knee_angle_from_results(res[0])
            with lock:
                result_holder["result0"] = res[0]
                result_holder["angle"] = angle
        except Exception:
            pass
    in_queue.task_done()


def _person_detected(keypoints_obj, min_conf_mean=0.15):
    """是否偵測到可繪製的人體（用於顯示「請退後/環境太暗」提示）。"""
    if keypoints_obj is None or keypoints_obj.conf is None:
        return False
    try:
        conf = keypoints_obj.conf[0].cpu().numpy()
        return float(conf.mean()) >= min_conf_mean
    except Exception:
        return False


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

def draw_progress_bar(img, progress):
    progress = float(np.clip(progress, 0, 1))
    bar_w, bar_x, bar_y, bar_h = int(DISPLAY_W * 0.7), int(DISPLAY_W * 0.15), DISPLAY_H - 40, 8
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (255, 150, 0), -1)

def _resolve_video_path(rel_path: str) -> str:
    """示範影片一律使用 exercise_videos 的無骨架影片。"""
    if os.path.isabs(rel_path):
        return rel_path
    for root in (_PROJECT_ROOT, os.path.dirname(_PROJECT_ROOT)):
        full = os.path.join(root, rel_path)
        if os.path.isfile(full):
            return full
    return os.path.join(_PROJECT_ROOT, rel_path)

def crop_and_resize(img, target_w, target_h):
    """【新增】確保畫面完美 1:1 的智慧裁切與縮放"""
    if img is None or img.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    img_aspect = w / h
    target_aspect = target_w / target_h
    
    if img_aspect > target_aspect:
        # 圖片太寬，裁切左右
        new_w = int(h * target_aspect)
        offset = (w - new_w) // 2
        cropped = img[:, offset:offset+new_w]
    else:
        # 圖片太高，裁切上下
        new_h = int(w / target_aspect)
        offset = (h - new_h) // 2
        cropped = img[offset:offset+new_h, :]
        
    return cv2.resize(cropped, (target_w, target_h))

def extract_right_knee_angle_from_results(results_obj):
    try:
        kpts = results_obj.keypoints
        if not kpts or kpts.xy is None or len(kpts.xy) == 0: return None
        pts = kpts.xy[0].cpu().numpy()
        if pts.shape[0] < 17: return None
        return calculate_angle(pts[12], pts[14], pts[16])
    except: return None




def draw_colored_pose(frame, keypoints_obj, color_bgr, scale_x=1.0, scale_y=1.0):
    """根據準確度顏色繪製骨架；支援 scale 以配合小圖推論後座標放大。"""
    if keypoints_obj is None or keypoints_obj.xy is None:
        return frame
    try:
        pts = keypoints_obj.xy[0].cpu().numpy().astype(np.float64)
        conf = keypoints_obj.conf[0].cpu().numpy()
    except Exception:
        return frame
    if pts.shape[0] < 17:
        return frame
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y

    connections = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (5, 11), (6, 12)
    ]
    th = POSE_CONF_THRESHOLD
    for start, end in connections:
        if conf[start] > th and conf[end] > th:
            p1 = (int(pts[start][0]), int(pts[start][1]))
            p2 = (int(pts[end][0]), int(pts[end][1]))
            cv2.line(frame, p1, p2, color_bgr, 3)
    for i in range(5, 17):
        if conf[i] > th:
            cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 5, (255, 255, 255), -1)
    return frame



# =========================
# 核心串流產生器 (有效次數與骨架變色強化版)
# =========================
def generate_frames(playlist, nyha_level="class_ii"):
    """
    nyha_level: "class_i" | "class_ii" | "class_iii"，可透過 API 的 URL 參數傳入。
    依 rom_standards 決定骨架變色與有效次數的判定門檻。
    """
    if not playlist: return

    playlist = [_resolve_video_path(p) for p in playlist]
    lib_path = os.path.join(_PROJECT_ROOT, "knowledge_base", "exercise_library.json")
    try:
        with open(lib_path, 'r', encoding='utf-8') as f:
            lib_data = json.load(f)
        exercise_list = lib_data.get("exercises", []) if isinstance(lib_data, dict) else lib_data
    except Exception:
        exercise_list = []

    model = YOLO(MODEL_PATH)
    # 背景 YOLO 用：queue + 結果暫存
    yolo_queue = Queue(maxsize=1) if USE_YOLO_THREAD else None
    yolo_result = {} if USE_YOLO_THREAD else None
    yolo_lock = threading.Lock() if USE_YOLO_THREAD else None
    yolo_stop = threading.Event() if USE_YOLO_THREAD else None
    yolo_thread = None
    if USE_YOLO_THREAD:
        yolo_thread = threading.Thread(
            target=_yolo_worker,
            args=(model, yolo_queue, yolo_result, yolo_lock, yolo_stop),
            daemon=True,
        )
        yolo_thread.start()
    # 跳幀用快取（未開 thread 時使用）
    cached_result0 = None
    cached_angle = None
    frame_counter = 0

    cap_user = cv2.VideoCapture(0)
    if not cap_user.isOpened():
        # 攝影機未開啟時持續回傳說明幀，避免前端全黑或串流結束，同時將次數統計歸零
        _update_session_stats(0, 0, ended=True)
        err_canvas = np.full((DISPLAY_H, DISPLAY_W, 3), (25, 25, 25), dtype=np.uint8)
        put_chinese_text(err_canvas, "無法開啟攝影機，請檢查權限或連接裝置", (DISPLAY_W//2 - 280, DISPLAY_H//2 - 30), 28, (255, 255, 255))
        put_chinese_text(err_canvas, "左：教練示範 | 右：您的畫面（需攝影機）", (DISPLAY_W//2 - 220, DISPLAY_H//2 + 20), 22, (180, 180, 180))
        while True:
            _, buf = cv2.imencode(".jpg", err_canvas)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.033)
        return

    current_vid_idx = 0
    ref_path = playlist[current_vid_idx]
    cap_ref = cv2.VideoCapture(ref_path)
    if not cap_ref.isOpened():
        # 示範影片開不起來時不進入主迴圈，否則會立刻被當成「影片結束」而跳出「我已訓練完畢」
        print(f"[stream_engine] 無法開啟示範影片: {ref_path}")
        _update_session_stats(0, 0, ended=True)
        err_canvas = np.full((DISPLAY_H, DISPLAY_W, 3), (25, 25, 25), dtype=np.uint8)
        put_chinese_text(err_canvas, "示範影片無法載入", (DISPLAY_W//2 - 120, DISPLAY_H//2 - 40), 28, (255, 255, 255))
        put_chinese_text(err_canvas, "請確認專案內 exercise_videos 資料夾有對應 .mp4", (DISPLAY_W//2 - 240, DISPLAY_H//2), 22, (180, 180, 180))
        put_chinese_text(err_canvas, f"路徑：{ref_path}", (DISPLAY_W//2 - 320, DISPLAY_H//2 + 40), 18, (120, 120, 120))
        while True:
            _, buf = cv2.imencode(".jpg", err_canvas)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.033)
        return

    # 狀態變數：優化為有效次數計算
    is_resting = False
    rest_start_time = 0
    user_series = []
    accuracy_history = []
    total_valid_reps = 0      # 總有效次數 (所有動作累計)
    current_valid_reps = 0    # 當前動作的有效次數
    total_raw_reps = 0        # 總動作次數（包含未達標）
    last_peak_count = 0       # 記錄上一次偵測到的原始動作次數

    session_end_flag = False
    session_end_time = None

    try:
        while True:
            # 1. 讀取並處理使用者畫面
            ret_user, frame_user = cap_user.read()
            if not ret_user: break
            if FLIP_USER_VIEW: frame_user = cv2.flip(frame_user, 1)
            scale_x = frame_user.shape[1] / YOLO_INPUT_W
            scale_y = frame_user.shape[0] / YOLO_INPUT_H

            # 執行 YOLO：背景執行緒（不擋編碼）或跳幀（每 N 幀跑一次）
            if USE_YOLO_THREAD and yolo_queue is not None:
                try:
                    yolo_queue.put_nowait(frame_user.copy())
                except Exception:
                    pass
                with yolo_lock:
                    result0 = yolo_result.get("result0")
                    user_angle = yolo_result.get("angle")
                if result0 is None:
                    small_frame = cv2.resize(frame_user, (YOLO_INPUT_W, YOLO_INPUT_H))
                    user_results = model(small_frame, verbose=False, imgsz=YOLO_IMGSZ)
                    user_angle = extract_right_knee_angle_from_results(user_results[0])
                    result0 = user_results[0]
                    with yolo_lock:
                        yolo_result["result0"] = result0
                        yolo_result["angle"] = user_angle
                user_results = [result0] if result0 is not None else []
            else:
                frame_counter += 1
                if frame_counter % YOLO_SKIP_FRAMES == 0 or cached_result0 is None:
                    small_frame = cv2.resize(frame_user, (YOLO_INPUT_W, YOLO_INPUT_H))
                    user_results = model(small_frame, verbose=False, imgsz=YOLO_IMGSZ)
                    cached_result0 = user_results[0]
                    cached_angle = extract_right_knee_angle_from_results(cached_result0)
                user_results = [cached_result0] if cached_result0 is not None else []
                user_angle = cached_angle

            # 2. 處理示範影片與即時比對邏輯
            display_acc = 0
            accuracy_color_bgr = (0, 0, 255) # 預設紅色 (角度不足)

            if is_resting:
                # 休息邏輯保持不變
                rest_elapsed = time.time() - rest_start_time
                remain = max(0, REST_DURATION - rest_elapsed)
                frame_ref_display = np.zeros((INNER_HEIGHT, HALF_W, 3), dtype=np.uint8)
                put_chinese_text(frame_ref_display, f"休息時間: {int(remain)} 秒", (HALF_W//2 - 120, INNER_HEIGHT//2), 40, (0, 255, 255))
                
                if remain <= 0:
                    is_resting = False
                    current_vid_idx += 1
                    cap_ref = cv2.VideoCapture(playlist[current_vid_idx])
                    user_series.clear()
                    total_valid_reps += current_valid_reps
                    current_valid_reps = 0
                    last_peak_count = 0 # 重置計數器
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
                            total_valid_reps += current_valid_reps
                            session_end_flag = True
                            session_end_time = time.time()
                        frame_ref_display = np.zeros((INNER_HEIGHT, HALF_W, 3), dtype=np.uint8)
                        put_chinese_text(frame_ref_display, " 示範影片結束或無法讀取 ", (HALF_W//2 - 140, INNER_HEIGHT//2 - 20), 22, (200, 200, 200))
                else:
                    frame_ref_display = frame_ref
                    
                    # --- 依 NYHA 分級抓取門檻角度與準確度判定 ---
                    curr_file = os.path.basename(playlist[current_vid_idx])
                    # 支援 *_yolo.mp4 檔名對應 library 的 video_filename（xxx.mp4）
                    curr_logical = curr_file.replace("_yolo.", ".") if "_yolo." in curr_file else curr_file
                    target_ex = next((x for x in exercise_list if x.get("video_filename") == curr_logical or x.get("video_filename") == curr_file), {})
                    standards = target_ex.get("rom_standards", {})
                    ref_standard_angle = float(standards.get(nyha_level, target_ex.get("rom_p5_p95", 160.0)))

                    if user_angle is not None:
                        angle_diff = abs(user_angle - ref_standard_angle)
                        # 誤差 ≤8 度或達到分級門檻即為達標（綠色）；否則依誤差算準確度（紅色）
                        if angle_diff <= 8 or user_angle >= ref_standard_angle:
                            display_acc = 100
                            accuracy_color_bgr = (0, 255, 0)
                        else:
                            display_acc = int(max(0, 100 - (angle_diff / 45 * 100)))
                            accuracy_color_bgr = (0, 0, 255)

                        # --- 有效次數：僅在「綠色骨架」當下偵測到新峰值才計入 ---
                        user_series.append(float(user_angle))
                        if len(user_series) > 50:
                            arr = np.array(user_series, dtype=np.float32)
                            peaks, _ = find_peaks(-arr, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
                            new_peak_count = len(peaks)
                            if new_peak_count > last_peak_count:
                                delta = new_peak_count - last_peak_count
                                total_raw_reps += max(0, delta)
                                if accuracy_color_bgr == (0, 255, 0):
                                    current_valid_reps += max(0, delta)
                                last_peak_count = new_peak_count

            # 3. 繪製自定義顏色骨架（座標從小圖放大回原畫面）
            kpts = user_results[0].keypoints if len(user_results) > 0 else None
            frame_user_pose = draw_colored_pose(
                frame_user.copy(), kpts, accuracy_color_bgr,
                scale_x=scale_x, scale_y=scale_y
            )

            # 4. 畫面裁切與拼接
            frame_ref_crop = crop_and_resize(frame_ref_display, HALF_W, INNER_HEIGHT)
            frame_user_crop = crop_and_resize(frame_user_pose, HALF_W, INNER_HEIGHT)

            put_chinese_text(frame_ref_crop, " 教練示範 ", (20, 50), 24, (255, 255, 255), thickness=2)
            put_chinese_text(frame_user_crop, " 您的畫面 ", (20, 50), 24, (255, 255, 255), thickness=2)
            # 健康行為學：偵測不到人時給予明確指引
            if not is_resting and not session_end_flag and not _person_detected(kpts):
                put_chinese_text(frame_user_crop, "請退後一點或改善光線", (HALF_W//2 - 140, INNER_HEIGHT//2 - 10), 22, (200, 200, 0))
            
            # 右側畫面顯示即時品質狀態
            if not is_resting and not session_end_flag and user_angle is not None:
                status_text = "動作標準" if display_acc >= 80 else "角度不足"
                put_chinese_text(frame_user_crop, f"品質：{status_text}", (20, 100), 28, accuracy_color_bgr, thickness=2)

            combined = np.hstack([frame_ref_crop, frame_user_crop])

            # 5. 繪製畫布資訊 (頂部同步顯示次數)
            canvas = np.full((DISPLAY_H, DISPLAY_W, 3), BACKGROUND_COLOR, dtype=np.uint8)
            canvas[80:80+INNER_HEIGHT, 0:DISPLAY_W] = combined

            # 頂部：有效次數顯示
            put_chinese_text(canvas, f"AI 智慧跟練 - 動作 {current_vid_idx+1}/{len(playlist)}", (40, 50), 26, (255, 255, 255))
            put_chinese_text(canvas, f"本組有效達標：{current_valid_reps} 次", (880, 50), 26, (0, 255, 255), thickness=2)

            # 底部與結算邏輯
            if session_end_flag:
                cv2.rectangle(canvas, (300, 200), (980, 520), (20, 20, 20), -1)
                put_chinese_text(canvas, "今日訓練完成！您的復健成果：", (420, 280), 30, (255, 255, 255))
                final_valid = total_valid_reps + current_valid_reps
                final_total = max(total_raw_reps, final_valid)
                put_chinese_text(canvas, f"總有效達標次數：{final_valid} 次", (460, 380), 36, (0, 255, 255), thickness=2)
            else:
                progress_text = "休息中..." if is_resting else f"累積總計：{total_valid_reps + current_valid_reps} 次"
                put_chinese_text(canvas, progress_text, (DISPLAY_W // 2 - 150, 640), 32, (200, 200, 200))

            draw_progress_bar(canvas, (current_vid_idx + (0.5 if is_resting else 0)) / len(playlist) if not session_end_flag else 1.0)

            # 每幀同步更新一次 session 統計，供前端查詢
            current_valid_total = total_valid_reps + current_valid_reps
            current_total = max(total_raw_reps, current_valid_total)
            _update_session_stats(current_valid_total, current_total, ended=session_end_flag)

            # 6. JPG 串流發送
            ret, buffer = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret: continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            if session_end_flag and (time.time() - session_end_time) >= END_HOLD_SECONDS:
                break
    finally:
        if USE_YOLO_THREAD and yolo_queue is not None:
            yolo_stop.set()
            try:
                yolo_queue.put(None, timeout=0.5)
            except Exception:
                pass
            if yolo_thread and yolo_thread.is_alive():
                yolo_thread.join(timeout=1.0)
        cap_user.release()
        if cap_ref: cap_ref.release()