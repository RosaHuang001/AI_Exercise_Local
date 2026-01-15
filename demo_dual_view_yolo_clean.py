#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ↑ 指定 Python 執行環境，並設定檔案編碼為 UTF-8，以避免中文註解亂碼

import cv2                  # 匯入 OpenCV，用於讀取影片、處理影像、顯示畫面
import numpy as np          # 匯入 NumPy，用於影像的矩陣運算與組合
from ultralytics import YOLO # 匯入 Ultralytics 套件中的 YOLO 模型（我們用 YOLOv11 Pose）
import os                   # Python 內建 OS 模組，用於與檔案系統互動（讀資料夾）
import time                 # 用於計算播放節奏

# ====== 參數設定 ======

MODEL_PATH = "yolo11n-pose.pt" 
# YOLOv11 pose 模型檔案名稱（需放在專案根目錄）

EXERCISE_DIR = "exercise_videos"
# 主人剪好的示範動作影片資料夾名稱

INNER_HEIGHT = 480
# 左右兩個影片（示範影片 + 使用者骨架）在合併前會被統一縮放到這個高度

DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720
# 組合後的畫面如果太大，就會再縮到螢幕可接受的最大尺寸

BACKGROUND_COLOR = (40, 40, 40)
# 背景顏色（深灰），用來當 1280x720 畫布的底色

FLIP_USER_VIEW = True
# True = 翻轉右側 webcam 畫面（像鏡子）→ 使用者看起來比較直覺

WINDOW_NAME = "HF Rehab Demo - 左：示範影片 / 右：使用者骨架"
# 影片視窗標題

# 播放速度設定
MIN_PLAYBACK_SPEED = 0.25
MAX_PLAYBACK_SPEED = 2.0
PLAYBACK_STEP = 0.25

# 滑鼠狀態（用來處理拖曳進度條）
mouse_x = 0
mouse_y = 0
mouse_event = None   # "down", "drag", "up"
mouse_is_down = False


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_event, mouse_is_down

    mouse_x = x
    mouse_y = y

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_is_down = True
        mouse_event = "down"
        print(f"[DEBUG] Mouse down at ({mouse_x}, {mouse_y})")
    elif event == cv2.EVENT_MOUSEMOVE and mouse_is_down:
        mouse_event = "drag"
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_is_down = False
        mouse_event = "up"



def load_all_videos():
    """
    從 exercise_videos 資料夾讀取所有 MP4 檔案
    依名稱排序（方便老師示範時切換動作）
    回傳：完整路徑的 list，例如：
    ["exercise_videos/squat.mp4", "exercise_videos/pushup.mp4", ...]
    """
    videos = [f for f in os.listdir(EXERCISE_DIR)
              if f.lower().endswith(".mp4")]
    # ↑ 讀所有檔案，並過濾出 .mp4 結尾的檔案

    videos.sort()
    # 排序：確保切換影片時順序一致

    full_paths = [os.path.join(EXERCISE_DIR, v) for v in videos]
    # 加上資料夾路徑 → 變成完整路徑

    return full_paths


def open_video(path):
    """
    開啟影片的 function
    輸入：影片路徑
    回傳：
        cap：OpenCV 影片物件
        total_frames：影片總影格數（用於計算進度）
        fps：影片每秒影格數（用於時間換算）
    """

    cap = cv2.VideoCapture(path)      # 嘗試打開影片
    if not cap.isOpened():            # 如果開啟失敗
        print(f"[ERROR] 無法開啟影片：{path}")
        return None, 0, 0.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    # 取得影片總影格，若取得失敗回傳 0（避免除以零）

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 取得 FPS，若影片 metadata 取不到，預設用 30 FPS（常見標準 FPS）

    return cap, total_frames, fps


def load_video_at_index(video_list, index, current_cap=None):
    """
    依序開啟指定 index 的影片並讀取第一幀，必要時釋放上一支影片。
    回傳：(cap, total_frames, fps, first_frame)
    """
    if current_cap is not None:
        current_cap.release()

    video_path = video_list[index]
    cap, total_frames, fps = open_video(video_path)
    if cap is None:
        return None, 0, 0.0, None

    ret, frame = cap.read()
    if not ret:
        print(f"[ERROR] 無法讀取影片第一幀：{video_path}")
        cap.release()
        return None, 0, 0.0, None

    return cap, total_frames, fps, frame


def seek_video_by_ratio(cap, total_frames, ratio):
    """
    依照 0~1 的比例跳轉影片並讀取該位置的影格。
    成功時回傳影像陣列，失敗回傳 None。
    """
    if cap is None or total_frames <= 0:
        return None

    ratio = float(np.clip(ratio, 0.0, 1.0))
    target_frame = int(max(0, min(total_frames - 1, total_frames * ratio)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        print("[WARN] 無法跳轉到指定影格，可能是影片結尾。")
        return None

    return frame


def point_in_progress_bar(x, y, bar_info, margin=0):
    """
    確認滑鼠位置是否落在進度條範圍（可附加容許 margin）。
    """
    if bar_info is None:
        return False

    bar_x, bar_y, bar_w, bar_h = bar_info
    return (
        bar_y - margin <= y <= bar_y + bar_h + margin and
        bar_x <= x <= bar_x + bar_w
    )


def ratio_from_mouse_x(x, bar_info):
    """
    根據滑鼠 x 座標計算進度比值（自動 clamp 在 0~1）。
    """
    if bar_info is None:
        return 0.0

    bar_x, _, bar_w, _ = bar_info
    ratio = (x - bar_x) / float(bar_w)
    return float(np.clip(ratio, 0.0, 1.0))


def format_time_by_frames(frame_idx, fps):
    """
    將影格數轉換成 mm:ss 對應時間格式
    用於顯示在進度條旁邊
    """
    if fps <= 0:                      # FPS 不能 <= 0
        return "00:00"

    seconds = int(frame_idx / fps)   # frame ÷ fps = 秒數
    minute = seconds // 60           # 分
    sec = seconds % 60               # 秒

    return f"{minute:02d}:{sec:02d}"


def draw_video_progress_bar(frame, current_frame, total_frames, fps):
    """
    畫出「影片播放器風格」進度條。
    包含：
    - 底部灰色軌道
    - 白色進度條
    - 圓形進度球 scrubber
    - 左側目前播放時間
    - 右側總長度時間

    回傳：
    (更新後的 frame, (bar_x, bar_y, bar_width, bar_height))
    → 第二個資料是給滑鼠偵測用的（知道進度條在哪裡）
    """

    # 取得畫面高度與寬度，用來決定進度條要放的位置
    h, w = frame.shape[:2]

    # ====== 進度條位置與尺寸設定 ======
    bar_width = int(w * 0.70)   
    # 進度條寬度 = 畫面寬度的 70%

    bar_height = 10             
    # 進度條高度 = 10 pixels（薄薄一條）

    bar_x = int(w * 0.15)       
    # 左側留 15% 畫面當邊界 → 讓進度條置中

    bar_y = h - 40              
    # 進度條高度位置：靠近畫面最下方（留 40px 以免貼太底）

    # ====== 計算影片進度比例 ======
    if total_frames > 0:
        progress = min(max(current_frame / total_frames, 0.0), 1.0)
        # progress = 當前影格 ÷ 影片總影格數
        # 並限制在 0～1 範圍，避免超出
    else:
        progress = 0.0

    progress_width = int(bar_width * progress)
    # 白色進度條寬度 = bar_width * 影片播放比例

    # ====== 畫灰色底條（整段軌道） ======
    cv2.rectangle(
        frame,
        (bar_x, bar_y),                     # 左上角座標
        (bar_x + bar_width, bar_y + bar_height),  # 右下角座標
        (100, 100, 100),                    # 顏色（灰色）
        thickness=-1                        # 填滿
    )

    # ====== 畫白色進度條（已播放的部分） ======
    if progress_width > 0:
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + progress_width, bar_y + bar_height),
            (255, 255, 255),                # 白色
            thickness=-1                    # 填滿
        )

    # ====== 畫白色圓形 scrubber（進度球） ======
    scrub_x = bar_x + progress_width         # 球的中心 x 位置
    scrub_y = bar_y + bar_height // 2        # 球中心 y 位置（進度條中心）

    cv2.circle(
        frame,
        (scrub_x, scrub_y),                   # 圓中心位置
        7,                                    # 圓半徑
        (255, 255, 255),                      # 白色
        thickness=-1                          # 填滿
    )

    # ====== 計算播放時間（格式 mm:ss） ======
    current_time_str = format_time_by_frames(current_frame, fps)
    # 左側：目前時間（example: 00:15）

    total_time_str = format_time_by_frames(total_frames, fps)
    # 右側：影片總長度（example: 01:27）


    # ====== 畫出左側播放時間文字 ======
    cv2.putText(
        frame,
        current_time_str,                    # 要顯示的文字
        (bar_x - 70, bar_y + bar_height + 5), # 位置（左側偏左一點）
        cv2.FONT_HERSHEY_SIMPLEX,           # 字體
        0.55,                                # 字體大小
        (255, 255, 255),                     # 白色文字
        2,                                   # 線條粗細
        cv2.LINE_AA                          # 抗鋸齒
    )

    # ====== 畫出右側影片總長度 ======
    cv2.putText(
        frame,
        total_time_str,
        (bar_x + bar_width + 15, bar_y + bar_height + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame, (bar_x, bar_y, bar_width, bar_height)
    # 回傳：
    # 1. 已畫上進度條的 frame
    # 2. 進度條的位置 → 用來偵測滑鼠是否點到 bar





def calculate_angle(a, b, c):
    """
    給三個點 a, b, c（每個點是 (x, y)）
    回傳在 b 這個點的夾角（度數），也就是 ∠ABC

    a、b、c 會是關節位置，例如：
    a = 髖關節
    b = 膝蓋
    c = 踝關節
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    # 向量 BA = A - B，BC = C - B
    ba = a - b
    bc = c - b

    # 算內積與長度
    dot = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        # 有點重疊或偵測不到，避免除以 0
        return None

    cos_angle = dot / (norm_ba * norm_bc)
    # 數值保護，避免浮點數誤差 > 1
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)      # 弧度
    angle_deg = np.degrees(angle_rad)     # 轉成度數

    return float(angle_deg)





def main():
    """
    主程式入口：
    - 載入 YOLO 模型
    - 載入示範影片列表
    - 打開第一支影片與攝影機
    - 建立視窗與滑鼠事件
    - 進入主迴圈進行畫面更新、YOLO 偵測、進度條與互動控制
    """
    global mouse_event, mouse_x, mouse_y, mouse_is_down  # 使用全域變數來處理滑鼠事件狀態

    print("[INFO] 載入 YOLOv11 Pose 模型中...")
    model = YOLO(MODEL_PATH)  # 讀取 YOLO 模型權重（yolo11n-pose.pt）

    # 取得所有示範影片的完整路徑列表
    video_list = load_all_videos()
    if len(video_list) == 0:
        # 如果資料夾裡沒有任何 mp4，就直接結束程式
        print("[ERROR] 資料夾 exercise_videos 裡沒有 mp4 影片！")
        return

    print(f"[INFO] 偵測到 {len(video_list)} 支影片。")
    current_index = 0  # 目前播放第幾支示範影片（從 0 開始）

    # ====== 開啟第一支影片 ======
    cap_ref, total_frames, fps, frame_ref = load_video_at_index(video_list, current_index)
    if cap_ref is None:
        # 如果第一支影片開不起來就直接結束
        return

    print(
        f"[INFO] 使用示範影片：{video_list[current_index]} "
        f"(共 {total_frames} 影格, fps={fps:.1f})"
    )

    # ====== 開啟攝影機（使用者畫面） ======
    cap_user = cv2.VideoCapture(0)  # 0 = 預設 webcam
    if not cap_user.isOpened():
        print("[ERROR] 無法開啟攝影機。")
        cap_ref.release()
        return

    # 建立顯示視窗，WINDOW_NORMAL 代表可以調整大小（但我們建議不要手動拉）
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # 註冊滑鼠事件 callback，讓這個視窗可以回應滑鼠點擊
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    # 播放控制變數
    paused = False        # 是否暫停示範影片（True=暫停）
    playback_speed = 1.0  # 播放速度倍率
    time_accumulator = 0.0
    last_time = time.perf_counter()

    # 顯示使用說明在 terminal
    print("[INFO] 已啟動 Demo。控制方式：")
    print("   q：退出")
    print("空白鍵：暫停 / 繼續示範影片")
    print("1 / 2 / 3：快速切換 0.5x / 1x / 1.5x 倍速")
    print("- / = ：減速 / 加速（範圍 0.25x ~ 2.0x，每次 0.25x）")
    print("← / → 或 A / D / [ / ]：切換上一支 / 下一支示範影片")
    print("用滑鼠點擊 / 拖曳下方進度條即可跳轉時間")

    # 拖曳與進度條相關狀態
    scrubbing_active = False
    scrub_was_paused = False
    last_bar_info = None
    SCRUB_MARGIN = 15

    # ====== 進入主迴圈：持續更新畫面 ======
    while True:
        now = time.perf_counter()
        elapsed = now - last_time
        last_time = now

        # --------- 處理進度條拖曳事件 ---------
        event = mouse_event
        if event is not None and last_bar_info is not None:
            if event == "down":
                if point_in_progress_bar(mouse_x, mouse_y, last_bar_info, SCRUB_MARGIN):
                    scrubbing_active = True
                    scrub_was_paused = paused
                    paused = True
                    frame_after_seek = seek_video_by_ratio(
                        cap_ref,
                        total_frames,
                        ratio_from_mouse_x(mouse_x, last_bar_info),
                    )
                    if frame_after_seek is not None:
                        frame_ref = frame_after_seek
                    time_accumulator = 0.0
                    last_time = time.perf_counter()
            elif event == "drag" and scrubbing_active:
                frame_after_seek = seek_video_by_ratio(
                    cap_ref,
                    total_frames,
                    ratio_from_mouse_x(mouse_x, last_bar_info),
                )
                if frame_after_seek is not None:
                    frame_ref = frame_after_seek
                time_accumulator = 0.0
                last_time = time.perf_counter()
            elif event == "up":
                if scrubbing_active:
                    frame_after_seek = seek_video_by_ratio(
                        cap_ref,
                        total_frames,
                        ratio_from_mouse_x(mouse_x, last_bar_info),
                    )
                    if frame_after_seek is not None:
                        frame_ref = frame_after_seek
                    scrubbing_active = False
                    paused = scrub_was_paused
                    time_accumulator = 0.0
                    last_time = time.perf_counter()
            mouse_event = None
        elif event is not None:
            mouse_event = None

        # --------- 示範影片更新 (暫停 + 倍速) ---------
        reached_end = False
        effective_fps = fps if fps > 0 else 30.0
        effective_speed = max(playback_speed, MIN_PLAYBACK_SPEED)

        if paused:
            time_accumulator = 0.0
        else:
            time_accumulator += elapsed
            seconds_per_frame = 1.0 / (effective_fps * effective_speed)
            frames_to_step = int(time_accumulator / seconds_per_frame)

            if frames_to_step >= 1:
                time_accumulator -= frames_to_step * seconds_per_frame
                for _ in range(frames_to_step):
                    ret_ref, frame_candidate = cap_ref.read()
                    if not ret_ref:
                        reached_end = True
                        break
                    frame_ref = frame_candidate

        if reached_end:
            next_index = (current_index + 1) % len(video_list)
            cap_ref, total_frames, fps, frame_ref = load_video_at_index(
                video_list,
                next_index,
                cap_ref,
            )
            if cap_ref is None:
                break

            current_index = next_index
            time_accumulator = 0.0
            last_time = time.perf_counter()
            scrubbing_active = False
            print(
                f"[INFO] 自動切換到下一支影片：{video_list[current_index]} "
                f"(共 {total_frames} 影格, fps={fps:.1f})"
            )

        # --------- Webcam + YOLO ---------
        # 讀取使用者 webcam 畫面
        ret_user, frame_user = cap_user.read()
        if not ret_user:
            print("[ERROR] 無法讀取 webcam。")
            break

        if FLIP_USER_VIEW:
            # 若設定為 True，將畫面左右翻轉，使用者鏡像效果
            frame_user = cv2.flip(frame_user, 1)

        # 將使用者畫面丟進 YOLOv11 Pose 模型做骨架偵測
        results = model(frame_user, verbose=False)

        # 先畫出骨架圖（含關節點）
        frame_user_pose = results[0].plot()



        # ====== 從 YOLO keypoints 抓出關節座標來「算角度」 ======
        # Ultralytics YOLO pose 的 keypoints 以 COCO 格式為例：
        # 0 nose, 5 left_shoulder, 6 right_shoulder, 11 left_hip, 12 right_hip,
        # 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle, ...
        # 這裡示範算「右膝關節角度」：髖關節(hip) - 膝蓋(knee) - 踝(ankle)

        knee_angle = None  # 預設沒有角度

        try:
            kpts_obj = results[0].keypoints  # Keypoints 物件
            if kpts_obj is not None and len(kpts_obj.xy) > 0:
                # 取第一個人（假設畫面中只有一個主要使用者）
                kpts_xy = kpts_obj.xy[0].cpu().numpy()  # shape: (num_kpts, 2)

                # 確保關節數量足夠（至少 17 個點）
                if kpts_xy.shape[0] >= 17:
                    # 右下肢：右髖(12)、右膝(14)、右踝(16)
                    right_hip   = kpts_xy[12]  # (x, y)
                    right_knee  = kpts_xy[14]
                    right_ankle = kpts_xy[16]

                    # 呼叫剛剛寫的 calculate_angle() 算 ∠(hip-knee-ankle)
                    knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                    if knee_angle is not None:
                        angle_text = f"Right knee angle: {knee_angle:.1f}°"
                        # 把角度文字畫在使用者骨架畫面左上角
                        cv2.putText(
                            frame_user_pose,
                            angle_text,
                            (30, 60),  # 文字位置（你可以之後自己調整）
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),  # 綠色
                            2,
                            cv2.LINE_AA,
                        )
        except Exception as e:
            # 這裡只是避免偶爾偵測不到 keypoints 時程式直接炸掉
            print(f"[WARN] 無法計算關節角度：{e}")




        # --------- 左右等高縮放（示範影片 + 使用者骨架） ---------

        # 取得原始示範影片畫面的高與寬
        h_ref, w_ref = frame_ref.shape[:2]
        # 取得使用者骨架畫面的高與寬
        h_user, w_user = frame_user_pose.shape[:2]

        # 將高度縮放成相同（INNER_HEIGHT），寬度依比例縮
        scale_ref = INNER_HEIGHT / h_ref
        scale_user = INNER_HEIGHT / h_user

        # 依 scale 計算縮放後的寬度
        new_w_ref = int(w_ref * scale_ref)
        new_w_user = int(w_user * scale_user)

        # 使用 OpenCV resize 將兩個畫面縮放到相同高度
        frame_ref_resized = cv2.resize(frame_ref, (new_w_ref, INNER_HEIGHT))
        frame_user_resized = cv2.resize(frame_user_pose, (new_w_user, INNER_HEIGHT))

        # 將示範影片（左）和使用者骨架畫面（右）水平拼接在一起
        combined = np.hstack([frame_ref_resized, frame_user_resized])
        # ↑ 左：示範影片、右：使用者骨架，兩個都已經縮到 INNER_HEIGHT 高度

        # --------- 建立 1280x720 的畫布，並把畫面置中 ---------
        h_c, w_c = combined.shape[:2]  # 取得左右合併後的原始高與寬

        # 等比例縮放係數（只縮小，不放大）
        scale_factor = min(
            DISPLAY_MAX_WIDTH / w_c,
            DISPLAY_MAX_HEIGHT / h_c,
            1.0,  # 不讓畫面被放大，避免太糊
        )

        new_w = int(w_c * scale_factor)  # 縮放後的寬度
        new_h = int(h_c * scale_factor)  # 縮放後的高度

        # 先把 combined 縮放到新的大小
        resized = cv2.resize(combined, (new_w, new_h))

        # 建立一張固定大小的畫布（1280x720），顏色用 BACKGROUND_COLOR（深灰）
        canvas = np.full(
            (DISPLAY_MAX_HEIGHT, DISPLAY_MAX_WIDTH, 3),
            BACKGROUND_COLOR,
            dtype=np.uint8,
        )

        # 計算要把縮放後畫面貼在畫布的哪個位置（置中）
        offset_x = (DISPLAY_MAX_WIDTH - new_w) // 2   # 左右置中
        offset_y = (DISPLAY_MAX_HEIGHT - new_h) // 2  # 上下置中

        # 把縮放後的畫面貼到畫布中央
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        # 之後所有東西（進度條、文字）都畫在這張 canvas 上
        display_frame = canvas


        # --------- 進度條 + 時間顯示 ---------

        # 從 cap_ref 取得當前播放到第幾個影格
        current_frame_idx = int(cap_ref.get(cv2.CAP_PROP_POS_FRAMES))

        # 在 display_frame 上畫播放器風格的進度條與時間文字
        # bar_info = (bar_x, bar_y, bar_w, bar_h) → 之後用來判斷滑鼠點的位置
        display_frame, bar_info = draw_video_progress_bar(
            display_frame,
            current_frame_idx,
            total_frames,
            fps,
        )
        last_bar_info = bar_info

        # --------- 狀態列（顯示影片編號 + 播放速度 + 暫停狀態） ---------
        status = (
            f"[影片 {current_index + 1}/{len(video_list)}] "
            f"Speed: {playback_speed:.2f}x | {'PAUSED' if paused else 'RUNNING'}"
        )

        # 把狀態文字畫在畫面左上角
        cv2.putText(
            display_frame,
            status,
            (10, 30),                   # 左上角起點座標
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,                        # 字體大小
            (255, 255, 255),            # 白色
            2,                          # 線條粗細
            cv2.LINE_AA,                # 抗鋸齒
        )

        # 將整個畫面顯示在視窗中
        cv2.imshow(WINDOW_NAME, display_frame)

        # --------- 處理鍵盤控制 ---------
        key = cv2.waitKey(1)

        if key == ord("q"):
            # 按下 q → 離開主迴圈，結束程式
            break

        elif key == 32:  # 空白鍵（ASCII code 32）
            # 切換暫停 / 繼續狀態
            paused = not paused
            time_accumulator = 0.0
            last_time = time.perf_counter()

        elif key == ord("1"):
            # 設定為 0.5 倍速播放
            playback_speed = 0.5
            print("[INFO] 播放速度設定為 0.5x")
            time_accumulator = 0.0
            last_time = time.perf_counter()

        elif key == ord("2"):
            # 設定為正常速度 1.0 倍
            playback_speed = 1.0
            print("[INFO] 播放速度設定為 1.0x")
            time_accumulator = 0.0
            last_time = time.perf_counter()

        elif key == ord("3"):
            # 設定為 1.5 倍速播放
            playback_speed = 1.5
            print("[INFO] 播放速度設定為 1.5x")
            time_accumulator = 0.0
            last_time = time.perf_counter()

        elif key in (ord("-"), ord("_")):
            new_speed = max(MIN_PLAYBACK_SPEED, playback_speed - PLAYBACK_STEP)
            new_speed = round(new_speed, 2)
            if new_speed != playback_speed:
                playback_speed = new_speed
                print(f"[INFO] 播放速度設定為 {playback_speed:.2f}x")
            time_accumulator = 0.0
            last_time = time.perf_counter()

        elif key in (ord("="), ord("+")):
            new_speed = min(MAX_PLAYBACK_SPEED, playback_speed + PLAYBACK_STEP)
            new_speed = round(new_speed, 2)
            if new_speed != playback_speed:
                playback_speed = new_speed
                print(f"[INFO] 播放速度設定為 {playback_speed:.2f}x")
            time_accumulator = 0.0
            last_time = time.perf_counter()

        # ====== 切換影片（左 / 右方向鍵） ======
        # 不同環境方向鍵的 key code 會不一樣，所以這裡用「多個可能值」
        elif key in (
            81,           # Linux：←
            2424832,      # Windows：←
            65361,        # X11：←
            ord("["), ord("{"),
            ord("a"), ord("A"),
        ):
            new_index = (current_index - 1) % len(video_list)
            cap_ref, total_frames, fps, frame_ref = load_video_at_index(
                video_list,
                new_index,
                cap_ref,
            )
            if cap_ref is None:
                break

            current_index = new_index
            scrubbing_active = False
            time_accumulator = 0.0
            last_time = time.perf_counter()
            print(
                f"[INFO] 切換到上一支影片：{video_list[current_index]} "
                f"(共 {total_frames} 影格, fps={fps:.1f})"
            )

        elif key in (
            83,           # Linux：→
            2555904,      # Windows：→
            65363,        # X11：→
            ord("]"), ord("}"),
            ord("d"), ord("D"),
        ):
            new_index = (current_index + 1) % len(video_list)
            cap_ref, total_frames, fps, frame_ref = load_video_at_index(
                video_list,
                new_index,
                cap_ref,
            )
            if cap_ref is None:
                break

            current_index = new_index
            scrubbing_active = False
            time_accumulator = 0.0
            last_time = time.perf_counter()
            print(
                f"[INFO] 切換到下一支影片：{video_list[current_index]} "
                f"(共 {total_frames} 影格, fps={fps:.1f})"
            )



    # ====== 迴圈結束，釋放資源 ======
    if cap_ref is not None:
        cap_ref.release()   # 釋放示範影片的 VideoCapture
    if cap_user is not None:
        cap_user.release()  # 釋放攝影機資源

    cv2.destroyAllWindows()  # 關閉所有 OpenCV 開啟的視窗
    print("[INFO] Demo 結束。")


# Python 腳本的標準入口檢查：
# 只有當此檔案「直接被執行」時，才會呼叫 main()
# 若是被別的程式 import，就不會自動執行 main()，方便未來重用
if __name__ == "__main__":
    main()
