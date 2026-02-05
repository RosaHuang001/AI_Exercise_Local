#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO Pose 單人運動分析 Demo

功能：
- 上肢 / 下肢 關節活動幅度、頻率、強度
- 頭部活動頻率（安全監測用）
- 體位判斷（站 / 坐 / 躺）

輸出：
- 逐支影片輸出一段「GPT 綜合運動摘要」(必備)
- 終端機輸出格式：與 CSV 欄位一致（逐欄列點顯示）
- CSV：完整結構化欄位輸出
"""

import os, csv, json, math, logging
import numpy as np
from collections import Counter
from scipy.signal import find_peaks, savgol_filter

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from modules.gpt_summary import call_openai_label




# ==================== 基本設定 ====================

LOGGER.setLevel(logging.WARNING)

VIDEO_BASE_DIR   = r"D:\AI_Exercise_Local\exercise_videos"
OUTPUT_VIDEO_DIR = r"D:\AI_Exercise_Local\yolo_exercise_videos"
OUTPUT_CSV_PATH  = r"D:\AI_Exercise_Local\yolo_gemini_demo.csv"
YOLO_MODEL_PATH  = "yolo11n-pose.pt"

IGNORE_FILES = set()

KPT_CONF_TH = 0.35
ROM_MIN_DEG = 20.0

TEXT_SCALE = 0.6
TEXT_THICKNESS = 2


# ==================== 上肢強度判定門檻（可調參） ====================
UPPER_HIGH_INTEN_P95_TH = 120
UPPER_HIGH_ENERGY_TH = 8000


# ==================== 體重區間（衝擊力換算用） ====================

# 每個區間用一個「代表體重」換算（kg）
WEIGHT_BINS = {
    "50 公斤以下": 45,
    "50–59 公斤": 55,
    "60–69 公斤": 65,
    "70–79 公斤": 75,
    "80–89 公斤": 85,
    "90 公斤以上": 95,
}

G_CONST = 9.81  # m/s^2



# ==================== 幾何計算 ====================

def compute_angle(a, b, c) -> float:
    """計算三點形成的關節角度（以 b 為中心）"""
    ba, bc = a - b, c - b
    mba, mbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if mba == 0 or mbc == 0:
        return 0.0
    cosv = np.clip(np.dot(ba, bc) / (mba * mbc), -1.0, 1.0)
    return math.degrees(math.acos(cosv))


def compute_head_angle(nose, shoulder_c) -> float:
    """計算頭部相對身體的偏移角度（用來看是否頻繁晃頭）"""
    v = nose - shoulder_c
    mag = np.linalg.norm(v)
    if mag == 0:
        return 0.0
    cosv = np.clip(np.dot(v, [0, 1]) / mag, -1.0, 1.0)
    return math.degrees(math.acos(abs(cosv)))


def classify_posture(j: np.ndarray) -> str:
    """根據肩、髖、膝的位置判斷站 / 坐 / 躺"""
    shoulder = (j[5] + j[6]) / 2
    hip      = (j[11] + j[12]) / 2
    knee     = (j[13] + j[14]) / 2

    body = shoulder - hip
    mag = np.linalg.norm(body)
    if mag == 0:
        return "Unknown"

    cosv = np.clip(np.dot(body, [0, 1]) / mag, -1.0, 1.0)
    theta = math.degrees(math.acos(abs(cosv)))

    if theta < 30 and shoulder[1] < hip[1] < knee[1]:
        return "Standing"
    if 30 <= theta < 60 and abs(hip[1] - knee[1]) < 40:
        return "Sitting"
    if theta >= 60:
        return "Lying"
    return "Unknown"


# ==================== 運動學計算 ====================

def smooth_angles(arr):
    """平滑角度時間序列，避免雜訊影響頻率計算"""
    if len(arr) < 5:
        return arr
    win = min(11, len(arr) | 1)
    return savgol_filter(arr, win, 2, mode="interp") if win >= 5 else arr


def calc_kinematics(angle_list, fps):
    """由角度序列計算 ROM、頻率、強度"""
    fps = fps or 25.0
    if not angle_list or len(angle_list) < int(fps * 1.0):
        return dict(
        rom_min=0.0, rom_max=0.0, rom=0.0, rom_p5_p95=0.0,
        intensity_mean=0.0, intensity_p95=0.0, intensity_energy=0.0,
        frequency_hz=0.0, reps=0
        )


    arr = smooth_angles(np.array(angle_list, float))

    p5, p95 = np.percentile(arr, [5, 95])
    rom_p = float(p95 - p5)
    rom_min, rom_max = float(arr.min()), float(arr.max())
    rom = float(rom_max - rom_min)

    # 幾乎沒動就直接回傳低活動結果
    if rom_p < ROM_MIN_DEG:
        return dict(
            rom_min=rom_min, rom_max=rom_max, rom=rom,
            rom_p5_p95=rom_p,
            intensity_mean=0.0, intensity_p95=0.0, intensity_energy=0.0,
            frequency_hz=0.0, reps=0
        )

    # 角速度（deg/s）
    w = np.gradient(arr, 1 / fps)

    # 用峰值數量估計動作次數
    peaks, _   = find_peaks(arr,  prominence=max(8, 0.15 * rom_p), distance=int(fps * 0.25))
    valleys, _ = find_peaks(-arr, prominence=max(8, 0.15 * rom_p), distance=int(fps * 0.25))

    duration = len(arr) / fps
    reps = int(round((len(peaks) + len(valleys)) / 2))
    freq = float(reps / duration) if duration > 0 else 0.0

    return dict(
        rom_min=rom_min,
        rom_max=rom_max,
        rom=rom,
        rom_p5_p95=rom_p,
        intensity_mean=float(np.mean(np.abs(w))),
        intensity_p95=float(np.percentile(np.abs(w), 95)),
        intensity_energy=float(np.mean(w ** 2)),
        frequency_hz=freq,
        reps=reps,
    )


# ==================== 小工具 ====================

def joint_rank_score(joint_stats: dict, mode: str = "rom_freq") -> float:
    """
    用於『主關節排序』的分數
    mode:
      - "rom_freq": ROM(p5–p95) * frequency (穩、好解釋)
      - "energy": intensity_energy (偏用力/爆發，但更吃噪音)
      - "hybrid": 0.7*(rom*freq) + 0.3*energy (折衷)
    """
    rom  = float(joint_stats.get("rom_p5_p95", 0) or 0)
    reps = int(joint_stats.get("reps", 0) or 0)
    freq = float(joint_stats.get("frequency_hz", 0) or 0)
    eng  = float(joint_stats.get("intensity_energy", 0) or 0)

    # 基本防抖：動作太少就不列為主關節候選
    if reps < 4:
        return 0.0

    # 抖動特徵：幅度小卻高頻（tracking jitter）
    if rom < 25.0 and freq > 1.0:
        return 0.0

    rom_freq = rom * freq

    if mode == "rom_freq":
        return rom_freq
    if mode == "energy":
        return eng
    if mode == "hybrid":
        return 0.7 * rom_freq + 0.3 * eng

    # 預設回 rom_freq
    return rom_freq



def classify_activity_level(stats: dict) -> str:
    """
    依運動學結果自動判定活動強度（低 / 中 / 高）
    ✅ 補齊上肢規則：Upper 用 elbow 的 ROM/freq/reps + intensity 判定
    ✅ Lower 維持下肢為主
    ✅ Core 等長保留
    """

    region = stats.get("primary_region", "Unknown")

    # ============ Core / Isometric ============
    if region == "Core":
        return "中"

    # ============ Upper rules ============
    if region == "Upper":
        # 取左右肘的最大值（代表上肢負荷）
        rom = max(stats.get("elbow_L_rom_p5_p95", 0), stats.get("elbow_R_rom_p5_p95", 0))
        freq = max(stats.get("elbow_L_frequency_hz", 0), stats.get("elbow_R_frequency_hz", 0))
        reps = max(stats.get("elbow_L_reps", 0), stats.get("elbow_R_reps", 0))
        inten_p95 = max(stats.get("elbow_L_intensity_p95", 0), stats.get("elbow_R_intensity_p95", 0))
        energy = max(stats.get("elbow_L_intensity_energy", 0), stats.get("elbow_R_intensity_energy", 0))

        # 等長（上肢）：幾乎沒 reps，但可能有微小調整
        if reps == 0 and rom < 15:
            return "中"

        # 低：幅度小 或 次數少 或 頻率低
        if rom < 30 or reps < 8 or freq < 0.4:
            return "低"

        # 高：上肢要判到「高」不能只靠 ROM/freq（容易被鏡頭放大），加上強度條件更穩
        # 這裡用 intensity_p95 或 energy 當加分門檻
        if (rom >= 70 and freq >= 0.8 and reps >= 12) and (
            inten_p95 >= UPPER_HIGH_INTEN_P95_TH or energy >= UPPER_HIGH_ENERGY_TH
        ):
            return "高"

        # 中：其餘動態上肢
        return "中"

    # ============ Lower rules (default) ============
    # 下肢最大負荷為主
    rom = max(
        stats.get("hip_L_rom_p5_p95", 0),
        stats.get("hip_R_rom_p5_p95", 0),
        stats.get("knee_L_rom_p5_p95", 0),
        stats.get("knee_R_rom_p5_p95", 0),
    )

    freq = max(
        stats.get("hip_L_frequency_hz", 0),
        stats.get("hip_R_frequency_hz", 0),
        stats.get("knee_L_frequency_hz", 0),
        stats.get("knee_R_frequency_hz", 0),
    )

    reps = max(
        stats.get("hip_L_reps", 0),
        stats.get("hip_R_reps", 0),
        stats.get("knee_L_reps", 0),
        stats.get("knee_R_reps", 0),
    )

    # 低
    if rom < 30 or freq < 0.4 or reps < 8:
        return "低"

    # 高（下肢更容易出現高負荷）
    if rom >= 80 and freq >= 0.9 and reps >= 12:
        return "高"

    return "中"






def collect_angle(frame, angles, key, a, b, c, label, color):
    """計算角度、存起來，順便畫在影片上（安全版）"""
    ang = compute_angle(a, b, c)
    if ang <= 0:
        return

    angles[key].append(float(ang))

    # === 文字位置防呆（避免畫出畫面外） ===
    x, y = b.astype(int)
    h, w = frame.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    cv2.putText(
        frame,
        f"{label}:{ang:.0f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        color,
        TEXT_THICKNESS
    )



def extract_joint_stats(feats, joints, keys):
    """把指定關節的指定指標攤平成 stats 欄位"""
    return {
        f"{j}_{k}": feats.get(j, {}).get(k, 0)
        for j in joints for k in keys
    }


def joint_activity_score(joint_stats: dict) -> float:
    """
    關節活動分數（避免抖動假動作）：
    - 必須有足夠 reps
    - 排除 ROM 很小但頻率很高（tracking jitter 常見型態）
    - 以 ROM * freq 當主要分數
    """
    rom  = float(joint_stats.get("rom_p5_p95", 0) or 0)
    reps = int(joint_stats.get("reps", 0) or 0)
    freq = float(joint_stats.get("frequency_hz", 0) or 0)

    if reps < 4:
        return 0.0

    # 抖動特徵：幅度小卻高頻（例如手肘晃動、追蹤跳點）
    if rom < 25.0 and freq > 1.0:
        return 0.0

    return rom * freq


def is_lower_dominant(knee_stats, hip_stats):
    """
    判定是否為『動態下肢運動』
    條件：有明顯 ROM + 重複性
    """
    for js in (knee_stats, hip_stats):
        if (
            js.get("rom_p5_p95", 0) >= 40 and
            js.get("reps", 0) >= 6 and
            js.get("frequency_hz", 0) >= 0.4
        ):
            return True
    return False




def pick_primary_joint(feats, rank_mode: str = "rom_freq"):
    """
    選『主關節/主區域』：
    - 下肢動態：hip/knee 用 joint_rank_score 排序
    - 上肢動態：elbow 用 joint_rank_score 排序
    - 等長收縮：Core/Isometric
    rank_mode: "rom_freq" / "energy" / "hybrid"
    """
    knee_L, knee_R = feats.get("knee_L", {}), feats.get("knee_R", {})
    hip_L, hip_R   = feats.get("hip_L", {}),  feats.get("hip_R", {})
    elbow_L, elbow_R = feats.get("elbow_L", {}), feats.get("elbow_R", {})

    # ---------- Step 1：判定是否為下肢動態運動 ----------
    if is_lower_dominant(knee_L, hip_L) or is_lower_dominant(knee_R, hip_R):
        candidates = {
            ("Lower","Hip","Left"): hip_L,
            ("Lower","Hip","Right"): hip_R,
            ("Lower","Knee","Left"): knee_L,
            ("Lower","Knee","Right"): knee_R,
        }

        # ✅ 改成用 score 排序（你可切 energy / hybrid）
        scored = [(k, v, joint_rank_score(v, mode=rank_mode)) for k, v in candidates.items()]
        best_k, best_v, best_s = max(scored, key=lambda x: x[2])

        # 如果全部 score 都是 0，就退回 ROM 最大（避免選不到）
        if best_s <= 0:
            best_k, best_v = max(candidates.items(), key=lambda x: x[1].get("rom_p5_p95", 0))

        region, joint, side = best_k
        return region, joint, side, best_v

    # ---------- Step 2：上肢動態運動 ----------
    elbow_candidates = {
        ("Upper","Elbow","Left"): elbow_L,
        ("Upper","Elbow","Right"): elbow_R,
    }

    scored = [(k, v, joint_rank_score(v, mode=rank_mode)) for k, v in elbow_candidates.items()]
    best_k, best_v, best_s = max(scored, key=lambda x: x[2])

    if best_s > 0:
        region, joint, side = best_k
        return region, joint, side, best_v

    # ---------- Step 3：等長收縮（平板支撐類） ----------
    if (
        feats.get("elbow_L", {}).get("reps", 0) == 0 and
        feats.get("elbow_R", {}).get("reps", 0) == 0 and
        feats.get("knee_L", {}).get("reps", 0) == 0 and
        feats.get("knee_R", {}).get("reps", 0) == 0
    ):
        return "Core", "Isometric", "Center", {}

    # ---------- Step 4：真的判不出來 ----------
    return "Unknown", "Unknown", "Center", {}





# ==================== 關節衝擊力換算 ====================
def estimate_relative_impact_bw(stats: dict):
    """
    估計『相對衝擊力（倍體重 BW）』範圍：回傳 (level_zh, (bw_low, bw_high))
    - 這是『影片負荷特性』的保守估計
    - 僅用已經算出來的 posture / primary_region / ROM / freq
    """
    posture = stats.get("posture", "Unknown")
    region = stats.get("primary_region", "Unknown")
    rom = float(stats.get("rom_p5_p95", 0) or 0)
    freq = float(stats.get("frequency_hz", 0) or 0)

    # 預設：低負荷（例如坐姿、躺姿、上肢小幅度）
    level = "低"
    bw_range = (0.8, 1.2)

    # 站姿 + 下肢：可能出現較高關節負荷（尤其是大幅度、快節奏）
    if posture == "Standing" and region == "Lower":
        if rom >= 70 and freq >= 0.8:
            level = "高"
            bw_range = (2.0, 3.0)
        elif rom >= 40 or freq >= 0.6:
            level = "中"
            bw_range = (1.3, 2.0)
        else:
            level = "低"
            bw_range = (1.0, 1.5)

    # 上肢動作通常關節衝擊相對小（以保守低～中）
    elif region == "Upper":
        if rom >= 60 and freq >= 0.8:
            level = "中"
            bw_range = (1.0, 1.6)

    # 核心等長（例如平板）：多為持續出力，衝擊不高但負擔可能累積
    elif region == "Core":
        level = "中"
        bw_range = (1.0, 1.4)

    return level, bw_range


def impact_newton_by_weight_bins(bw_range, weight_bins=WEIGHT_BINS):
    """
    將 bw_range (倍體重) 轉成各體重區間的衝擊力 N 範圍
    回傳 dict：{ "50–59 公斤": {"min_N":..., "max_N":...}, ... }
    """
    bw_low, bw_high = bw_range
    out = {}
    for label, w_kg in weight_bins.items():
        low_n = bw_low * w_kg * G_CONST
        high_n = bw_high * w_kg * G_CONST
        out[label] = {"min_N": round(low_n, 1), "max_N": round(high_n, 1)}
    return out


def impact_text_zh(impact_by_weight: dict) -> str:
    """把衝擊力 dict 轉成一行可讀中文（方便終端機印）"""
    parts = []
    for k, v in impact_by_weight.items():
        parts.append(f"{k}:{v['min_N']:.0f}–{v['max_N']:.0f}N")
    return "；".join(parts)




# ---------- 計算（終端機輸出用） ----------
def format_value(v):
    """終端機與 CSV 一致：遇到 dict/list 就轉字串；None 轉空字串"""
    if v is None:
        return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return v



# ==================== YOLO 主流程 ====================
def yolo_process_one_video(model, video_path, out_dir):
    """對單一影片做 YOLO → 運動分析 → 輸出結果（cap 只開一次，加速）"""
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_s = float(frame_count / fps) if fps > 0 else 0.0

    angles = dict(
        hip_L=[], hip_R=[],
        knee_L=[], knee_R=[],
        elbow_L=[], elbow_R=[],
        head=[]
    )
    postures = []

    out_path = os.path.join(out_dir, os.path.basename(video_path).replace(".mp4", "_yolo.mp4"))
    writer = None

    while True:
        ok_read, frame0 = cap.read()
        if not ok_read:
            break

        # 單幀推論（⚠️ 必須在 while 內）
        res = model.predict(source=frame0, verbose=False, conf=0.25, imgsz=640)[0]
        frame = res.plot()

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 沒有 keypoints / 沒有人
        if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
            writer.write(frame)
            continue

        # ====== 取「單人」：取 person box conf 最高的那個 ======
        idx_best = 0
        if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) > 0:
            idx_best = int(torch.argmax(res.boxes.conf).item())  # 需要 import torch

        j = res.keypoints.xy.cpu().numpy()[idx_best]
        kp_conf = res.keypoints.conf.cpu().numpy()[idx_best] if res.keypoints.conf is not None else None
        ok = (lambda i: True) if kp_conf is None else (lambda i: kp_conf[i] >= KPT_CONF_TH)

        # 體位判斷
        p = classify_posture(j) if all(ok(i) for i in (5,6,11,12,13,14)) else "Unknown"
        postures.append(p)
        cv2.putText(frame, f"Posture:{p}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # 髖角（shoulder - hip - knee）
        if all(ok(i) for i in (5,11,13)):
            collect_angle(frame, angles, "hip_L", j[5], j[11], j[13], "HipL", (0,255,255))
        if all(ok(i) for i in (6,12,14)):
            collect_angle(frame, angles, "hip_R", j[6], j[12], j[14], "HipR", (0,255,255))

        # 膝角（hip - knee - ankle）
        if all(ok(i) for i in (11,13,15)):
            collect_angle(frame, angles, "knee_L", j[11], j[13], j[15], "KneeL", (0,255,0))
        if all(ok(i) for i in (12,14,16)):
            collect_angle(frame, angles, "knee_R", j[12], j[14], j[16], "KneeR", (0,255,0))

        # 肘角（shoulder - elbow - wrist）
        if all(ok(i) for i in (5,7,9)):
            collect_angle(frame, angles, "elbow_L", j[5], j[7], j[9], "ElbL", (255,255,0))
        if all(ok(i) for i in (6,8,10)):
            collect_angle(frame, angles, "elbow_R", j[6], j[8], j[10], "ElbR", (255,255,0))

        # 頭部活動（安全監測）
        if all(ok(i) for i in (0,5,6)):
            sc = (j[5] + j[6]) / 2
            ha = compute_head_angle(j[0], sc)
            if ha > 0:
                angles["head"].append(float(ha))

        writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    feats = {k: calc_kinematics(v, fps) for k, v in angles.items()}

    PRIMARY_RANK_MODE = "rom_freq"
    region, joint, side, primary = pick_primary_joint(feats, rank_mode=PRIMARY_RANK_MODE)
    primary_score = joint_rank_score(primary, mode=PRIMARY_RANK_MODE) if isinstance(primary, dict) else 0.0

    stats = dict(
        posture=Counter(postures).most_common(1)[0][0] if postures else "Unknown",
        primary_region=region,
        primary_joint=joint,
        primary_side=side,
        primary_rank_mode=PRIMARY_RANK_MODE,
        primary_score=float(primary_score),
        **{k: primary.get(k, 0) for k in (
            "rom_p5_p95","frequency_hz","reps",
            "intensity_mean","intensity_p95","intensity_energy")}
    )

    stats.update(
        extract_joint_stats(
            feats,
            ["hip_L","hip_R","elbow_L","elbow_R","knee_L","knee_R","head"],
            ["rom_p5_p95","reps","frequency_hz","intensity_mean","intensity_p95","intensity_energy"]
        )
    )

    impact_level, bw_range = estimate_relative_impact_bw(stats)
    impact_by_weight = impact_newton_by_weight_bins(bw_range)

    stats.update({
        "impact_level": impact_level,
        "impact_bw_low": float(bw_range[0]),
        "impact_bw_high": float(bw_range[1]),
        "impact_by_weight_bins": impact_by_weight,
        "impact_by_weight_bins_text": impact_text_zh(impact_by_weight),
    })

    return out_path, stats, duration_s




# ==================== Main ====================

def main():
    """批次處理資料夾內所有影片（終端機輸出與 CSV 欄位一致 + GPT 摘要必備）"""

    model = YOLO(YOLO_MODEL_PATH)
    rows = []

    video_files = sorted(
        x for x in os.listdir(VIDEO_BASE_DIR)
        if x.lower().endswith(".mp4") and x not in IGNORE_FILES
    )

    if not video_files:
        print(">>> 沒有找到任何 mp4 影片")
        return

    print(f">>> 共找到 {len(video_files)} 支影片，開始分析...\n")

    for idx, f in enumerate(video_files, start=1):
        path = os.path.join(VIDEO_BASE_DIR, f)

        out_video, stats, duration_s = yolo_process_one_video(model, path, OUTPUT_VIDEO_DIR)
        yolo_result = pack_yolo_result(out_video, stats, duration_s)
        activity_level = classify_activity_level(stats)


        # GPT（必備摘要）
        gpt_pack = call_openai_label(f, duration_s, stats, activity_level)

        row = dict(
            file_name=f,
            yolo_output_video=out_video,
            activity_level=activity_level,
            **stats,
            **gpt_pack
        )
        rows.append(row)

        print("=" * 60)
        print(f"[{idx}/{len(video_files)}] 影片：{f}\n")

        print("▶ 標準運動學指標")
        print(f"- 系統判定活動強度：{activity_level}強度")
        print(f"- 體位：{stats.get('posture')}")
        print(f"- 主要動作區域：{stats.get('primary_region')}")
        print(f"- 主要關節：{stats.get('primary_joint')}（{stats.get('primary_side')}）")
        print(f"- 主關節排序分數（{stats.get('primary_rank_mode', '')}）：{stats.get('primary_score', 0):.2f}")
        print(f"- 關節活動幅度 ROM(p5–p95)：{stats.get('rom_p5_p95', 0):.1f} 度")
        print(f"- 動作次數：{stats.get('reps', 0)} 下")
        print(f"- 動作頻率：{stats.get('frequency_hz', 0):.2f} 次/秒")
        print(
            f"- 動作強度（角速度 mean / p95）："
            f"{stats.get('intensity_mean', 0):.1f} / {stats.get('intensity_p95', 0):.1f}"
        )
        print("\n▶ 關節衝擊力（依體重區間換算，參考用）")
        print(f"- 衝擊等級：{stats.get('impact_level', '未知')}")
        print(f"- 相對負荷：{stats.get('impact_bw_low', 0):.1f}–{stats.get('impact_bw_high', 0):.1f} × 體重")
        print(f"- 各體重區間（N）：{stats.get('impact_by_weight_bins_text', '')}")

        print("\n▶ GPT 綜合運動摘要")
        print(gpt_pack.get("gpt_summary", "").strip())

        print("=" * 60 + "\n")

    # ✅ 這裡才檢查 rows 是否有成功產生資料
    if not rows:
        print(">>> 沒有任何影片成功產生結果，CSV 不輸出")
        return

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: format_value(r.get(k)) for k in rows[0].keys()})

    print(f">>> 全部完成，CSV 已輸出：{OUTPUT_CSV_PATH}")




def pack_yolo_result(out_path, stats, duration_s):
    """
    封裝給主系統 / RAG / 規則 / GPT 用的 YOLO 回傳格式
    """
    return {
        # --- 影片層級 ---
        "video_output": out_path,
        "duration_s": duration_s,

        # --- 高階語意（系統判斷主力）---
        "posture": stats.get("posture"),
        "primary_region": stats.get("primary_region"),
        "primary_joint": stats.get("primary_joint"),
        "primary_side": stats.get("primary_side"),
        "activity_level": classify_activity_level(stats),

        # --- 主關節運動學（避免用全關節雜訊）---
        "primary_kinematics": {
            "rom_p5_p95": stats.get("rom_p5_p95"),
            "frequency_hz": stats.get("frequency_hz"),
            "reps": stats.get("reps"),
            "intensity_mean": stats.get("intensity_mean"),
            "intensity_p95": stats.get("intensity_p95"),
            "intensity_energy": stats.get("intensity_energy"),
        },

        # --- 醫療安全 ---
        "impact": {
            "level": stats.get("impact_level"),
            "bw_low": stats.get("impact_bw_low"),
            "bw_high": stats.get("impact_bw_high"),
            "by_weight_bins": stats.get("impact_by_weight_bins"),
            "by_weight_bins_text": stats.get("impact_by_weight_bins_text", ""),
        }
    }




if __name__ == "__main__":
    main()

