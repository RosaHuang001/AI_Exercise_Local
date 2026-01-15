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
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from openai import OpenAI
from dotenv import load_dotenv

# ==================== OpenAI 設定 ====================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
OPENAI_MODEL = "gpt-4o"


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
    if not angle_list or len(angle_list) < int(fps * 1.0):  # 至少 1 秒資料
        return {}

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

def classify_activity_level(stats: dict) -> str:
    """
    依運動學結果自動判定活動強度（低 / 中 / 高）
    以「下肢最大負荷」為主，以及等長收縮判定（如平板支撐）
    """
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

    # 等長收縮（如平板支撐）
    if (
        stats.get("primary_region") == "Upper"
        and reps == 0
        and rom < 15
    ):
        return "中"

    # 一般低強度動態運動
    if rom < 30 or freq < 0.4 or reps < 8:
        return "低"

    # 高強度動態運動
    if rom >= 80 and freq >= 0.9:
        return "高"

    return "中"




def get_video_duration_s(video_path: str) -> float:
    """抓影片秒數（用於 GPT prompt）"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(frame_count / fps) if fps > 0 else 0.0


def collect_angle(frame, angles, key, a, b, c, label, color):
    """計算角度、存起來，順便畫在影片上"""
    ang = compute_angle(a, b, c)
    if ang > 0:
        angles[key].append(float(ang))
        cv2.putText(
            frame, f"{label}:{ang:.0f}",
            tuple(b.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE, color, TEXT_THICKNESS
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




def pick_primary_joint(feats):
    knee_L, knee_R = feats.get("knee_L", {}), feats.get("knee_R", {})
    hip_L, hip_R   = feats.get("hip_L", {}),  feats.get("hip_R", {})
    elbow_L, elbow_R = feats.get("elbow_L", {}), feats.get("elbow_R", {})

    # ---------- Step 1：先判定是否為下肢動態運動 ----------
    if is_lower_dominant(knee_L, hip_L) or is_lower_dominant(knee_R, hip_R):
        candidates = {
            ("Lower","Hip","Left"): hip_L,
            ("Lower","Hip","Right"): hip_R,
            ("Lower","Knee","Left"): knee_L,
            ("Lower","Knee","Right"): knee_R,
        }
        best = max(candidates.items(), key=lambda x: x[1].get("rom_p5_p95", 0))
        region, joint, side = best[0]
        return region, joint, side, best[1]

    # ---------- Step 2：上肢動態運動 ----------
    elbow_scores = {
        ("Upper","Elbow","Left"): joint_activity_score(elbow_L),
        ("Upper","Elbow","Right"): joint_activity_score(elbow_R),
    }

    if max(elbow_scores.values()) > 0:
        best = max(elbow_scores, key=elbow_scores.get)
        region, joint, side = best
        primary = elbow_L if side == "Left" else elbow_R
        return region, joint, side, primary

    # ---------- Step 3：等長收縮（平板支撐類） ----------
    # 無明顯動態關節，但存在穩定出力
    if (
        feats.get("elbow_L", {}).get("reps", 0) == 0 and
        feats.get("elbow_R", {}).get("reps", 0) == 0 and
        feats.get("knee_L", {}).get("reps", 0) == 0 and
        feats.get("knee_R", {}).get("reps", 0) == 0
    ):
        return "Core", "Isometric", "Center", {}

    # ---------- Step 4：真的判不出來 ----------
    return "Unknown", "Unknown", "Center", {}





def format_value(v):
    """終端機與 CSV 一致：遇到 dict/list 就轉字串；None 轉空字串"""
    if v is None:
        return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return v


# ==================== YOLO 主流程 ====================

def yolo_process_one_video(model, video_path, out_dir):
    """對單一影片做 YOLO → 運動分析 → 輸出結果"""
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # ✅ 加入 hip_L / hip_R
    angles = dict(
        hip_L=[], hip_R=[],
        knee_L=[], knee_R=[],
        elbow_L=[], elbow_R=[],
        head=[]
    )
    postures = []

    out_path = os.path.join(out_dir, os.path.basename(video_path).replace(".mp4", "_yolo.mp4"))
    writer = None

    for res in model(video_path, stream=True, verbose=False):
        frame = res.plot()

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        if res.keypoints is None:
            writer.write(frame)
            continue

        j = res.keypoints.xy.cpu().numpy()[0]
        conf = res.keypoints.conf.cpu().numpy()[0] if res.keypoints.conf is not None else None
        ok = lambda i: True if conf is None else conf[i] >= KPT_CONF_TH

        # 體位判斷
        p = classify_posture(j) if all(ok(i) for i in (5,6,11,12,13,14)) else "Unknown"
        postures.append(p)
        cv2.putText(frame, f"Posture:{p}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # ✅ 髖角（shoulder - hip - knee）
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

    if writer is not None:
        writer.release()

    feats = {k: calc_kinematics(v, fps) for k, v in angles.items()}
    region, joint, side, primary = pick_primary_joint(feats)

    stats = dict(
        posture=Counter(postures).most_common(1)[0][0] if postures else "Unknown",
        primary_region=region,
        primary_joint=joint,
        primary_side=side,
        **{k: primary.get(k, 0) for k in (
            "rom_p5_p95","frequency_hz","reps",
            "intensity_mean","intensity_p95","intensity_energy")}
    )

    # ✅ 把 hip 也輸出到 CSV 欄位中
    stats.update(
        extract_joint_stats(
            feats,
            ["hip_L","hip_R","elbow_L","elbow_R","knee_L","knee_R","head"],
            ["rom_p5_p95","reps","frequency_hz",
             "intensity_mean","intensity_p95","intensity_energy"]
        )
    )

    return out_path, stats


# ==================== GPT 文案（Demo 用後處理說明） ====================

def call_openai_label(file_name: str, duration_s: float, stats: dict, activity_level: str) -> dict:
    """
    【Demo 用：GPT 白話說明模組】
    - 產出「可交付品質」綜合摘要（必備）
    - 同時保留活動程度、風險提醒、安全建議（可寫入 CSV）
    - 僅做文字說明，不回寫、不更動 stats
    """
    fallback = {
        "gpt_summary": (
            "本影片呈現受試者進行重複性或維持姿勢的基礎動作。"
            "建議量力而為並保持身體穩定，若出現不適請停止。"
        ),
        "gpt_activity_level": "中",
        "gpt_risk_notice": "若動作速度過快或幅度過大，可能增加關節負擔。",
        "gpt_safety_tip": "建議放慢速度、保持核心穩定，若疼痛或頭暈請停止。",
        "gpt_long_text": ""
    }

    if openai_client is None:
        return fallback

    posture = stats.get("posture", "Unknown")
    primary_region = stats.get("primary_region", "Unknown")
    primary_joint = stats.get("primary_joint", "Unknown")
    primary_side = stats.get("primary_side", "Unknown")

    rom = float(stats.get("rom_p5_p95", 0) or 0)
    reps = int(stats.get("reps", 0) or 0)
    freq = float(stats.get("frequency_hz", 0) or 0)

    head_rom = float(stats.get("head_rom_p5_p95", 0) or 0)
    head_freq = float(stats.get("head_frequency_hz", 0) or 0)

    if reps == 0 and rom < 12:
        motion_type_hint = "此動作較像『維持姿勢』或小幅度調整，次數接近 0 屬正常現象。"
    else:
        motion_type_hint = "此動作屬於『重複進行』的運動，可用次數與頻率描述。"

    if primary_region == "Upper":
        region_hint = "上肢（手臂、肩膀、手肘相關）"
    elif primary_region == "Lower":
        region_hint = "下肢（髖、膝、踝與站穩能力）"
    else:
        region_hint = "全身或混合動作"

    system_prompt = """
你是一位「復健專業人員」，正在向病患說明一支『運動示範影片』，
協助病患了解這支影片適合怎樣的運動強度，以及回家照著做時可以怎麼安排。

這支影片不是病患本人拍攝，而是提供病患模仿進行的運動示範。

你的任務是：
根據系統已計算完成的數值結果，
用病患聽得懂的白話中文，說明這支影片在示範什麼動作，
動作大不大、快不快、會不會累，
並依據 美國運動醫學會(American College of Sports Medicine,ACSM) 的 FITT 原則(包含頻率 (Frequency)、強度 (Intensity)、時間 (Time)、類型 (Type))，
提供「建議的運動方式描述」（非醫療處方），用來確保病患同時獲得運動效果與安全，避免受傷。

說明時請包含：
- 動作主要使用的身體部位與關節
- 動作幅度與速度的整體感覺（避免只給數字）
- 影片中示範的大約次數與節奏（用生活化方式描述）
- 此影片屬於低 / 中 / 高強度運動示範
- 病患照著影片進行時，可參考的組數、次數或時間建議（以保守、安全為原則）

嚴格規則：
1) 不假設病患已經做過這個動作。
2) 不使用「你剛剛完成了…」等回饋語氣。
3) 所有建議須以「可參考」、「建議從…開始」描述，不可使用命令語。
4) 不做醫療診斷，不給治療或處方。
5) 不提及 AI、模型、YOLO、演算法或分析流程。
6) 僅依據提供的數值進行合理解讀，不可自行捏造數據。
7) 避免專業術語，必要時請用病患能理解的方式解釋。
8) 語氣溫和、鼓勵、實際可執行。
9) 只輸出 JSON，不要任何多餘文字。
"""


    user_prompt = f"""
【系統自動判定之活動強度（請直接使用，不需自行推論）】
- 本次運動強度等級：{activity_level}強度
（此強度由系統依據 ROM、動作頻率與次數自動判定，
請直接在摘要中明確標示，不需修改或重新判斷。）
請嚴格依照「主要動作區域」與「主要參考關節」描述，
不得自行推翻或更換系統判定結果。

【影片基本資訊】
- 檔名：{file_name}
- 影片長度：約 {duration_s:.1f} 秒

【整體判定】
- 體位：{posture}
- 主要動作區域：{region_hint}
- 主要參考關節：{primary_joint}
- 動作主側：{primary_side}

【主要運動學指標（已計算完成）】
- ROM(p5–p95)：{rom:.1f} 度
- 次數：{reps} 下
- 頻率：{freq:.2f} 次/秒

【頭部活動（安全監測用途）】
- 頭部 ROM(p5–p95)：{head_rom:.1f} 度
- 頭部活動頻率：{head_freq:.2f} 次/秒

【動作型態提示】
- {motion_type_hint}

請輸出以下 JSON（只輸出 JSON）：

{{
  "gpt_summary": "必填。請以『專業復健人員向病患介紹一支運動示範影片』的方式，撰寫一段病患友善的白話摘要（約 5–7 句）。請說明影片示範的是什麼運動、主要會用到哪個身體部位與關節，並將動作的次數、節奏與幅度轉譯成病患能理解的描述（避免只列數字）。請**明確標示這是一支低 / 中 / 高強度的運動示範影片**，並依 ACSM FITT 原則，以保守、安全的方式，提供病患照著影片進行時可參考的組數、次數或時間建議。最後加入 1–2 句安全提醒，幫助病患知道什麼情況下應放慢或停止。請勿使用列點。",
  "gpt_activity_level": "低 / 中 / 高（三選一）",
  "gpt_risk_notice": "一句話風險提醒（務必具體、可理解）",
  "gpt_safety_tip": "一句話安全建議（務必可操作）",
  "gpt_long_text": "可留空字串或提供較長說明（可選）"
}}
"""

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            timeout=30,
        )

        data = json.loads(response.choices[0].message.content.strip())
        return {
            "gpt_summary": data.get("gpt_summary", fallback["gpt_summary"]),
            "gpt_activity_level": data.get("gpt_activity_level", fallback["gpt_activity_level"]),
            "gpt_risk_notice": data.get("gpt_risk_notice", fallback["gpt_risk_notice"]),
            "gpt_safety_tip": data.get("gpt_safety_tip", fallback["gpt_safety_tip"]),
            "gpt_long_text": data.get("gpt_long_text", fallback["gpt_long_text"]),
        }

    except Exception as e:
        print(f"[WARN] GPT 產生失敗，使用 fallback：{e}")
        return fallback



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

        out_video, stats = yolo_process_one_video(model, path, OUTPUT_VIDEO_DIR)
        duration_s = get_video_duration_s(path)
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
        print(f"- 關節活動幅度 ROM(p5–p95)：{stats.get('rom_p5_p95', 0):.1f} 度")
        print(f"- 動作次數：{stats.get('reps', 0)} 下")
        print(f"- 動作頻率：{stats.get('frequency_hz', 0):.2f} 次/秒")
        print(
            f"- 動作強度（角速度 mean / p95）："
            f"{stats.get('intensity_mean', 0):.1f} / {stats.get('intensity_p95', 0):.1f}"
        )

        print("\n▶ GPT 綜合運動摘要")
        print(gpt_pack.get("gpt_summary", "").strip())

        print("=" * 60 + "\n")

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: format_value(r.get(k)) for k in rows[0].keys()})

    print(f">>> 全部完成，CSV 已輸出：{OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()

