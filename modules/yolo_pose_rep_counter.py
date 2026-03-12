#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【三總心臟衰竭復健系統 - 線下數據預處理引擎】
這支程式的角色是系統的「數據工廠」，負責把原始的運動示範影片轉化為科學化的醫學數據。

🚀 初學者學習重點：
1. 環境路徑 (Path)：如何讓程式在不同目錄下都能正確抓到其他模組。
2. 向量幾何 (Geometry)：利用座標點計算關節角度。
3. 訊號處理 (Signal Processing)：如何把 AI 震顫的數據變平滑，並精準計算次數。
4. 全自動化 (Automation)：自動建立與同步 JSON 資料庫，完全取代手動對照表。
"""

import os
import sys
import uuid

# --- 系統環境配置 ---
# 取得目前這個 Python 檔案所在的資料夾路徑 (例如 D:\AI_Exercise_Local\modules)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 取得專案的根目錄 (例如 D:\AI_Exercise_Local)
_ROOT = os.path.dirname(_CURRENT_DIR)

# 💡 關鍵修正：將根目錄與模組目錄加入 Python 搜尋路徑
# 這樣做可以確保我們在執行此程式時，能夠順利 import 到根目錄下的 rag 或 modules 內容
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

import csv, json, math, logging
import numpy as np
from collections import Counter
from scipy.signal import find_peaks, savgol_filter  # 引入科學計算庫，用於訊號處理
import cv2
import torch
from ultralytics import YOLO

# 從 GPT 模組中匯入標籤生成函數，用於為每部影片生成白話文建議
try:
    from gpt_summary import call_openai_label
except ImportError:
    from modules.gpt_summary import call_openai_label

# ==========================================
# 1. 基本路徑與參數設定
# ==========================================
BASE_PATH = _ROOT

# 設定資料夾路徑，使用 os.path.join 確保在 Windows/Mac/Linux 都能正確運行
VIDEO_IN_DIR = os.path.join(BASE_PATH, "exercise_videos")        # 存放原始影片
VIDEO_OUT_DIR = os.path.join(BASE_PATH, "yolo_exercise_videos")  # 存放分析後帶骨架的影片
JSON_LIB_PATH = os.path.join(BASE_PATH, "knowledge_base", "exercise_library.json") # 核心資料庫
MODEL_PATH = os.path.join(BASE_PATH, "yolo11n-pose.pt")          # YOLO11 模型檔

# 確保輸出目錄存在，如果沒有的話會自動建立
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

# ==========================================
# 2. 核心運算邏輯：幾何與運動學
# ==========================================

def compute_angle(a, b, c) -> float:
    """
    【幾何運算：計算三點夾角】
    輸入 a, b, c 三個座標點，計算以 b 為頂點的夾角。
    原理：利用向量內積公式算出 cos 值，再轉回角度。
    """
    ba, bc = a - b, c - b
    # 計算向量的長度 (歐幾里得距離)
    mba, mbc = np.linalg.norm(ba), np.linalg.norm(bc)
    
    # 防止分母為 0 (當點重合時)，避免數學崩潰
    if mba == 0 or mbc == 0: return 0.0
    
    # 計算 cos 值並透過 clip 限制在 -1 ~ 1 之間
    # 💡 為什麼要 clip？因為浮點運算誤差可能產生 1.00000001，會導致 acos 報錯
    cosv = np.clip(np.dot(ba, bc) / (mba * mbc), -1.0, 1.0)
    
    # 透過反餘弦函數轉為角度 (度數)
    return math.degrees(math.acos(cosv))

def calc_kinematics(angle_list, fps):
    """
    【運動學特徵萃取 - 白話邏輯檢查版】
    將 AI 抓到的角度序列，轉化為看得懂的活動度、頻率與衝擊力數據。
    """
    fps = fps or 25.0
    # 樣本長度檢查：如果影片不到 0.5 秒，代表數據不夠，不進行計算。
    if not angle_list or len(angle_list) < int(fps * 0.5):
        return {
            "rom_p5_p95": 0, "reps": 0, "frequency_hz": 0, 
            "intensity_energy": 0, "impact_bw_high": 1.0, 
            "impact_level": "低衝擊", "primary_region": "Lower"
        }

    # --- 1. 數據平滑化 (濾波處理) ---
    # 目的：把 AI 偵測時像「鋸齒」一樣的抖動磨平，留下平滑的運動曲線。
    arr = savgol_filter(np.array(angle_list, float), min(11, len(angle_list)|1), 2)
    
    # --- 2. ROM (活動幅度) 計算 ---
    # 白話公式：動作最大處的數值(第95%) 減去 動作最小處的數值(第5%)。
    # 註解：不直接用最大減最小，是為了怕 AI 突然閃現一個錯誤的極大值毀掉數據。
    p5, p95 = np.percentile(arr, [5, 95])
    rom_p = float(p95 - p5)

    # --- 3. 動作次數 (Reps) 偵測 ---
    # 邏輯：數出曲線中有幾個大波峰與大波谷。
    # 判定條件：波峰必須高於「總幅度的 15%」，避免把微小的晃動當成一次動作。
    peaks, _ = find_peaks(arr, prominence=max(8, 0.15*rom_p), distance=int(fps*0.25))
    valleys, _ = find_peaks(-arr, prominence=max(8, 0.15*rom_p), distance=int(fps*0.25))
    reps = int(round((len(peaks) + len(valleys)) / 2))

    # --- 4. 運動頻率 (Frequency) 計算 ---
    # 白話公式：總共做的次數 除以 真正有在動的時間。
    # 註解：有在動的時間是指「第一個動作開始」到「最後一個動作結束」的時間段。
    all_extrema = np.sort(np.concatenate((peaks, valleys)))
    if len(all_extrema) >= 2:
        active_sec = (all_extrema[-1] - all_extrema[0]) / fps + (1.0/max(1, reps))
        freq = reps / active_sec
    else:
        freq = reps / (len(arr)/fps)

    # --- 5. 物理負荷 (Impact BW) 衝擊力計算 ---
    # (A) 計算速度：每一幀角度變化的快慢 (角速度)。
    w = np.gradient(arr, 1 / fps) 
    
    # (B) 計算能量：把所有速度「平方後取平均」，代表這個動作有多「猛」。
    avg_w_sq = float(np.mean(w ** 2))
    
    # (C) 倍體重負荷估算 (Impact Body Weight)：
    # 白話公式：基礎體重(1.0) + (速度能量影響) + (動作頻率補償)。
    # 邏輯：動作越快、頻率越高，關節承受的壓力倍數就越高。
    impact_bw = 1.0 + (avg_w_sq / 5000) + (freq * 0.2)
    impact_bw_high = round(min(2.5, impact_bw), 2) # 安全考量，設定上限為 2.5 倍體重。
    
    # --- 6. 衝擊等級判定 ---
    # 低衝擊：負荷 < 1.2 倍體重。
    # 中衝擊：負荷在 1.2 到 1.4 倍體重之間。
    # 高衝擊：負荷 > 1.4 倍體重 (這會觸發後端自動降低 RPE 運動建議)。
    impact_level = "低衝擊"
    if impact_bw_high > 1.4: impact_level = "高衝擊"
    elif impact_bw_high > 1.2: impact_level = "中衝擊"

    return {
        "rom_p5_p95": rom_p, 
        "reps": reps, 
        "frequency_hz": freq,
        "intensity_energy": avg_w_sq,
        "impact_bw_high": impact_bw_high, # 這個數字會給 GPT 決定 RPE 建議文字內容
        "impact_level": impact_level,
        "primary_region": "Lower"
    }



# ==========================================
# 3. [優化邏輯] 全自動資料庫同步 (免手動 Map)
# ==========================================
def sync_to_json_library(results_data):
    """
    【全自動資料庫建置 - 醫學門檻增強版】
    將計算出的物理指標與「三段式 ROM 門檻」寫入 JSON。
    """
    if not os.path.exists(JSON_LIB_PATH) or os.path.getsize(JSON_LIB_PATH) == 0:
        lib_data = {"version": "1.2", "exercises": []}
    else:
        with open(JSON_LIB_PATH, "r", encoding="utf-8") as f:
            try:
                lib_data = json.load(f)
            except:
                lib_data = {"version": "1.2", "exercises": []}

    for result in results_data:
        pure_name = os.path.splitext(result["file_name"])[0]
        target_ex = next((ex for ex in lib_data["exercises"] if ex.get("name_zh") == pure_name), None)

        
        # 💡 新增：計算三段式 ROM 門檻
        # 以教練角度為 100% 基準，依 NYHA 分級給予不同寬鬆限度
        coach_rom = round(result["rom_p5_p95"], 1)
        rom_standards = {
            "class_i": round(coach_rom * 0.90, 1),   # I級：需達到教練的 90%
            "class_ii": round(coach_rom * 0.75, 1),   # II級：需達到教練的 75%
            "class_iii": round(coach_rom * 0.60, 1),   # III級：需達到教練的 60%
        }

        update_fields = {
            "video_filename": result["file_name"],
            "reps": result["reps"],                  # 保留次數
            "frequency_hz": round(result["frequency_hz"], 2),
            "rom_p5_p95": coach_rom,               # 保留教練角度
            "rom_standards": rom_standards,        # 寫入多級門檻資料
            "impact_bw_high": result["impact_bw_high"], # 保留衝擊力
            "impact_level": result["impact_level"], # 保留衝擊等級
            "primary_region": result["primary_region"], # 保留部位摘要
            "gpt_summary": result["gpt_summary"],      # 保留 GPT 摘要
            "gpt_safety_tip": result["safety_tip"],    # 保留 GPT 安全提示
        }

        if target_ex:
            target_ex.update(update_fields)
        else:
            new_item = {
                "exercise_id": f"auto_{str(uuid.uuid4())[:4]}", # 生成唯一 ID
                "name_zh": pure_name,                # 保留原動作名稱
                "posture": "unknown",
                "primary_focus": [result["primary_region"]], # 保留部位摘要
                "nyha_allowed": ["I", "II", "III"], # 允許所有 NYHA 分級
                **update_fields,                  # 直接展開包含所有新欄位
            }
            lib_data["exercises"].append(new_item)

    with open(JSON_LIB_PATH, "w", encoding="utf-8") as f:
        json.dump(lib_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 資料庫已同步，已為 {len(results_data)} 個動作建立三段式醫學門檻。")



# ==========================================
# 4. YOLO 核心分析流程 (影像處理)
# ==========================================

def process_video_and_save_skeleton(model, filename):
    """
    逐幀讀取影片，執行 YOLO 推論，收集關節點，最後存成帶骨架的影片。
    """
    in_path = os.path.join(VIDEO_IN_DIR, filename)
    out_path = os.path.join(VIDEO_OUT_DIR, filename.replace(".mp4", "_yolo.mp4"))
    
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    angles = []
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret: break # 讀取失敗或影片播完即結束
        
        # 執行姿態估計：device=0 強制使用第一張 GPU；boxes=False 表示不畫人物方框
        res = model.predict(source=frame, verbose=False, conf=0.3, imgsz=640, device=0)[0]
        plot_frame = res.plot(boxes=False, labels=False)

        # 初始化影片寫入器
        if writer is None:
            h, w = plot_frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 抓取第一個人 (通常示範影片只有一位教練) 的關節座標
        if res.keypoints is not None and len(res.keypoints.xy) > 0:
            pts = res.keypoints.xy[0].cpu().numpy()
            if len(pts) > 16:
                # 這裡預設以右膝的角度作為特徵數據 (12:髖, 14:膝, 16:踝)
                angles.append(compute_angle(pts[12], pts[14], pts[16]))
        
        writer.write(plot_frame)
    
    cap.release()
    if writer: writer.release()

    # 分析這一整段角度序列
    stats = calc_kinematics(angles, fps)
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps)
    return stats, duration

# ==========================================
# 5. 主程式執行入口
# ==========================================

def main():
    print("\n" + "="*55)
    print("🚀 三總復健系統：YOLO 運動分析與自動建檔管線 啟動中...")
    print("="*55 + "\n")
    
    # 強制使用 GPU：若環境未安裝 CUDA，將直接在後續初始化/推論時爆掉（不允許偷跑 CPU）
    print(f"[yolo_pose_rep_counter] CUDA device(0): {torch.cuda.get_device_name(0)}")

    # 載入 YOLO 模型並移到 GPU
    model = YOLO(MODEL_PATH)
    try:
        model.to("cuda")
    except Exception:
        # ultralytics 版本差異時仍可透過 predict(device=0) 強制 GPU
        pass
    
    # 掃描原始影片資料夾
    video_files = [f for f in os.listdir(VIDEO_IN_DIR) if f.lower().endswith(".mp4")]
    
    if not video_files:
        print(f"📍 在 {VIDEO_IN_DIR} 找不到任何影片檔案。")
        return

    all_processed_data = []
    
    for f in video_files:
        # 【增量更新邏輯】：檢查是否已經辨識過 (輸出目錄已有帶 yolo 字樣影片)
        target_skeleton_vid = os.path.join(VIDEO_OUT_DIR, f.replace(".mp4", "_yolo.mp4"))
        
        if os.path.exists(target_skeleton_vid):
            print(f"⏭️  跳過已分析影片: {f}")
            continue

        print(f"🎬 發現新動作，開始進行 AI 分析: {f}")
        # 1. 執行影像分析與物理運算 (回傳包含 impact_bw_high 的字典)
        stats, duration = process_video_and_save_skeleton(model, f)
        
        # 2. 呼叫 GPT 進行語意分析
        gpt_info = call_openai_label(
            file_name=f, 
            duration_s=duration, 
            stats=stats, 
            activity_level=stats["impact_level"], # 動態傳入計算出的強度
            user_condition={}, 
            risk_assessment={}
        )
        
        # 3. 打包成一筆記錄 (透過 **stats 將所有新欄位自動展開)
        combined_row = {
            "file_name": f,
            **stats,  # 這裡會包含 impact_bw_high, impact_level, primary_region 等
            "gpt_summary": gpt_info["gpt_summary"],
            "safety_tip": gpt_info["gpt_safety_tip"]
        }
        all_processed_data.append(combined_row)
        
    # 如果有新算出的數據，就執行自動同步
    if all_processed_data:
        sync_to_json_library(all_processed_data)
    else:
        print("\n✨ 任務完成！所有影片皆已處理，資料庫已是最新狀態。")

if __name__ == "__main__":
    main()