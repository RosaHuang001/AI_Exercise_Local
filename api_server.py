# -*- coding: utf-8 -*-
# api_server.py
# Python 3.x | flask, flask-cors

import json
import os
import sys
import traceback
import webbrowser
from threading import Timer

import cv2
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from rag_module.user_condition_mapper import build_user_context
from stream_engine import generate_frames, get_last_session_stats
from modules.recommender_filter import (
    UserState,
    load_exercise_library,
    hard_filter_exercises,
    soft_rank_exercises,
)
try:
    from modules.gpt_summary import (
        generate_today_summary,
        generate_7_day_plan,
        generate_rpe_instruction,
    )
except ImportError:
    from gpt_summary import generate_today_summary, generate_7_day_plan, generate_rpe_instruction

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 資料輔助解析與格式轉換
# ==========================================
def _get(ui_data, *keys):
    for k in keys:
        v = ui_data.get(k)
        if v is not None and v != "":
            return v
    return None


def map_ui_to_rag_format(ui_data):
    """
    將前端 UI JSON 轉換為 RAG / user_condition_mapper 要求的格式
    此處的 Key 必須與 rag_module.user_condition_mapper.map_user_conditions 完全對應
    """
    gender_map = {"male": "男性", "female": "女性"}
    age_map = {
        "18-39": "18–39歲",
        "40-54": "40–54歲",
        "55-64": "55–64歲",
        "65-74": "65–74歲",
        "75+": "75歲以上",
    }
    nyha_map = {"I": "I 級", "II": "II 級", "III": "III 級", "IV": "IV 級"}

    current_symptom = _get(ui_data, "currentSymptom") or "none"
    sys_bp = int(_get(ui_data, "sysBP") or 120)
    dia_bp = int(_get(ui_data, "diaBP") or 80)
    heart_rate = int(_get(ui_data, "heartRate") or 75)
    chronic = ui_data.get("chronic") or []
    if not isinstance(chronic, list):
        chronic = []
    has = lambda label: any(label in str(x) for x in chronic)

    # 穩定度判斷：收縮壓 > 180 / 舒張壓 > 110 / 心跳 > 120 / 有症狀 => 不穩定
    is_stable = "穩定"
    if sys_bp > 180 or dia_bp > 110 or heart_rate > 120 or current_symptom != "none":
        is_stable = "不穩定"

    # 呼吸困難判定
    dyspnea_level = "無"
    if current_symptom == "breathless":
        dyspnea_level = "輕度"

    chest_pain = "是" if current_symptom == "chest_pain" else "否"
    dizziness = "是" if current_symptom == "dizzy" else "否"

    return {
        "基本資料": {
            "族群": "心臟衰竭",
            "年齡層": age_map.get(_get(ui_data, "age"), "65–74歲"),
            "性別": gender_map.get(_get(ui_data, "gender"), "男性"),
        },
        "心臟衰竭狀態": {
            "NYHA 心臟功能分級": nyha_map.get(_get(ui_data, "nyhaclass"), "II 級"),
            "目前是否穩定": is_stable,
            "生理數值監測": {
                "收縮壓": sys_bp,
                "舒張壓": dia_bp,
                "安靜心跳": heart_rate,
            },
            # 以下為預設或前端未提供的欄位
            "心臟衰竭類型": "射出分率降低型",
            "是否使用心室輔助器（LVAD）": "否",
        },
        "疾病史": {
            "是否有高血壓": "是" if has("高血壓") else "否",
            "是否有糖尿病": "是" if has("糖尿病") else "否",
            "是否有冠狀動脈疾病": "是" if (has("冠心症") or has("心肌梗塞") or has("冠狀動脈")) else "否",
            "是否有心律不整": "是" if has("心律不整") else "否",
            "是否有肺部疾病": "是" if (has("肺") or has("COPD") or has("慢性阻塞")) else "否",
        },
        "開刀史": {
            "是否曾接受心臟支架或繞道手術": "否",
            "是否曾接受心臟瓣膜手術": "否",
            "是否曾接受人工關節置換": "否",
            "是否有近期大型手術史": "否",
        },
        "用藥史": {
            "是否使用鈣離子通道阻斷劑": "否",
            "是否使用抗心律不整藥物": "否",
            "是否使用影響運動耐受性的藥物": "否",
        },
        "個人自覺症狀評估": {
            "呼吸困難程度": dyspnea_level,
            "是否出現胸痛": chest_pain,
            "是否出現頭暈或接近昏厥": dizziness,
            "疲勞程度": "輕度" if current_symptom != "none" else "無",
        },
        "運動情境": {
            "運動目的": "復健訓練",
        },
    }


@app.route("/api/analyze", methods=["POST"])
def analyze_data():
    """接收病患資訊 -> 篩選動作 -> 回傳 GPT 建議與影片"""
    ui_data = request.json
    try:
        # Layer 1：嚴格安全門檻（中止/不宜運動條件）
        sys_bp = int(ui_data.get("sysBP") or 0)
        dia_bp = int(ui_data.get("diaBP") or 0)
        heart_rate = int(ui_data.get("heartRate") or 0)
        symptom = ui_data.get("currentSymptom") or "none"
        if symptom in ("chest_pain", "dizzy"):
            return jsonify({
                "status": "error",
                "message": "安全性分析未通過：偵測到胸悶/胸痛或頭暈症狀。請先停止運動、緩慢深呼吸並聯繫醫護人員。",
            }), 400
        if sys_bp >= 180 or dia_bp >= 110 or heart_rate >= 120:
            return jsonify({
                "status": "error",
                "message": "安全性分析未通過：血壓/心跳偏高（屬於高風險區間）。請先休息 10 分鐘後再量測，必要時請回診評估。",
            }), 400
        if symptom == "breathless" and (sys_bp >= 160 or heart_rate >= 110):
            return jsonify({
                "status": "error",
                "message": "安全性分析未通過：合併喘促且生理負擔偏高。建議先做 3–5 分鐘呼吸放鬆與坐姿休息，待心跳平穩後再評估是否運動。",
            }), 400

        # 1. 安全評估 (RAG Engine + Mapper)
        rag_formatted_data = map_ui_to_rag_format(ui_data)
        user_context = build_user_context(rag_formatted_data)
        user_condition = user_context.get("user_conditions", {})
        risk_assessment = user_context.get("risk_assessment", {})

        if not risk_assessment.get("allow_exercise", True):
            msg = risk_assessment.get("reason", "數值異常，不建議運動")
            return jsonify({"status": "error", "message": msg}), 400

        # 2. 運動推薦與硬性過濾
        user_state = UserState(
            nyha=user_condition.get("nyha", ""),
            contraindications=risk_assessment.get("risk_flags", []),
        )
        lib_path = os.path.join(SCRIPT_DIR, "knowledge_base", "exercise_library.json")
        exercise_library = load_exercise_library(lib_path)
        filtered = hard_filter_exercises(user_state, exercise_library)
        ranked_exercises = soft_rank_exercises(user_state, filtered["included"])

        videos_for_html = []
        selected_top_4 = ranked_exercises[:4]
        for ex in selected_top_4:
            f_name = ex.get("video_filename")
            if not f_name:
                continue
            stats_list = [
                {"icon": "refresh-cw", "text": f"{ex.get('reps', 0)} 次/組"},
                {"icon": "timer", "text": f"節奏 {float(ex.get('frequency_hz', 0)):.1f}/s"},
                {"icon": "move-horizontal", "text": f"幅度 {float(ex.get('rom_p5_p95', 0)):.0f}°"},
            ]
            videos_for_html.append({
                "title": ex.get("name_zh"),
                "filename": f"exercise_videos/{f_name}",
                "tags": [f"NYHA {user_state.nyha}", ex.get("impact_level", "低衝擊")],
                "stats": stats_list,
                "tip": ex.get("gpt_safety_tip") or "請依個人節奏進行運動",
            })

        # 3. GPT 生成建議
        # (1) 生成今日摘要與一週計畫
        gpt_summary_text = generate_today_summary(user_condition, risk_assessment, selected_top_4)
        current_vitals = {
            "sysBP": sys_bp,
            "heartRate": heart_rate,
            "currentSymptom": symptom,
            "chronic": ui_data.get("chronic") or [],
        }
        gpt_weekly_result = generate_7_day_plan(user_condition, risk_assessment, selected_top_4, current_vitals)
        plan_text = gpt_weekly_result.get("plan_text", ["運動 2 組"] * 7)[:7]
        suggested_reps = gpt_weekly_result.get("suggested_reps", 8)
        suggested_rest = int(gpt_weekly_result.get("suggested_rest", 60) or 60)
        gpt_daily = gpt_weekly_result.get("daily_tasks")
        if isinstance(gpt_daily, list) and len(gpt_daily) == 7 and gpt_daily[0]:
            # 注入 rest（讓前端組間休息秒數與處方一致）
            daily_tasks = []
            for day in gpt_daily:
                if not isinstance(day, list):
                    daily_tasks.append([])
                    continue
                daily_tasks.append([
                    {
                        "file": t.get("file", ""),
                        "reps": int(t.get("reps", suggested_reps) or suggested_reps),
                        "name": t.get("name", ""),
                        "rest": suggested_rest,
                    }
                    for t in day if isinstance(t, dict)
                ])
        else:
            today_tasks_raw = [
                {"file": f"exercise_videos/{ex.get('video_filename')}", "reps": suggested_reps, "name": ex.get("name_zh", ""), "rest": suggested_rest}
                for ex in selected_top_4 if ex.get("video_filename")
            ]
            daily_tasks = [today_tasks_raw] + [[]] * 6

        # (2) 生成 RPE 專業強度描述
        try:
            rpe_instruction = generate_rpe_instruction(user_condition, selected_top_4)
        except Exception:
            rpe_instruction = "根據 ACSM 標準，請維持 [RPE:11-13] 的運動強度。"

        # (3) 今日任務結構（相容舊前端）：filename / target / rest；target 與 daily_tasks[0] 一致
        day0 = daily_tasks[0] if daily_tasks else []
        rep_by_file = {t["file"]: t["reps"] for t in day0} if day0 else {}
        today_tasks = [
            {"filename": f"exercise_videos/{ex.get('video_filename')}", "target": rep_by_file.get(f"exercise_videos/{ex.get('video_filename')}", suggested_reps), "rest": suggested_rest}
            for ex in selected_top_4 if ex.get("video_filename")
        ]

        # (4) 回傳整合資料
        return jsonify({
            "status": "success",
            "videos": videos_for_html,
            "seven_day_plan": plan_text,
            "today_summary": gpt_summary_text,
            "rpe_text": rpe_instruction,
            "today_tasks": today_tasks,
            "daily_tasks": daily_tasks,
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/thumbnail")
def thumbnail():
    """從影片抓取第一幀作為縮圖"""
    path = request.args.get("path", "").strip()
    if not path or ".." in path:
        return "缺失 path 參數", 400
    full_path = os.path.join(SCRIPT_DIR, path)
    if not os.path.isfile(full_path):
        return "檔案不存在", 404
    try:
        cap = cv2.VideoCapture(full_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return "無法讀取影片幀", 404
        _, buf = cv2.imencode(".jpg", frame)
        return Response(buf.tobytes(), mimetype="image/jpeg")
    except Exception as e:
        traceback.print_exc()
        return str(e), 500


@app.route("/")
def index():
    """載入前端介面檔案"""
    ui_path = os.path.join(SCRIPT_DIR, "rehab_app.html")
    if not os.path.isfile(ui_path):
        return "rehab_app.html 不存在", 404
    with open(ui_path, "r", encoding="utf-8") as f:
        return f.read()


@app.route("/video_feed")
def video_feed():
    """跟練串流：接收影片與 NYHA 分級，啟動視覺引擎"""
    videos_param = request.args.get("videos", "")
    if not videos_param:
        return "沒有影片路徑", 400
    playlist = videos_param.split(",")
    nyha = request.args.get("nyha", "II").strip()
    nyha_upper = nyha.upper()
    if nyha_upper in ("I", "II", "III"):
        nyha_level = {"I": "class_i", "II": "class_ii", "III": "class_iii"}[nyha_upper]
    elif nyha.lower() in ("class_i", "class_ii", "class_iii"):
        nyha_level = nyha.lower()
    else:
        nyha_level = "class_ii"
    return Response(
        generate_frames(playlist, nyha_level=nyha_level),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/session_stats")
def session_stats():
    """回傳最近一次 AI 跟練的統計數據"""
    return jsonify(get_last_session_stats())


if __name__ == "__main__":
    print("\n✅ AI 心衰復健 API 伺服器啟動中...")
    print("🚀 前端介面網址：http://127.0.0.1:5000/ \n")

    def open_browser():
        try:
            webbrowser.open("http://127.0.0.1:5000/")
        except Exception:
            pass

    Timer(2.0, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)