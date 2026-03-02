# 檔案名稱：api_server.py
# Python 3.x | flask, flask-cors

import json
import os
import sys
import traceback
import webbrowser
from threading import Timer

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from rag.user_condition_mapper import build_user_context
from stream_engine import generate_frames
from modules.recommender_filter import (
    UserState,
    load_exercise_library,
    hard_filter_exercises,
    soft_rank_exercises,
)
try:
    from modules.gpt_summary import generate_today_summary, generate_7_day_plan
except ImportError:
    from gpt_summary import generate_today_summary, generate_7_day_plan

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 資料格式轉換邏輯
# ==========================================
def _get(ui_data, *keys):
    for k in keys:
        v = ui_data.get(k)
        if v is not None and v != "":
            return v
    return None


def map_ui_to_rag_format(ui_data):
    """將 UI 資料翻譯成 RAG 期待格式，並加入新數據判斷。"""
    gender_map = {"male": "男性", "female": "女性"}
    age_map = {
        "18-39": "18–39歲", "40-54": "40–54歲", "55-64": "55–64歲",
        "65-74": "65–74歲", "75+": "75歲以上",
    }
    nyha_map = {"I": "I 級", "II": "II 級", "III": "III 級", "IV": "IV 級"}

    current_symptom = _get(ui_data, "currentSymptom") or "none"
    sys_bp = int(_get(ui_data, "sysBP") or 120)
    dia_bp = int(_get(ui_data, "diaBP") or 80)
    heart_rate = int(_get(ui_data, "heartRate") or 75)

    # 穩定性：收縮壓>180 / 舒張壓>110 / 心跳>120 / 有症狀 → 不穩定
    is_stable = "穩定"
    if sys_bp > 180 or dia_bp > 110 or heart_rate > 120 or current_symptom != "none":
        is_stable = "不穩定"

    return {
        "基本資料": {
            "族群": "心臟衰竭",
            "年齡層": age_map.get(_get(ui_data, "age"), "65–74歲"),
            "性別": gender_map.get(_get(ui_data, "gender"), "男性"),
        },
        "心臟衰竭狀態": {
            "NYHA 心臟功能分級": nyha_map.get(_get(ui_data, "nyhaclass"), "II 級"),
            "目前是否穩定": is_stable,
            "生理數值紀錄": {
                "收縮壓": sys_bp,
                "舒張壓": dia_bp,
                "安靜心跳": heart_rate,
            },
        },
        "個人自覺症狀評估": {
            "呼吸困難程度": "中度" if current_symptom == "breathless" else "無",
            "是否出現胸痛": "是" if current_symptom == "chest_pain" else "否",
            "是否出現頭暈": "是" if current_symptom == "dizzy" else "否",
        },
        "運動情境": {"運動目的": "復健訓練"},
    }


@app.route("/api/analyze", methods=["POST"])
def analyze_data():
    """接收數據 ➡️ 篩選動作 ➡️ 回傳物理指標與影片"""
    ui_data = request.json
    try:
        # 1. 安全評估 (RAG Engine)
        rag_formatted_data = map_ui_to_rag_format(ui_data)
        user_context = build_user_context(rag_formatted_data)
        user_condition = user_context.get("user_conditions", {})
        risk_assessment = user_context.get("risk_assessment", {})

        if not risk_assessment.get("allow_exercise", True):
            msg = risk_assessment.get("reason", "數值異常")
            return jsonify({"status": "error", "message": msg}), 400

        # 2. 動作推薦與數據讀取
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
                "tip": ex.get("gpt_safety_tip") or "請依個人節奏進行。",
            })

        # 3. GPT 今日總評與計畫
        gpt_summary_text = generate_today_summary(
            user_condition, risk_assessment, selected_top_4
        )
        gpt_weekly_plan = generate_7_day_plan(
            user_condition, risk_assessment, selected_top_4
        )

        return jsonify({
            "status": "success",
            "videos": videos_for_html,
            "plan": gpt_weekly_plan,
            "gpt_summary": gpt_summary_text,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/video_feed")
def video_feed():
    """跟練串流：接收影片清單並啟動雙畫面引擎。"""
    videos_param = request.args.get("videos", "")
    if not videos_param:
        return "沒有提供影片", 400
    playlist = videos_param.split(",")
    return Response(
        generate_frames(playlist),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    ui_path = os.path.join(SCRIPT_DIR, "rehab_app.html")
    file_url = "file:///" + os.path.abspath(ui_path).replace("\\", "/")

    print("\n🚀 AI 心衰復健 API 伺服器啟動完成")
    print("📍 API：http://127.0.0.1:5000")
    print("📍 請手動開啟前端：" + file_url + "\n")

    def open_browser():
        if os.path.exists(ui_path):
            try:
                webbrowser.open(file_url)
            except Exception:
                pass

    Timer(2.0, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)