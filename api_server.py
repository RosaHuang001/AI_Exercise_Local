# 檔案名稱：api_server.py
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import sys
import traceback
import webbrowser
import json

# 引入影像串流引擎
from stream_engine import generate_frames 

# 引入條件整理與風險評估
from rag.user_condition_mapper import build_user_context

# 引入推薦引擎函數與資料結構
from modules.recommender_filter import (
    UserState, 
    load_exercise_library, 
    hard_filter_exercises, 
    soft_rank_exercises
)

# 引入 GPT 模組
try:
    from modules.gpt_summary import generate_today_summary, generate_7_day_plan
except ImportError:
    print("⚠️ 警告：無法載入 gpt_summary 模組，將使用預設文字。")
    def generate_today_summary(*args): return "系統已依據狀況為您安排課表。"
    def generate_7_day_plan(*args): return ["預設動作"] * 7

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 資料格式轉換邏輯
# ==========================================
def _get(ui_data, *keys):
    for k in keys:
        v = ui_data.get(k)
        if v is not None and v != "": return v
    return None

def map_ui_to_rag_format(ui_data):
    """將 UI 資料翻譯成 RAG 期待格式"""
    gender_map = {"male": "男性", "female": "女性"}
    age_map = {"18-39": "18–39歲", "40-54": "40–54歲", "55-64": "55–64歲", "65-74": "65–74歲", "75+": "75歲以上"}
    nyha_map = {"I": "I 級", "II": "II 級", "III": "III 級", "IV": "IV 級"}
    
    current_symptom = _get(ui_data, "currentSymptom") or "none"
    sys_bp = _get(ui_data, "sysBP")
    is_stable = "不穩定" if (int(sys_bp or 120) > 180 or current_symptom != "none") else "穩定"

    return {
        "基本資料": {
            "族群": "心臟衰竭",
            "年齡層": age_map.get(_get(ui_data, "age"), "65–74歲"),
            "性別": gender_map.get(_get(ui_data, "gender"), "男性")
        },
        "心臟衰竭狀態": {
            "NYHA 心臟功能分級": nyha_map.get(_get(ui_data, "nyhaclass"), "II 級"),
            "目前是否穩定": is_stable
        },
        "個人自覺症狀評估": {
            "呼吸困難程度": "中度" if current_symptom == "breathless" else "無",
            "是否出現胸痛": "是" if current_symptom == "chest_pain" else "否"
        },
        "運動情境": {"運動目的": "復健訓練"}
    }

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """接收數據 ➡️ 篩選動作 ➡️ 回傳原始影片路徑與運動指標"""
    ui_data = request.json
    try:
        # 1. 安全評估
        rag_formatted_data = map_ui_to_rag_format(ui_data)
        user_context = build_user_context(rag_formatted_data)
        user_condition = user_context.get("user_conditions", {})
        risk_assessment = user_context.get("risk_assessment", {})

        if not risk_assessment.get('allow_exercise', True):
            return jsonify({"status": "error", "message": risk_assessment.get('reason', '數值異常')}), 400

        # 2. 動作推薦
        user_state = UserState(nyha=user_condition.get("nyha", ""), contraindications=risk_assessment.get("risk_flags", []))
        exercise_library = load_exercise_library(os.path.join(SCRIPT_DIR, "knowledge_base", "exercise_library.json"))
        
        with open(os.path.join(SCRIPT_DIR, "knowledge_base", "exercise_video_map.json"), "r", encoding="utf-8") as f:
            video_map = json.load(f)
            
        filtered = hard_filter_exercises(user_state, exercise_library)
        ranked_exercises = soft_rank_exercises(user_state, filtered["included"])
        
        videos_for_html = []
        color_themes = ["bg-blue-100 text-blue-600", "bg-green-100 text-green-600", "bg-purple-100 text-purple-600", "bg-orange-100 text-orange-600"]
        selected_top_4 = ranked_exercises[:4]
        
        for idx, ex in enumerate(selected_top_4):
            ex_id = ex["exercise_id"]
            raw_vids = video_map.get(ex_id)
            filenames = [raw_vids] if isinstance(raw_vids, str) else (raw_vids if isinstance(raw_vids, list) else [])
            
            # 擷取線下算好的精準運動學數據 (Reps/Freq/ROM)
            stats_list = [
                {"icon": "refresh-cw", "text": f"{ex.get('reps', 0)} 次/組"},
                {"icon": "timer", "text": f"節奏 {float(ex.get('frequency_hz', 0)):.1f}/s"},
                {"icon": "move-horizontal", "text": f"幅度 {float(ex.get('rom_p5_p95', 0)):.0f}°"}
            ]

            for f_name in filenames:
                if not f_name: continue
                
                # 🔥 關鍵修正：確保 filename 指向原始的 exercise_videos 資料夾
                # 使用者在跟練時看到的是最清晰的原片，但顯示的數據是 YOLO 算出來的精華
                clean_vid_path = f"exercise_videos/{f_name}"
                
                videos_for_html.append({
                    "title": ex.get("name_zh", ex_id),
                    "filename": clean_vid_path,
                    "tags": [f"NYHA {user_state.nyha}", ex.get("impact_level", "安全")],
                    "stats": stats_list,
                    "color": color_themes[idx % len(color_themes)],
                    "tip": ex.get("gpt_safety_tip") or "請依個人節奏進行，保持呼吸平穩。"
                })
        
        # 3. GPT 生成文案
        gpt_summary_text = generate_today_summary(user_condition, risk_assessment, selected_top_4)
        gpt_weekly_plan = generate_7_day_plan(user_condition, risk_assessment, selected_top_4)

        return jsonify({
            "status": "success",
            "videos": videos_for_html,
            "plan": gpt_weekly_plan,
            "gpt_summary": gpt_summary_text
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """啟動跟練引擎：左側顯示原始影片，右側運算使用者骨架"""
    videos_param = request.args.get('videos', '')
    if not videos_param: return "沒有提供影片", 400
    playlist = videos_param.split(',')
    return Response(generate_frames(playlist), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🚀 心臟衰竭復健運動推薦系統 API 伺服器啟動中...")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)