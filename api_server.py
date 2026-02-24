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

# 引入您寫好的條件整理與風險評估
from rag.user_condition_mapper import build_user_context

# 引入推薦引擎函數與資料結構
from modules.recommender_filter import (
    UserState, 
    load_exercise_library, 
    hard_filter_exercises, 
    soft_rank_exercises
)

# 【新增】引入 GPT 模組的 OpenAI 客戶端
try:
    from modules.gpt_summary import openai_client, OPENAI_MODEL
except ImportError:
    openai_client = None
    OPENAI_MODEL = "gpt-4o"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 輔助函數：生成 GPT 完整課表總說明
# ==========================================
def generate_today_summary(user_condition, risk_assessment, selected_exercises):
    """將選出的多個動作打包，請 GPT 產生一段給病患的綜合說明"""
    if not openai_client:
        return "【系統提醒】本課表已依據您的狀況進行個人化調整，請量力而為。若有任何不適請立即停止運動。"

    ex_names = [ex.get('name_zh', '復健運動') for ex in selected_exercises]
    nyha = user_condition.get("nyha", "未知")
    risk = risk_assessment.get("risk_level", "中等")

    prompt = f"""你是一位充滿同理心的復健指導員。
病患目前狀況：心臟衰竭 NYHA {nyha} 分級，系統風險評估為 {risk}。
系統剛剛為他安排了今日的「連續跟練影片課表」，包含以下動作串聯：{', '.join(ex_names)}。

任務：
請用 3~4 句溫暖、白話的文字，向病患總結「這整套課表」的重點。
告訴病患這套動作主要鍛鍊哪裡、整體的強度感受，以及安全提醒（例如何時該暫停）。
絕對不要列點，不要使用醫學術語，請直接輸出一段自然流暢的口語說明。"""

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            timeout=15
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT 總說明生成失敗: {e}")
        return "【系統提醒】本課表已依據您的狀況進行個人化調整，請量力而為。若有任何不適請立即停止運動。"

# ==========================================
# 資料格式轉換（與前端 rehab_app.html 完全對應）
# 前端送出: age, gender, nyhaclass, baseWeight, exerciseHabit, improveGoal, sysBP, diaBP, heartRate, todayWeight, currentSymptom
# ==========================================
def _get(ui_data, *keys):
    """取第一個存在的 key（前後端對應，可混用 camelCase / snake_case）"""
    for k in keys:
        v = ui_data.get(k)
        if v is not None and v != "":
            return v
    return None

def map_ui_to_rag_format(ui_data):
    """將網頁 (HTML) 傳來的 JSON 翻譯成 RAG 期待的巢狀中文"""
    gender_map = {"male": "男性", "female": "女性"}
    
    age_raw = _get(ui_data, "age") or ""
    age_map = {
        "18-39": "18–39歲", "40-54": "40–54歲",
        "55-64": "55–64歲", "65-74": "65–74歲", "75+": "75歲以上"
    }
    age_str = age_map.get(age_raw, "65–74歲") 

    nyha_raw = _get(ui_data, "nyhaclass", "nyhaClass")
    nyha_map = {
        "I": "I 級（日常活動無症狀）", "II": "II 級（一般活動輕微症狀）",
        "III": "III 級（輕度活動即有症狀）", "IV": "IV 級（休息時仍有症狀）"
    }
    nyha_str = nyha_map.get(nyha_raw, "II 級（一般活動輕微症狀）")

    chronic = _get(ui_data, "chronic") or []
    surgery = _get(ui_data, "surgery") or []
    symptoms = _get(ui_data, "symptoms") or []
    sys_bp = _get(ui_data, "sysBP", "sys_bp")
    sysBP = int(sys_bp) if sys_bp is not None else 120
    current_symptom = _get(ui_data, "currentSymptom", "current_symptom") or "none"
    
    is_stable = "不穩定" if (sysBP > 180 or sysBP < 90 or current_symptom != "none") else "穩定"

    rag_input = {
        "基本資料": {
            "族群": "心臟衰竭",
            "年齡層": age_str,
            "性別": gender_map.get(_get(ui_data, "gender"), "男性")
        },
        "心臟衰竭狀態": {
            "NYHA 心臟功能分級": nyha_str,
            "目前是否穩定": is_stable,
            "心臟衰竭類型": "射出分率降低型", 
            "是否曾發生急性惡化": "否", 
            "是否使用心室輔助器（LVAD）": "是" if surgery and "左心室輔助器 (LVAD)" in surgery else "否"
        },
        "疾病史": {
            "是否有高血壓": "是" if chronic and "高血壓" in chronic else "否",
            "是否有糖尿病": "是" if chronic and "糖尿病" in chronic else "否",
            "是否有冠狀動脈疾病": "是" if chronic and "冠心症/心肌梗塞" in chronic else "否",
            "是否有心律不整": "是" if chronic and "心律不整" in chronic else "否",
            "是否有肺部疾病": "否"
        },
        "開刀史": {
            "是否曾接受心臟支架或繞道手術": "是" if surgery and ("心導管/支架置放" in surgery or "冠狀動脈繞道手術" in surgery) else "否",
            "是否曾接受心臟瓣膜手術": "否",
            "是否曾接受人工關節置換": "是" if surgery and "人工關節置換" in surgery else "否",
            "是否有近期大型手術史": "否"
        },
        "用藥史": {
            "是否使用鈣離子通道阻斷劑": "否",
            "是否使用抗心律不整藥物": "否",
            "是否使用影響運動耐受性的藥物": "否"
        },
        "個人自覺症狀評估": {
            "呼吸困難程度": "中度" if (symptoms and "活動時呼吸喘促 (喘)" in symptoms) or current_symptom == "breathless" else "無",
            "是否出現胸痛": "是" if current_symptom == "chest_pain" else "否",
            "是否出現頭暈或接近昏厥": "是" if current_symptom == "chest_pain" else "否",
            "下肢水腫程度": "明顯" if symptoms and "下肢水腫 (腫)" in symptoms else "無",
            "疲勞程度": "中等" if (symptoms and "容易疲倦無力 (累)" in symptoms) or current_symptom == "mild_tired" else "無"
        },
        "運動情境": {
            "運動目的": "復健訓練"
        },
        "運動中即時回饋（可選填）": {
            "是否出現不適症狀": "否",
            "自覺運動強度（RPE）": "中等",
            "是否出現異常心悸": "否"
        }
    }
    return rag_input


@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """接收網頁數據 ➡️ 風險評估 ➡️ 知識庫篩選 (Hard/Soft) ➡️ GPT總摘要 ➡️ 回傳網頁"""
    ui_data = request.json
    print("\n📥 [1] 收到前端網頁的原始資料...")
    
    try:
        # --- 步驟 A：格式翻譯 ---
        rag_formatted_data = map_ui_to_rag_format(ui_data)
        
        # --- 步驟 B：建立 Context 與 運動前風險篩選 (Risk Gate) ---
        user_context = build_user_context(rag_formatted_data)
        user_condition = user_context.get("user_conditions", {})
        risk_assessment = user_context.get("risk_assessment", {})
        
        print(f"🛡️ [2] Risk Gate 評估結果: {risk_assessment.get('risk_level')} (允許運動: {risk_assessment.get('allow_exercise')})")

        if not risk_assessment.get('allow_exercise', True):
            reason = risk_assessment.get('reason', '今日生理數值異常，不建議進行運動。')
            return jsonify({"status": "error", "message": reason}), 400

        # --- 步驟 C：真實的推薦模組 ---
        print("🧠 [3] 讀取知識庫並進行 Hard Filter / Soft Ranking...")
        
        contraindications = (
            user_condition.get("contraindications") or 
            risk_assessment.get("contraindications") or 
            risk_assessment.get("risk_flags") or []
        )
        
        user_state = UserState(
            nyha=user_condition.get("nyha", ""),
            contraindications=contraindications
        )
        
        lib_path = os.path.join(SCRIPT_DIR, "knowledge_base", "exercise_library.json")
        map_path = os.path.join(SCRIPT_DIR, "knowledge_base", "exercise_video_map.json")
        
        exercise_library = load_exercise_library(lib_path)
        with open(map_path, "r", encoding="utf-8") as f:
            video_map = json.load(f)
            
        filtered = hard_filter_exercises(user_state, exercise_library)
        ranked_exercises = soft_rank_exercises(user_state, filtered["included"])
        
        # --- 步驟 D：將結果包裝成網頁需要的 JSON 格式（與前端完全對應）---
        videos_for_html = []
        color_themes = ["bg-blue-100 text-blue-600", "bg-green-100 text-green-600", "bg-purple-100 text-purple-600", "bg-orange-100 text-orange-600"]
        
        selected_top_4 = ranked_exercises[:4]
        action_descriptions = [f"動作 {idx+1}：{ex.get('name_zh', ex['exercise_id'])} (建議 {ex.get('intensity_band', '適度')} 強度)" for idx, ex in enumerate(selected_top_4)]
        
        # 一週課表固定 7 筆，對應前端週一～週日 (plan[0]..plan[6])
        plan_for_html = [
            f"依據您的 NYHA {user_state.nyha} 分級與身體狀況，今日建議：",
            action_descriptions[0] if len(action_descriptions) > 0 else "溫和活動",
            action_descriptions[1] if len(action_descriptions) > 1 else "休息或伸展",
            action_descriptions[2] if len(action_descriptions) > 2 else "休息或伸展",
            action_descriptions[3] if len(action_descriptions) > 3 else "休息或伸展",
            "休息日（恢復）",
            "休息日（恢復）"
        ]

        for idx, ex in enumerate(selected_top_4):
            ex_id = ex["exercise_id"]
            title_zh = ex.get("name_zh", ex_id)
            score = ex.get("soft_rank_score", 0)
            
            raw_vids = video_map.get(ex_id)
            filenames = [raw_vids] if isinstance(raw_vids, str) else (raw_vids if isinstance(raw_vids, list) else [])
            
            for f_name in filenames:
                if not f_name: continue
                vid_path = f"exercise_videos/{f_name}" if not f_name.startswith("exercise_videos") else f_name
                
                videos_for_html.append({
                    "title": f"{title_zh} (AI評分:{score})",
                    "filename": vid_path,
                    "duration": "0:30", 
                    "tags": [f"NYHA {user_state.nyha}", ex.get("impact_level", "安全")],
                    "color": color_themes[idx % len(color_themes)]
                })
            
        if not videos_for_html:
            videos_for_html.append({
                "title": "溫和伸展 (系統預設)",
                "filename": "exercise_videos/單邊抬腳.mp4",
                "duration": "10:00",
                "tags": ["安全", "舒緩"],
                "color": "bg-gray-100 text-gray-600"
            })

        # --- 步驟 E：產生 GPT 總說明 ---
        print("💬 [4] 正在請求 GPT 生成今日課表總說明...")
        gpt_summary_text = generate_today_summary(user_condition, risk_assessment, selected_top_4)

        print(f"✅ [5] 成功推薦 {len(videos_for_html)} 部專屬影片與摘要，準備回傳給網頁！\n")
        
        return jsonify({
            "status": "success",
            "message": "評估分析完成",
            "videos": videos_for_html,
            "plan": plan_for_html,
            "gpt_summary": gpt_summary_text  # 將 GPT 總摘要傳給前端
        })
        
    except Exception as e:
        print("❌ 系統分析時發生異常:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/video_feed')
def video_feed():
    """提供即時影像串流給 HTML (OpenCV MJPEG Stream)"""
    videos_param = request.args.get('videos', '')
    if not videos_param:
        return "沒有提供影片", 400
        
    playlist = videos_param.split(',')
    print(f"🎥 開始即時影像串流，清單: {playlist}")
    
    return Response(
        generate_frames(playlist),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    print("=========================================")
    print("🚀 三總心臟衰竭復健系統 API 伺服器啟動中...")
    print("=========================================")
    
    html_path = os.path.join(SCRIPT_DIR, "rehab_app.html")
    webbrowser.open(f"file://{html_path}")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)