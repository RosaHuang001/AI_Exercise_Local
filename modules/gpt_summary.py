# -*- coding: utf-8 -*-
# modules/gpt_summary.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from rag_module.rag_engine import ACSMRagEngine
from rag_module.rule_controller import RuleController


# ==================== OpenAI 設定 ====================

load_dotenv()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

OPENAI_MODEL = "gpt-4o"


# ==================== RAG 設定 ====================

RAG_ENGINE = ACSMRagEngine()  # 內部已預設路徑，不必傳入參數

RULE_CONTROLLER = RuleController(
    max_rules=4,
    debug=False
)


def normalize(v, fallback_level="中"):
    s = str(v or "")
    if "高" in s:
        return "高"
    if "低" in s:
        return "低"
    if ("中" in s) or ("等" in s):
        return "中"
    return fallback_level


def build_impact_paragraph(stats: dict) -> str:
    """將物理精準數據轉化為白話描述段落"""
    impact_level = stats.get("impact_level", "未知")
    bw_low = float(stats.get("impact_bw_low", 0) or 0)
    bw_high = float(stats.get("impact_bw_high", 0) or 0)
    bins_text = stats.get("impact_by_weight_bins_text", "")

    primary_region = str(stats.get("primary_region", "Unknown") or "Unknown")
    primary_joint = str(stats.get("primary_joint", "Unknown") or "Unknown")
    primary_side = str(stats.get("primary_side", "Unknown") or "Unknown")

    if bw_low <= 0 or bw_high <= 0:
        return ""

    if primary_region == "Upper": region_zh = "上肢"
    elif primary_region == "Lower": region_zh = "下肢"
    elif primary_region == "Core": region_zh = "核心與軀幹"
    else: region_zh = "主要運動部位"

    side_zh = ""
    if "Right" in primary_side or primary_side in ("R", "Right"): side_zh = "右側"
    elif "Left" in primary_side or primary_side in ("L", "Left"): side_zh = "左側"

    joint_map = {
        "Hip": "髖關節", "Knee": "膝關節", "Ankle": "踝關節",
        "Elbow": "手肘", "Shoulder": "肩關節", "Wrist": "手腕", "Head": "頭頸"
    }
    joint_zh = primary_joint
    for k, v in joint_map.items():
        if k in primary_joint:
            joint_zh = v
            break

    target = f"{region_zh}{side_zh}{joint_zh}".replace("主要運動部位", "").strip()
    if not target: target = f"{side_zh}{joint_zh}".strip() or "主要受力關節"

    return (
        f"此動作主要訓練「{target}」，屬於{impact_level}衝擊負荷。"
        f"下肢局部預計最大承受{bw_low:.1f}至{bw_high:.1f}倍體重的壓力（個人體重換算約為 {bins_text}）。"
        "此數值為系統基於安全考量之保守推估，供您參考。"
    )


# =========================================================================
# ==================== 線下 Offline 專用：批次處理模組 ====================
# =========================================================================

def call_openai_label(file_name: str, duration_s: float, stats: dict, activity_level: str, user_condition: dict, risk_assessment: dict) -> dict:
    """用於線下批次建置用戶產出單一影片之 FITT 詳細描述，用於擴充知識庫"""
    fallback = {
        "gpt_summary": "本影片適合嘗試者進行訓練，請維持姿勢基礎動作。建議量力而為並確保身體穩定，以避免不適或運動終止。",
        "gpt_activity_level": "中",
        "gpt_risk_notice": "請注意動作速度不宜過快、角度不宜過大，以免增加關節負擔。",
        "gpt_safety_tip": "建議放慢速度、維持核心穩定，若有疼痛或不適請停止。",
        "gpt_long_text": ""
    }

    if openai_client is None:
        return fallback

    posture = stats.get("posture", "Unknown")
    primary_region = stats.get("primary_region", "Unknown")
    rom = float(stats.get("rom_p5_p95", 0) or 0)
    reps = int(stats.get("reps", 0) or 0)

    if reps == 0 and rom < 12:
        motion_type_hint = "此動作屬於「維持姿勢類」，包含小幅度調整，次數顯示為 0 屬正常現象。"
    else:
        motion_type_hint = "此動作屬「重複進行類」運動，可以針對次數進行描述。"

    region_map = {
        "Upper": "上肢（手臂、肩膀相關運動）",
        "Lower": "下肢（腿部與平衡能力穩定）",
        "Core": "核心（軀幹穩定、支撐性動作）",
    }
    region_zh = region_map.get(stats.get("primary_region"), "全身運動")

    posture_map = {
        "standing": "站姿",
        "sitting": "坐姿",
        "lying": "臥姿",
    }
    posture_zh = posture_map.get(str(stats.get("posture", "")).lower(), "一般姿勢")

    query_text = (
        f"心臟衰竭 NYHA {user_condition.get('nyha')} 級病患，"
        f"進行{posture_zh}下之{region_zh}運動建議與安全規範"
    )
    hf_rules_raw = RAG_ENGINE.retrieve_rules(query_text=query_text, k=4)

    user_profile = {
        "risk_level": risk_assessment.get("risk_level", "moderate"),
        "nyha": user_condition.get("nyha"),
        "is_stable": user_condition.get("hf_stable"),
        "posture": stats.get("posture"),
        "weight_bearing": stats.get("weight_bearing")
    }
    hf_rules = RULE_CONTROLLER.process(rules=hf_rules_raw, user_profile=user_profile)

    seen_content = set()
    unique_rules = []
    for r in hf_rules or []:
        content = (r.get("rule") or "").strip()
        if content and content not in seen_content:
            unique_rules.append(r)
            seen_content.add(content)

    rag_text = "\n".join(f"- {r.get('rule', '')}" for r in unique_rules) if unique_rules else ""

    system_prompt = f"""你是一位「復健專業人員」。你收到的資料包含影片的物理指標與多段 ACSM 醫療指引。
你的核心任務：將複雜的數據提煉為 3-4 句直觀、可執行的白話中文建議，說明示範影片的動作特徵。

【重點擷取與融合原則】：
1) 嚴禁逐字翻譯或條列所有的 ACSM 規則。請針對病患的 NYHA 等級與動作特徵（如：{posture_zh}、{region_zh}），僅擷取最關鍵的 1-2 個數值（如 RPE、MAP 或 FITT 核心數值）。
2) 數據融合：將數值自然融入建議中，例如：「運動時請感覺輕鬆（RPE 11-13），這能維持心肺穩定」。
3) 避免冗長：直接說明動作大小與快慢，不使用「首先」、「此外」、「總結」等瑣碎的連接詞。

【你的任務與嚴格規則】：
1. 嚴禁任何社交性問候語（Greeting-free zone），例如：午安、你好、加油、維持心肺穩定等。請直接從核心建議開始。
2. 必須明確告訴病患這支影片主要鍛鍊哪個身體部位（{region_zh}）。
3. 擷取核心數據：請從 ACSM 指引中僅擷取最關鍵的數值（如 RPE 11-13 或 MAP 70-90），並將其融入描述中。

【個人化與指令規則】：
1) 不假設病患已做過動作，不使用「你剛剛完成了」等回饋語氣。
2) 嚴禁社交賀詞（午安、加油等）。
3) 不做醫療診斷，不給治療或處方，語句使用「建議從...開始」。
4) 絕對禁止提及「AI」、「評分」、「模型」、「YOLO」、「分數」或「達成率」。
5) 避免專業術語，語氣溫和。
6) 只輸出 JSON，不要 markdown 或解釋。
7) 品質預測：利用生理數值產出 40 字內「疲勞警報」，格式為「【品質預測】：」。
"""
    if rag_text:
        system_prompt += f"\n【ACSM 指引補充】：\n{rag_text}"

    user_prompt = f"""
【系統自動判定活動強度】：
- 本次動作強度等級：{activity_level}強度

【影片基本資訊】：
- 檔案：{file_name}
- 影片長度：約 {duration_s:.1f} 秒
- 姿勢：{posture}
- 主要訓練區域：{region_hint}

【主要運動學指標】：
- ROM：{rom:.1f} 度
- 次數：{reps} 次

【動作類型提示】：
- {motion_type_hint}

【使用者個人情況】：
- 族群：{user_condition.get("population")}
- 心臟功能分級（NYHA）：{user_condition.get("nyha")}
- 風險評估：{risk_assessment.get("risk_level")}

請輸出以下 JSON（僅輸出 JSON）：
{{
  "gpt_summary": "撰寫一段約 3-4 句的白話描述，說明這是什麼動作、用於哪裡，並提供起步建議。",
  "gpt_activity_level": "低 / 中 / 高",
  "gpt_risk_notice": "一句話風險提醒",
  "gpt_safety_tip": "一句話安全建議",
  "gpt_long_text": ""
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

        content = response.choices[0].message.content or "{}"
        data = json.loads(content.strip())
        
        impact_para = build_impact_paragraph(stats)
        summary = data.get("gpt_summary", fallback["gpt_summary"])
        if impact_para and ("衝擊" not in summary and "體重" not in summary):
            summary = summary.rstrip() + " " + impact_para

        return {
            "gpt_summary": summary,
            "gpt_activity_level": normalize(data.get("gpt_activity_level"), fallback["gpt_activity_level"]),
            "gpt_risk_notice": data.get("gpt_risk_notice", fallback["gpt_risk_notice"]),
            "gpt_safety_tip": data.get("gpt_safety_tip", fallback["gpt_safety_tip"]),
            "gpt_long_text": data.get("gpt_long_text", fallback["gpt_long_text"]),
        }

    except Exception as e:
        print(f"[WARN] GPT 處理失敗，使用 fallback：{e}")
        return fallback


# =========================================================================
# ==================== 線上 Online Web API 專用：今日建議 ====================
# =========================================================================

def generate_today_summary(user_condition: dict, risk_assessment: dict, selected_exercises: list) -> str:
    """用於網頁呈現之今日建議摘要"""
    if not openai_client:
        return "【系統提示】本課表已依據您的情況進行調整，請量力而為。若有任何不適請立即停止。"

    ex_names = [ex.get('name_zh', '復健運動') for ex in selected_exercises]
    regions = set()
    max_bw = 0.0
    
    for ex in selected_exercises:
        reg = ex.get('primary_region', 'Unknown')
        if reg != "Unknown":
            region_map = {"Upper": "上肢", "Lower": "下肢", "Core": "核心與背部"}
            regions.add(region_map.get(reg, "全身"))
        bw_high = float(ex.get('impact_bw_high', 0))
        if bw_high > max_bw: max_bw = bw_high

    region_str = "及".join(regions) if regions else "全身部位"
    max_bw_str = f"{max_bw:.1f}" if max_bw > 0 else "1.2"
    nyha = user_condition.get("nyha", "未知")
    risk = risk_assessment.get("risk_level", "中等")

    # 判斷推薦的影片是否有分左右側示範 (資料預處理時，已經將檔案名稱包含左或右)
    has_lateral = any("左" in ex.get('name_zh', '') or "右" in ex.get('name_zh', '') for ex in selected_exercises)
    lateral_note = "『部分動作分為左右側示範，請務必成對完成，以保持身體兩側的平衡與穩定』" if has_lateral else ""

    prompt = f"""你是一位「專業復健治療師」，正在為心臟衰竭病患解說「今日運動示範影片」的整合建議。
【核心目標】：將運動細節與部位說明，轉化為「安全、專業、好懂」的執行指引。

【重要呈現內容】：
1. **明確指出訓練部位**：請務必在第一段開頭明確說明：『今日訓練重點為：{region_str}（包含：{', '.join(ex_names)}）』。
2. **左右對稱提醒**：{lateral_note}。

【寫作風格與邏輯】：
1. **數據因果化**：解釋數值（如 {max_bw_str} 倍體重）對關節的壓力意義，建議保持穩定。
2. **劑量指引**：明確說明建議從 1-2 組、每組 8-10 下開始（左右兩側各算一下），並解釋這是為了讓心臟逐步適應。
3. **安全溫柔化**：將警示自然融入（如：微微喘氣即可），若感胸悶頭暈須立即停止。

【嚴格規則】：
- **嚴禁問候語**：不可出現「午安」、「你好」、「辛苦了」。請直接從「今日訓練重點為...」開始。
- **隱藏技術痕跡**：絕對禁止提及 AI、YOLO、分數、演算法。
- **數據完整性**：必須包含具體組數（1-2組）、次數（8-10下）及關節負重（{max_bw_str}倍）的說明。
- **品質預測**：依病患年齡與 NYHA {nyha} 級，產出 40 字內疲勞預警（以「【品質預測】：」開頭）。
"""

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            timeout=15
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return "【系統提示】課表已產生，請依建議強度進行，若有不適請停止。"


def calculate_dynamic_reps(user_condition: dict, current_vitals: dict | None) -> int:
    """
    依 NYHA、今日生理數據與當下症狀計算建議次數，符合 ACSM 安全邊界。
    下限 6 次以維持肌肉幫浦作用。
    """
    nyha = str(user_condition.get("nyha", "II")).strip()
    # 基礎次數（依 NYHA）
    if "IV" in nyha:
        reps = 6
    elif "III" in nyha:
        reps = 8
    elif "II" in nyha:
        reps = 10
    else:
        reps = 12

    if not current_vitals:
        return max(6, reps)

    # 生理數據微調：收縮壓 > 140 或心跳 > 100 視為負擔較重，次數減 2
    sys_bp = int(current_vitals.get("sysBP") or 0)
    heart_rate = int(current_vitals.get("heartRate") or 0)
    if sys_bp > 140 or heart_rate > 100:
        reps -= 2

    # 症狀微調：當下有輕微喘，次數再減 2，維持 RPE 安全區間
    symptom = (current_vitals.get("currentSymptom") or user_condition.get("currentSymptom") or "none")
    if symptom == "breathless":
        reps -= 2

    # 慢性病史加權：用保守扣分，上限 -2，避免過度懲罰
    chronic = current_vitals.get("chronic") or user_condition.get("chronic") or []
    if not isinstance(chronic, list):
        chronic = []
    risk_points = 0
    if any("高血壓" in str(x) for x in chronic): risk_points += 1
    if any("糖尿病" in str(x) for x in chronic): risk_points += 1
    if any(("冠心症" in str(x)) or ("心肌梗塞" in str(x)) or ("冠狀動脈" in str(x)) for x in chronic): risk_points += 1
    if any("心律不整" in str(x) for x in chronic): risk_points += 1
    if any("慢性腎臟病" in str(x) for x in chronic): risk_points += 1

    # 0-1 項不扣，2-3 項扣 1，>=4 項扣 2
    if risk_points >= 4:
        reps -= 2
    elif risk_points >= 2:
        reps -= 1

    return max(6, reps)


def calculate_dynamic_rest_seconds(user_condition: dict, current_vitals: dict | None) -> int:
    """
    Layer 2：在安全門檻內，依 NYHA / 當日負擔 / 慢性病史微調組間休息秒數。
    """
    nyha = str(user_condition.get("nyha", "II")).strip()
    rest = 60
    if "IV" in nyha:
        rest = 120
    elif "III" in nyha:
        rest = 90

    if not current_vitals:
        return rest

    sys_bp = int(current_vitals.get("sysBP") or 0)
    heart_rate = int(current_vitals.get("heartRate") or 0)
    symptom = current_vitals.get("currentSymptom") or "none"

    if sys_bp > 140 or heart_rate > 100:
        rest += 30
    if symptom == "breathless":
        rest += 30

    chronic = current_vitals.get("chronic") or []
    if not isinstance(chronic, list):
        chronic = []
    chronic_points = 0
    if any("高血壓" in str(x) for x in chronic): chronic_points += 1
    if any("糖尿病" in str(x) for x in chronic): chronic_points += 1
    if any(("冠心症" in str(x)) or ("心肌梗塞" in str(x)) or ("冠狀動脈" in str(x)) for x in chronic): chronic_points += 1
    if any("心律不整" in str(x) for x in chronic): chronic_points += 1
    if any("慢性腎臟病" in str(x) for x in chronic): chronic_points += 1
    if chronic_points >= 2:
        rest += 30

    return min(180, max(60, rest))


def generate_7_day_plan(
    user_condition: dict,
    risk_assessment: dict,
    selected_exercises: list,
    current_vitals: dict | None = None,
) -> dict:
    """生成專業 7 天運動排程，回傳 plan_text（顯示用）與 daily_tasks 由 api_server 組裝"""
    default_plan = ["運動 2 組", "運動 2 組", "休息日", "運動 2 組", "休息日", "散步 15 分", "伸展"]
    if not openai_client:
        return {"plan_text": default_plan, "daily_tasks": None, "suggested_reps": 8}

    nyha = str(user_condition.get("nyha", "II")).strip()

    # 動態次數：納入今日生理數據與症狀的加權判定
    suggested_reps = calculate_dynamic_reps(user_condition, current_vitals)
    suggested_rest = calculate_dynamic_rest_seconds(user_condition, current_vitals)
    reps_str = f"{suggested_reps}-{suggested_reps + 2} 次"
    pressure_label = "高" if suggested_reps < 8 else "正常"

    # 動作檔案映射表：供 GPT 僅使用以下路徑產出 daily_tasks
    tasks_metadata = []
    for ex in selected_exercises:
        f_name = ex.get("video_filename")
        if not f_name:
            continue
        tasks_metadata.append({
            "name": ex.get("name_zh", "復健動作"),
            "file": f"exercise_videos/{f_name}",
        })

    # 取得動作部位與最大負荷
    regions = set()
    max_bw = 0.0
    for ex in selected_exercises:
        reg = ex.get('primary_region', 'Unknown')
        if reg != "Unknown":
            region_map = {"Upper": "上肢", "Lower": "下肢", "Core": "核心與背部"}
            regions.add(region_map.get(reg, "全身"))
        bw_high = float(ex.get('impact_bw_high', 0))
        if bw_high > max_bw:
            max_bw = bw_high
    region_str = "及".join(regions) if regions else "全身部位"
    max_bw_str = f"{max_bw:.1f}" if max_bw > 0 else "1.2"
    base_names = [ex.get("name_zh", "復健動作") for ex in selected_exercises]
    allowed_tasks_json = json.dumps(tasks_metadata, ensure_ascii=False)

    prompt = f"""你是一位「專業復健專家」。請為 NYHA {nyha} 級病患規劃 7 天詳細菜單。

    【病患當前狀態評估】
    - NYHA 分級：{nyha}
    - 當前生理壓力：{pressure_label}（已依今日血壓/心跳/症狀微調建議次數）

    【重要指令】
    1. 針對該病患「今日」身體反應，運動日建議次數鎖定為「{reps_str}」。
    2. 須在第一天（或摘要）中簡短說明原因，例如：考量今日血壓偏高／有輕微喘促，已將每組次數設為 {suggested_reps} 次，以確保運動安全性。

    核心動作：{', '.join(base_names)}。本週訓練部位：{region_str}。建議負荷約 {max_bw_str} 倍體重。

    【輸出規範】
    1. plan_text：7 個字串，每一天需包含「動作名稱＋1組＋{reps_str}」及 RPE 提醒。休息日寫清楚內容（如：休息日：主動式散步 10 分鐘，監測體重與水腫）。
    2. daily_tasks：長度為 7 的陣列，對應週一至週日。每個元素為「當日動作物件陣列」。僅可使用以下動作物件（file 路徑不可改）：
    {allowed_tasks_json}
    動作物件格式：{{"file": "上表內的 file 字串", "reps": {suggested_reps}, "name": "上表內的 name 字串"}}。休息日為空陣列 []。運動日可從上表選 1～4 個動作放入。

    【輸出 JSON 格式】
    {{"plan": ["第1天內容", "第2天內容", ...], "daily_tasks": [[動作物件,...], [], ...]}}
    嚴禁問候語。plan 與 daily_tasks 皆須 7 個元素。
    """

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_object"},
            timeout=25
        )
        data = json.loads(resp.choices[0].message.content)
        plan = data.get("plan", default_plan)
        if len(plan) > 7:
            plan = plan[:7]
        raw_daily = data.get("daily_tasks")
        daily_tasks = None
        if isinstance(raw_daily, list) and len(raw_daily) == 7:
            daily_tasks = []
            for day_list in raw_daily:
                if not isinstance(day_list, list):
                    daily_tasks = None
                    break
                day_tasks = []
                for t in day_list:
                    if isinstance(t, dict) and t.get("file"):
                        day_tasks.append({
                            "file": str(t.get("file", "")),
                            "reps": int(t.get("reps", suggested_reps)),
                            "name": str(t.get("name", "")),
                        })
                daily_tasks.append(day_tasks)
        if daily_tasks is None:
            daily_tasks = None
        return {"plan_text": plan, "daily_tasks": daily_tasks, "suggested_reps": suggested_reps, "suggested_rest": suggested_rest}
    except Exception as e:
        print(f"[ERROR] 7天計畫生成失敗: {e}")
        return {"plan_text": default_plan, "daily_tasks": None, "suggested_reps": suggested_reps, "suggested_rest": suggested_rest}


def generate_rpe_instruction(user_condition: dict, selected_exercises: list) -> str:
    """根據 NYHA 與負荷給出白話 RPE 建議"""
    nyha = str(user_condition.get("nyha", "II")).strip()
    max_bw = max([float(ex.get('impact_bw_high', 1.2)) for ex in selected_exercises])

    if "III" in nyha:
        rpe_range = "[RPE: 11]"
        desc = "體感應該是「很輕鬆」的。您可以一邊做一邊輕鬆唱歌，呼吸完全不費力。"
    elif max_bw > 1.4:
        rpe_range = "[RPE: 11-12]"
        desc = "感覺「稍微有一點點動到」的費力。您可以順暢地講完一句話，不需要停下來換氣。"
    else:
        rpe_range = "[RPE: 11-13]"
        desc = "感覺「微喘但舒服」。說話時會稍微斷斷續續，但仍能維持交談。"
    
    # 最終輸出的白話化包裝
    return f"ACSM 強度建議：根據您的狀況與動作負荷（約 {max_bw:.1f} 倍體重），建議強度為 {rpe_range}。也就是說，{desc} 如果感到胸悶或頭暈，請先停下來休息。"