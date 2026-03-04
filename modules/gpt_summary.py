# modules/gpt_summary.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from rag.rag_engine import ACSMRagEngine
from rag.rule_controller import RuleController


# ==================== OpenAI 設定 ====================

load_dotenv()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

OPENAI_MODEL = "gpt-4o"


# ==================== RAG 設定 ====================

RAG_ENGINE = ACSMRagEngine(
    knowledge_path="knowledge_base/hf_chunks.json"
)

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
    """【精煉版】線下批次處理用的關節衝擊力段落"""
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
    elif primary_region == "Core": region_zh = "核心／軀幹"
    else: region_zh = "主要動作"

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

    target = f"{region_zh}{side_zh}{joint_zh}".replace("主要動作", "").strip()
    if not target: target = f"{side_zh}{joint_zh}".strip() or "主要參考關節"

    # 簡潔有力的說明，不再冗長解釋物理名詞
    return (
        f"本動作主要鍛鍊「{target}」，屬於{impact_level}衝擊負荷。 "
        f"運動時該關節大約會承受 {bw_low:.1f}–{bw_high:.1f} 倍體重的壓力（依體重換算約為 {bins_text}）。 "
        "此數值為系統基於安全考量的保守推估，供您參考。"
    )


# =========================================================================
# ==================== 線下 Offline 專用：批次生成模組 ====================
# =========================================================================

def call_openai_label(file_name: str, duration_s: float, stats: dict, activity_level: str, user_condition: dict, risk_assessment: dict) -> dict:
    """【線下批次建置用】產出單支影片的 FITT 詳細摘要，用於擴充知識庫"""
    fallback = {
        "gpt_summary": "本影片呈現受試者進行重複性或維持姿勢的基礎動作。建議量力而為並保持身體穩定，若出現不適請停止。",
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

    impact_level = stats.get("impact_level", "未知")
    impact_bw_low = float(stats.get("impact_bw_low", 0) or 0)
    impact_bw_high = float(stats.get("impact_bw_high", 0) or 0)
    impact_bins_text = stats.get("impact_by_weight_bins_text", "")

    if reps == 0 and rom < 12:
        motion_type_hint = "此動作較像『維持姿勢』或小幅度調整，次數接近 0 屬正常現象。"
    else:
        motion_type_hint = "此動作屬於『重複進行』的運動，可用次數與頻率描述。"

    if primary_region == "Upper": region_hint = "上肢（手臂、肩膀、手肘相關）"
    elif primary_region == "Lower": region_hint = "下肢（髖、膝、踝與站穩能力）"
    elif primary_region == "Core": region_hint = "核心（軀幹穩定、支撐類動作）"
    else: region_hint = "全身或混合動作"

    user_profile = {
        "risk_level": risk_assessment.get("risk_level", "moderate"),
        "nyha": user_condition.get("nyha"),
        "is_stable": user_condition.get("hf_stable"),
        "posture": stats.get("posture"),
        "weight_bearing": stats.get("weight_bearing")
    }
    
    rag_condition = {
        "region": primary_region if primary_region != "Unknown" else None,
        "posture": posture if posture != "Unknown" else None,
        "weight_bearing": stats.get("weight_bearing") if "weight_bearing" in stats else None
    }

    hf_rules_raw = RAG_ENGINE.retrieve_rules(
        population="Heart Failure",
        condition=rag_condition,
        topics=["Exercise Intensity", "Joint Impact", "Safety", "FITT", "Individualization"]
    )

    hf_rules = RULE_CONTROLLER.process(rules=hf_rules_raw, user_profile=user_profile)
    rag_text = "\n".join(f"- {r.get('rule', '')} ({r.get('source', r.get('id', 'ACSM guideline'))})" for r in hf_rules) if hf_rules else ""

    system_prompt = """你是一位「復健專業人員」，正在向病患說明一支『運動示範影片』，協助病患了解這支影片適合怎樣的運動強度。
你的任務是：根據系統數值，用病患聽得懂的白話中文說明這支影片在示範什麼動作，動作大不大、快不快，並依據 ACSM FITT 原則，提供「建議的運動方式描述」（非醫療處方）。

嚴格規則：
1) 不假設病患已經做過這個動作。
2) 不使用「你剛剛完成了…」等回饋語氣。
3) 所有建議須以「可參考」、「建議從…開始」描述，不可使用命令語。
4) 不做醫療診斷，不給治療或處方。
5) 不提及 AI、模型、YOLO、演算法或分析流程。
6) 避免專業術語，語氣溫和、鼓勵、實際可執行。
7) 只輸出 JSON，不要任何多餘文字。
"""
    if rag_text:
        system_prompt += f"\n【ACSM 指引補充（僅限參考）】\n{rag_text}"

    user_prompt = f"""
【系統自動判定之活動強度】
- 本次運動強度等級：{activity_level}強度

【影片基本資訊】
- 檔名：{file_name}
- 影片長度：約 {duration_s:.1f} 秒
- 體位：{posture}
- 主要動作區域：{region_hint}

【主要運動學指標】
- ROM(p5–p95)：{rom:.1f} 度
- 次數：{reps} 下
- 頻率：{freq:.2f} 次/秒

【動作型態提示】
- {motion_type_hint}

【使用者個人狀況】
- 族群：{user_condition.get("population")}
- 心臟功能分級（NYHA）：{user_condition.get("nyha")}
- 系統整體風險評估：{risk_assessment.get("risk_level")}

請輸出以下 JSON（只輸出 JSON）：
{{
  "gpt_summary": "撰寫一段約 3-4 句的白話摘要，說明這是什麼動作、用到哪裡，並提供保守的起步建議。",
  "gpt_activity_level": "低 / 中 / 高（三選一）",
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
        if impact_para and ("衝擊" not in summary and "倍體重" not in summary and "體重" not in summary):
            summary = summary.rstrip() + " " + impact_para

        return {
            "gpt_summary": summary,
            "gpt_activity_level": normalize(data.get("gpt_activity_level"), fallback["gpt_activity_level"]),
            "gpt_risk_notice": data.get("gpt_risk_notice", fallback["gpt_risk_notice"]),
            "gpt_safety_tip": data.get("gpt_safety_tip", fallback["gpt_safety_tip"]),
            "gpt_long_text": data.get("gpt_long_text", fallback["gpt_long_text"]),
        }

    except Exception as e:
        print(f"[WARN] GPT 產生失敗，使用 fallback：{e}")
        return fallback


def generate_weekly_plan(user_condition: dict, risk_assessment: dict, results: list) -> dict:
    """【線下批次建置用】依據分析結果產出使用者個人化一週 7 日運動計畫 (長篇文字版)"""
    fallback = {
        "plan_text": "（本週計畫需依個人狀況調整，建議與醫療人員討論後執行。本次因故無法自動產出，請稍後再試或手動安排。）",
        "plan_intro": "",
        "days": {},
    }

    if openai_client is None:
        return fallback

    user_desc = (
        f"族群：{user_condition.get('population', '—')}；"
        f"心臟功能分級（NYHA）：{user_condition.get('nyha', '—')}；"
        f"風險等級：{risk_assessment.get('risk_level', '—')}；"
        f"是否建議運動：{risk_assessment.get('allow_exercise', True)}。"
    )
    if risk_assessment.get("note"):
        user_desc += f" 備註：{risk_assessment['note']}"

    exercises_desc = []
    for i, res in enumerate(results, 1):
        video_name = res.get("video", "")
        name_zh = os.path.splitext(os.path.basename(video_name))[0] if video_name else f"影片{i}"
        decision = res.get("decision", "RECOMMEND")
        gpt = res.get("gpt_summary", {}) if isinstance(res.get("gpt_summary"), dict) else {}
        summary = gpt.get("gpt_summary", "") if isinstance(gpt, dict) else ""
        yolo = res.get("yolo_result", {})
        intensity = yolo.get("activity_level") or (yolo.get("impact", {}) or {}).get("level") or "—"
        exercises_desc.append(f"【{name_zh}】系統建議：{decision}；強度：{intensity}。摘要：{summary[:200]}")

    exercises_block = "\n\n".join(exercises_desc)

    system_prompt = """你是一位復健／運動指導專業人員，正在為一位使用者制定「個人化一週 7 日運動計畫」。
你的任務是：依據使用者狀況與分析結果，決定每天適合做哪些動作、做多久、幾組，以及哪天休息。
輸出格式：先一段簡短「本週計畫說明」，接著依序「週一：…」「週二：…」…「週日：…」。
語氣友善、具體、可操作，避免專業術語。不要輸出 JSON，只輸出純文字計畫。"""

    user_prompt = f"【使用者狀況】\n{user_desc}\n\n【已分析之運動影片與建議】\n{exercises_block}"

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.4,
            timeout=45,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content: return fallback
        return {"plan_text": content, "plan_intro": "", "days": {}}
    except Exception as e:
        print(f"[WARN] 一週計畫產生失敗：{e}")
        return fallback


# =========================================================================
# ==================== 線上 Online Web API 專用：即時生成模組 ====================
# =========================================================================

def generate_today_summary(user_condition: dict, risk_assessment: dict, selected_exercises: list) -> str:
    """【網頁即時用】將選出的多個動作打包，產出具備深度、含具體組數建議的總說明"""
    if not openai_client:
        return "【系統提醒】本課表已依據您的狀況進行個人化調整，請量力而為。若有任何不適請立即停止運動。"

    # 1. 萃取這套課表的「綜合物理特徵」
    ex_names = []
    regions = set()
    max_bw = 0.0
    
    for ex in selected_exercises:
        ex_names.append(ex.get('name_zh', '復健運動'))
        
        reg = ex.get('primary_region', 'Unknown')
        if reg != "Unknown":
            region_map = {"Upper": "上肢", "Lower": "下肢", "Core": "核心與軀幹"}
            regions.add(region_map.get(reg, "全身"))
            
        bw_high = float(ex.get('impact_bw_high', 0))
        if bw_high > max_bw:
            max_bw = bw_high

    region_str = "與".join(regions) if regions else "全身各部位"
    max_bw_str = f"{max_bw:.1f}" if max_bw > 0 else "1.2"
    
    nyha = user_condition.get("nyha", "未知")
    risk = risk_assessment.get("risk_level", "中等")
    stable = "穩定" if user_condition.get("hf_stable") else "不穩定"

    # 2. 【核心修改】：精煉 Prompt，要求 GPT 給出具體的運動處方(組數/次數)
    prompt = f"""你是一位「專業復健人員」，正在向病患說明今日為他量身打造的『連續跟練影片課表』。
這套課表的特色在於，它是基於電腦視覺算出的精準關節數據與病患的身體狀況所生成的。

【使用者個人狀況】
- 族群：心臟衰竭
- 心臟功能分級（NYHA）：{nyha} 級
- 疾病穩定狀態：{stable}
- 系統整體風險評估：{risk}

【課表內容與物理特徵】
- 包含動作串聯：{', '.join(ex_names)}
- 主要活動區域：{region_str}
- 系統估算最大關節相對負荷：約 {max_bw_str} 倍體重

【你的任務與嚴格規則（必須遵守）】
請用 4~5 句溫暖、白話的中文，向病患總結「這整套課表」的重點：
1. 必須明確告訴病患這套課表主要鍛鍊哪個身體部位（{region_str}）。
2. 【獨家要求 - 簡潔有力】：請自然地融入一句「關節負荷」的說明，例如：「這套動作大約會讓關節承受 {max_bw_str} 倍體重的壓力」。請勿做多餘的物理名詞解釋（不需要再解釋什麼是倍體重），保持句子短巧好讀。
3. 依據該病患的 NYHA 分級與風險，給予具體且保守的「起步運動量建議」（請明確說出類似「建議您可以先從 1 到 2 組，每組 8 到 10 下開始」或「建議每個動作進行 30 秒」的具體數字），並提醒保持微喘即可、若有頭暈胸悶請立即停止。
4. 語氣限制：不假設病患已經做過這些動作；不可使用命令語，請用「可參考」、「建議從…開始」；不做醫療診斷，不給治療處方。
5. 嚴禁提及 AI、模型、YOLO、演算法或分析流程。
6. 絕對不要列點，請直接輸出一段自然流暢、充滿同理心的口語說明。
7. **動作品質預測 (New)**：請根據使用者的年齡 {user_condition.get('age')}、血壓 {user_condition.get('sysBP')} mmHg、以及 NYHA {nyha} 分級，產出一句約 40 字內的「疲勞預警」。
    - **格式限制**：請將這句話放在整個總結的最末端，並用「【品質預測】：」作為開頭。
    - **邏輯參考**：若年齡 > 65 或 NYHA III，提醒肌肉耐力可能較早衰退；若血壓偏高，提醒注意運動中的呼吸調節以維持動作穩定度。
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
        print(f"GPT 總說明生成失敗: {e}")
        return "【系統提醒】本課表已依據您的狀況進行個人化調整，請量力而為。若有任何不適請立即停止運動。"


def generate_7_day_plan(user_condition: dict, risk_assessment: dict, selected_exercises: list) -> list:
    """【網頁即時用】請 GPT 針對選出的動作，規劃 7 天的排程陣列 (供 UI 時間軸渲染)"""
    default_plan = [
        "建議進行所選動作 2-3 組", "建議進行所選動作 2-3 組", "休息日 (溫和伸展)", 
        "建議進行所選動作 2-3 組", "休息日 (腹式呼吸)", "室內散步 15 分鐘", "全身溫和伸展"
    ]
    
    if not openai_client: return default_plan

    ex_names = [ex.get('name_zh', '復健運動') for ex in selected_exercises]
    nyha = user_condition.get("nyha", "未知")
    risk = risk_assessment.get("risk_level", "中等")

    prompt = f"""你是一位復健指導員。病患狀況：心臟衰竭 NYHA {nyha} 分級，風險評估為 {risk}。
本週系統推薦的動作有：{', '.join(ex_names)}。

請為病患規劃一週 7 天的運動排程。
規定：
1. 必須輸出一個 JSON 格式的物件，包含 "plan" 陣列，陣列內剛好 7 個字串（代表週一到週日）。
2. 合理安排 2-3 天的「休息日」。(字串請包含"休息"兩字)
3. 每個字串長度控制在 15~25 個字內，白話好懂。
格式範例：{{"plan": ["平板支撐與深蹲各2組", "休息日 (搭配腹式呼吸)", ...]}}"""

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_object"},
            timeout=20
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("plan", default_plan)
    except Exception as e:
        print(f"GPT 7日計畫生成失敗: {e}")
        return default_plan





def generate_rpe_instruction(user_condition: dict, selected_exercises: list) -> str:
    """根據 NYHA 分級與 YOLO 物理負荷，動態生成 RPE 強度建議"""
    
    # 1. 取得 NYHA 分級 (例如: "I", "II", "III", "IV")
    nyha = str(user_condition.get("nyha", "II"))
    
    # 2. 找出推薦動作中的最大衝擊力 (YOLO 辨識數據)
    max_bw = 0.0
    for ex in selected_exercises:
        # 從 impact_bw_high 欄位直接獲取，避免解析 stats 陣列文字出錯
        bw_val = float(ex.get('impact_bw_high', 1.2)) 
        if bw_val > max_bw: max_bw = bw_val

    # 3. 判定邏輯：NYHA III 級強制保護邏輯
    if nyha == "III" or "III" in nyha:
        rpe_range = "[RPE:11] (輕鬆至輕微喘)"
        safety_note = "由於您的心臟功能分級為 Class III，建議採取最保守強度。"
    elif max_bw > 1.4:
        rpe_range = "[RPE:11]-12 (微喘但可輕鬆對話)"
        safety_note = "考量今日動作負荷較大，強度不宜過高。"
    else:
        rpe_range = "[RPE:11]-13 (微喘但可持續對話)"
        safety_note = "目前動作負荷適中，請維持在目標強度區間。"

    return f"ACSM 強烈建議：根據您的 NYHA {nyha} 分級與今日動作負荷（約 {max_bw:.1f} 倍體重），運動時自覺費力感應控制在 {rpe_range}。{safety_note}"