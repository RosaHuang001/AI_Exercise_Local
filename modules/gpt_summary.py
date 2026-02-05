

# gpt_summary.py
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
    """
    固定版：關節衝擊力段落（民眾看得懂 + 指向主要肢體/關節 + 不讓 GPT 自由發揮）
    白話規則：
    - 解釋「倍體重」= 大約等於自己體重幾倍的壓力
    - 解釋「體重越重 → 換算出的力量數字越大」用於對照理解
    - 不講公式、不解釋牛頓定義、不做計算步驟
    - 一定要講清楚：這是針對「主要活動肢體／主要參考關節」的保守估計，不是全身每個關節
    """
    impact_level = stats.get("impact_level", "未知")
    bw_low = float(stats.get("impact_bw_low", 0) or 0)
    bw_high = float(stats.get("impact_bw_high", 0) or 0)
    bins_text = stats.get("impact_by_weight_bins_text", "")

    # 取主要肢體/關節資訊
    primary_region = str(stats.get("primary_region", "Unknown") or "Unknown")
    primary_joint = str(stats.get("primary_joint", "Unknown") or "Unknown")
    primary_side = str(stats.get("primary_side", "Unknown") or "Unknown")

    # 沒資料就回空字串（避免塞出怪句）
    if bw_low <= 0 or bw_high <= 0:
        return ""

    # 區域白話（跟你上面 region_hint 的邏輯一致，但這裡自己做一份避免依賴外部變數）
    if primary_region == "Upper":
        region_zh = "上肢"
    elif primary_region == "Lower":
        region_zh = "下肢"
    elif primary_region == "Core":
        region_zh = "核心／軀幹"
    else:
        region_zh = "主要動作"

    # 主側白話
    side_zh = ""
    if "Right" in primary_side or primary_side in ("R", "Right"):
        side_zh = "右側"
    elif "Left" in primary_side or primary_side in ("L", "Left"):
        side_zh = "左側"

    # 關節白話（把英文關節名翻成民眾好懂的詞，翻不到就原樣）
    joint_map = {
        "Hip": "髖關節",
        "Knee": "膝關節",
        "Ankle": "踝關節",
        "Elbow": "手肘",
        "Shoulder": "肩關節",
        "Wrist": "手腕",
        "Head": "頭頸"
    }
    joint_zh = primary_joint
    for k, v in joint_map.items():
        if k in primary_joint:
            joint_zh = v
            break

    # 把「衝擊力」明確綁到動作的    「主要肢體/關節」
    target = f"{region_zh}{side_zh}{joint_zh}".replace("主要動作", "").strip()
    if not target:
        target = f"{side_zh}{joint_zh}".strip() or "主要參考關節"

    return (
        f"在這支影片中，主要活動集中在「{target}」，"
        "以下的關節衝擊力估計是針對這個主要參考關節在動作過程中的負荷做保守推估，並不是代表全身每個關節都一樣。"
        f"本次估計屬於{impact_level}等負荷；以相對負荷來看，約為 {bw_low:.1f}–{bw_high:.1f} 倍體重的壓力，"
        "也就是『大約等於自己體重幾倍的壓力』。"
        f"若依不同體重區間換算成力量大小，約相當於 {bins_text}；"
        "體重越重，換算出來的力量數字通常會越大，這只是讓不同體重的人方便對照理解。"
        "（以上為系統依影片動作特性所做的保守估計，僅供參考）"
    )






# ==================== GPT 文案（Demo 用後處理說明） ====================

def call_openai_label(
    file_name: str,
    duration_s: float,
    stats: dict,
    activity_level: str,
    user_condition: dict,
    risk_assessment: dict
) -> dict:
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

    impact_level = stats.get("impact_level", "未知")
    impact_bw_low = float(stats.get("impact_bw_low", 0) or 0)
    impact_bw_high = float(stats.get("impact_bw_high", 0) or 0)
    impact_bins_text = stats.get("impact_by_weight_bins_text", "")


    if reps == 0 and rom < 12:
        motion_type_hint = "此動作較像『維持姿勢』或小幅度調整，次數接近 0 屬正常現象。"
    else:
        motion_type_hint = "此動作屬於『重複進行』的運動，可用次數與頻率描述。"

    if primary_region == "Upper":
        region_hint = "上肢（手臂、肩膀、手肘相關）"
    elif primary_region == "Lower":
        region_hint = "下肢（髖、膝、踝與站穩能力）"
    elif primary_region == "Core":
        region_hint = "核心（軀幹穩定、支撐類動作）"
    else:
        region_hint = "全身或混合動作"



    # ==================== STEP 2 : RAG 規則檢索 ====================

    user_profile = {
        "risk_level": risk_assessment.get("risk_level", "moderate"),
        "nyha": user_condition.get("nyha"),             # I / II / III / IV（與 rule_controller 一致）
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
    topics=[
        "Exercise Intensity",
        "Joint Impact",
        "Safety",
        "FITT",
        "Individualization"
    ]
    )



    hf_rules = RULE_CONTROLLER.process(
        rules=hf_rules_raw,
        user_profile=user_profile
    )



    rag_text = "\n".join(
    f"- {r.get('rule', '')} ({r.get('source', r.get('id', 'ACSM guideline'))})"
    for r in hf_rules
    )  if hf_rules else ""




    if hf_rules:
        print(f"[RAG] Retrieved {len(hf_rules)} ACSM rules")

    else:
        print(f"[INFO] No ACSM rules found")





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
10) 必須在摘要中加入 1–2 句「關節衝擊力（依體重區間換算）」的白話說明，衝擊力描述必須明確指出是針對「主要動作區域 + 主要參考關節（含主側）」的估計，不得寫成全身關節皆相同。不可自行捏造或重算數值。
    白話規則：
    - 必須用一句話解釋「倍體重」的意思（例如：1.5 倍體重 = 大約等於身體重量的 1.5 倍壓力）。
    - 必須用一句話解釋「為什麼不同體重會有不同 N 值」（因為體重越重，換算成力就越大）。
    - 不要提「牛頓」的物理定義與公式，不要講重力加速度，不要做任何計算步驟。
    - 不要用專業術語；若提到 N，只要說「換算成力量大小」即可。
"""



    if rag_text:
        system_prompt += f"""

【ACSM 指引補充（RAG 檢索，僅限參考）】
以下內容已依據本次運動情境，自 ACSM 指引中自動篩選。
你在撰寫說明時，只能在下列規則範圍內進行白話轉譯，
不得新增、推翻或延伸未提及的運動建議：

{rag_text}
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

【關節衝擊力（依體重區間換算，參考用）】
- 衝擊等級（低/中/高）：{impact_level}
- 相對負荷（倍體重 BW）：{impact_bw_low:.1f}–{impact_bw_high:.1f} × 體重
- 各體重區間換算（牛頓 N）：{impact_bins_text}
（以上為系統依影片負荷特性之保守估計，請直接引用，不可推翻、不可自行重算。）

【白話表達要求（必須照做）】
- 你在摘要中提到「倍體重 BW」時，必須加上白話翻譯：用「大約等於自己體重的幾倍壓力」來說明。
- 你在摘要中提到「依體重區間換算」時，必須補一句：體重越重，換算出來的力量數字會越大，這只是幫不同體重的人對照理解。
- 不要解釋牛頓(N)是什麼單位；可以說「換算成力量大小」或「換算成大約的受力範圍」。
- 不要新增任何沒有在這裡提供的數字、也不要自己重新計算或改寫區間。


【動作型態提示】
- {motion_type_hint}


【使用者個人狀況（請務必納入解讀）】
- 族群：{user_condition["population"]}
- 心臟功能分級（NYHA）：{user_condition.get("nyha")}
- 疾病穩定狀態：{user_condition.get("hf_stable")}
- 是否有心臟手術史：{user_condition.get("cardiac_surgery")}
- 系統整體風險評估：{risk_assessment["risk_level"]}

說明要求：
- 請在摘要中明確說明「對這位使用者而言」
- 評估目前影片的動作頻率與次數，是否對該使用者來說偏快、適中或需保守調整
- 若需調整，請以「可考慮從…開始」、「可適度降低…」描述
- 不可新增任何數值，只能調整『建議做法的保守程度』



請輸出以下 JSON（只輸出 JSON）：

{{
  "gpt_summary": "必填。請以『專業復健人員向病患介紹一支運動示範影片』的方式，撰寫一段病患友善的白話摘要（約 5–7 句）。請說明影片示範的是什麼運動、主要會用到哪個身體部位與關節，並將動作的次數、節奏與幅度轉譯成病患能理解的描述（避免只列數字）。請**明確標示這是一支低 / 中 / 高強度的運動示範影片**，並依 ACSM FITT 原則，以保守、安全的方式，提供病患照著影片進行時可參考的組數、次數或時間建議。最後加入 1–2 句安全提醒，摘要中關於「關節衝擊力」必須用民眾聽得懂的說法解釋「倍體重」與「體重區間換算」的意義（不講公式、不做計算）。幫助病患知道什麼情況下應放慢或停止。請勿使用列點。你在說明時，必須同時站在「這支影片的動作特性」與「該使用者的身體與風險狀況」兩個角度進行解讀。
",
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

        content = response.choices[0].message.content
        content = content if isinstance(content, str) else ""
        data = json.loads(content.strip() or "{}")

        
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


# ==================== 一週 7 日個人化運動計畫 ====================

def generate_weekly_plan(
    user_condition: dict,
    risk_assessment: dict,
    results: list,
) -> dict:
    """
    依使用者條件與各影片分析結果，產出「使用者個人化一週 7 日運動計畫」。
    回傳：{"plan_text": 完整計畫文字, "plan_intro": 簡短說明（可選）, "days": {1..7: 當日內容}}
    """
    fallback = {
        "plan_text": "（本週計畫需依個人狀況調整，建議與醫療人員討論後執行。本次因故無法自動產出，請稍後再試或手動安排。）",
        "plan_intro": "",
        "days": {},
    }

    if openai_client is None:
        return fallback

    # 使用者狀況摘要
    user_desc = (
        f"族群：{user_condition.get('population', '—')}；"
        f"心臟功能分級（NYHA）：{user_condition.get('nyha', '—')}；"
        f"風險等級：{risk_assessment.get('risk_level', '—')}；"
        f"是否建議運動：{risk_assessment.get('allow_exercise', True)}。"
    )
    if risk_assessment.get("note"):
        user_desc += f" 備註：{risk_assessment['note']}"

    # 各影片分析摘要（供 GPT 判斷適合度與安排）
    exercises_desc = []
    for i, res in enumerate(results, 1):
        video_name = res.get("video", "")
        name_zh = os.path.splitext(os.path.basename(video_name))[0] if video_name else f"影片{i}"
        decision = res.get("decision", "RECOMMEND")
        gpt = res.get("gpt_summary", {}) if isinstance(res.get("gpt_summary"), dict) else {}
        summary = gpt.get("gpt_summary", "") if isinstance(gpt, dict) else ""
        risk_notice = gpt.get("gpt_risk_notice", "") if isinstance(gpt, dict) else ""
        safety_tip = gpt.get("gpt_safety_tip", "") if isinstance(gpt, dict) else ""
        yolo = res.get("yolo_result", {})
        intensity = yolo.get("activity_level") or (yolo.get("impact", {}) or {}).get("level") or "—"
        exercises_desc.append(
            f"【{name_zh}】系統建議：{decision}；強度：{intensity}。"
            f"摘要：{summary[:200]}{'…' if len(str(summary)) > 200 else ''}。"
            f"風險提醒：{risk_notice}。安全建議：{safety_tip}。"
        )

    exercises_block = "\n\n".join(exercises_desc)

    system_prompt = """你是一位復健／運動指導專業人員，正在為一位使用者制定「個人化一週 7 日運動計畫」。
你的任務是：
1. 依據「使用者狀況」與「各支影片的分析結果與建議」，決定一週內每天適合做哪些動作、做多久、幾組，以及哪天休息。
2. 計畫必須符合該使用者的風險等級與心臟功能分級，保守、可執行，且明確寫出「週一」到「週日」的內容。
3. 只使用上述分析過的影片／動作，不要發明未提供的運動。
4. 若同一動作有「左側」「右側」兩支影片（檔名會標示左側、右側），請在計畫中寫明「該動作左、右各一組」或「先做左側一組，再做右側一組」等說法，讓使用者知道要雙側都做。
5. 輸出格式：先一段簡短「本週計畫說明」（2–3 句），接著依序「週一：…」「週二：…」…「週日：…」，每 day 內寫清楚：做哪個動作、建議時間或組數、注意事項（若有）。
6. 語氣友善、具體、可操作，避免專業術語。不要輸出 JSON，只輸出純文字計畫。"""

    user_prompt = f"""
【使用者狀況】
{user_desc}

【已分析之運動影片與建議】
{exercises_block}

請根據以上資訊，產出「使用者個人化一週 7 日運動計畫」（週一至週日，含休息日與每日建議動作／時間／注意事項）。只輸出計畫內容，不要前言或結尾多餘說明。
"""

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
        if not content:
            return fallback
        return {
            "plan_text": content,
            "plan_intro": "",
            "days": {},
        }
    except Exception as e:
        print(f"[WARN] 一週計畫產生失敗：{e}")
        return fallback
