# rag/user_condition_mapper.py

"""
功能說明：
1. 將 user_input.py 產生的中文結構資料轉換為系統內部條件
2. 執行運動前風險篩選（risk pre-check）
3. 回傳可供 RAG / Rule Controller 使用的使用者狀態 context

設計原則：
- 安全優先（Risk Gate 先於 RAG）
- 不產生建議、不碰 GPT
- 所有邏輯可解釋、可對應 ACSM
- ★ 系統內條件一律使用「正規化英文 enum」
"""

from typing import Dict, Any


# =========================
# 工具函式
# =========================
def _map(value, mapping):
    if value is None:
        return None
    return mapping.get(value)


def normalize_nyha(nyha_raw) -> str:
    """
    Normalize NYHA class into canonical form: I / II / III / IV

    可接受：
    - int: 1–4
    - str: "II", "II 級（一般活動輕微症狀）", "NYHA III", "3", "三"
    """
    if nyha_raw is None:
        return None

    s = str(nyha_raw).strip().upper()

    # 移除括號內說明（中英文）
    for sep in ["（", "(", "[", "{"]:
        if sep in s:
            s = s.split(sep, 1)[0]

    # 移除常見字樣
    s = s.replace("NYHA", "").replace("級", "").replace("期", "").replace(" ", "")

    mapping = {
        "1": "I", "I": "I", "一": "I",
        "2": "II", "II": "II", "二": "II",
        "3": "III", "III": "III", "三": "III",
        "4": "IV", "IV": "IV", "四": "IV",
    }

    return mapping.get(s)


# =========================
# 使用者條件轉換
# =========================
def map_user_conditions(user_input: Dict[str, Any]) -> Dict[str, Any]:
    conditions = {}

    # ---------- 基本資料 ----------
    basic = user_input.get("基本資料", {})

    # ★ 目前系統僅支援 HF，與 hf_chunks.json 完全對齊
    conditions["population"] = "Heart Failure"

    conditions["age_group"] = _map(
        basic.get("年齡層"),
        {
            "18–39歲": "18_39",
            "40–54歲": "40_54",
            "55–64歲": "55_64",
            "65–74歲": "65_74",
            "75歲以上": "75_plus"
        }
    )

    conditions["sex"] = _map(
        basic.get("性別"),
        {"男性": "male", "女性": "female"}
    )

    # ---------- 心臟衰竭狀態 ----------
    hf = user_input.get("心臟衰竭狀態", {})

    conditions["nyha"] = normalize_nyha(
        hf.get("NYHA 心臟功能分級")
    )

    conditions["hf_stable"] = _map(
        hf.get("目前是否穩定"),
        {"穩定": True, "不穩定": False}
    )

    conditions["hf_type"] = _map(
        hf.get("心臟衰竭類型"),
        {
            "射出分率降低型": "HFrEF",
            "射出分率保留型": "HFpEF"
        }
    )

    conditions["has_lvad"] = hf.get("是否使用心室輔助器（LVAD）") == "是"

    # ---------- 疾病史 ----------
    disease = user_input.get("疾病史", {})
    conditions["hypertension"] = disease.get("是否有高血壓") == "是"
    conditions["diabetes"] = disease.get("是否有糖尿病") == "是"
    conditions["cad"] = disease.get("是否有冠狀動脈疾病") == "是"
    conditions["arrhythmia"] = disease.get("是否有心律不整") == "是"
    conditions["lung_disease"] = disease.get("是否有肺部疾病") == "是"

    # ---------- 開刀史 ----------
    surgery = user_input.get("開刀史", {})
    conditions["cardiac_surgery"] = surgery.get("是否曾接受心臟支架或繞道手術") == "是"
    conditions["valve_surgery"] = surgery.get("是否曾接受心臟瓣膜手術") == "是"
    conditions["joint_replacement"] = surgery.get("是否曾接受人工關節置換") == "是"
    conditions["recent_surgery"] = surgery.get("是否有近期大型手術史") == "是"

    # ---------- 用藥史 ----------
    meds = user_input.get("用藥史", {})
    conditions["ccb"] = meds.get("是否使用鈣離子通道阻斷劑") == "是"
    conditions["anti_arrhythmic"] = meds.get("是否使用抗心律不整藥物") == "是"
    conditions["exercise_affecting_drugs"] = meds.get("是否使用影響運動耐受性的藥物") == "是"

    # ---------- 症狀 ----------
    symptom = user_input.get("個人自覺症狀評估", {})

    conditions["dyspnea"] = _map(
        symptom.get("呼吸困難程度"),
        {"無": "none", "輕度": "mild", "中度": "moderate", "重度": "severe"}
    )

    conditions["chest_pain"] = symptom.get("是否出現胸痛") == "是"
    conditions["dizziness"] = symptom.get("是否出現頭暈或接近昏厥") == "是"

    conditions["fatigue"] = _map(
        symptom.get("疲勞程度"),
        {"無": "none", "輕度": "mild", "中等": "moderate", "嚴重": "severe"}
    )

    # ---------- 運動目的 ----------
    exercise = user_input.get("運動情境", {})
    conditions["exercise_goal"] = _map(
        exercise.get("運動目的"),
        {"健康促進": "health", "復健訓練": "rehab", "日常活動": "daily"}
    )

    # ---------- 禁忌條件（給 recommender Hard Filter 用，與 exercise_library / STANDARD 對齊）----------
    contraindications: list = []
    if conditions.get("dizziness"):
        contraindications.append("balance_disorder")
    if conditions.get("joint_replacement"):
        contraindications.append("hip_pain")
    if conditions.get("hypertension"):
        contraindications.append("uncontrolled_hypertension")
    conditions["contraindications"] = contraindications

    return conditions


# =========================
# 運動前風險篩選（Risk Gate）
# =========================
def risk_precheck(conditions: Dict[str, Any]) -> Dict[str, Any]:
    """
    根據 ACSM HF 原則進行運動前安全判斷
    ★ 只使用正規化後的 nyha（I / II / III / IV）
    """

    nyha = conditions.get("nyha")

    # --- 防呆：非 HF（目前僅保留安全出口） ---
    if conditions.get("population") != "Heart Failure":
        return {
            "allow_exercise": True,
            "risk_level": "low",
            "note": "非心臟衰竭族群，套用一般 ACSM 運動安全原則",
            "nyha": None
        }

    # --- HF 不穩定：直接阻擋 ---
    if conditions.get("hf_stable") is False:
        return {
            "allow_exercise": False,
            "risk_level": "high",
            "reason": "心臟衰竭狀態不穩定",
            "nyha": nyha
        }

    # --- NYHA 分級 ---
    if nyha == "IV":
        return {
            "allow_exercise": False,
            "risk_level": "high",
            "reason": "NYHA IV 級不建議進行運動",
            "nyha": nyha
        }

    if nyha == "III":
        return {
            "allow_exercise": True,
            "risk_level": "moderate",
            "note": "NYHA III 級，建議低強度、密切監測",
            "nyha": nyha
        }

    if nyha == "II":
        return {
            "allow_exercise": True,
            "risk_level": "low",
            "note": "NYHA II 級，屬穩定型心臟衰竭",
            "nyha": nyha
        }

    # --- 危險症狀 ---
    if conditions.get("chest_pain") or conditions.get("dizziness"):
        return {
            "allow_exercise": False,
            "risk_level": "high",
            "reason": "出現胸痛或接近昏厥等高風險症狀",
            "nyha": nyha
        }

    if conditions.get("dyspnea") == "severe":
        return {
            "allow_exercise": False,
            "risk_level": "high",
            "reason": "嚴重呼吸困難",
            "nyha": nyha
        }

    # --- 預設：低風險 ---
    return {
        "allow_exercise": True,
        "risk_level": "low",
        "nyha": nyha
    }


# =========================
# 對外統一接口
# =========================
def build_user_context(user_input: Dict[str, Any]) -> Dict[str, Any]:
    conditions = map_user_conditions(user_input)
    risk = risk_precheck(conditions)

    # 防呆：確保 nyha 一定存在
    if risk is not None:
        risk.setdefault("nyha", conditions.get("nyha"))

    return {
        "user_conditions": conditions,
        "risk_assessment": risk
    }
