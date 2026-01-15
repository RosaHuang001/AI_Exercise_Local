# modules/user_profile.py

from typing import Dict, List

# =========================
# 中文顯示對應（UI → 系統）
# =========================

AGE_MAP = {
    "20 歲以下": "<20",
    "20–29 歲": "20–29",
    "30–39 歲": "30–39",
    "40–59 歲": "40–59",
    "60–74 歲": "60–74",
    "75–84 歲": "75–84",
    "85 歲以上": "≥85",
}


SEX_MAP = {
    "男性": "Male",
    "女性": "Female",
}

NYHA_MAP = {
    "第一級（日常活動不受限，如：走路、爬樓梯）": "I",
    "第二級（活動時可能出現疲倦、呼吸急促，但休息後可緩解）": "II",
    "第三級（輕微活動即感到不適，但休息時狀態尚可）": "III",
    "第四級（休息時也感到不適，無法進行任何活動）": "IV",
}

COMORBIDITY_MAP = {
    "高血壓": "Hypertension",
    "糖尿病": "Diabetes",
    "關節退化或疼痛": "JointProblem",
    "無／不確定": "NoneOrUnknown",
}

SURGERY_MAP = {
    "下肢手術（膝、髖、踝）": "LowerLimb",
    "上肢手術（肩、肘、腕）": "UpperLimb",
    "脊椎／核心相關": "SpineOrCore",
    "無／不確定": "NoneOrUnknown",
}

GOAL_MAP = {
    "增加肌力": "IncreaseStrength",
    "改善活動能力": "ImproveMobility",
    "維持現有功能": "MaintainFunction",
    "預防退化": "PreventDecline",
}

# =========================
# 建立使用者資料（匿名）
# =========================

def create_user_profile(
    subject_id: str, 
    age_label: str,
    sex_label: str,
    nyha_label: str,
    comorbidity_labels: List[str],
    surgery_labels: List[str],
    goal_label: str,
) -> Dict:
    """
    將『中文選項』轉成系統內部使用的標準化資料
    所有欄位皆為必選，若不符合定義將直接拋出錯誤
    """

    if age_label not in AGE_MAP:
        raise ValueError("年齡區間為必選欄位，且必須使用系統提供的選項")

    if sex_label not in SEX_MAP:
        raise ValueError("性別為必選欄位")

    if nyha_label not in NYHA_MAP:
        raise ValueError("心臟功能狀態（NYHA）為必選欄位")

    if goal_label not in GOAL_MAP:
        raise ValueError("運動目標為必選欄位")

    profile = {
        "subject_id": subject_id.strip().upper(),
        "age_group": AGE_MAP[age_label],
        "sex": SEX_MAP[sex_label],
        "nyha_group": NYHA_MAP[nyha_label],
        "comorbidities": [
            COMORBIDITY_MAP[c]
            for c in comorbidity_labels
            if c in COMORBIDITY_MAP
        ],
        "surgery_history": [
            SURGERY_MAP[s]
            for s in surgery_labels
            if s in SURGERY_MAP
        ],
        "exercise_goal": GOAL_MAP[goal_label],
    }

    return profile
