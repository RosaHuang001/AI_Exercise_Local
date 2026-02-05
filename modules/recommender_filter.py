# recommender_filter.py
# ======================================================
# 個人化運動推薦系統－前置安全篩選模組
#
# 功能說明：
# 1. 載入運動動作庫（exercise_library.json）
# 2. 接收使用者狀態（NYHA、禁忌條件）
# 3. 進行禁忌條件正規化（normalize）
# 4. 依規則進行 Hard Filter（安全性前置篩選）
# 5. 輸出可解釋的排除原因（Explainable Recommendation）
# 6. Soft Ranking 時輸出「結構化推薦理由 recommendation_reason」（最終版）
# ======================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Set


# ======================================================
# 一、使用者狀態定義（User State）
# ======================================================
@dataclass
class UserState:
    """
    使用者目前狀態（系統決策依據）
    """
    nyha: str  # 心臟衰竭功能分級："I" | "II" | "III" | "IV"
    contraindications: List[str]  # 使用者自身禁忌條件（可為自然語言或系統代碼）


# ======================================================
# 二、禁忌條件正規化（Contraindication Normalization）
# ======================================================
# 說明：
# - 系統內部使用「標準禁忌詞（canonical form）」
# - 將不同寫法、同義詞映射為統一代碼
# - 未識別詞彙會標記為 unknown，避免資料靜默遺失

# --- 系統內部標準禁忌詞 ---
STANDARD_CONTRAINDICATIONS: Set[str] = {
    "low_back_pain",
    "neck_pain",
    "shoulder_pain",
    "shoulder_instability",
    "hip_pain",
    "balance_disorder",
    "spinal_rotation_pain",
    "uncontrolled_hypertension"
}

# --- 同義詞 / 變體 對照表 ---
SYNONYM_MAP: Dict[str, str] = {
    # 下背相關
    "acute_low_back_pain": "low_back_pain",
    "chronic_low_back_pain": "low_back_pain",
    "low_back_instability": "low_back_pain",

    # 肩部
    "severe_shoulder_pain": "shoulder_pain",

    # 平衡 / 眩暈
    "dizziness": "balance_disorder",
    "vertigo": "balance_disorder",

    # 髖／關節（與動作庫對齊）
    "hip_joint_pain": "hip_pain",
}


def normalize_contraindications(raw_items: List[str]) -> Set[str]:
    """
    將禁忌條件正規化為系統內部標準禁忌詞

    參數：
        raw_items: 原始禁忌條件列表（使用者輸入或動作標註）

    回傳：
        Set[str]：正規化後的禁忌條件集合
    """
    normalized: Set[str] = set()

    for item in raw_items or []:
        key = (item or "").strip()
        if not key:
            continue

        # 同義詞映射
        if key in SYNONYM_MAP:
            normalized.add(SYNONYM_MAP[key])
        # 已是系統標準詞
        elif key in STANDARD_CONTRAINDICATIONS:
            normalized.add(key)
        # 未識別詞彙，保留並標記
        else:
            normalized.add(f"unknown:{key}")

    return normalized


# ======================================================
# 三、載入運動動作庫（exercise_library.json）
# ======================================================
def load_exercise_library(json_path: str) -> Dict[str, Any]:
    """
    載入運動動作庫（JSON 格式）

    回傳格式需包含：
    {
        "version": "...",
        "exercises": [ {...}, {...} ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "exercises" not in data or not isinstance(data["exercises"], list):
        raise ValueError("運動動作庫格式錯誤，缺少 exercises 欄位")

    return data


# ======================================================
# 四、Hard Filter：前置安全篩選（核心邏輯）
# ======================================================
def hard_filter_exercises(
    user: UserState,
    library: Dict[str, Any],
    *,
    allow_if_missing_nyha: bool = False
) -> Dict[str, Any]:
    """
    根據使用者狀態，對運動動作進行硬性安全篩選（Hard Filter）

    篩選規則：
    1. 使用者 NYHA 不在動作允許範圍 → 排除
    2. 使用者禁忌條件與動作禁忌條件重疊 → 排除

    回傳：
    - included：通過安全篩選的動作
    - excluded：被排除的動作 + 結構化排除原因
    """
    included: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []

    user_nyha = (user.nyha or "").strip().upper()
    user_cis = normalize_contraindications(user.contraindications)

    for ex in library["exercises"]:
        reasons: List[Dict[str, str]] = []

        # ---------- 規則一：NYHA 分級 ----------
        nyha_allowed = ex.get("nyha_allowed")
        if nyha_allowed is None:
            if not allow_if_missing_nyha:
                reasons.append({
                    "type": "nyha",
                    "code": "missing_nyha_allowed"
                })
        else:
            allowed = {x.strip().upper() for x in nyha_allowed}
            if user_nyha not in allowed:
                reasons.append({
                    "type": "nyha",
                    "code": f"not_allowed_for_{user_nyha}"
                })

        # ---------- 規則二：禁忌條件 ----------
        ex_cis = normalize_contraindications(ex.get("contraindications", []))
        overlap = user_cis.intersection(ex_cis)

        for ci in overlap:
            reasons.append({
                "type": "contraindication",
                "code": ci,
                "source": "user_and_exercise_overlap"
            })

        # ---------- 決策 ----------
        if reasons:
            excluded.append({
                "exercise": ex,
                "exclusion_reasons": reasons
            })
        else:
            included.append(ex)

    return {
        "user": asdict(user),
        "library_version": library.get("version"),
        "included": included,
        "excluded": excluded,
        "counts": {
            "included": len(included),
            "excluded": len(excluded)
        }
    }


# ======================================================
# 五、可解釋層：排除原因轉為人類可讀說明
# ======================================================
def explain_exclusion(reasons: List[Dict[str, str]]) -> List[str]:
    """
    將結構化排除原因轉換為中文說明文字
    （用於使用者介面或研究展示）
    """
    messages: List[str] = []

    for r in reasons:
        if r["type"] == "nyha":
            messages.append("此動作不適合您目前的心臟功能分級（NYHA）")
        elif r["type"] == "contraindication":
            messages.append(f"此動作與您的身體狀況不建議併用（{r['code']}）")
        else:
            messages.append("不符合系統安全條件")

    return messages


# ======================================================
# 六、輔助功能：展開單側動作（左右）
# ======================================================
def expand_unilateral_sides(included: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    若動作標註為 unilateral，則自動展開為 left / right 兩筆
    """
    expanded: List[Dict[str, Any]] = []

    for ex in included:
        if ex.get("sides") != "unilateral":
            expanded.append(ex)
            continue

        for side in ("left", "right"):
            ex_copy = dict(ex)
            ex_copy["exercise_id"] = f"{ex['exercise_id']}_{side}"
            ex_copy["side"] = side
            expanded.append(ex_copy)

    return expanded


# ======================================================
# 七、推薦理由（Recommendation Reason）— 內部產生器
# ======================================================
def _build_hard_filter_pass_reasons(
    *,
    user_nyha: str,
    user_cis: Set[str],
    exercise: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    產生「通過 Hard Filter 的理由」（結構化）
    注意：此函式假設該動作已通過 hard_filter_exercises（即不應有 overlap）
    """
    reasons: List[Dict[str, str]] = []

    # --- NYHA 通過理由 ---
    nyha_allowed = exercise.get("nyha_allowed")
    if not nyha_allowed:
        # 動作庫沒有標註 nyha_allowed 的情況（視你的設計可能不會發生）
        reasons.append({
            "type": "nyha",
            "code": "missing_nyha_allowed",
            "description": "動作未標註 NYHA 適用範圍（需人工確認）"
        })
    else:
        allowed = {x.strip().upper() for x in nyha_allowed}
        if user_nyha and user_nyha in allowed:
            reasons.append({
                "type": "nyha",
                "code": f"allowed_for_{user_nyha}",
                "description": f"此動作符合使用者目前 NYHA {user_nyha} 等級之建議運動範圍"
            })
        elif not user_nyha:
            # 使用者未提供 NYHA（保守設計下通常會被 hard filter 排除，但這裡仍做完整描述）
            reasons.append({
                "type": "nyha",
                "code": "user_nyha_missing",
                "description": "使用者未提供 NYHA 分級，系統無法以 NYHA 進行適配判斷"
            })
        else:
            # 理論上不應到這裡（因為若不在 allowed 應被 hard filter 排除）
            reasons.append({
                "type": "nyha",
                "code": f"allowed_range_contains_{','.join(sorted(allowed))}",
                "description": "此動作標註之 NYHA 適用範圍與使用者分級相符"
            })

    # --- 禁忌條件通過理由 ---
    ex_cis = normalize_contraindications(exercise.get("contraindications", []))
    overlap = user_cis.intersection(ex_cis)

    if not overlap:
        reasons.append({
            "type": "contraindication",
            "code": "no_overlap",
            "description": "未偵測到與使用者禁忌條件之衝突"
        })
    else:
        # 理論上不應到這裡（重疊應被 hard filter 排除）
        for ci in overlap:
            reasons.append({
                "type": "contraindication",
                "code": ci,
                "description": f"注意：偵測到禁忌條件重疊（{ci}），需再確認 hard filter 設定"
            })

    return reasons


def _humanize_positive_factor(type_: str, code: str) -> str:
    """
    將正向因素轉為穩定中文敘述（避免每次輸出句子亂飄）
    """
    mapping = {
        ("impact", "low_impact"): "屬於低衝擊動作，對心臟衰竭族群較為安全",
        ("impact", "medium_impact"): "衝擊程度中等，整體可接受但需注意個別狀況",
        ("balance", "low_balance_requirement"): "平衡需求低，降低跌倒與不穩定風險",
        ("balance", "medium_balance_requirement"): "平衡需求中等，可視能力循序漸進",
        ("intensity", "low_intensity"): "動作強度偏低，較符合保守安全原則",
        ("intensity", "low_to_medium_intensity"): "動作強度偏低到中等，適合作為漸進訓練",
        ("nyha_flexibility", "wide_nyha_range"): "動作適用 NYHA 範圍較廣，彈性高"
    }
    return mapping.get((type_, code), "符合個人化適配條件，整體優先度較高")


def _humanize_penalty_factor(type_: str, code: str) -> str:
    """
    將懲罰因素轉為穩定中文敘述（用於「相對不優先」而非「不可做」）
    """
    mapping = {
        ("impact", "high_impact"): "動作衝擊較高，建議降低強度或改採替代動作",
        ("balance", "high_balance_requirement"): "平衡需求較高，對目前狀態可能較具挑戰",
        ("intensity", "medium_or_high_for_nyha_III"): "動作強度略高於 NYHA III 建議之理想區間"
    }
    return mapping.get((type_, code), "此動作存在相對不利因素，因此排序優先度降低")


# ======================================================
# 八、Soft Ranking：個人化排序（非排除） + 結構化推薦理由
# ======================================================
def soft_rank_exercises(
    user: UserState,
    exercises: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    對已通過 Hard Filter 的動作進行個人化排序（Soft Ranking）

    排序依據（加權分數）【維持你原本邏輯，不改】：
    - NYHA 適配度（允許範圍越廣越高）
    - 動作衝擊程度（low > medium > high）
    - 平衡需求（low > medium > high，且 NYHA III/IV 對 high 額外懲罰）
    - 動作強度區間（low > low_to_medium > medium > high，且 NYHA III 對 medium/high 懲罰）

    新增輸出：
    - recommendation_reason（結構化推薦理由，最終 schema）
    """
    ranked: List[Dict[str, Any]] = []

    user_nyha = (user.nyha or "").strip().upper()
    user_cis = normalize_contraindications(user.contraindications)

    for ex in exercises:
        score = 0.0
        score_detail = {}  # 保留各項得分，方便解釋與 debug（原本就有）

        # 這裡開始蒐集「結構化推薦理由」用的因素
        soft_positive: List[Dict[str, str]] = []
        soft_penalty: List[Dict[str, str]] = []

        # ---------- 1. NYHA 適配度 ----------
        nyha_range = ex.get("nyha_allowed", [])
        score_detail["nyha_range_size"] = len(nyha_range)
        score += len(nyha_range) * 1.5

        # 正向因素：nyha 允許範圍越大 → 彈性越高
        if len(nyha_range) >= 3:
            soft_positive.append({
                "type": "nyha_flexibility",
                "code": "wide_nyha_range",
                "description": _humanize_positive_factor("nyha_flexibility", "wide_nyha_range")
            })

        # ---------- 2. 衝擊程度 ----------
        impact = ex.get("impact_level", "low")
        impact_score_map = {
            "low": 2.0,
            "medium": 1.0,
            "high": 0.0
        }
        score_detail["impact"] = impact
        score += impact_score_map.get(impact, 0.5)

        if impact == "low":
            soft_positive.append({
                "type": "impact",
                "code": "low_impact",
                "description": _humanize_positive_factor("impact", "low_impact")
            })
        elif impact == "medium":
            soft_positive.append({
                "type": "impact",
                "code": "medium_impact",
                "description": _humanize_positive_factor("impact", "medium_impact")
            })
        elif impact == "high":
            soft_penalty.append({
                "type": "impact",
                "code": "high_impact",
                "description": _humanize_penalty_factor("impact", "high_impact")
            })

        # ---------- 3. 平衡需求 ----------
        balance = ex.get("balance_requirement", "low")
        balance_score_map = {
            "low": 2.0,
            "medium": 1.0,
            "high": 0.0
        }

        balance_score = balance_score_map.get(balance, 0.5)

        # NYHA III 以上，對高平衡需求動作給予額外懲罰（維持你原本邏輯）
        if user_nyha in ["III", "IV"] and balance == "high":
            balance_score -= 1.0
            soft_penalty.append({
                "type": "balance",
                "code": "high_balance_requirement",
                "description": _humanize_penalty_factor("balance", "high_balance_requirement")
            })
        else:
            if balance == "low":
                soft_positive.append({
                    "type": "balance",
                    "code": "low_balance_requirement",
                    "description": _humanize_positive_factor("balance", "low_balance_requirement")
                })
            elif balance == "medium":
                soft_positive.append({
                    "type": "balance",
                    "code": "medium_balance_requirement",
                    "description": _humanize_positive_factor("balance", "medium_balance_requirement")
                })

        score_detail["balance"] = balance
        score += balance_score

        # ---------- 4. 強度區間 ----------
        intensity = ex.get("intensity_band", "medium")
        intensity_score_map = {
            "low": 2.0,
            "low_to_medium": 1.5,
            "medium": 1.0,
            "high": 0.0
        }

        intensity_score = intensity_score_map.get(intensity, 0.5)

        # NYHA III 偏好低強度（維持你原本邏輯）
        if user_nyha == "III" and intensity in ["medium", "high"]:
            intensity_score -= 0.5
            soft_penalty.append({
                "type": "intensity",
                "code": "medium_or_high_for_nyha_III",
                "description": _humanize_penalty_factor("intensity", "medium_or_high_for_nyha_III")
            })
        else:
            if intensity == "low":
                soft_positive.append({
                    "type": "intensity",
                    "code": "low_intensity",
                    "description": _humanize_positive_factor("intensity", "low_intensity")
                })
            elif intensity == "low_to_medium":
                soft_positive.append({
                    "type": "intensity",
                    "code": "low_to_medium_intensity",
                    "description": _humanize_positive_factor("intensity", "low_to_medium_intensity")
                })

        score_detail["intensity"] = intensity
        score += intensity_score

        # ---------- Hard Filter 通過理由（結構化） ----------
        hard_pass_reasons = _build_hard_filter_pass_reasons(
            user_nyha=user_nyha,
            user_cis=user_cis,
            exercise=ex
        )

        # ---------- 結構化推薦理由（最終 schema） ----------
        recommendation_reason = {
            "included": True,
            "hard_filter_pass_reasons": hard_pass_reasons,
            "soft_rank_positive_factors": soft_positive,
            "soft_rank_penalty_factors": soft_penalty,
            "final_rank_score": round(score, 2)
        }

        # ---------- 總結（維持你原本輸出 + 新增 recommendation_reason） ----------
        ex_with_score = dict(ex)
        ex_with_score["soft_rank_score"] = round(score, 2)
        ex_with_score["soft_rank_detail"] = score_detail
        ex_with_score["recommendation_reason"] = recommendation_reason

        ranked.append(ex_with_score)

    # 依分數由高到低排序
    ranked.sort(key=lambda x: x["soft_rank_score"], reverse=True)
    return ranked


# ======================================================
# 九、範例執行（測試用）
# ======================================================
if __name__ == "__main__":
    # 讓單獨執行更不容易踩路徑雷（不影響你 main.py）
    candidate_paths = [
        "exercise_library.json",
        os.path.join("knowledge_base", "exercise_library.json"),
        os.path.join("knowledge_base", "exercise_library", "exercise_library.json"),
    ]
    lib_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if not lib_path:
        raise FileNotFoundError("找不到 exercise_library.json，請確認路徑。")

    library = load_exercise_library(lib_path)

    user = UserState(
        nyha="III",
        contraindications=["low_back_instability", "dizziness"]
    )

    filtered = hard_filter_exercises(user, library)
    print("=== 篩選結果統計 ===")
    print(filtered["counts"])

    print("\n=== 可推薦動作（含推薦理由） ===")
    ranked = soft_rank_exercises(user, filtered["included"])
    for ex in ranked[:10]:
        print("-", ex["exercise_id"], ex.get("name_zh"), "score=", ex["soft_rank_score"])
        rr = ex.get("recommendation_reason", {})
        for item in rr.get("hard_filter_pass_reasons", []):
            print("  [Hard]", item.get("description"))
        for item in rr.get("soft_rank_positive_factors", []):
            print("  [+]", item.get("description"))
        for item in rr.get("soft_rank_penalty_factors", []):
            print("  [-]", item.get("description"))

    print("\n=== 被排除動作（含原因） ===")
    for item in filtered["excluded"]:
        ex = item["exercise"]
        print("-", ex["exercise_id"], ex.get("name_zh"))
        for msg in explain_exclusion(item["exclusion_reasons"]):
            print("  •", msg)
