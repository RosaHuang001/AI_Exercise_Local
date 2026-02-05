# rag/rag_engine.py
import json
from typing import Dict, List, Optional, Any


class ACSMRagEngine:
    """
    STEP 2：ACSM 規則檢索引擎（Rule-based RAG Retriever）

    功能說明：
    - 載入 ACSM 心臟衰竭（HF）相關規則 JSON
    - 根據「使用者條件（condition）」進行嚴格比對
    - 回傳『符合條件、可被後續 RuleController 使用』的規則集合

    設計原則：
    - 本模組只負責「檢索」，不做風險判斷
    - 不產生建議、不介入 LLM 推論
    - 採取保守條件比對，避免誤用醫學指引
    """

    def __init__(self, knowledge_path: str):
        """
        參數說明：
        knowledge_path：ACSM HF 規則 JSON 檔案路徑
        """
        with open(knowledge_path, "r", encoding="utf-8") as f:
            self.rules: List[Dict[str, Any]] = json.load(f)

    # =====================================================
    # 內部工具：條件比對（Conservative Matching）
    # =====================================================
    def _match_condition(
        self,
        rule_condition: Optional[Dict[str, Any]],
        query_condition: Dict[str, Any]
    ) -> bool:
        """
        判斷單一規則是否適用於目前使用者運動情境。

        比對原則（醫學保守策略）：
        1. 規則未指定 condition（None 或 {}）→ 視為通用規則，可套用
        2. 規則指定的每一個條件，都必須在使用者情境中『明確存在且完全相符』
        3. 若使用者缺少任何規則要求的欄位 → 視為不適用（避免誤套）

        rule_condition 範例：
        {
            "region": "Lower",
            "posture": "Standing",
            "weight_bearing": True
        }

        query_condition 範例：
        {
            "region": "Lower",
            "posture": "Standing",
            "weight_bearing": True
        }
        """

        # 規則未設定任何條件 → 通用規則
        if not rule_condition:
            return True

        for key, rule_value in rule_condition.items():
            query_value = query_condition.get(key)

            # 使用者條件中缺少此欄位 → 不適用
            if query_value is None:
                return False

            # 規則值為 list → 使用者值須在清單內
            if isinstance(rule_value, list):
                if query_value not in rule_value:
                    return False
            elif isinstance(rule_value, bool):
                if query_value is not rule_value:
                    return False
            else:
                if rule_value != query_value:
                    return False

        return True

    # =====================================================
    # 對外介面：規則檢索
    # =====================================================
    def retrieve_rules(
        self,
        population: str,
        condition: Dict[str, Any],
        topics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        依據族群與運動情境，檢索符合條件的 ACSM 規則。

        參數說明：
        - population：
          系統內部族群代碼（例如："heart_failure"）
        - condition：
          使用者運動情境條件（如 posture / weight_bearing / region）
        - topics：
          （可選）僅檢索特定主題的規則

        回傳：
        - 符合條件的 ACSM 規則清單（尚未做風險排序）
        """

        selected_rules: List[Dict[str, Any]] = []

        for rule in self.rules:
            # ---------- 族群不符，直接略過 ----------
            if rule.get("population") != population:
                continue

            # ---------- 主題篩選（若有指定） ----------
            if topics is not None and rule.get("topic") not in topics:
                continue

            # ---------- 條件比對 ----------
            if not self._match_condition(
                rule_condition=rule.get("condition"),
                query_condition=condition
            ):
                continue

            selected_rules.append(rule)

        return selected_rules
