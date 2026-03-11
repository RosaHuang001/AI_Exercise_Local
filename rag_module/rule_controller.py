from typing import List, Dict, Any


class RuleController:

    TOPIC_PRIORITY = {
        "Scope": 100,
        "Safety": 95,
        "Exercise_Termination": 95,
        "Exercise_Testing_Safety": 90,
        "Arrhythmia_Risk": 90,
        "Medication_Exercise_Prescription": 85,
        "Monitoring_RPE": 80,
        "Monitoring_Symptoms": 80,
        "Exercise Intensity": 75,
        "FITT": 70,
        "Movement Pattern": 65,
        "Joint Impact": 65,
        "Lower Limb Exercise": 60,
        "Upper Limb Exercise": 55,
        "Core Exercise": 50,
        "Individualization": 45,
        "Background": 10
    }

    RISK_PRIORITY_MULTIPLIER = {
        "low": 1.0,
        "moderate": 1.2,
        "high": 1.5,
        "very_high": 2.0
    }

    def __init__(self, max_rules: int = 6, debug: bool = False):
        self.max_rules = max_rules
        self.debug = debug

    # =====================================================
    # CONDITION MATCHING（等值 / list / bool 通吃）
    # =====================================================
    def _match_condition(
        self,
        rule_condition: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> bool:

        if not rule_condition:
            return True

        for key, rule_value in rule_condition.items():
            user_value = user_profile.get(key)

            if user_value is None:
                continue

            if isinstance(rule_value, list):
                if user_value not in rule_value:
                    return False
            elif isinstance(rule_value, bool):
                if user_value is not rule_value:
                    return False
            else:
                if user_value != rule_value:
                    return False

        return True

    # =====================================================
    # STEP 1｜風險導向排序
    # =====================================================
    def prioritize_rules(
        self,
        rules: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        risk_level = user_profile.get("risk_level", "moderate")
        risk_weight = self.RISK_PRIORITY_MULTIPLIER.get(risk_level, 1.0)

        def rule_score(rule: Dict[str, Any]) -> float:
            topic = rule.get("topic", "")
            base = self.TOPIC_PRIORITY.get(topic, 0)
            condition = rule.get("condition", {}) or {}
            specificity = sum(1 for v in condition.values() if v is not None)
            return (base + specificity * 2) * risk_weight

        return sorted(rules, key=rule_score, reverse=True)

    # =====================================================
    # STEP 2｜風險修正（Safety Modifiers）
    # =====================================================
    def apply_risk_modifiers(
        self,
        rules: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        filtered = []
        risk_level = user_profile.get("risk_level")
        posture = user_profile.get("posture")
        weight_bearing = user_profile.get("weight_bearing")
        nyha = user_profile.get("nyha")

        for rule in rules:

            condition = rule.get("condition", {}) or {}
            if not self._match_condition(condition, user_profile):
                continue

            topic = rule.get("topic", "")
            text = rule.get("rule", "").lower()

            if risk_level in ("high", "very_high"):
                if topic in ("Joint Impact", "Exercise Intensity"):
                    continue

            if nyha in ("III", "IV", 3, 4):
                if topic in ("Exercise Intensity", "Lower Limb Exercise", "Joint Impact", "FITT"):
                    rule = dict(rule)
                    rule["modifier_note"] = f"NYHA {nyha}，套用保守運動解讀"

            if posture == "standing" and topic in ("Movement Pattern", "Lower Limb Exercise"):
                rule = dict(rule)
                rule["modifier_note"] = "站姿運動，增加平衡風險考量"

            if weight_bearing and topic == "Joint Impact" and risk_level in ("moderate", "high", "very_high"):
                rule = dict(rule)
                rule["modifier_note"] = "負重運動，套用關節保護原則"

            filtered.append(rule)

        return filtered

    # =====================================================
    # STEP 3｜對外流程
    # =====================================================
    def process(
        self,
        rules: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        prioritized = self.prioritize_rules(rules, user_profile)
        modified = self.apply_risk_modifiers(prioritized, user_profile)
        return modified[: self.max_rules]
