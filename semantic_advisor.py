#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI_Exercise_Local - Semantic Advisor (v3, tqdm progress)
功能：
  - 使用 OpenAI GPT 模型生成動作分析與安全建議
  - tqdm 進度顯示（建立提示 → 生成階段）
  - 同時輸出 JSON + Markdown
  - 完全支援 openai>=1.0 語法
  - 與 DatabaseHandler_v3、MainPipeline_v3 完全整合
"""

import os
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv


class SemanticAdvisor:
    """GPT 語意建議生成器（tqdm 強化版，SQL 兼容版）"""

    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv()
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.logger = self._setup_logger()
        self.client = None

    # === 日誌設定 ===
    def _setup_logger(self):
        logger = logging.getLogger("SemanticAdvisor")
        logger.setLevel(logging.INFO)
        log_file = Path(self.config["paths"]["logs_dir"]) / f"semantic_advisor_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    # === 初始化 OpenAI ===
    def initialize_openai(self):
        api_key = os.getenv("OPENAI_API_KEY") or self.config["api"]["openai"].get("api_key", "")
        assert api_key != "", "❌ 未設定 OPENAI_API_KEY"
        self.client = OpenAI(api_key=api_key)
        self.logger.info("✅ OpenAI GPT 初始化成功")
        return True

    # === 建立 Prompt ===
    def _create_prompt(self, action: str, confidence: float, classification_result: dict) -> str:
        return f"""
你是一位專業的運動醫學顧問。根據以下動作辨識結果，請生成完整的安全分析與訓練建議：

- 動作名稱：{action}
- 模型信心值：{confidence:.3f}
- 關節點數：{classification_result.get('keypoints_count', 'N/A')}
- 序列長度：{classification_result.get('sequence_length', 'N/A')} 幀

請以繁體中文、條列方式輸出以下段落：
1. 動作分析（主要肌群、正確姿勢）
2. 技術要點（呼吸與協調）
3. 修正建議（常見錯誤與修正方法）
4. 安全提醒（常見傷害與預防）
5. 訓練建議（建議組數、次數、頻率）
"""

    # === 呼叫 GPT ===
    def _call_gpt_api(self, prompt: str, model_name: str, max_tokens: int, temperature: float) -> str:
        with tqdm(total=2, desc="🧠 GPT 生成建議中", ncols=90, leave=False) as bar:
            bar.set_postfix_str("建立提示中...")
            bar.update(1)
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一位運動醫學與復健訓練專家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            bar.set_postfix_str("生成完成！")
            bar.update(1)
        return response.choices[0].message.content.strip()

    # === 解析 GPT 回應 ===
    def _parse_response(self, response_text: str) -> dict:
        sections = {
            "action_analysis": "",
            "technical_points": "",
            "correction_suggestions": "",
            "safety_reminders": "",
            "training_recommendations": ""
        }
        current_key = "action_analysis"
        for line in response_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "動作分析" in line:
                current_key = "action_analysis"
            elif "技術要點" in line:
                current_key = "technical_points"
            elif "修正建議" in line:
                current_key = "correction_suggestions"
            elif "安全提醒" in line:
                current_key = "safety_reminders"
            elif "訓練建議" in line:
                current_key = "training_recommendations"
            sections[current_key] += line + "\n"
        return sections

    # === 生成完整報告 ===
    def generate_advice(self, classification_result: dict) -> dict:
        action = classification_result["predicted_action"]
        confidence = classification_result["confidence"]
        model_name = self.config["api"]["openai"]["model"]
        temperature = self.config["api"]["openai"]["temperature"]
        max_tokens = self.config["api"]["openai"]["max_tokens"]

        # === 生成 Prompt ===
        prompt = self._create_prompt(action, confidence, classification_result)

        # === 呼叫 GPT ===
        response_text = self._call_gpt_api(prompt, model_name, max_tokens, temperature)

        # === 解析文本 ===
        parsed = self._parse_response(response_text)

        # === 組成輸出報告 ===
        report = {
            "video_info": classification_result.get("video_info", {}),
            "classification_summary": {
                "predicted_action": action,
                "confidence": confidence,
                "model_info": classification_result.get("model_info", {}),
            },
            "ai_advice": parsed,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model_name": model_name,
                "temperature": temperature,
            },
        }

        # === 儲存 JSON + Markdown ===
        results_dir = Path(self.config["paths"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        filename_prefix = f"{action}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        json_path = results_dir / f"{filename_prefix}_advice.json"
        md_path = results_dir / f"{filename_prefix}_advice.md"

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(report, jf, ensure_ascii=False, indent=2)

        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write(self._to_markdown(report))

        # === 回傳與 SQL 對應 ===
        report["advice_json_path"] = str(json_path)
        report["advice_markdown_path"] = str(md_path)

        # 若分類結果中含 JSON 路徑，帶入（SQL 用）
        if "classification_json_path" in classification_result:
            report["classification_json_path"] = classification_result["classification_json_path"]

        self.logger.info(f"✅ GPT 建議生成完成：{json_path.name}")
        return report

    # === Markdown 匯出 ===
    def _to_markdown(self, report: dict) -> str:
        ai = report["ai_advice"]
        return f"""# 運動動作建議報告

## 動作分類結果
- 預測動作：{report['classification_summary']['predicted_action']}
- 信心值：{report['classification_summary']['confidence']:.3f}

## AI 建議
### 動作分析
{ai.get("action_analysis", "")}

### 技術要點
{ai.get("technical_points", "")}

### 修正建議
{ai.get("correction_suggestions", "")}

### 安全提醒
{ai.get("safety_reminders", "")}

### 訓練建議
{ai.get("training_recommendations", "")}

---
*生成時間：{report['metadata']['generated_at']}*
*模型：{report['metadata']['model_name']}*
"""


def main():
    """測試單次生成"""
    advisor = SemanticAdvisor()
    advisor.initialize_openai()

    test_result = {
        "predicted_action": "深蹲",
        "confidence": 0.88,
        "sequence_length": 120,
        "keypoints_count": 17,
        "classification_json_path": "results/test_pose_classification.json"
    }

    report = advisor.generate_advice(test_result)
    print("✅ 測試完成，輸出：", report["advice_json_path"])


if __name__ == "__main__":
    main()
