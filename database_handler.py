#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI_Exercise_Local - Database Handler (v3)
與現有 MySQL 結構（pose_results、pose_reference、pose_keypoints_17…）完全對齊。

功能：
- 連線 MySQL
- 將 GPT/分類結果寫入 pose_results
- 自動更新 action_statistics
- 查詢與匯出
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv


class DatabaseHandler:
    """MySQL 資料庫處理器"""

    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv()
        self.config = self._load_config(config_path)
        self.connection = None
        self.logger = self._setup_logger()

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"無法載入設定檔 {config_path}: {e}")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("DatabaseHandler")
        logger.setLevel(getattr(logging, self.config["system"]["log_level"]))
        log_file = Path(self.config["paths"]["logs_dir"]) / f"database_handler_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    # --------------------------------------------------
    # 基本連線
    # --------------------------------------------------
    def connect(self) -> bool:
        try:
            db_config = self.config["database"]["mysql"]
            self.connection = mysql.connector.connect(
                host=db_config["host"],
                port=db_config["port"],
                user=db_config["user"],
                password=db_config["password"],
                database=db_config["database"],
                charset="utf8mb4",
                collation="utf8mb4_unicode_ci",
                autocommit=True
            )
            self.logger.info(f"✅ 連接 MySQL 成功: {db_config['host']}:{db_config['port']}")
            return True
        except Error as e:
            self.logger.error(f"❌ 資料庫連接錯誤: {e}")
            return False

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.logger.info("🔌 資料庫連線已關閉")

    # --------------------------------------------------
    # 建表（僅建立 action_statistics）
    # --------------------------------------------------
    def create_tables(self) -> bool:
        try:
            if not self.connection or not self.connection.is_connected():
                return False
            cursor = self.connection.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_statistics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                action_name VARCHAR(100) UNIQUE,
                total_count INT DEFAULT 0,
                avg_confidence DECIMAL(5,3) DEFAULT 0.000,
                max_confidence DECIMAL(5,3) DEFAULT 0.000,
                min_confidence DECIMAL(5,3) DEFAULT 0.000,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            self.logger.info("✅ action_statistics 表確認/建立完成")
            return True
        except Error as e:
            self.logger.error(f"❌ 建立表錯誤: {e}")
            return False

    # --------------------------------------------------
    # 插入分析結果（對應 pose_results）
    # --------------------------------------------------
    def insert_pose_result(self, report: Dict) -> bool:
        """插入 SemanticAdvisor/Classifier 的整合結果"""
        try:
            if not self.connection or not self.connection.is_connected():
                return False

            cursor = self.connection.cursor()

            classification = report.get("classification_summary", {})
            ai_advice = report.get("ai_advice", {})

            video_name = report.get("video_info", {}).get("filename", "unknown")
            predicted_action = classification.get("predicted_action", "")
            confidence = float(classification.get("confidence", 0.0))

            classification_json_path = report.get("classification_json_path", "")
            advice_json_path = report.get("advice_json_path", "")

            semantic_summary = ai_advice.get("action_analysis", "")[:300]
            advice_summary = ai_advice.get("correction_suggestions", "")[:300]

            safety_status = "unknown"
            risk_level = "medium"
            safety_reason = ""

            insert_sql = """
            INSERT INTO pose_results (
                video_name, predicted_action, confidence,
                safety_status, risk_level, safety_reason,
                semantic_summary, advice_summary,
                classification_json_path, advice_json_path, processed_by
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """
            values = (
                video_name,
                predicted_action,
                confidence,
                safety_status,
                risk_level,
                safety_reason,
                semantic_summary,
                advice_summary,
                classification_json_path,
                advice_json_path,
                "system"
            )

            cursor.execute(insert_sql, values)
            self.connection.commit()
            cursor.close()

            self._update_action_statistics(predicted_action, confidence)
            self.logger.info(f"✅ 寫入 pose_results 成功: {video_name} / {predicted_action}")
            return True
        except Error as e:
            self.logger.error(f"❌ 插入 pose_results 錯誤: {e}")
            return False

    # --------------------------------------------------
    # 更新動作統計
    # --------------------------------------------------
    def _update_action_statistics(self, action_name: str, confidence: float):
        try:
            if not self.connection or not self.connection.is_connected():
                return
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO action_statistics (action_name, total_count, avg_confidence, max_confidence, min_confidence)
                VALUES (%s, 1, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    total_count = total_count + 1,
                    avg_confidence = (avg_confidence + VALUES(avg_confidence)) / 2,
                    max_confidence = GREATEST(max_confidence, VALUES(max_confidence)),
                    min_confidence = LEAST(min_confidence, VALUES(min_confidence))
            """, (action_name, confidence, confidence, confidence))
            self.connection.commit()
            cursor.close()
        except Error as e:
            self.logger.error(f"更新 action_statistics 失敗: {e}")

    # --------------------------------------------------
    # 批次插入（從 *_advice.json 檔匯入）
    # --------------------------------------------------
    def batch_insert_results(self, results_dir: str) -> int:
        results_dir = Path(results_dir)
        if not results_dir.exists():
            return 0
        json_files = list(results_dir.glob("*_advice.json"))
        if not json_files:
            return 0

        success = 0
        for js in json_files:
            try:
                with open(js, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["advice_json_path"] = str(js)
                if self.insert_pose_result(data):
                    success += 1
            except Exception as e:
                self.logger.error(f"批次插入錯誤: {e}")
        self.logger.info(f"批次插入完成：{success}/{len(json_files)} 成功")
        return success

    # --------------------------------------------------
    # 查詢 pose_results
    # --------------------------------------------------
    def query_results(self, limit: int = 100) -> List[Dict]:
        try:
            if not self.connection or not self.connection.is_connected():
                return []
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM pose_results
                ORDER BY processed_at DESC
                LIMIT %s
            """, (limit,))
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except Error as e:
            self.logger.error(f"查詢 pose_results 失敗: {e}")
            return []

    # --------------------------------------------------
    # 匯出結果為 CSV
    # --------------------------------------------------
    def export_to_csv(self, output_path: str, limit: int = 1000) -> bool:
        try:
            rows = self.query_results(limit)
            if not rows:
                return False
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            self.logger.info(f"成功匯出 {len(rows)} 筆記錄到 {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"匯出 CSV 失敗: {e}")
            return False


def main():
    db = DatabaseHandler()
    if not db.connect():
        print("❌ 資料庫連接失敗")
        return
    db.create_tables()
    results = db.query_results(limit=5)
    print(f"查詢結果筆數：{len(results)}")
    db.disconnect()


if __name__ == "__main__":
    main()
