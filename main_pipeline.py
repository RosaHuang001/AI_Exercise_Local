#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI_Exercise_Local - Main Pipeline (v3, tqdm progress)
整合：
  YOLOv11 Pose → PoseC3D (finGYM/K400) → GPT Semantic → MySQL
特性：
  - 單一長條顯示批次進度
  - 每支影片顯示子模組進度（YOLO→PoseC3D→GPT）
  - GPU only, 無防呆
"""

import os
import time
import json
import yaml
import torch
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# === 模組導入 ===
from video_processor import VideoProcessor
from pose_classifier import PoseClassifier
from semantic_advisor import SemanticAdvisor
from database_handler import DatabaseHandler


class MainPipelineV3:
    """AI 運動分析主流程（YOLO→PoseC3D→GPT→SQL）"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        assert torch.cuda.is_available(), "❌ CUDA 不可用，請確認 GPU 驅動或環境設定"
        self.device = "cuda"

        # === 初始化各模組 ===
        self.video_processor = VideoProcessor(config_path)
        self.pose_classifier = PoseClassifier(config_path)
        self.semantic_advisor = SemanticAdvisor(config_path)
        self.database_handler = DatabaseHandler(config_path)

        self.stats = {
            "total_videos": 0,
            "successful_poses": 0,
            "successful_classifications": 0,
            "successful_advice": 0,
            "successful_db_inserts": 0,
            "processing_time": 0.0,
        }

    # === 初始化 ===
    def initialize(self, no_db: bool = False):
        print("\n🚀 初始化所有模組中...")
        self.video_processor.load_model()
        self.pose_classifier.load_model()
        self.semantic_advisor.initialize_openai()
        if not no_db:
            self.database_handler.connect()
            self.database_handler.create_tables()
        print("✅ 所有模組初始化完成\n")

    # === 單部影片流程 ===
    def process_single_video(self, video_path: str, no_db: bool = False, pbar=None):
        video_path = Path(video_path)
        self.stats["total_videos"] += 1

        with tqdm(total=3, desc=f"🎬 {video_path.name}", leave=False, ncols=80) as subbar:
            # Step 1: YOLO Pose 偵測
            subbar.set_postfix_str("YOLO Pose 偵測中")
            pose_json_path = self.video_processor.process_video(str(video_path))
            self.stats["successful_poses"] += 1
            subbar.update(1)

            # Step 2: PoseC3D 動作分類
            subbar.set_postfix_str("PoseC3D 分類中")
            classification_result = self.pose_classifier.classify_pose_json(pose_json_path)
            pose_json = json.load(open(pose_json_path, "r", encoding="utf-8"))
            classification_result["video_info"] = pose_json.get("video_info", {})
            classification_result["sequence_length"] = len(pose_json.get("pose_sequence", []))
            classification_result["keypoints_count"] = (
                len(pose_json["pose_sequence"][0]["keypoints"])
                if classification_result["sequence_length"] > 0 else 0
            )
            self.stats["successful_classifications"] += 1
            subbar.update(1)

            # Step 3: GPT 語意分析
            subbar.set_postfix_str("GPT 語意分析中")
            advice_report = self.semantic_advisor.generate_advice(classification_result)

            # 把分類 JSON 路徑補進報告（供 SQL 寫入）
            if "classification_json_path" in classification_result:
                advice_report["classification_json_path"] = classification_result["classification_json_path"]

            self.stats["successful_advice"] += 1
            subbar.update(1)

        # Step 4: 寫入 MySQL
        if not no_db:
            ok = self.database_handler.insert_pose_result(advice_report)
            if ok:
                self.stats["successful_db_inserts"] += 1

        # 更新主進度
        if pbar:
            pbar.update(1)

    # === 批次處理 ===
    def batch_process(self, input_dir: str, no_db: bool = False):
        video_dir = Path(input_dir)
        exts = (".mp4", ".avi", ".mov", ".mkv")
        files = [f for ext in exts for f in video_dir.glob(f"*{ext}")]
        if not files:
            print(f"⚠️  未在 {input_dir} 找到影片。")
            return

        print(f"📂 偵測到 {len(files)} 支影片，開始批次分析...\n")
        with tqdm(total=len(files), desc="🔥 系統總進度", ncols=90) as pbar:
            for vf in files:
                self.process_single_video(str(vf), no_db=no_db, pbar=pbar)

    # === 結尾 ===
    def finalize(self):
        try:
            self.database_handler.disconnect()
        except Exception:
            pass

    # === 統計摘要 ===
    def print_summary(self):
        print("\n" + "=" * 70)
        print("🎯 AI 運動分析系統 (v3) - 處理摘要")
        print("=" * 70)
        print(f"📊 總影片數: {self.stats['total_videos']}")
        print(f"🟢 姿勢偵測成功: {self.stats['successful_poses']}")
        print(f"🟢 動作分類成功: {self.stats['successful_classifications']}")
        print(f"🟢 GPT 建議成功: {self.stats['successful_advice']}")
        print(f"🟢 資料庫寫入成功: {self.stats['successful_db_inserts']}")
        print(f"⏱️  總耗時: {self.stats['processing_time']:.2f} 秒")
        if self.stats["total_videos"] > 0:
            rate = (self.stats["successful_advice"] / self.stats["total_videos"]) * 100
            print(f"✅ 成功率: {rate:.1f}%")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="AI Exercise Main Pipeline (v3, tqdm)")
    parser.add_argument("--input", "-i", type=str, required=True, help="輸入影片檔案或資料夾")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="設定檔路徑")
    parser.add_argument("--batch", "-b", action="store_true", help="批次模式")
    parser.add_argument("--no-db", action="store_true", help="跳過資料庫")
    args = parser.parse_args()

    start = time.time()
    pipeline = MainPipelineV3(args.config)
    pipeline.initialize(no_db=args.no_db)

    input_path = Path(args.input)
    if args.batch or input_path.is_dir():
        pipeline.batch_process(str(input_path), no_db=args.no_db)
    else:
        pipeline.process_single_video(str(input_path), no_db=args.no_db)

    pipeline.stats["processing_time"] = time.time() - start
    pipeline.print_summary()
    pipeline.finalize()


if __name__ == "__main__":
    main()
