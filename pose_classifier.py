#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI_Exercise_Local - Pose Classifier (v3)
功能：
  - 使用 MMAction2 PoseC3D 模型進行動作分類
  - 自動儲存分類結果 JSON 檔（供 SQL 寫入）
  - tqdm 顯示初始化與推論進度
  - 與 YOLOv11-Pose JSON 完整對接
"""

import os
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from mmaction.apis import init_recognizer, inference_recognizer
from dotenv import load_dotenv


class PoseClassifier:
    """PoseC3D 動作分類器（GPU only, tqdm 強化版）"""

    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv()
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # === 載入模型 ===
    def load_model(self):
        print("\n⚙️ 載入 PoseC3D 模型中...")
        cfg_file = self.config["models"]["posec3d"]["config_file"]
        checkpoint = self.config["models"]["posec3d"]["checkpoint_file"]

        with tqdm(total=3, desc="PoseC3D 初始化", ncols=80) as bar:
            self.model = init_recognizer(cfg_file, checkpoint, device=self.device)
            bar.update(3)

        print(f"✅ PoseC3D 模型載入成功 ({self.device})\n")
        return True

    # === YOLO Pose JSON 轉換為 PoseC3D 可讀格式 ===
    def _convert_yolo_to_posec3d(self, yolo_json_path: str) -> str:
        """
        將 YOLOv11 Pose 的輸出格式轉換成 MMAction2 可用的 skeleton JSON。
        """
        with open(yolo_json_path, "r", encoding="utf-8") as f:
            yolo_data = json.load(f)

        frames = []
        for frame_data in yolo_data["pose_sequence"]:
            keypoints = np.array([[kp["x"], kp["y"]] for kp in frame_data["keypoints"]])
            frames.append(keypoints.tolist())

        converted = {
            "data": [
                {
                    "frame_dir": yolo_json_path,
                    "total_frames": len(frames),
                    "keypoint": [frames],
                    "label": -1,
                }
            ]
        }

        converted_path = str(Path(yolo_json_path).with_name(Path(yolo_json_path).stem + "_mmaction.json"))
        with open(converted_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

        return converted_path

    # === 分類推論 ===
    def classify_pose_json(self, json_path: str) -> dict:
        """
        Args:
            json_path: YOLOv11 Pose 輸出的 skeleton JSON
        Returns:
            dict: 動作分類結果（含儲存路徑）
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"❌ 找不到輸入 JSON：{json_path}")

        # === 轉換格式 ===
        mmaction_input = self._convert_yolo_to_posec3d(str(json_path))

        # === 推論 ===
        with tqdm(total=3, desc=f"PoseC3D 推論 {json_path.stem}", ncols=80, leave=False) as bar:
            result = inference_recognizer(self.model, mmaction_input)
            bar.update(3)

        # === 解析結果 ===
        if isinstance(result, (list, tuple)) and len(result) > 0:
            top_result = result[0]
            pred_label, pred_score = top_result[0], float(top_result[1])
        elif hasattr(result, "pred_label"):
            pred_label, pred_score = int(result.pred_label), float(result.pred_score)
        else:
            pred_label, pred_score = 0, 0.0

        label_map_path = self.config["models"]["posec3d"].get("label_map", {})
        label_name = label_map_path.get(str(pred_label), f"class_{pred_label}")

        classification_result = {
            "predicted_action": label_name,
            "predicted_label": pred_label,
            "confidence": pred_score,
            "inference_time": datetime.now().isoformat(),
            "model_info": {
                "config": self.config["models"]["posec3d"]["config_file"],
                "checkpoint": self.config["models"]["posec3d"]["checkpoint_file"],
            },
        }

        # === 儲存分類 JSON ===
        result_dir = Path("results")
        result_dir.mkdir(parents=True, exist_ok=True)
        output_path = result_dir / f"{json_path.stem}_classification.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(classification_result, f, ensure_ascii=False, indent=2)

        classification_result["classification_json_path"] = str(output_path)
        print(f"🧩 動作辨識完成：{label_name}（信心值 {pred_score:.3f}）")
        return classification_result

    # === 批次分類 ===
    def batch_classify(self, pose_json_dir: str) -> list:
        pose_json_dir = Path(pose_json_dir)
        json_files = list(pose_json_dir.glob("*_pose.json"))
        if not json_files:
            print(f"⚠️ 目錄 {pose_json_dir} 無可分析 JSON 檔案")
            return []

        print(f"📂 偵測到 {len(json_files)} 個 pose JSON，開始批次分類...\n")

        outputs = []
        with tqdm(total=len(json_files), desc="PoseC3D 批次分類", ncols=90) as bar:
            for jf in json_files:
                result = self.classify_pose_json(str(jf))
                outputs.append(result)
                bar.update(1)
        return outputs


def main():
    """簡易測試"""
    classifier = PoseClassifier()
    classifier.load_model()

    test_json = "pose_json/test_pose.json"
    if os.path.exists(test_json):
        result = classifier.classify_pose_json(test_json)
        print("✅ 測試結果：", result)
    else:
        print("❌ 找不到測試檔案")


if __name__ == "__main__":
    main()
