#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI_Exercise_Local - Video Processor (v3, tqdm progress)
YOLOv11 Pose 主角偵測模組
特性：
  - 單支影片逐幀進度條 (tqdm)
  - 主角選擇：最大面積 + 最接近中心
  - JSON & pose video 同步輸出
  - GPU only, 無防呆
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv


class VideoProcessor:
    """YOLOv11 Pose 影片處理器（含 tqdm 進度顯示與標準化 JSON 輸出）"""

    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv()
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.logger = self._setup_logger()
        self.model = None

    # === 日誌 ===
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("VideoProcessor")
        logger.setLevel(logging.INFO)
        log_file = Path(self.config["paths"]["logs_dir"]) / f"video_processor_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    # === 模型載入 ===
    def load_model(self) -> bool:
        model_path = self.config["models"]["yolov11_pose"]["model_path"]
        device = self.config["models"]["yolov11_pose"].get("device", "cuda")
        assert torch.cuda.is_available(), "❌ CUDA 不可用，請確認 GPU 驅動"
        self.logger.info(f"正在載入 YOLOv11 Pose 模型: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.logger.info(f"✅ 模型載入成功，使用設備: {device}")
        return True

    # === 主角選擇邏輯 ===
    def _get_main_person(self, results, frame_w, frame_h):
        boxes = results[0].boxes.xywh.cpu().numpy()
        keypoints_all = results[0].keypoints.xy.cpu().numpy()
        frame_center = np.array([frame_w / 2, frame_h / 2])

        main_idx, max_score = 0, -1
        for i, box in enumerate(boxes):
            x, y, w, h = box
            area = w * h
            center = np.array([x, y])
            dist_center = np.linalg.norm(center - frame_center)
            score = area / (dist_center + 1)
            if score > max_score:
                max_score = score
                main_idx = i
        return keypoints_all[main_idx]

    # === 處理單支影片 ===
    def process_video(self, video_path: str) -> Optional[str]:
        video_path = Path(video_path)
        self.logger.info(f"🎬 開始處理影片: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        max_duration = self.config["system"]["max_video_duration"]
        frame_skip = self.config["system"]["frame_skip"]

        # === 輸出影片設定 ===
        result_dir = Path(self.config["paths"]["results_dir"]) / "pose_videos"
        result_dir.mkdir(parents=True, exist_ok=True)
        out_path = result_dir / f"{video_path.stem}_pose.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        pose_sequence, frame_count, processed_frames = [], 0, 0

        # === tqdm 逐幀進度條 ===
        with tqdm(total=total_frames, desc=f"YOLO Pose {video_path.name}", ncols=80, leave=False) as bar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count / fps > max_duration:
                    break

                if frame_count % frame_skip == 0:
                    results = self.model(frame, conf=0.45, iou=0.45, max_det=5)
                    h, w, _ = frame.shape
                    main_person = self._get_main_person(results, w, h)
                    pose_data = {
                        "frame_number": frame_count,
                        "timestamp": frame_count / fps,
                        "keypoints": [{"id": i, "x": float(x), "y": float(y)} for i, (x, y) in enumerate(main_person)],
                    }
                    pose_sequence.append(pose_data)
                    processed_frames += 1

                    # 畫面輸出
                    annotated = results[0].plot()
                    out.write(annotated)
                    bar.update(frame_skip)
                else:
                    frame_count += 1
                    bar.update(1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # === 儲存 JSON ===
        output_path = self._save_pose_sequence(video_path, pose_sequence, {
            "fps": fps,
            "frames_total": total_frames,
            "frames_processed": processed_frames,
            "duration": duration,
            "pose_video": str(out_path),
        })

        self.logger.info(f"✅ 完成: {video_path.name} -> {output_path}")
        return str(output_path)

    # === 儲存 JSON ===
    def _save_pose_sequence(self, video_path: Path, pose_sequence: List[Dict], video_info: Dict) -> Path:
        output_dir = Path(self.config["paths"]["results_dir"]) / "pose_json"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{video_path.stem}_{timestamp}_pose.json"
        output_path = output_dir / output_filename

        output_data = {
            "video_info": {
                "filename": video_path.name,
                "original_path": str(video_path),
                **video_info,
            },
            "pose_sequence": pose_sequence,
            "summary": {
                "total_frames": video_info["frames_total"],
                "processed_frames": video_info["frames_processed"],
                "keypoints_per_frame": len(pose_sequence[0]["keypoints"]) if pose_sequence else 0,
            },
            "metadata": {
                "processed_at": datetime.now().isoformat(),
                "model": "YOLOv11 Pose (Main Person Mode)",
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return output_path

    # === 批次處理 ===
    def batch_process_videos(self, video_dir: str) -> List[str]:
        video_dir = Path(video_dir)
        exts = [".mp4", ".avi", ".mov", ".mkv"]
        files = [f for ext in exts for f in video_dir.glob(f"*{ext}")]
        outputs = []
        for fpath in files:
            result = self.process_video(str(fpath))
            if result:
                outputs.append(result)
        return outputs


def main():
    processor = VideoProcessor()
    processor.load_model()

    test_video = "videos/test.mp4"
    if os.path.exists(test_video):
        processor.process_video(test_video)
    else:
        print(f"❌ 找不到影片: {test_video}")


if __name__ == "__main__":
    main()
