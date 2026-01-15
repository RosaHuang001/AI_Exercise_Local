#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_video_metrics.py

功能：
- 讀取 YOLO / Pose 分析後的 JSON 檔（例如 pose_json/ 底下）
- 依每支影片計算：
    - rep_count：總次數
    - total_duration：總長度(秒)
    - avg_cadence：每分鐘次數
    - avg_cycle_sec：每一下平均秒數
- 寫入 MySQL 資料表：hf_exercise.video_motion_metrics
    - 透過 videos.file_name 對應到 video_id
"""

import os
import json
import math
import argparse
from typing import Optional, Dict, Any, Tuple

import mysql.connector
from mysql.connector import Error

from dotenv import load_dotenv

load_dotenv()

# ================== DB 設定 ==================

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",        # 若有密碼請修改這裡
    "password": "",
    "database": "hf_exercise",
}


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


# ================== JSON → 指標計算核心 ==================

def compute_metrics_from_pose_json(json_data: Dict[str, Any]) -> Tuple[int, float]:
    """
    根據「你的」 pose JSON 結構，計算：
    - rep_count：總次數
    - total_duration：總長度 (秒)

    ⚠️ 這裡一定要依照你實際的 JSON 格式做修改。
    狗狗先示範兩種常見寫法，你選一種改成符合你實際欄位即可。
    """

    # ---- 範例 1：若 JSON 已經直接給你 rep_count & duration_sec ----
    # 例如：
    # {
    #   "video_path": "...",
    #   "rep_count": 14,
    #   "total_duration_sec": 32.5,
    #   ...
    # }
    if "rep_count" in json_data and "total_duration_sec" in json_data:
        rep_count = int(json_data["rep_count"])
        total_duration = float(json_data["total_duration_sec"])
        return rep_count, total_duration

    # ---- 範例 2：若 JSON 是 frame-based，裡面有 fps & 最後一個 frame 時間 ----
    # 例如：
    # {
    #   "fps": 30,
    #   "frames": [
    #       {"frame_index": 0, "is_rep": false},
    #       {"frame_index": 15, "is_rep": true},
    #       ...
    #   ]
    # }
    if "frames" in json_data and "fps" in json_data:
        fps = float(json_data.get("fps", 30.0))
        frames = json_data["frames"]

        # 假設：每一個 frame 裡，若 is_rep == true 就代表「動作峰值」
        rep_count = 0
        max_frame_index = 0

        for fr in frames:
            idx = int(fr.get("frame_index", 0))
            max_frame_index = max(max_frame_index, idx)
            if fr.get("is_rep"):
                rep_count += 1

        total_duration = max_frame_index / fps if fps > 0 else 0.0
        return rep_count, total_duration

    # ---- 如果都不是上述格式，就先丟錯，提醒你要來改這一段 ----
    raise ValueError(
        "無法從 JSON 推算 rep_count / total_duration，請修改 compute_metrics_from_pose_json() "
        "以符合你的 pose JSON 結構。"
    )


# ================== 寫入 MySQL 的工具 ==================

def get_video_id_by_filename(cursor, file_name: str) -> Optional[int]:
    """
    透過 videos.file_name 找 video_id
    """
    sql = "SELECT id FROM videos WHERE file_name = %s"
    cursor.execute(sql, (file_name,))
    row = cursor.fetchone()
    if row:
        return int(row[0])
    return None


def upsert_video_metrics(
    cursor,
    video_id: int,
    model_name: str,
    model_version: Optional[str],
    rep_count: int,
    total_duration: float,
    rep_label: Optional[str] = None,
    rom_score: Optional[float] = None,
    quality_score: Optional[float] = None,
):
    """
    將計算好的指標寫入 video_motion_metrics，如果同一支影片同一個 model_name+version 已存在就更新
    """
    if total_duration <= 0 or rep_count <= 0:
        avg_cadence = None
        avg_cycle_sec = None
    else:
        avg_cycle_sec = total_duration / rep_count
        avg_cadence = (rep_count / total_duration) * 60.0

    sql = """
    INSERT INTO video_motion_metrics (
        video_id, model_name, model_version,
        rep_label,
        rep_count, total_duration, avg_cadence, avg_cycle_sec,
        rom_score, quality_score
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        rep_label      = VALUES(rep_label),
        rep_count      = VALUES(rep_count),
        total_duration = VALUES(total_duration),
        avg_cadence    = VALUES(avg_cadence),
        avg_cycle_sec  = VALUES(avg_cycle_sec),
        rom_score      = VALUES(rom_score),
        quality_score  = VALUES(quality_score),
        updated_at     = CURRENT_TIMESTAMP
    """

    params = (
        video_id,
        model_name,
        model_version,
        rep_label,
        rep_count,
        total_duration,
        avg_cadence,
        avg_cycle_sec,
        rom_score,
        quality_score,
    )
    cursor.execute(sql, params)


# ================== 主流程 ==================

def main():
    parser = argparse.ArgumentParser(
        description="從 YOLO pose JSON 計算影片動作頻率，寫入 hf_exercise.video_motion_metrics"
    )
    parser.add_argument(
        "--pose-dir",
        type=str,
        default="pose_json",
        help="存放 YOLO pose JSON 的資料夾 (default: pose_json)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolov11n-pose",
        help="寫入 DB 的 model_name (default: yolov11n-pose)",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="寫入 DB 的 model_version (可留空)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只印出結果，不寫入資料庫",
    )

    args = parser.parse_args()

    pose_dir = args.pose_dir
    model_name = args.model_name
    model_version = args.model_version
    dry_run = args.dry_run

    if not os.path.isdir(pose_dir):
        print(f"[ERROR] 找不到 pose_json 目錄：{pose_dir}")
        return

    # 收集所有 .json 檔
    json_files = [
        os.path.join(pose_dir, f)
        for f in os.listdir(pose_dir)
        if f.lower().endswith(".json")
    ]

    if not json_files:
        print(f"[WARN] 目錄 {pose_dir} 裡沒有找到任何 JSON 檔。")
        return

    print(f"[INFO] 在 {pose_dir} 找到 {len(json_files)} 個 JSON 檔。")

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        processed = 0
        skipped_no_video = 0
        skipped_error = 0

        for path in json_files:
            file_name_only = os.path.basename(path)
            # 假設 JSON 檔名與影片檔名一樣（含 .mp4）
            # 如果實際情況是「仰臥踢腿_pose.json」，可在這裡做對應處理
            video_file_name = file_name_only.replace("_pose", "").replace(".json", ".mp4")

            print(f"\n[INFO] 處理 JSON：{file_name_only} -> 對應影片檔名：{video_file_name}")

            video_id = get_video_id_by_filename(cursor, video_file_name)
            if video_id is None:
                print(f"  [WARN] 在 videos 表找不到檔名 = {video_file_name}，略過。")
                skipped_no_video += 1
                continue

            # 讀 JSON
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [ERROR] 讀取 JSON 失敗：{e}")
                skipped_error += 1
                continue

            # 計算 rep_count & total_duration
            try:
                rep_count, total_duration = compute_metrics_from_pose_json(data)
            except Exception as e:
                print(f"  [ERROR] 計算指標失敗：{e}")
                skipped_error += 1
                continue

            print(f"  [OK] rep_count = {rep_count}, total_duration = {total_duration:.2f} 秒")

            if dry_run:
                print("  [DRY-RUN] 不寫入資料庫。")
            else:
                upsert_video_metrics(
                    cursor,
                    video_id=video_id,
                    model_name=model_name,
                    model_version=model_version,
                    rep_count=rep_count,
                    total_duration=total_duration,
                    rep_label=None,       # 若你有在 JSON 裡存動作名稱，可改成 data["label"]
                    rom_score=None,
                    quality_score=None,
                )
                processed += 1

        if not dry_run:
            conn.commit()
            print(f"\n[INFO] 完成寫入，共寫入 {processed} 筆。")

        print(f"\n[SUMMARY] 處理 JSON 檔：{len(json_files)}")
        print(f"          成功寫入：{processed}")
        print(f"          找不到對應影片：{skipped_no_video}")
        print(f"          讀檔 / 計算失敗：{skipped_error}")

    except Error as e:
        if conn:
            conn.rollback()
        print(f"[ERROR] MySQL 錯誤：{e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
