#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — AI 個人化運動推薦系統「介面層」
- 跟練兩種方式：① 本機彈窗（subprocess 跑 demo_guided_training_ui）② 瀏覽器攝影機（WebRTC，分享連結時用）。
- 需求：streamlit, streamlit-webrtc, av, opencv, ultralytics, scipy, fastdtw（WebRTC 跟練時）

執行：streamlit run app.py
"""

import os
import sys
import json
import glob
import subprocess
from typing import Optional

import streamlit as st

# WebRTC 與 YOLO 僅在「瀏覽器攝影機跟練」時使用
try:
    from streamlit_webrtc import webrtc_streamer
    from ultralytics import YOLO
    from webrtc_guided_processor import GuidedTrainingProcessor
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False


# ==============================
# 專案路徑（介面用）
# ==============================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
EXERCISE_DIR = os.path.join(PROJECT_ROOT, "exercise_videos")
MODEL_PATH = os.path.join(PROJECT_ROOT, "yolo11n-pose.pt")
REFERENCE_JSON_DIR = os.path.join(PROJECT_ROOT, "yolo_exercise_videos")
DEMO_SCRIPT = os.path.join(PROJECT_ROOT, "demo_guided_training_ui.py")


@st.cache_resource
def get_yolo_model():
    """YOLO 模型僅載入一次（供 WebRTC 跟練用）。"""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"找不到 YOLO 模型：{MODEL_PATH}")
    return YOLO(MODEL_PATH)


def launch_guided_training(video_path: Optional[str] = None) -> bool:
    """
    啟動跟練模組（demo_guided_training_ui.py）。
    video_path: 相對專案根或絕對路徑；None 則由 demo 自行從 exercise_videos 取第一支。
    回傳是否成功啟動。
    """
    if not os.path.isfile(DEMO_SCRIPT):
        return False
    cmd = [sys.executable, DEMO_SCRIPT]
    if video_path:
        path = video_path if os.path.isabs(video_path) else os.path.join(PROJECT_ROOT, video_path)
        if os.path.isfile(path):
            cmd.append(path)
    subprocess.Popen(cmd, cwd=PROJECT_ROOT)
    return True


# ==============================
# Streamlit 設定
# ==============================

st.set_page_config(page_title="AI 個人化運動推薦系統", layout="wide")


# ==============================
# Session 初始化
# ==============================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "recommendation_results" not in st.session_state:
    st.session_state.recommendation_results = [
        {
            "exercise_id": "深蹲訓練",
            "decision": "RECOMMENDED",
            "video": os.path.join("exercise_videos", "demo.mp4"),
            "recommendation_reason": {
                "hard_filter_pass_reasons": [{"description": "符合心血管安全條件"}],
                "soft_rank_positive_factors": [{"description": "強化下肢肌力"}],
                "soft_rank_penalty_factors": [{"description": "膝關節需注意角度"}],
            },
        }
    ]


# ==============================
# Sidebar 導覽
# ==============================

st.sidebar.title("🧠 系統導覽")

pages = ["Dashboard", "運動推薦", "跟練模式", "運動紀錄", "系統監控"]
try:
    default_index = pages.index(st.session_state.page)
except ValueError:
    default_index = 0

selected_page = st.sidebar.radio("功能選單", pages, index=default_index)
st.session_state.page = selected_page


# ==============================
# 各頁面（僅介面串接）
# ==============================

if st.session_state.page == "Dashboard":
    st.title("📊 系統總覽")

    col1, col2, col3 = st.columns(3)
    col1.metric("可用運動數量", len(st.session_state.recommendation_results))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    col2.metric("訓練紀錄數", len(glob.glob(os.path.join(RESULTS_DIR, "*.json"))))
    col3.metric("系統狀態", "正常運行")

    st.markdown("---")
    st.subheader("系統流程")
    st.markdown("""
    1️⃣ 使用者輸入健康條件  
    2️⃣ RAG 規則引擎篩選運動  
    3️⃣ YOLO 動作分析  
    4️⃣ 生成個人化推薦  
    5️⃣ 跟練與即時評估（由 **demo_guided_training_ui** 執行）  
    """)


elif st.session_state.page == "運動推薦":
    st.title("📋 系統推薦運動")

    results = st.session_state.recommendation_results
    if not results:
        st.info("尚無推薦結果")
    else:
        for r in results:
            with st.container():
                st.markdown(f"### 🏋 {r['exercise_id']}")
                st.markdown(f"**決策結果：** `{r['decision']}`")

                rr = r.get("recommendation_reason", {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("#### ✔ 通過條件")
                    for item in rr.get("hard_filter_pass_reasons", []):
                        st.write("-", item["description"])
                with col2:
                    st.markdown("#### ➕ 加分因素")
                    for item in rr.get("soft_rank_positive_factors", []):
                        st.write("-", item["description"])
                with col3:
                    st.markdown("#### ⚠ 注意事項")
                    for item in rr.get("soft_rank_penalty_factors", []):
                        st.write("-", item["description"])

                if st.button("開始跟練", key=f"start_{r['exercise_id']}"):
                    st.session_state.selected_video = r
                    st.session_state.page = "跟練模式"
                    st.rerun()
                st.markdown("---")


elif st.session_state.page == "跟練模式":
    st.title("🏃 跟練模式")

    selected = st.session_state.get("selected_video")
    if not selected:
        st.warning("請先從推薦頁選擇運動")
    else:
        st.subheader(f"目前運動：{selected['exercise_id']}")

        video_rel = selected["video"]
        video_path = video_rel if os.path.isabs(video_rel) else os.path.join(PROJECT_ROOT, video_rel)

        col_v, col_btn = st.columns([1, 1])
        with col_v:
            st.markdown("### 🎬 示範影片")
            if os.path.isfile(video_path):
                st.video(video_path)
            else:
                st.error(f"找不到示範影片：{video_path}")

        with col_btn:
            st.markdown("### ▶️ 啟動跟練")
            session_sec = st.slider("訓練秒數", 10, 120, 30, 5, key="session_sec")

            st.markdown("**方式一：本機視窗**（需在本機執行）")
            st.caption("開啟獨立跟練視窗，使用本機攝影機。")
            if st.button("啟動 AI 跟練（本機視窗）", key="launch_desktop"):
                if not os.path.isfile(video_path):
                    st.error("示範影片不存在，無法啟動。")
                elif launch_guided_training(video_path):
                    st.success("已啟動跟練視窗，請在彈出的視窗中操作。")
                else:
                    st.error("無法啟動跟練程式（請確認 demo_guided_training_ui.py 存在）。")

            st.markdown("---")
            st.markdown("**方式二：瀏覽器攝影機**（分享連結時用）")
            st.caption("使用您裝置的攝影機，在瀏覽器內即時跟練，適合分享連結給他人使用。")
            if not HAS_WEBRTC:
                st.warning("請安裝 streamlit-webrtc 與 av：`pip install streamlit-webrtc av`")
            elif not os.path.isfile(video_path):
                st.error("示範影片不存在。")
            else:
                if "show_webrtc" not in st.session_state:
                    st.session_state.show_webrtc = False
                if "webrtc_processor" in st.session_state and getattr(
                    st.session_state.webrtc_processor, "video_path", None
                ) != video_path:
                    st.session_state.webrtc_processor = None
                    st.session_state.show_webrtc = False

                if st.button("開始即時跟練（使用我的攝影機）", type="primary", key="start_webrtc"):
                    try:
                        model = get_yolo_model()
                        st.session_state.webrtc_processor = GuidedTrainingProcessor(
                            video_path=video_path,
                            model=model,
                            session_duration=int(session_sec),
                            results_dir=RESULTS_DIR,
                            project_root=PROJECT_ROOT,
                        )
                        st.session_state.show_webrtc = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"無法啟動：{e}")

                if st.session_state.get("show_webrtc") and st.session_state.get("webrtc_processor"):
                    st.info("請允許使用攝影機。訓練時間到會自動顯示結果並寫入紀錄。")
                    webrtc_streamer(
                        key="guided_webrtc",
                        video_frame_callback=lambda f: st.session_state.webrtc_processor.process(f),
                        media_stream_constraints={"video": True, "audio": False},
                        async_processing=False,
                    )


elif st.session_state.page == "運動紀錄":
    st.title("📈 訓練紀錄")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")), reverse=True)

    if not result_files:
        st.info("尚無訓練紀錄")
    else:
        for file in result_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            timestamp = os.path.basename(file).replace("session_", "").replace(".json", "")
            with st.expander(f"訓練時間：{timestamp}"):
                st.metric("平均相似度", f"{int(data.get('average_similarity', 0))}%")
                st.metric("完成次數", data.get("repetition_count", 0))
                st.metric("訓練時長", f"{int(data.get('duration_seconds', 0))} 秒")
                if data.get("video"):
                    st.write("示範影片：", data.get("video"))


elif st.session_state.page == "系統監控":
    st.title("🛠 系統監控")

    st.metric("影片資料夾", "正常" if os.path.exists(EXERCISE_DIR) else "缺失")
    st.metric("YOLO 模型", "存在" if os.path.exists(MODEL_PATH) else "缺失")
    st.metric("結果資料夾", "正常" if os.path.exists(RESULTS_DIR) else "尚未建立")
    st.metric("Reference JSON 資料夾", "正常" if os.path.exists(REFERENCE_JSON_DIR) else "缺失")
    st.metric("跟練模組", "存在" if os.path.isfile(DEMO_SCRIPT) else "缺失")

    st.markdown("---")
    st.subheader("系統資訊")
    st.write("Python 版本:", sys.version)
    st.write("專案目錄:", PROJECT_ROOT)
