import hashlib
from datetime import datetime
from contextlib import contextmanager

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules import user_profile
from modules.planner_agent import generate_planner_conditions
from modules.recommendation_engine import generate_weekly_plan, get_weekly_plan
from modules.training_session import run_training_session
from modules.database import execute, fetch_all
from modules.exercise_library import get_exercise
from modules.coach_agent import coach_response


load_dotenv()
st.set_page_config(page_title="HF Rehab Coach", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
<style>
body {
    background-color: #FFF6F2;
}
header[data-testid="stHeader"] {
    display: none !important;
}
section[data-testid="stSidebar"] {
    background-color: #FFF1EB !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}
.main {
    background-color: #FFF6F2;
}
.hf-container {
    max-width: 520px;
    margin: 0 auto;
    padding: 10px;
}
.hf-section {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 18px;
    border: 1px solid #FAD8D3;
    box-shadow: 0 8px 20px rgba(250, 162, 143, 0.18);
}
.hf-section h3 {
    margin: 0 0 10px 0;
    font-weight: 700;
    color: #B8483A;
}
.hf-item {
    margin: 6px 0;
    font-size: 15px;
}
.hf-badge-info {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 12px;
    background: #E6F6FF;
    color: #2A6C94;
    font-weight: 600;
    margin-top: 8px;
}
.hf-badge-warn {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 12px;
    background: #FFF5F0;
    color: #B53030;
    font-weight: 600;
    margin-top: 8px;
}
.hf-button {
    width: 100%;
    border-radius: 16px;
    border: none;
    padding: 12px;
    font-weight: 700;
    background: linear-gradient(135deg, #FFB4A2, #FF8F70);
    color: white;
    box-shadow: 0 6px 14px rgba(255, 143, 112, 0.36);
    cursor: pointer;
}
.hf-button:active {
    transform: scale(0.98);
}
.hf-tabs {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 12px;
}
.hf-tab {
    flex: 1;
    text-align: center;
    padding: 12px 12px;
    border-radius: 20px;
    font-weight: 600;
    cursor: pointer;
    background: #FFE6DF;
    color: #B8483A;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}
.hf-tab.active {
    background: #FFB4A2;
    color: white;
}
.hf-hamburger {
    position: fixed;
    top: 16px;
    left: 16px;
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: rgba(255, 180, 162, 0.9);
    box-shadow: 0 6px 18px rgba(255, 143, 112, 0.35);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    z-index: 999;
}
.hf-hamburger span {
    width: 60%;
    height: 3px;
    background: white;
    border-radius: 2px;
}
.stTabs [role="tab"] {
    background: #FFE6DF;
    color: #B8483A;
    border-radius: 24px;
    padding: 16px 28px;
    font-weight: 700;
    font-size: 18px;
    box-shadow: 0 6px 18px rgba(255, 143, 112, 0.25);
    margin-right: 12px;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: #FFB4A2;
    color: white;
    box-shadow: 0 10px 24px rgba(255, 143, 112, 0.35);
}
div.stButton>button {
    width: 100%;
    border-radius: 16px;
    padding: 12px;
    font-weight: 700;
    background: linear-gradient(135deg, #FFB4A2, #FF8F70);
    color: white;
    border: none;
    box-shadow: 0 6px 14px rgba(255, 143, 112, 0.36);
}
div.stButton>button:focus:not(:active) {
    box-shadow: 0 0 0 0.2rem rgba(255, 143, 112, 0.35);
}
input, textarea, select {
    border-radius: 10px !important;
    border: 1px solid #F5B5B5 !important;
}
@media (max-width: 640px) {
    .css-18e3th9 {
        padding: 0 !important;
    }
    .hf-container {
        padding-top: 10px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


@contextmanager
def hf_container():
    st.markdown('<div class="hf-container">', unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)


@contextmanager
def hf_section(title: str | None = None):
    st.markdown('<div class="hf-section">', unsafe_allow_html=True)
    if title:
        st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)


def render_hamburger_button():
    st.markdown(
        """
        <script>
        function toggleSidebar(){
            const doc = window.parent.document;
            const btn = doc.querySelector('button[kind="header"]');
            if(btn){btn.click();}
        }
        </script>
        <div class="hf-hamburger" onclick="toggleSidebar()">
            <span></span>
            <span></span>
            <span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "latest_filters" not in st.session_state:
    st.session_state["latest_filters"] = None
if "planner_chat" not in st.session_state:
    st.session_state["planner_chat"] = [
        {
            "role": "assistant",
            "content": "👋 歡迎使用心衰竭個人化運動規劃助手！請描述目前的身體狀況、希望調整的運動內容或特別注意事項，我會依據您的資料給出建議。",
        }
    ]
if "prescription_generated_at" not in st.session_state:
    st.session_state["prescription_generated_at"] = None
if "checkup_values" not in st.session_state:
    st.session_state["checkup_values"] = {"systolic": 110, "diastolic": 70, "spo2": 98}




def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def log_usage(user_id: int, page: str, action: str):
    execute(
        "INSERT INTO usage_logs (user_id, page_name, action, timestamp) VALUES (%s,%s,%s,%s)",
        (user_id, page, action, datetime.utcnow()),
    )


def log_login(user_id: int, device_info: str):
    execute(
        "INSERT INTO login_logs (user_id, login_time, device_info) VALUES (%s,%s,%s)",
        (user_id, datetime.utcnow(), device_info),
    )


def update_session_user():
    if "user" in st.session_state and st.session_state["user"]:
        st.session_state["user"] = user_profile.get_user(st.session_state["user"]["id"])


def login_view():
    with hf_section("登入"):
        username = st.text_input("使用者帳號")
        password = st.text_input("密碼", type="password")
        if st.button("登入"):
            user = user_profile.get_user_by_username(username)
            if user and user["password_hash"] == hash_password(password):
                st.session_state["user"] = user
                log_login(user["id"], "web")
                st.success("登入成功")
                st.rerun()
            else:
                st.error("帳號或密碼錯誤")


def register_view():
    with hf_section("註冊"):
        username = st.text_input("新帳號")
        password = st.text_input("設定密碼", type="password")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("年齡", min_value=18, max_value=100, value=50)
            nyha = st.selectbox("NYHA 分級", ["I", "II", "III", "IV"])
        with col2:
            sex = st.selectbox("性別", ["Male", "Female"])
            goals = st.text_input("運動目標")
        comorbidities = st.text_area("共病")
        surgery = st.text_area("術後部位 / 手術史")
        if st.button("建立帳號"):
            profile = {
                "age": age,
                "sex": sex,
                "nyha_class": nyha,
                "comorbidities": comorbidities,
                "surgery_history": surgery,
                "exercise_goals": goals,
            }
            user_profile.create_user(username, hash_password(password), profile)
            st.success("註冊成功，請登入")


def render_profile_sidebar():
    user = user_profile.get_user(st.session_state["user"]["id"])
    st.sidebar.markdown("### 個人資料")
    with st.sidebar.form("profile_form"):
        age = st.number_input("年齡", 18, 100, user["age"] or 50, key="profile_age")
        sex_value = user.get("sex") or "Male"
        sex = st.selectbox("性別", ["Male", "Female"], index=0 if sex_value == "Male" else 1, key="profile_sex")
        nyha_value = user.get("nyha_class") or "II"
        nyha = st.selectbox(
            "NYHA", ["I", "II", "III", "IV"], index=["I", "II", "III", "IV"].index(nyha_value), key="profile_nyha"
        )
        comorbidities = st.text_area("共病", value=user.get("comorbidities", ""), key="profile_comorbidities")
        surgery = st.text_area("術後部位", value=user.get("surgery_history", ""), key="profile_surgery")
        goals = st.text_area("運動目標", value=user.get("exercise_goals", ""), key="profile_goals")
        submitted = st.form_submit_button("儲存資料")
    if submitted:
        user_profile.update_user_profile(
            user["id"],
            {
                "age": age,
                "sex": sex,
                "nyha_class": nyha,
                "comorbidities": comorbidities,
                "surgery_history": surgery,
                "exercise_goals": goals,
            },
        )
        update_session_user()
        log_usage(user["id"], "Profile", "update")
        st.sidebar.success("已更新")


def render_history_sidebar():
    st.sidebar.markdown("### 歷程紀錄")
    sessions = fetch_all(
        "SELECT session_id, start_time, end_time FROM sessions WHERE user_id=%s ORDER BY start_time DESC LIMIT 10",
        (st.session_state["user"]["id"],),
    )
    if sessions:
        st.sidebar.write(pd.DataFrame(sessions))
    else:
        st.sidebar.info("尚無訓練紀錄")


def rehab_assistant_page():
    with hf_section("復健助理"):
        for msg in st.session_state["planner_chat"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_prompt = st.chat_input("輸入希望調整的運動內容...", key="planner_chat_input")
        if user_prompt:
            st.session_state["planner_chat"].append({"role": "user", "content": user_prompt})
            user = st.session_state["user"]
            filters = generate_planner_conditions(user, user_prompt)
            weekly_plan = generate_weekly_plan(user["id"], filters, filters.get("fitt", {}))
            st.session_state["latest_filters"] = filters
            st.session_state["prescription_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            log_usage(user["id"], "Rehab Assistant", "generate_plan")
            fitt = filters.get("fitt", {})
            response = "\n".join(
                [
                    "以下是根據您輸入的建議：",
                    f"- 頻率：{fitt.get('frequency_per_week', '3')} 天/週",
                    f"- 強度：{fitt.get('intensity_level', '低強度')}",
                    f"- 時間：{fitt.get('session_duration_minutes', '15')} 分鐘/次",
                    f"- 類型：{filters.get('acsm_type', 'resistance')} / {filters.get('body_region', 'core')}",
                    f"- 避免：{filters.get('must_avoid_tags', '無特別限定')}",
                    f"- 優先：{filters.get('preferred_tags', '依現有動作')}",
                    "已同步更新一週行程，請留意任何胸痛、頭暈或呼吸困難。",
                ]
            )
            st.session_state["planner_chat"].append({"role": "assistant", "content": response})
            st.rerun()


def plan_page():
    with hf_container():
        with hf_section("個人化運動處方"):
            generated_at = st.session_state.get("prescription_generated_at")
            if generated_at:
                st.markdown(f"<div class='hf-item'>📅 最近更新日期：{generated_at}</div>", unsafe_allow_html=True)
            filters = st.session_state.get("latest_filters")
            if filters:
                fitt = filters.get("fitt", {})
                st.markdown(
                    f"""
                    <div class="hf-item"><b>頻率：</b>{fitt.get("frequency_per_week", "3")} 天/週</div>
                    <div class="hf-item"><b>強度：</b>{fitt.get("intensity_level", "低強度")}</div>
                    <div class="hf-item"><b>時間：</b>{fitt.get("session_duration_minutes", "15")} 分鐘/次</div>
                    <div class="hf-item"><b>類型：</b>{filters.get("acsm_type", "resistance")} / {filters.get("body_region", "core")}</div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("尚未產生任何處方，請先到「復健助理」頁面輸入需求。")
            st.markdown(
                '<div class="hf-badge-warn">若出現胸痛、頭暈或呼吸困難，請立即停止並就醫。</div>',
                unsafe_allow_html=True,
            )


def checkup_page():
    with hf_container():
        with hf_section("運動前自我檢核"):
            values = st.session_state["checkup_values"]
            col_sys, col_dia, col_spo2 = st.columns(3)
            values["systolic"] = col_sys.number_input("今日收縮壓 (mmHg)", 80, 200, values["systolic"])
            values["diastolic"] = col_dia.number_input("今日舒張壓 (mmHg)", 40, 140, values["diastolic"])
            values["spo2"] = col_spo2.number_input("今日血氧 (%)", 70, 100, values["spo2"])
            st.session_state["checkup_values"] = values

            checks = [
                "無發燒、無呼吸道感染",
                "最近三日體重未增加超過 2 公斤",
                "無胸痛或不尋常呼吸困難",
                "無明顯頭暈 / 暈厥史",
                f"血壓 {values['systolic']}-{values['diastolic']} mmHg",
                f"血氧 {values['spo2']}%",
            ]
            for item in checks:
                st.markdown(f'<div class="hf-item">✅ {item}</div>', unsafe_allow_html=True)
            within_bp = 90 <= values["systolic"] <= 180 and 60 <= values["diastolic"] <= 110
            within_spo2 = values["spo2"] >= 92
            if within_bp and within_spo2:
                st.markdown('<div class="hf-badge-info">可進行今天的運動，請持續監測症狀</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="hf-badge-warn">血壓或血氧超出安全範圍，請先諮詢醫療人員</div>', unsafe_allow_html=True)


def video_page():
    with hf_container():
        with hf_section("推薦影片"):
            plan = get_weekly_plan(st.session_state["user"]["id"])
            if not plan:
                st.info("尚未建立週計畫，請先在「復健助理」頁面輸入需求。")
            else:
                for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
                    slots = plan.get(day, [])
                    if slots:
                        st.markdown(f"**{day}**")
                        for slot in slots:
                            st.markdown(
                                f"<div class='hf-item'>🎬 {slot['name']} · {slot['reps_or_duration']} · 強度 {slot['intensity_level']}</div>",
                                unsafe_allow_html=True,
                            )
    training_page()
    coach_page()


def training_page():
    with hf_container():
        with hf_section("訓練模式"):
            plan = get_weekly_plan(st.session_state["user"]["id"])
            options = []
            for day, items in plan.items():
                for item in items:
                    options.append((day, item))
            if not options:
                st.info("請先建立每週計畫")
                return
            labels = [f"{d} - {item['name']}" for d, item in options]
            idx = st.selectbox("選擇訓練內容", range(len(labels)), format_func=lambda i: labels[i], key="train_select_main")
            holder = st.empty()
            if st.button("開始訓練", key="start_training_main"):
                day, selected = options[idx]
                exercise = get_exercise(selected["exercise_id"])
                video_path = exercise["local_path"]
                plan_id = selected["plan_id"]
                run_training_session(
                    st.session_state["user"]["id"],
                    video_path,
                    plan_id=plan_id,
                    streamlit_slot=holder,
                )
                log_usage(st.session_state["user"]["id"], "Training", f"session_{plan_id}")


def coach_page():
    with hf_container():
        with hf_section("醫療教練"):
            plan_dict = get_weekly_plan(st.session_state["user"]["id"])
            flat_plan = []
            for day, entries in plan_dict.items():
                for item in entries:
                    flat_plan.append(
                        {
                            "day": day,
                            "exercise": item["name"],
                            "sets": item["sets"],
                            "duration": item["reps_or_duration"],
                            "intensity": item["intensity_level"],
                        }
                    )
            user_message = st.text_area("想對教練說的話（可留空）", key="coach_input_main")
            if st.button("取得建議", key="coach_button_main"):
                response = coach_response(flat_plan, user_message or None)
                log_usage(st.session_state["user"]["id"], "Coach", "message")
                st.write(response)


def main_view():
    render_hamburger_button()
    st.sidebar.title("HF Rehab System")
    render_profile_sidebar()
    render_history_sidebar()

    tabs = st.tabs(["💬 復健助理", "📋 運動處方", "🛡️ 自我檢核", "🎬 推薦影片"])
    with tabs[0]:
        rehab_assistant_page()
    with tabs[1]:
        plan_page()
    with tabs[2]:
        checkup_page()
    with tabs[3]:
        video_page()
    if st.sidebar.button("登出"):
        st.session_state.clear()
        st.rerun()


def main():
    if "user" not in st.session_state or not st.session_state["user"]:
        with hf_container():
            tab1, tab2 = st.tabs(["登入", "註冊"])
            with tab1:
                login_view()
            with tab2:
                register_view()
    else:
        main_view()


if __name__ == "__main__":
    main()
