# main.py
# --------------------------------------------------
# AI 個人化健康運動推薦系統（工程師直線版）
# --------------------------------------------------

import os
import sys
import json
import hashlib
from pprint import pprint

# ===== 推薦模組（前置安全篩選 + 個人化排序）=====
from modules.recommender_filter import (
    UserState,
    load_exercise_library,
    hard_filter_exercises,
    soft_rank_exercises
)

# ===== RAG =====
from rag.user_input import get_user_input
from rag.user_condition_mapper import build_user_context
from rag.rag_engine import ACSMRagEngine
from rag.rule_controller import RuleController

# ===== YOLO =====
from ultralytics import YOLO
from modules.yolo_pose_rep_counter import (
    yolo_process_one_video,
    pack_yolo_result
)

# ===== GPT =====
from modules.gpt_summary import call_openai_label, generate_weekly_plan


# ===== 基本設定 =====
VIDEO_DIR = "exercise_videos"
EXERCISE_LIBRARY_PATH = "knowledge_base/exercise_library.json"
EXERCISE_VIDEO_MAP_PATH = "knowledge_base/exercise_video_map.json"

KNOWLEDGE_PATH = "knowledge_base/hf_chunks.json"
YOLO_MODEL_PATH = "modules/yolo11n-pose.pt"
YOLO_OUTPUT_DIR = "results/yolo_videos"
OUTPUT_DIR = "results"
MAX_RULES = 4
CACHE_DIR = os.path.join(OUTPUT_DIR, "video_cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print("======================================")
print(" AI 個人化健康運動推薦系統 啟動")
print("======================================")


# --------------------------------------------------
# Step 1. 使用者輸入
# --------------------------------------------------
print("\n[Step 1] 收集使用者輸入")
user_input = get_user_input()
pprint(user_input)


# --------------------------------------------------
# Step 2. 使用者條件 mapping
# --------------------------------------------------
print("\n[Step 2] 使用者條件轉換")
user_context = build_user_context(user_input)
user_condition = user_context.get("user_conditions", {})
risk_assessment = user_context.get("risk_assessment", {})
pprint(user_context)


# --------------------------------------------------
# Step 2.5 前置動作推薦（Hard Filter + Soft Ranking）
# --------------------------------------------------
print("\n[Step 2.5] 前置安全篩選與個人化動作排序")

# 禁忌條件來源彈性整合（避免 key 不一致）
contraindications = (
    user_condition.get("contraindications")
    or risk_assessment.get("contraindications")
    or risk_assessment.get("risk_flags")
    or []
)

# 建立 UserState（給推薦引擎使用）
user_state = UserState(
    nyha=user_condition.get("nyha", ""),
    contraindications=contraindications
)

# 載入動作庫
exercise_library = load_exercise_library(EXERCISE_LIBRARY_PATH)

# Hard Filter：安全篩選
filtered = hard_filter_exercises(
    user=user_state,
    library=exercise_library
)

print(f"通過安全篩選的動作數量：{filtered['counts']['included']}")

# Soft Ranking：個人化排序
ranked_exercises = soft_rank_exercises(
    user=user_state,
    exercises=filtered["included"]
)

if not ranked_exercises:
    print("⚠️ 無任何動作通過安全篩選，停止後續分析")
    sys.exit(1)

print("\n【個人化排序後的動作推薦（含推薦理由）】")
for ex in ranked_exercises:
    print(f"\n- {ex['exercise_id']} | {ex['name_zh']} | score={ex['soft_rank_score']}")

    rr = ex.get("recommendation_reason", {})

    # Hard Filter 通過理由
    for item in rr.get("hard_filter_pass_reasons", []):
        print("  [Hard 通過]", item.get("description"))

    # Soft Ranking 正向因素
    for item in rr.get("soft_rank_positive_factors", []):
        print("  [+]", item.get("description"))

    # Soft Ranking 懲罰因素
    for item in rr.get("soft_rank_penalty_factors", []):
        print("  [-]", item.get("description"))



# --------------------------------------------------
# Step 3. RAG 撈 ACSM / HF 規則
# --------------------------------------------------
print("\n[Step 3] RAG 擷取 ACSM / HF 規則")

rag_engine = ACSMRagEngine(knowledge_path=KNOWLEDGE_PATH)
rule_controller = RuleController(max_rules=MAX_RULES)

population = user_condition.get("population")
condition = {**user_condition, "risk_level": risk_assessment.get("risk_level")}

rag_results = rag_engine.retrieve_rules(
    population=population,
    condition=condition
)

rules = rule_controller.process(
    rag_results,
    user_profile=risk_assessment
)

print(f"取得規則數量：{len(rules)}")


# --------------------------------------------------
# Step 4. 依推薦結果選擇影片（只分析被推薦者）
# --------------------------------------------------
print("\n[Step 4] 依個人化推薦結果選擇影片")

with open(EXERCISE_VIDEO_MAP_PATH, "r", encoding="utf-8") as f:
    exercise_video_map = json.load(f)

video_files = []
missing_videos = []

exercise_reason_map = {
    ex["exercise_id"]: ex.get("recommendation_reason") or {}
    for ex in ranked_exercises
}

# 存 (exercise_id, video_path)；支援一動作多支影片（如左/右側）
# exercise_video_map 值可為字串 "a.mp4" 或 陣列 ["左.mp4", "右.mp4"]
video_jobs = []
for ex in ranked_exercises:
    raw = exercise_video_map.get(ex["exercise_id"])
    filenames = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
    filenames = [f for f in filenames if f]
    if not filenames:
        missing_videos.append(ex["exercise_id"])
        continue
    added = 0
    for filename in filenames:
        video_path = os.path.join(VIDEO_DIR, filename)
        if os.path.exists(video_path):
            video_jobs.append({"exercise_id": ex["exercise_id"], "video_path": video_path})
            added += 1
    if added == 0:
        missing_videos.append(ex["exercise_id"])

video_files = [j["video_path"] for j in video_jobs]

print(f"實際分析影片數量：{len(video_files)}")
if missing_videos:
    print("⚠️ 找不到對應影片的動作：", missing_videos)


# --------------------------------------------------
# Step 5. 初始化 YOLO
# --------------------------------------------------
print("\n[Step 5] 初始化 YOLO Pose 模型")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("YOLO 初始化完成")


# --------------------------------------------------
# 快取：YOLO + GPT 結果（key = 影片路徑 + 模型版本；GPT 再加使用者 hash）
# --------------------------------------------------
def _cache_key_yolo(video_path: str, model_path: str) -> str:
    mtime = str(os.path.getmtime(model_path)) if os.path.exists(model_path) else "0"
    raw = f"{os.path.abspath(video_path)}|{mtime}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_key_gpt(video_path: str, model_path: str, user_condition: dict, risk_assessment: dict) -> str:
    yolo_part = _cache_key_yolo(video_path, model_path)
    # 只取會影響 GPT 輸出的欄位
    payload = {
        "nyha": user_condition.get("nyha"),
        "population": user_condition.get("population"),
        "risk_level": risk_assessment.get("risk_level"),
        "allow_exercise": risk_assessment.get("allow_exercise"),
    }
    raw = yolo_part + "|" + json.dumps(payload, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _json_safe(obj):
    """讓 numpy / 不可序列化型別可寫入 JSON。"""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def _load_cached_yolo(cache_key: str) -> dict | None:
    path = os.path.join(CACHE_DIR, f"yolo_{cache_key}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cached_yolo(cache_key: str, data: dict) -> None:
    path = os.path.join(CACHE_DIR, f"yolo_{cache_key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(data), f, ensure_ascii=False)
    except Exception:
        pass


def _load_cached_gpt(cache_key: str) -> dict | None:
    path = os.path.join(CACHE_DIR, f"gpt_{cache_key}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cached_gpt(cache_key: str, data: dict) -> None:
    path = os.path.join(CACHE_DIR, f"gpt_{cache_key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(data), f, ensure_ascii=False)
    except Exception:
        pass


# --------------------------------------------------
# 工具：格式化單支影片輸出
# --------------------------------------------------
def _one_line_takeaway(gpt_summary_text: str) -> str:
    """從 GPT 摘要取一句話重點（給使用者先看）。"""
    if not gpt_summary_text or not isinstance(gpt_summary_text, str):
        return ""
    s = gpt_summary_text.strip()
    for sep in "。", "；", ".", "\n":
        if sep in s:
            s = s.split(sep)[0].strip()
            if sep != "\n" and s and not s.endswith("。"):
                s += "。"
            break
    return (s[:100] + "…") if len(s) > 100 else s


def _video_side_label(video_path: str) -> str:
    """從檔名解析左/右側標示，供報告與計畫使用。"""
    name = os.path.splitext(os.path.basename(video_path))[0]
    if "左側" in name or "左" in name:
        return "左側"
    if "右側" in name or "右" in name:
        return "右側"
    return ""


def format_video_report(result: dict, index: int, total: int) -> str:
    video_name = os.path.basename(result["video"])
    side_label = _video_side_label(result["video"])
    title_suffix = f" · {side_label}" if side_label else ""
    yolo = result.get("yolo_result", {})
    gpt = result.get("gpt_summary", {})

    kinematics = yolo.get("primary_kinematics", {})
    impact = yolo.get("impact", {}) or {}

    rom = kinematics.get("rom_p5_p95") or 0
    reps_val = kinematics.get("reps") or 0
    freq = kinematics.get("frequency_hz") or 0
    intensity_m = kinematics.get("intensity_mean") or 0
    intensity_p = kinematics.get("intensity_p95") or 0

    gpt_raw = gpt.get("gpt_summary", "（無摘要）") or "（無摘要）"
    # 顯示用：重複的「體重越重…」只保留一次，避免刷屏
    _weight_phrase = "體重越重，換算出來的力量數字會越大，這只是幫不同體重的人對照理解"
    if _weight_phrase in gpt_raw:
        parts = gpt_raw.split(_weight_phrase)
        gpt_display = parts[0].strip() + _weight_phrase + "".join(p.strip().lstrip("。，") for p in parts[1:])
    else:
        gpt_display = gpt_raw

    takeaway = _one_line_takeaway(gpt_raw)

    lines = [
        "",
        "=" * 60,
        f"[{index}/{total}] 影片：{video_name}{title_suffix}",
        "=" * 60,
        "",
        "▶ 本支重點",
        f"- 系統建議：{result.get('decision', '')}",
        f"- 給您的建議：{takeaway}" if takeaway else "",
        "",
        "▶ 運動學與摘要",
        f"- 體位／區域：{yolo.get('posture', '—')} · {yolo.get('primary_region', '—')}",
        f"- 次數／頻率：{reps_val} 次 · {freq:.2f} 次/秒 · 關節衝擊 {impact.get('level', '—')}",
        "",
        gpt_display,
        "",
    ]
    rr = result.get("recommendation_reason") or {}
    if rr.get("soft_rank_positive_factors") or rr.get("soft_rank_penalty_factors"):
        lines.append("▶ 為何推薦此動作")
        for item in rr.get("soft_rank_positive_factors", []):
            lines.append(f"  [+] {item.get('description', '')}")
        for item in rr.get("soft_rank_penalty_factors", []):
            lines.append(f"  [-] {item.get('description', '')}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------
# 決策函式（安全寫法 + RAG 規則整合）
# --------------------------------------------------
def decide_exercise(user_condition, risk_assessment, yolo_result, rules):
    decision = "RECOMMEND"
    reasons = []

    primary_region = yolo_result.get("primary_region", "")
    impact_level = (yolo_result.get("impact") or {}).get("level", "未知")
    posture = yolo_result.get("posture", "")
    head_rom = float(yolo_result.get("head_rom_p5_p95") or 0)
    risk_level = risk_assessment.get("risk_level") or ""

    # 既有：下肢高衝擊
    if primary_region == "Lower" and impact_level == "高":
        decision = "CAUTION"
        reasons.append("偵測到下肢高衝擊運動")

    # 既有：高風險族群
    if risk_level in ("high", "very_high"):
        decision = "CAUTION"
        reasons.append("使用者屬於高風險族群")

    # RAG 整合：頭頸活動過大 + 規則有提及頭/眩暈/終止
    rule_texts = " ".join((r.get("rule") or "").lower() for r in (rules or []))
    if head_rom > 50 and any(kw in rule_texts for kw in ("head", "lightheaded", "dizziness", "termination", "terminate")):
        decision = "CAUTION"
        reasons.append("頭頸活動幅度較大，依安全規則建議謹慎")

    # RAG 整合：站姿 + 高風險，規則有平衡/下肢/動作模式
    rule_topics = {r.get("topic") or "" for r in (rules or [])}
    if posture == "Standing" and risk_level in ("high", "very_high"):
        if rule_topics & {"Movement Pattern", "Lower Limb Exercise", "Safety"}:
            decision = "CAUTION"
            reasons.append("站姿運動，高風險族群請注意平衡與症狀")

    if not reasons:
        reasons.append("未偵測到明顯禁忌條件")

    return decision, reasons


# --------------------------------------------------
# Step 6. YOLO 分析 + 規則判斷
# --------------------------------------------------
print("\n[Step 6] 分析影片並進行安全判斷")

all_results = []

for job in video_jobs:
    video_path = job["video_path"]
    exercise_id = job["exercise_id"]
    key_yolo = _cache_key_yolo(video_path, YOLO_MODEL_PATH)
    key_gpt = _cache_key_gpt(video_path, YOLO_MODEL_PATH, user_condition, risk_assessment)

    # 快取：YOLO
    yolo_from_cache = False
    yolo_result = _load_cached_yolo(key_yolo)
    if yolo_result is None:
        print(f"\n[YOLO] 分析影片：{video_path}")
        out_video, stats, duration_s = yolo_process_one_video(
            yolo_model,
            video_path,
            YOLO_OUTPUT_DIR
        )
        yolo_result = pack_yolo_result(out_video, stats, duration_s)
    else:
        yolo_from_cache = True
        print(f"\n[YOLO] 使用快取：{os.path.basename(video_path)}")

    # 攤平給 gpt_summary / decide（與快取無關，每次都做）
    pk = yolo_result.get("primary_kinematics") or {}
    imp = yolo_result.get("impact") or {}
    stats_ref = yolo_result
    for k in ("rom_p5_p95", "reps", "frequency_hz", "intensity_mean", "intensity_p95"):
        yolo_result.setdefault(k, pk.get(k) if k in pk else stats_ref.get(k))
    for k in ("head_rom_p5_p95", "head_frequency_hz", "weight_bearing"):
        yolo_result.setdefault(k, stats_ref.get(k))
    yolo_result.setdefault("impact_level", imp.get("level") or stats_ref.get("impact_level"))
    yolo_result.setdefault("impact_bw_low", imp.get("bw_low") or stats_ref.get("impact_bw_low"))
    yolo_result.setdefault("impact_bw_high", imp.get("bw_high") or stats_ref.get("impact_bw_high"))
    yolo_result.setdefault("impact_by_weight_bins_text", imp.get("by_weight_bins_text") or stats_ref.get("impact_by_weight_bins_text", ""))

    if not yolo_from_cache:
        _save_cached_yolo(key_yolo, yolo_result)

    decision, reasons = decide_exercise(
        user_condition,
        risk_assessment,
        yolo_result,
        rules
    )

    # 快取：GPT
    gpt_summary = _load_cached_gpt(key_gpt)
    if gpt_summary is None:
        gpt_summary = call_openai_label(
            file_name=video_path,
            duration_s=yolo_result.get("duration_s"),
            stats=yolo_result,
            activity_level=decision,
            user_condition=user_condition,
            risk_assessment=risk_assessment
        )
        _save_cached_gpt(key_gpt, gpt_summary)

    result = {
        "video": video_path,
        "exercise_id": exercise_id,
        "decision": decision,
        "reasons": reasons,
        "recommendation_reason": exercise_reason_map.get(exercise_id) or {},
        "final_decision_reason": {
            "exercise_level": exercise_reason_map.get(exercise_id) or {},
            "video_level": reasons,
        },
        "yolo_result": yolo_result,
        "gpt_summary": gpt_summary
    }


    all_results.append(result)
    print(format_video_report(result, len(all_results), len(video_files)))


# --------------------------------------------------
# Step 7. 一週 7 日個人化運動計畫 + 儲存
# --------------------------------------------------
print("\n[Step 7] 產出使用者個人化一週運動計畫")
weekly_plan = generate_weekly_plan(
    user_condition,
    risk_assessment,
    all_results
)

output_path = os.path.join(OUTPUT_DIR, "final_output.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "user_input": user_input,
            "user_condition": user_condition,
            "rules": rules,
            "results": all_results,
            "weekly_plan": weekly_plan,
        },
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n" + "=" * 60)
print("【您的本週運動計畫】")
print("=" * 60)
plan_text = (weekly_plan or {}).get("plan_text") or ""
if plan_text:
    print(plan_text)
else:
    print("（本週計畫已寫入 JSON，此處無文字摘要）")
print("=" * 60)
print("結果已輸出至：", output_path)
print("系統執行完成！！！")
