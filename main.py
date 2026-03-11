# main.py
# --------------------------------------------------
# AI ?‹äºº?–å¥åº·é??•æ¨?¦ç³»çµ±ï?å·¥ç?å¸«ç›´ç·šç?ï¼?
# --------------------------------------------------

import os
import sys
import json
import hashlib
from pprint import pprint

# ===== ?¨è–¦æ¨¡ç?ï¼ˆå?ç½®å??¨ç¯©??+ ?‹äºº?–æ?åºï?=====
from modules.recommender_filter import (
    UserState,
    load_exercise_library,
    hard_filter_exercises,
    soft_rank_exercises
)

# ===== RAG =====
from rag_module.user_input import get_user_input
from rag_module.user_condition_mapper import build_user_context
from rag_module.rag_engine import ACSMRagEngine
from rag_module.rule_controller import RuleController

# ===== YOLO =====
from ultralytics import YOLO
from modules.yolo_pose_rep_counter import (
    yolo_process_one_video,
    pack_yolo_result
)

# ===== GPT =====
from modules.gpt_summary import call_openai_label, generate_weekly_plan


# ===== ?ºæœ¬è¨­å? =====
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
print(" AI ?‹äºº?–å¥åº·é??•æ¨?¦ç³»çµ??Ÿå?")
print("======================================")


# --------------------------------------------------
# Step 1. ä½¿ç”¨?…è¼¸??
# --------------------------------------------------
print("\n[Step 1] ?¶é?ä½¿ç”¨?…è¼¸??)
user_input = get_user_input()
pprint(user_input)


# --------------------------------------------------
# Step 2. ä½¿ç”¨?…æ?ä»?mapping
# --------------------------------------------------
print("\n[Step 2] ä½¿ç”¨?…æ?ä»¶è???)
user_context = build_user_context(user_input)
user_condition = user_context.get("user_conditions", {})
risk_assessment = user_context.get("risk_assessment", {})
pprint(user_context)


# --------------------------------------------------
# Step 2.5 ?ç½®?•ä??¨è–¦ï¼ˆHard Filter + Soft Rankingï¼?
# --------------------------------------------------
print("\n[Step 2.5] ?ç½®å®‰å…¨ç¯©é¸?‡å€‹äºº?–å?ä½œæ?åº?)

# ç¦å?æ¢ä»¶ä¾†æ?å½ˆæ€§æ•´?ˆï??¿å? key ä¸ä??´ï?
contraindications = (
    user_condition.get("contraindications")
    or risk_assessment.get("contraindications")
    or risk_assessment.get("risk_flags")
    or []
)

# å»ºç? UserStateï¼ˆçµ¦?¨è–¦å¼•æ?ä½¿ç”¨ï¼?
user_state = UserState(
    nyha=user_condition.get("nyha", ""),
    contraindications=contraindications
)

# è¼‰å…¥?•ä?åº?
exercise_library = load_exercise_library(EXERCISE_LIBRARY_PATH)

# Hard Filterï¼šå??¨ç¯©??
filtered = hard_filter_exercises(
    user=user_state,
    library=exercise_library
)

print(f"?šé?å®‰å…¨ç¯©é¸?„å?ä½œæ•¸?ï?{filtered['counts']['included']}")

# Soft Rankingï¼šå€‹äºº?–æ?åº?
ranked_exercises = soft_rank_exercises(
    user=user_state,
    exercises=filtered["included"]
)

if not ranked_exercises:
    print("? ï? ?¡ä»»ä½•å?ä½œé€šé?å®‰å…¨ç¯©é¸ï¼Œå?æ­¢å?çºŒå???)
    sys.exit(1)

print("\n?å€‹äºº?–æ?åºå??„å?ä½œæ¨?¦ï??«æ¨?¦ç??±ï???)
for ex in ranked_exercises:
    print(f"\n- {ex['exercise_id']} | {ex['name_zh']} | score={ex['soft_rank_score']}")

    rr = ex.get("recommendation_reason", {})

    # Hard Filter ?šé??†ç”±
    for item in rr.get("hard_filter_pass_reasons", []):
        print("  [Hard ?šé?]", item.get("description"))

    # Soft Ranking æ­??? ç?
    for item in rr.get("soft_rank_positive_factors", []):
        print("  [+]", item.get("description"))

    # Soft Ranking ?²ç½°? ç?
    for item in rr.get("soft_rank_penalty_factors", []):
        print("  [-]", item.get("description"))



# --------------------------------------------------
# Step 3. RAG ??ACSM / HF è¦å?
# --------------------------------------------------
print("\n[Step 3] RAG ?·å? ACSM / HF è¦å?")

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

print(f"?–å?è¦å??¸é?ï¼š{len(rules)}")


# --------------------------------------------------
# Step 4. ä¾æ¨?¦ç??œé¸?‡å½±?‡ï??ªå??è¢«?¨è–¦?…ï?
# --------------------------------------------------
print("\n[Step 4] ä¾å€‹äºº?–æ¨?¦ç??œé¸?‡å½±??)

with open(EXERCISE_VIDEO_MAP_PATH, "r", encoding="utf-8") as f:
    exercise_video_map = json.load(f)

video_files = []
missing_videos = []

exercise_reason_map = {
    ex["exercise_id"]: ex.get("recommendation_reason") or {}
    for ex in ranked_exercises
}

# å­?(exercise_id, video_path)ï¼›æ”¯?´ä??•ä?å¤šæ”¯å½±ç?ï¼ˆå?å·??³å´ï¼?
# exercise_video_map ?¼å¯?ºå?ä¸?"a.mp4" ????? ["å·?mp4", "??mp4"]
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

print(f"å¯¦é??†æ?å½±ç??¸é?ï¼š{len(video_files)}")
if missing_videos:
    print("? ï? ?¾ä??°å??‰å½±?‡ç??•ä?ï¼?, missing_videos)


# --------------------------------------------------
# Step 5. ?å???YOLO
# --------------------------------------------------
print("\n[Step 5] ?å???YOLO Pose æ¨¡å?")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("YOLO ?å??–å???)


# --------------------------------------------------
# å¿«å?ï¼šYOLO + GPT çµæ?ï¼ˆkey = å½±ç?è·¯å? + æ¨¡å??ˆæœ¬ï¼›GPT ?å?ä½¿ç”¨??hashï¼?
# --------------------------------------------------
def _cache_key_yolo(video_path: str, model_path: str) -> str:
    mtime = str(os.path.getmtime(model_path)) if os.path.exists(model_path) else "0"
    raw = f"{os.path.abspath(video_path)}|{mtime}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_key_gpt(video_path: str, model_path: str, user_condition: dict, risk_assessment: dict) -> str:
    yolo_part = _cache_key_yolo(video_path, model_path)
    # ?ªå??ƒå½±??GPT è¼¸å‡º?„æ?ä½?
    payload = {
        "nyha": user_condition.get("nyha"),
        "population": user_condition.get("population"),
        "risk_level": risk_assessment.get("risk_level"),
        "allow_exercise": risk_assessment.get("allow_exercise"),
    }
    raw = yolo_part + "|" + json.dumps(payload, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _json_safe(obj):
    """è®?numpy / ä¸å¯åºå??–å??¥å¯å¯«å…¥ JSON??""
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
# å·¥å…·ï¼šæ ¼å¼å??®æ”¯å½±ç?è¼¸å‡º
# --------------------------------------------------
def _one_line_takeaway(gpt_summary_text: str) -> str:
    """å¾?GPT ?˜è??–ä??¥è©±?é?ï¼ˆçµ¦ä½¿ç”¨?…å??‹ï???""
    if not gpt_summary_text or not isinstance(gpt_summary_text, str):
        return ""
    s = gpt_summary_text.strip()
    for sep in "??, "ï¼?, ".", "\n":
        if sep in s:
            s = s.split(sep)[0].strip()
            if sep != "\n" and s and not s.endswith("??):
                s += "??
            break
    return (s[:100] + "??) if len(s) > 100 else s


def _video_side_label(video_path: str) -> str:
    """å¾æ??è§£?å·¦/?³å´æ¨™ç¤ºï¼Œä??±å??‡è??«ä½¿?¨ã€?""
    name = os.path.splitext(os.path.basename(video_path))[0]
    if "å·¦å´" in name or "å·? in name:
        return "å·¦å´"
    if "?³å´" in name or "?? in name:
        return "?³å´"
    return ""


def format_video_report(result: dict, index: int, total: int) -> str:
    video_name = os.path.basename(result["video"])
    side_label = _video_side_label(result["video"])
    title_suffix = f" Â· {side_label}" if side_label else ""
    yolo = result.get("yolo_result", {})
    gpt = result.get("gpt_summary", {})

    kinematics = yolo.get("primary_kinematics", {})
    impact = yolo.get("impact", {}) or {}

    rom = kinematics.get("rom_p5_p95") or 0
    reps_val = kinematics.get("reps") or 0
    freq = kinematics.get("frequency_hz") or 0
    intensity_m = kinematics.get("intensity_mean") or 0
    intensity_p = kinematics.get("intensity_p95") or 0

    gpt_raw = gpt.get("gpt_summary", "ï¼ˆç„¡?˜è?ï¼?) or "ï¼ˆç„¡?˜è?ï¼?
    # é¡¯ç¤º?¨ï??è??„ã€Œé??è??â€¦ã€åªä¿ç?ä¸€æ¬¡ï??¿å??·å?
    _weight_phrase = "é«”é?è¶Šé?ï¼Œæ?ç®—å‡ºä¾†ç??›é??¸å??ƒè?å¤§ï??™åª?¯å¹«ä¸å?é«”é??„äººå°ç…§?†è§£"
    if _weight_phrase in gpt_raw:
        parts = gpt_raw.split(_weight_phrase)
        gpt_display = parts[0].strip() + _weight_phrase + "".join(p.strip().lstrip("?‚ï?") for p in parts[1:])
    else:
        gpt_display = gpt_raw

    takeaway = _one_line_takeaway(gpt_raw)

    lines = [
        "",
        "=" * 60,
        f"[{index}/{total}] å½±ç?ï¼š{video_name}{title_suffix}",
        "=" * 60,
        "",
        "???¬æ”¯?é?",
        f"- ç³»çµ±å»ºè­°ï¼š{result.get('decision', '')}",
        f"- çµ¦æ‚¨?„å»ºè­°ï?{takeaway}" if takeaway else "",
        "",
        "???‹å?å­¸è??˜è?",
        f"- é«”ä?ï¼å??Ÿï?{yolo.get('posture', '??)} Â· {yolo.get('primary_region', '??)}",
        f"- æ¬¡æ•¸ï¼é »?‡ï?{reps_val} æ¬?Â· {freq:.2f} æ¬?ç§?Â· ?œç?è¡æ? {impact.get('level', '??)}",
        "",
        gpt_display,
        "",
    ]
    rr = result.get("recommendation_reason") or {}
    if rr.get("soft_rank_positive_factors") or rr.get("soft_rank_penalty_factors"):
        lines.append("???ºä??¨è–¦æ­¤å?ä½?)
        for item in rr.get("soft_rank_positive_factors", []):
            lines.append(f"  [+] {item.get('description', '')}")
        for item in rr.get("soft_rank_penalty_factors", []):
            lines.append(f"  [-] {item.get('description', '')}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------
# æ±ºç??½å?ï¼ˆå??¨å¯«æ³?+ RAG è¦å??´å?ï¼?
# --------------------------------------------------
def decide_exercise(user_condition, risk_assessment, yolo_result, rules):
    decision = "RECOMMEND"
    reasons = []

    primary_region = yolo_result.get("primary_region", "")
    impact_level = (yolo_result.get("impact") or {}).get("level", "?ªçŸ¥")
    posture = yolo_result.get("posture", "")
    head_rom = float(yolo_result.get("head_rom_p5_p95") or 0)
    risk_level = risk_assessment.get("risk_level") or ""

    # ?¢æ?ï¼šä??¢é?è¡æ?
    if primary_region == "Lower" and impact_level == "é«?:
        decision = "CAUTION"
        reasons.append("?µæ¸¬?°ä??¢é?è¡æ??‹å?")

    # ?¢æ?ï¼šé?é¢¨éšª?ç¾¤
    if risk_level in ("high", "very_high"):
        decision = "CAUTION"
        reasons.append("ä½¿ç”¨?…å±¬?¼é?é¢¨éšª?ç¾¤")

    # RAG ?´å?ï¼šé ­?¸æ´»?•é?å¤?+ è¦å??‰æ??Šé ­/?©æ?/çµ‚æ­¢
    rule_texts = " ".join((r.get("rule") or "").lower() for r in (rules or []))
    if head_rom > 50 and any(kw in rule_texts for kw in ("head", "lightheaded", "dizziness", "termination", "terminate")):
        decision = "CAUTION"
        reasons.append("?­é ¸æ´»å?å¹…åº¦è¼ƒå¤§ï¼Œä?å®‰å…¨è¦å?å»ºè­°è¬¹æ?")

    # RAG ?´å?ï¼šç?å§?+ é«˜é¢¨?ªï?è¦å??‰å¹³è¡?ä¸‹è‚¢/?•ä?æ¨¡å?
    rule_topics = {r.get("topic") or "" for r in (rules or [])}
    if posture == "Standing" and risk_level in ("high", "very_high"):
        if rule_topics & {"Movement Pattern", "Lower Limb Exercise", "Safety"}:
            decision = "CAUTION"
            reasons.append("ç«™å§¿?‹å?ï¼Œé?é¢¨éšª?ç¾¤è«‹æ³¨?å¹³è¡¡è??‡ç?")

    if not reasons:
        reasons.append("?ªåµæ¸¬åˆ°?é¡¯ç¦å?æ¢ä»¶")

    return decision, reasons


# --------------------------------------------------
# Step 6. YOLO ?†æ? + è¦å??¤æ–·
# --------------------------------------------------
print("\n[Step 6] ?†æ?å½±ç?ä¸¦é€²è?å®‰å…¨?¤æ–·")

all_results = []

for job in video_jobs:
    video_path = job["video_path"]
    exercise_id = job["exercise_id"]
    key_yolo = _cache_key_yolo(video_path, YOLO_MODEL_PATH)
    key_gpt = _cache_key_gpt(video_path, YOLO_MODEL_PATH, user_condition, risk_assessment)

    # å¿«å?ï¼šYOLO
    yolo_from_cache = False
    yolo_result = _load_cached_yolo(key_yolo)
    if yolo_result is None:
        print(f"\n[YOLO] ?†æ?å½±ç?ï¼š{video_path}")
        out_video, stats, duration_s = yolo_process_one_video(
            yolo_model,
            video_path,
            YOLO_OUTPUT_DIR
        )
        yolo_result = pack_yolo_result(out_video, stats, duration_s)
    else:
        yolo_from_cache = True
        print(f"\n[YOLO] ä½¿ç”¨å¿«å?ï¼š{os.path.basename(video_path)}")

    # ?¤å¹³çµ?gpt_summary / decideï¼ˆè?å¿«å??¡é?ï¼Œæ?æ¬¡éƒ½?šï?
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

    # å¿«å?ï¼šGPT
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
# Step 7. ä¸€??7 ?¥å€‹äºº?–é??•è???+ ?²å?
# --------------------------------------------------
print("\n[Step 7] ?¢å‡ºä½¿ç”¨?…å€‹äºº?–ä??±é??•è???)
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
print("?æ‚¨?„æœ¬?±é??•è??«ã€?)
print("=" * 60)
plan_text = (weekly_plan or {}).get("plan_text") or ""
if plan_text:
    print(plan_text)
else:
    print("ï¼ˆæœ¬?±è??«å·²å¯«å…¥ JSONï¼Œæ­¤?•ç„¡?‡å??˜è?ï¼?)
print("=" * 60)
print("çµæ?å·²è¼¸?ºè‡³ï¼?, output_path)
print("ç³»çµ±?·è?å®Œæ?ï¼ï?ï¼?)
