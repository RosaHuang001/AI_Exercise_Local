import hashlib
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from modules import user_profile
from modules.coach_agent import coach_response
from modules.database import execute, fetch_all, fetch_one
from modules.exercise_library import get_exercise, list_exercises
from modules.planner_agent import generate_planner_conditions
from modules.recommendation_engine import generate_weekly_plan, get_weekly_plan
from modules.training_session import close_session, ingest_pose_frame, start_session, summarize_session


def get_settings() -> Dict[str, Any]:
  """Load API runtime settings from environment variables."""
  return {
      "secret_key": os.getenv("API_SECRET", "hf-exercise-secret"),
      "token_exp_hours": int(os.getenv("API_TOKEN_EXP_HOURS", "12")),
      "cors_origins": [origin.strip() for origin in os.getenv("API_CORS_ORIGINS", "*").split(",")],
  }


SETTINGS = get_settings()
app = FastAPI(title="HF Rehab Coach API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory="exercise_videos"), name="media")

security = HTTPBearer()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
  return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_access_token(user_id: int) -> str:
  payload = {
      "sub": str(user_id),
      "exp": datetime.utcnow() + timedelta(hours=SETTINGS["token_exp_hours"]),
  }
  return jwt.encode(payload, SETTINGS["secret_key"], algorithm="HS256")


def decode_token(token: str) -> dict:
  try:
    return jwt.decode(token, SETTINGS["secret_key"], algorithms=["HS256"])
  except jwt.PyJWTError as exc:
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
  payload = decode_token(credentials.credentials)
  user = user_profile.get_user(int(payload["sub"]))
  if not user:
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
  return user


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
  username: str = Field(min_length=3, max_length=64)
  password: str = Field(min_length=4, max_length=128)
  age: Optional[int] = Field(default=None, ge=18, le=100)
  sex: Optional[str] = Field(default=None, description="Male / Female")
  nyha_class: Optional[str] = Field(default="II")
  comorbidities: Optional[str] = None
  surgery_history: Optional[str] = None
  exercise_goals: Optional[str] = None


class TokenResponse(BaseModel):
  access_token: str
  token_type: str = "bearer"


class LoginRequest(BaseModel):
  username: str
  password: str


class UpdateProfileRequest(BaseModel):
  age: Optional[int] = Field(default=None, ge=18, le=100)
  sex: Optional[str] = None
  nyha_class: Optional[str] = None
  comorbidities: Optional[str] = None
  surgery_history: Optional[str] = None
  exercise_goals: Optional[str] = None


class PlannerRequest(BaseModel):
  message: str


class GeneratePlanRequest(BaseModel):
  filters: Dict[str, Any]


class CoachRequest(BaseModel):
  message: Optional[str] = None


class StartSessionRequest(BaseModel):
  plan_id: Optional[int] = None
  exercise_id: Optional[int] = None


class PoseFrameRequest(BaseModel):
  frame_base64: str = Field(min_length=10)
  exercise_id: Optional[int] = None
  source: Optional[str] = Field(default="unknown")


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------


@app.post("/auth/register", status_code=201)
def register(payload: RegisterRequest):
  if user_profile.get_user_by_username(payload.username):
    raise HTTPException(status_code=400, detail="Username already exists")

  profile = {
      "age": payload.age,
      "sex": payload.sex,
      "nyha_class": payload.nyha_class,
      "comorbidities": payload.comorbidities or "",
      "surgery_history": payload.surgery_history or "",
      "exercise_goals": payload.exercise_goals or "",
  }
  user_profile.create_user(payload.username, hash_password(payload.password), profile)
  user = user_profile.get_user_by_username(payload.username)
  return {"id": user["id"], "username": user["username"]}


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest):
  user = user_profile.get_user_by_username(payload.username)
  if not user or user["password_hash"] != hash_password(payload.password):
    raise HTTPException(status_code=401, detail="Invalid credentials")
  return TokenResponse(access_token=create_access_token(user["id"]))


# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------


@app.get("/users/me")
def get_profile(current_user=Depends(get_current_user)):
  return current_user


@app.put("/users/me")
def update_profile(payload: UpdateProfileRequest, current_user=Depends(get_current_user)):
  data = payload.model_dump(exclude_none=True)
  user_profile.update_user_profile(current_user["id"], data)
  return user_profile.get_user(current_user["id"])


# ---------------------------------------------------------------------------
# Exercise library
# ---------------------------------------------------------------------------


@app.get("/exercises")
def list_exercise_endpoint(
    acsm_type: Optional[str] = None,
    body_region: Optional[str] = None,
    difficulty_level: Optional[str] = None,
    current_user=Depends(get_current_user),
):
  filters = {
      "acsm_type": acsm_type,
      "body_region": body_region,
      "difficulty_level": difficulty_level,
  }
  filters = {k: v for k, v in filters.items() if v}
  return list_exercises(filters)


@app.get("/exercises/{exercise_id}")
def get_exercise_endpoint(exercise_id: int, current_user=Depends(get_current_user)):
  exercise = get_exercise(exercise_id)
  if not exercise:
    raise HTTPException(status_code=404, detail="Exercise not found")
  return exercise


# ---------------------------------------------------------------------------
# Planner & Plans
# ---------------------------------------------------------------------------


@app.post("/planner")
def planner(payload: PlannerRequest, current_user=Depends(get_current_user)):
  filters = generate_planner_conditions(current_user, payload.message)
  return filters


@app.post("/plans/generate")
def generate_plan(payload: GeneratePlanRequest, current_user=Depends(get_current_user)):
  filters = payload.filters or {}
  fitt = filters.get("fitt", {})
  plan = generate_weekly_plan(current_user["id"], filters, fitt)
  return {"plan": plan}


@app.get("/plans/current")
def current_plan(current_user=Depends(get_current_user)):
  plan = get_weekly_plan(current_user["id"])
  return plan


# ---------------------------------------------------------------------------
# Coach
# ---------------------------------------------------------------------------


@app.post("/coach")
def coach(payload: CoachRequest, current_user=Depends(get_current_user)):
  plan = get_weekly_plan(current_user["id"])
  flat_plan = [item for day in plan.values() for item in day]
  message = coach_response(flat_plan, payload.message)
  return {"message": message}


# ---------------------------------------------------------------------------
# Sessions & pose data
# ---------------------------------------------------------------------------


def _session_for_user(session_id: int, user_id: int):
  session = fetch_one(
      "SELECT session_id, exercise_id FROM sessions WHERE session_id=%s AND user_id=%s",
      (session_id, user_id),
  )
  if not session:
    raise HTTPException(status_code=404, detail="Session not found")
  return session


@app.get("/sessions/history")
def session_history(current_user=Depends(get_current_user)):
  rows = fetch_all(
      """
      SELECT s.session_id,
             s.plan_id,
             s.exercise_id,
             el.name AS exercise_name,
             s.start_time,
             s.end_time,
             COALESCE(pose.pose_samples, 0) AS pose_samples,
             pose.last_pose_time
      FROM sessions s
      LEFT JOIN exercise_library el ON s.exercise_id = el.exercise_id
      LEFT JOIN (
          SELECT session_id, COUNT(*) AS pose_samples, MAX(timestamp) AS last_pose_time
          FROM pose_time_series
          GROUP BY session_id
      ) pose ON pose.session_id = s.session_id
      WHERE s.user_id=%s
      ORDER BY s.start_time DESC
      """,
      (current_user["id"],),
  )
  for row in rows:
    if row.get("pose_samples") is not None:
      row["pose_samples"] = int(row["pose_samples"])
  return rows


@app.post("/sessions/start")
def start_session_endpoint(payload: StartSessionRequest, current_user=Depends(get_current_user)):
  session_id = start_session(current_user["id"], payload.plan_id, payload.exercise_id)
  return {"session_id": session_id}


@app.post("/sessions/{session_id}/end")
def end_session(session_id: int, current_user=Depends(get_current_user)):
  _session_for_user(session_id, current_user["id"])
  close_session(session_id)
  return {"session_id": session_id, "status": "closed"}


@app.get("/sessions/{session_id}/pose")
def pose_series(session_id: int, current_user=Depends(get_current_user)):
  _session_for_user(session_id, current_user["id"])
  rows = fetch_all(
      "SELECT timestamp, joint_name, angle_value FROM pose_time_series WHERE session_id=%s ORDER BY timestamp",
      (session_id,),
  )
  return rows


@app.post("/sessions/{session_id}/pose/frame")
def pose_frame(session_id: int, payload: PoseFrameRequest, current_user=Depends(get_current_user)):
  session = _session_for_user(session_id, current_user["id"])
  exercise_id = payload.exercise_id or session.get("exercise_id")
  result = ingest_pose_frame(session_id, payload.frame_base64, exercise_id)
  return result


@app.get("/sessions/{session_id}/pose/summary")
def pose_summary(session_id: int, current_user=Depends(get_current_user)):
  _session_for_user(session_id, current_user["id"])
  return summarize_session(session_id)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
  return {"status": "ok", "time": datetime.utcnow().isoformat()}

