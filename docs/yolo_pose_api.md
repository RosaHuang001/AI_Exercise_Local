# YOLO 姿勢推論 API 介面規格

本文件定義 Flutter 前端與 FastAPI 後端之間，用於即時訓練監測的 REST 介面。設計目標：

- 允許行動裝置定期上傳使用者訓練畫面，後端利用 YOLOv11 Pose 模型計算關節角度。
- 將角度資料寫入 `pose_time_series`，提供歷程查詢與安全風險判斷。
- 回傳即時計算結果，供前端顯示教練提醒。

## 共用欄位

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `session_id` | int | 對應 `/sessions/start` 建立的訓練場次 |
| `timestamp` | ISO8601 | 後端接收影格的 UTC 時間 |
| `angles` | object | key 為關節名稱，value 為角度 (degrees) |
| `risk_level` | string | `low` / `medium` / `high`，依照建議角度區間判斷 |

## 1. 上傳單張影格

- **Route**: `POST /sessions/{session_id}/pose/frame`
- **Header**: `Authorization: Bearer <token>`
- **Body**:

```json
{
  "frame_base64": "<JPEG base64 string>",
  "exercise_id": 12,
  "source": "front_camera"
}
```

- **Responses**:
  - `200 OK`

```json
{
  "session_id": 45,
  "angles": {
    "left_knee": 132.4,
    "right_knee": 135.1,
    "left_elbow": 88.3,
    "right_elbow": 90.7
  },
  "risk_level": "medium",
  "timestamp": "2025-11-19T08:12:31.012345Z"
}
```

- **錯誤**
  - `400`: base64 格式錯誤或影格無法解析
  - `404`: session 不存在或不屬於使用者

## 2. 取得即時摘要

- **Route**: `GET /sessions/{session_id}/pose/summary`
- **Response**:

```json
{
  "session_id": 45,
  "last_pose_time": "2025-11-19T08:13:00Z",
  "avg_angles": {
    "left_knee": 128.1,
    "right_knee": 130.6
  },
  "max_deviation": {
    "left_knee": 6.4,
    "right_knee": 4.9
  },
  "risk_breakdown": {
    "high": 1,
    "medium": 4,
    "low": 12
  }
}
```

## 3. Pose 時序資料

- **Route**: `GET /sessions/{session_id}/pose` （既有 API）
- 用於下載完整時序資料，可搭配前端圖表顯示。

## 推論流程

1. Flutter 啟動訓練 → 呼叫 `/sessions/start`，附上 `exercise_id`。
2. App 取得相機畫面，每 2 秒擷取一張 JPEG，呼叫 `/pose/frame`。
3. FastAPI 將影格交給 `modules.training_session.ingest_pose_frame`，透過 YOLO 模型產生關節角度並寫入 DB。
4. App 每次收到回應後更新 UI，同時週期性呼叫 `/pose/summary` 顯示趨勢。
5. 訓練結束 → `/sessions/{id}/end`，前端停止上傳，相機釋放。

## 安全與效能建議

- 單筆影格限制大小 512 KB，過大請先在端側壓縮。
- 建議頻率 0.5~1 fps，以降低行動網路與後端 GPU 負載。
- 後端可針對同一 session 啟用批次寫入，以減少資料庫壓力。

## 未來擴充

- 支援 WebSocket 連線，降低 HTTP 開銷。
- 在回應中加入 AI 語意建議，例如「請放慢速度」。
- 支援多機位上傳並同步推論。 

