# 🧠 AI_Exercise_Local - 地端 AI 運動分析系統

一個基於 YOLOv11 Pose + MMAction2 PoseC3D + GPT + MySQL 的完整地端 AI 運動分析解決方案。

## 📋 專案概述

本專案提供了一個完整的運動動作分析系統，能夠：
- 🎬 自動偵測影片中的人體姿勢
- 🏃 識別並分類運動動作類型
- 🤖 使用 AI 生成專業的運動建議
- 💾 將結果安全地儲存到資料庫

## 🏗️ 系統架構

```
AI_Exercise_Local/
├── video_processor.py        # YOLOv11 Pose 關節偵測
├── pose_classifier.py        # MMAction2 PoseC3D 動作分類
├── semantic_advisor.py       # GPT 語意分析與建議
├── database_handler.py       # MySQL 資料庫操作
├── main_pipeline.py          # 主控流程整合
├── config.yaml               # 系統設定檔
├── requirements.txt           # Python 依賴套件
├── videos/                   # 原始影片存放目錄
├── pose_json/                # 姿勢 JSON 暫存
├── results/                  # 分析結果輸出
├── cache/                    # 中間結果暫存
└── logs/                     # 系統執行日誌
```

## 🚀 快速開始

### 1. 環境需求

- **Python**: 3.10 或更高版本
- **GPU**: NVIDIA GPU (建議，支援 CUDA 11.8+)
- **記憶體**: 至少 8GB RAM
- **儲存空間**: 至少 10GB 可用空間
- **資料庫**: MySQL 8.0+

### 2. 安裝依賴

```bash
# 複製專案
git clone <repository-url>
cd AI_Exercise_Local

# 安裝 Python 依賴
pip install -r requirements.txt

# 或使用 conda
conda install --file requirements.txt
```

### 3. 設定配置

編輯 `config.yaml` 檔案：

```yaml
# API 設定
api:
  openai:
    api_key: "your-openai-api-key-here"  # 替換為您的 OpenAI API Key

# 資料庫設定
database:
  mysql:
    host: "localhost"
    user: "root"
    password: "your-password"
    database: "ai_exercise"
```

### 4. 準備資料庫

```sql
-- 建立資料庫
CREATE DATABASE ai_exercise CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 系統會自動建立必要的資料表
```

### 5. 執行系統

```bash
# 處理單一影片
python main_pipeline.py --input videos/squat.mp4

# 批次處理目錄中的所有影片
python main_pipeline.py --input videos/ --batch

# 使用自訂設定檔
python main_pipeline.py --input videos/test.mp4 --config my_config.yaml
```

## 📖 使用說明

### 命令列參數

| 參數 | 簡寫 | 說明 | 範例 |
|------|------|------|------|
| `--input` | `-i` | 輸入影片檔案或目錄 | `--input videos/squat.mp4` |
| `--config` | `-c` | 設定檔路徑 | `--config config.yaml` |
| `--batch` | `-b` | 批次處理模式 | `--batch` |
| `--no-db` | | 跳過資料庫寫入 | `--no-db` |

### 處理流程

1. **姿勢偵測** (`video_processor.py`)
   - 使用 YOLOv11 Pose 模型偵測人體關鍵點
   - 輸出骨架序列 JSON 檔案

2. **動作分類** (`pose_classifier.py`)
   - 使用 MMAction2 PoseC3D 模型分析動作
   - 識別運動類型並計算信心值

3. **AI 建議** (`semantic_advisor.py`)
   - 呼叫 GPT API 生成專業建議
   - 包含技術要點、修正建議、安全提醒

4. **資料儲存** (`database_handler.py`)
   - 將完整結果寫入 MySQL 資料庫
   - 提供查詢和統計功能

## 🔧 模組說明

### VideoProcessor
- **功能**: YOLOv11 Pose 姿勢偵測
- **輸入**: 影片檔案 (MP4, AVI, MOV 等)
- **輸出**: 姿勢序列 JSON 檔案
- **特點**: 支援 GPU 加速、批次處理

### PoseClassifier
- **功能**: MMAction2 PoseC3D 動作分類
- **輸入**: 姿勢序列 JSON 檔案
- **輸出**: 動作分類結果
- **支援動作**: 深蹲、伏地挺身、引體向上等 10 種動作

### SemanticAdvisor
- **功能**: GPT 語意分析與建議生成
- **輸入**: 動作分類結果
- **輸出**: Markdown 和 JSON 格式的建議報告
- **內容**: 技術要點、修正建議、安全提醒、訓練建議

### DatabaseHandler
- **功能**: MySQL 資料庫操作
- **功能**: 結果儲存、查詢、統計、匯出
- **表格**: `pose_results`, `action_statistics`

## 📊 輸出格式

### JSON 結果檔案
```json
{
  "video_info": {
    "filename": "squat.mp4",
    "duration": 30.5,
    "processed_frames": 915
  },
  "classification_summary": {
    "predicted_action": "squat",
    "confidence": 0.85
  },
  "ai_advice": {
    "action_analysis": "深蹲是複合性運動...",
    "technical_points": "正確姿勢要領...",
    "correction_suggestions": "常見錯誤修正...",
    "safety_reminders": "安全注意事項...",
    "training_recommendations": "訓練建議..."
  }
}
```

### Markdown 報告
系統會自動生成易讀的 Markdown 格式報告，包含：
- 影片資訊摘要
- 動作分類結果
- 詳細的 AI 建議
- 技術要點和安全提醒

## 🛠️ 進階設定

### GPU 設定
在 `config.yaml` 中設定：
```yaml
models:
  yolov11_pose:
    device: "cuda"  # 或 "cpu"
  posec3d:
    device: "cuda"  # 或 "cpu"
```

### 模型參數調整
```yaml
models:
  yolov11_pose:
    confidence_threshold: 0.5
  posec3d:
    batch_size: 1
```

### API 設定
```yaml
api:
  openai:
    model: "gpt-3.5-turbo"
    max_tokens: 1000
    temperature: 0.7
```

## 🔍 故障排除

### 常見問題

1. **CUDA 記憶體不足**
   ```bash
   # 降低批次大小或使用 CPU
   # 在 config.yaml 中設定 device: "cpu"
   ```

2. **OpenAI API 錯誤**
   ```bash
   # 檢查 API Key 是否正確設定
   # 確認 API 額度是否充足
   ```

3. **資料庫連接失敗**
   ```bash
   # 檢查 MySQL 服務是否啟動
   # 確認資料庫設定是否正確
   ```

4. **模型載入失敗**
   ```bash
   # 檢查模型檔案是否存在
   # 確認依賴套件是否正確安裝
   ```

### 日誌檔案
系統會在 `logs/` 目錄中產生詳細的執行日誌：
- `video_processor_YYYYMMDD.log`
- `pose_classifier_YYYYMMDD.log`
- `semantic_advisor_YYYYMMDD.log`
- `database_handler_YYYYMMDD.log`
- `main_pipeline_YYYYMMDD.log`

## 📈 效能優化

### 硬體建議
- **GPU**: NVIDIA RTX 3080 或更高
- **CPU**: Intel i7 或 AMD Ryzen 7
- **RAM**: 16GB 或更多
- **儲存**: SSD 硬碟

### 軟體優化
- 使用 CUDA 11.8+ 版本
- 啟用混合精度訓練
- 調整批次大小以適應硬體

## 🤝 貢獻指南

1. Fork 專案
2. 建立功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 📞 支援與聯絡

- **問題回報**: [GitHub Issues](https://github.com/your-repo/issues)
- **功能建議**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **技術支援**: 請透過 GitHub Issues 聯絡

## 🔄 版本更新

### v1.0.0 (2024-10-23)
- ✨ 初始版本發布
- 🎯 支援 10 種基本運動動作
- 🤖 整合 GPT-3.5-turbo 建議生成
- 💾 完整的 MySQL 資料庫支援
- 📊 詳細的處理統計和日誌

---

**🎉 感謝使用 AI_Exercise_Local！**

如有任何問題或建議，歡迎隨時聯絡我們。
