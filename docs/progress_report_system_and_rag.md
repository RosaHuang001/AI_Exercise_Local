# AI 個人化健康運動推薦系統 — 進度報告（系統架構、流程、核心技術與 RAG 資料串接）

本文件以**初學者易懂**的方式說明：系統在做什麼、怎麼一步步跑、用了哪些技術，以及 **RAG（檢索增強生成）的資料如何從知識庫一路串到決策與 GPT**。

---

## 一、系統在做什麼（一句話）

本系統針對**心臟衰竭患者**（尤其是 NYHA 心臟功能分級 I～III 級），依其**個人狀況**與**醫學規則**，先篩選「安全可做」的動作，再依**影片內容**（姿勢、強度、衝擊）判斷每支影片是否建議做，最後產出**個人化一週運動計畫**與**可解釋的推薦理由**。

---

## 二、整體架構（模組分工）

可以把系統想成「一條流水線」：**使用者輸入 → 條件轉換 → 動作篩選與排序 → 規則檢索（RAG）→ 影片選擇 → 影片分析（YOLO）→ 安全判斷與摘要（規則 + GPT）→ 一週計畫**。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           使用者（心臟衰竭患者）                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1  使用者輸入 (rag/user_input)                                          │
│          問卷：年齡、性別、NYHA 分級、症狀、疾病史、用藥、開刀史…               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2  使用者條件轉換 (rag/user_condition_mapper)                           │
│          中文 → 系統內部代碼（nyha=I/II/III/IV、risk_level、contraindications） │
│          + 運動前風險篩選（不穩定 / NYHA IV / 胸痛頭暈 → 不建議運動）             │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2.5  動作推薦 (modules/recommender_filter)                              │
│            Hard Filter：依 NYHA、禁忌症篩掉不適合的動作                          │
│            Soft Ranking：依個人偏好排序，並產出「推薦理由」                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3  RAG 規則檢索 (rag/rag_engine + rag/rule_controller)  ★ 本報告重點     │
│          從知識庫 hf_chunks.json 依「族群 + 使用者條件」撈出 ACSM 規則          │
│          再依風險與主題排序、篩選，取前 N 條給後續使用                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4  選擇影片                                                             │
│          通過篩選的動作 → 對照 exercise_video_map → 得到要分析的影片列表         │
│          （支援一動作多支影片，例如左側 / 右側）                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 5  初始化 YOLO 姿勢模型                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 6  影片分析與安全判斷                                                    │
│          YOLO：體位、主要關節、ROM、次數、頻率、衝擊等級                         │
│          decide_exercise：用「規則 + YOLO 結果」決定 RECOMMEND / CAUTION        │
│          GPT：依規則與 YOLO 數據產出「民眾看得懂的運動摘要」                     │
│          （YOLO / GPT 結果可快取，同一影片、同一使用者可不重算）                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 7  一週運動計畫                                                         │
│          GPT 依使用者狀況與各影片分析結果，產出週一～週日的個人化計畫             │
│          結果寫入 results/final_output.json，並在終端機顯示                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **RAG** 的角色：在 Step 3 提供「符合條件的醫學規則」，並在 Step 6 被用在 **(1) 影片層安全判斷（decide_exercise）** 與 **(2) GPT 摘要的參考依據**；同一套規則在兩處串接，讓決策與說明都對齊 ACSM/心衰指引。

---

## 三、核心技術簡介

| 技術／模組 | 用途 | 說明 |
|-----------|------|------|
| **規則式條件比對** | 使用者條件、動作篩選、RAG 檢索 | 不用深度學習，用「條件欄位是否相符」決定規則是否套用、動作是否通過，可解釋性高。 |
| **RAG（檢索增強）** | 從知識庫撈出與「族群 + 條件」相符的條文 | 知識存在 `hf_chunks.json`，依 population、nyha、risk_level 等篩選；不把整份知識塞進 GPT，只傳「相關規則」。 |
| **YOLO Pose** | 影片中姿勢與關節角度 | 辨識關鍵點、算 ROM、次數、頻率、衝擊等級，供後續「規則 + 數據」判斷。 |
| **GPT（LLM）** | 運動摘要、一週計畫 | 輸入包含「YOLO 數據 + RAG 規則文字」，產出白話說明與計畫，語氣與內容受規則約束。 |
| **快取** | 加速重複執行 | 以「影片路徑 + 模型版本」快取 YOLO；再加「使用者狀態 hash」快取 GPT，避免同一情境重複呼叫 API。 |

---

## 四、RAG 的資料串接方式（重點說明）

RAG 在本系統中代表：**先把「和當前使用者、當前影片情境有關」的醫學規則從知識庫裡挑出來，再交給後面的「決策邏輯」和「GPT」使用**。下面依**資料從哪裡來、怎麼被選、傳到哪裡**說明。

### 4.1 知識庫長什麼樣（資料來源）

- **檔案**：`knowledge_base/hf_chunks.json`
- **內容**：一條一條的「規則」物件，每條大致包含：
  - `population`：適用族群（本系統固定為 `"Heart Failure"`）
  - `topic`：規則主題（例如 Safety、Exercise_Termination、FITT、Joint Impact）
  - `condition`：**套用條件**（例如 `{"nyha": ["II", "III"]}`、`{"risk_level": ["low", "moderate"]}`）
  - `rule`：規則的**英文條文內容**（給 GPT 與決策邏輯參考）

範例：

```json
{
  "id": "HF_P3_WARN_01",
  "population": "Heart Failure",
  "topic": "Exercise_Termination",
  "condition": { "risk_level": ["high"] },
  "rule": "Exercise should be terminated if abnormal symptoms such as chest pain, severe dyspnea, or lightheadedness occur."
}
```

也就是說：**RAG 的「資料」就是這些結構化規則；串接的起點就是「誰來讀這個 JSON、依什麼條件篩選」**。

### 4.2 第一步：使用者條件從哪裡來（RAG 的查詢條件）

- **Step 1**：使用者在問卷填寫的內容 → `user_input`（中文結構）。
- **Step 2**：`rag/user_condition_mapper.py` 的 `build_user_context(user_input)` 會：
  - 把中文轉成系統內部代碼（`map_user_conditions`）：例如 NYHA「II 級」→ `nyha: "II"`，年齡層、性別、疾病史、症狀、禁忌症等。
  - 做**運動前風險篩選**（`risk_precheck`）：得到 `risk_level`（low / moderate / high）、`allow_exercise`、`nyha` 等，放在 `risk_assessment` 裡。
- **輸出**：`user_context = { "user_conditions": {...}, "risk_assessment": {...} }`。

這些 **user_conditions** 和 **risk_assessment** 就是後面 RAG「依使用者條件篩選規則」的依據；例如 `condition` 裡會帶入 `nyha`、`risk_level` 等，和知識庫裡每條規則的 `condition` 做比對。

### 4.3 第二步：主程式怎麼呼叫 RAG（main.py Step 3）

主程式在 **Step 3** 組出「要傳給 RAG 的查詢條件」，並呼叫檢索與排序：

```python
# 查詢條件 = 使用者條件 + 風險等級（與 hf_chunks 的 condition 欄位對齊）
population = user_condition.get("population")   # "Heart Failure"
condition = {
    **user_condition,
    "risk_level": risk_assessment.get("risk_level")
}

# 檢索：從知識庫篩出「符合 population 且 condition 相符」的規則
rag_results = rag_engine.retrieve_rules(
    population=population,
    condition=condition
)

# 排序與篩選：依主題重要性、風險加權，取前 MAX_RULES 條
rules = rule_controller.process(
    rag_results,
    user_profile=risk_assessment
)
```

- **rag_engine.retrieve_rules**：只做「檢索」（篩選），回傳所有符合的規則，尚未排序。
- **rule_controller.process**：做**風險導向排序**與**風險修正**（例如高風險時隱藏部分主題、為規則加註 modifier_note），再取前 `max_rules` 條（例如 4 條）。

得到的 **rules** 會一路被傳到：
1. **Step 6 的 decide_exercise**（影片層安全判斷）
2. **GPT 摘要**（在 gpt_summary 裡會再依「當前影片」做一次 RAG，見 4.5）

也就是說：**RAG 的「輸出」= 這份 rules 列表；串接方式 = 把這份列表當作參數傳給「決策函式」和「GPT 呼叫」**。

### 4.4 第三步：RAG 引擎怎麼篩選規則（rag_engine.py）

- **輸入**：`population`（例如 `"Heart Failure"`）、`condition`（例如 `{ "nyha": "II", "risk_level": "low", ... }`），以及可選的 `topics`。
- **邏輯**：
  1. 只保留 `rule["population"] == population` 的規則。
  2. 若有 `topics`，只保留 `rule["topic"]` 在 topics 內的規則。
  3. 對每條規則的 **condition** 做**嚴格比對**（`_match_condition`）：
     - 規則的 `condition` 為空 → 視為通用，通過。
     - 規則條件裡的每個 key，在 `query_condition` 裡都要**存在且相符**：
       - 規則值是 list（例如 `["II","III"]`）→ 使用者值須在 list 內。
       - 規則值是 bool → 必須相等。
       - 否則必須字串/數值相等。
  4. 通過的規則全部放入清單回傳。

因此：**RAG 的資料串接在這裡 = 「使用者條件（condition）+ 族群（population）」從主程式傳入 → 與知識庫每條的 condition / population 比對 → 輸出「符合的規則列表」**。

### 4.5 第四步：RuleController 怎麼排序與修正（rule_controller.py）

- **輸入**：RAG 回傳的規則列表 + `user_profile`（本系統用 risk_assessment 當 user_profile）。
- **步驟**：
  1. **prioritize_rules**：依 `topic` 的主題優先級與 `condition` 的具體程度打分，再乘上 `risk_level` 的權重（高風險時安全相關規則權重更高），排序。
  2. **apply_risk_modifiers**：再依使用者風險、體位、負重等，決定是否隱藏某些主題（例如高風險時略過 Joint Impact、Exercise Intensity），或為規則加上 `modifier_note`（例如「NYHA III，套用保守運動解讀」）。
  3. 取前 **max_rules** 條（例如 4 條）回傳。

這樣 **rules** 不但是「符合條件」的，而且是「依風險與主題排過序、修過註解」的，方便後續決策與 GPT 使用。

### 4.6 第五步：rules 怎麼接到「影片安全判斷」（main.py decide_exercise）

在 Step 6 每支影片分析完後，會呼叫：

```python
decision, reasons = decide_exercise(
    user_condition,
    risk_assessment,
    yolo_result,
    rules   # ← Step 3 產出的 RAG 規則
)
```

**decide_exercise** 會：

- 用 **YOLO 結果**（體位、下肢衝擊、頭頸 ROM 等）做基本判斷（例如下肢高衝擊 → CAUTION）。
- **再搭配 rules**：
  - 把規則的 `rule` 文字串起來，若出現 head / lightheaded / dizziness / termination 等關鍵字，且頭頸 ROM > 50° → 加入 CAUTION 理由：「頭頸活動幅度較大，依安全規則建議謹慎」。
  - 若體位為站姿、風險為 high，且規則主題中有 Movement Pattern / Lower Limb Exercise / Safety → 加入「站姿運動，高風險族群請注意平衡與症狀」。

因此：**RAG 的規則在這裡的串接 = 規則的「主題」與「條文內容」直接參與 if 條件，產生結構化的 CAUTION 理由**，與知識庫條文一致、可解釋。

### 4.7 第六步：rules 怎麼接到「GPT 摘要」（gpt_summary.py）

在 **每支影片** 要產生 GPT 摘要時，`call_openai_label` 內部會：

1. **再跑一次 RAG**（與 main 的 Step 3 獨立）：  
   - 用 **當前影片的 YOLO 結果** 組出 `rag_condition`（例如 `primary_region`、`posture`、`weight_bearing`）。  
   - 用 **使用者狀態** 組出 `user_profile`（risk_level、nyha、posture、weight_bearing 等）。  
   - 呼叫 `RAG_ENGINE.retrieve_rules(population="Heart Failure", condition=rag_condition, topics=[...])`，再經 `RULE_CONTROLLER.process(rules, user_profile)` 得到該影片專用的 **hf_rules**。
2. 把 **hf_rules** 轉成一段文字 **rag_text**（每條規則一行）。
3. 把 **rag_text** 放進給 GPT 的 **system / user prompt**，要求 GPT 依這些 ACSM 規則與 YOLO 數據，用白話說明動作與建議。

所以：**RAG 在 GPT 端的串接 = 「每支影片」依「該影片的體位／區域／負重 + 使用者風險」再檢索一次規則 → 規則文字當作 GPT 的輸入 → GPT 產出與規則一致的摘要**。這樣主流程的「整體規則」與「每支影片的規則」都來自同一套知識庫與檢索邏輯。

### 4.8 RAG 資料串接總整理（從資料到使用）

| 階段 | 資料從哪來 | 傳到哪裡 | 用途 |
|------|------------|----------|------|
| 知識庫 | `hf_chunks.json` | RAG 引擎載入 | 規則的唯讀來源 |
| 查詢條件 | `user_condition` + `risk_assessment`（Step 2 產出） | main → `retrieve_rules(condition=...)` | 篩選「誰適用」 |
| 檢索結果 | `rag_engine.retrieve_rules(...)` | `rule_controller.process(...)` | 排序、修正、取前 N 條 |
| 規則列表 | `rule_controller.process(...)` → **rules** | main 保留，傳入 Step 6 | 供 decide_exercise 使用 |
| 同份 rules | main 的 **rules** | Step 6 每支影片的 decide_exercise | 影片層 CAUTION 理由（頭頸、站姿等） |
| 規則文字 | rules 的 `rule` 欄位 | （decide 內）組成 rule_texts / rule_topics | 關鍵字與主題判斷 |
| 影片專用 RAG | gpt_summary 內依「該影片 YOLO + 使用者」再查一次 | RAG_ENGINE + RULE_CONTROLLER → hf_rules | 每支影片的 GPT 摘要參考 |
| 規則 → GPT | hf_rules 轉成 rag_text | 寫入 GPT prompt | 約束摘要內容與 ACSM 一致 |

整體來說：**RAG 的資料流 = 知識庫（JSON）→ 依使用者與影片條件檢索 → 規則列表 → 決策邏輯（decide_exercise）與 GPT（call_openai_label）兩處使用**，兩處都與同一套知識庫與檢索邏輯對齊，形成可解釋、可追溯的推薦與說明。

---

## 五、流程總覽（七大步驟對應到 RAG）

| 步驟 | 名稱 | 與 RAG 的關係 |
|------|------|----------------|
| 1 | 使用者輸入 | 問卷資料，稍後轉成 RAG 的「查詢條件」的一部分。 |
| 2 | 使用者條件轉換 | 產出 `user_condition`、`risk_assessment`，提供 `population`、`nyha`、`risk_level` 等給 RAG 檢索與 RuleController。 |
| 3 | **RAG 規則檢索** | **直接使用 RAG**：從 hf_chunks 依 condition 篩選 → RuleController 排序與修正 → 得到 **rules**。 |
| 4 | 選擇影片 | 用 Step 2.5 的推薦結果對照影片對照表，與 RAG 無直接相依。 |
| 5 | 初始化 YOLO | 與 RAG 無直接相依。 |
| 6 | 影片分析與安全判斷 | **使用 Step 3 的 rules** 在 decide_exercise 中；GPT 摘要內**再依該影片做一次 RAG**，規則文字傳入 GPT。 |
| 7 | 一週計畫 | GPT 依各影片分析與使用者狀況產出計畫；規則透過「摘要內容」間接影響計畫風格與保守程度。 |

---

## 六、小結

- **系統架構**：從使用者輸入到一週計畫的一條龍流水線，模組分工清楚（輸入、條件、推薦、RAG、影片、YOLO、決策、GPT、計畫）。
- **核心技術**：規則比對、RAG 檢索、YOLO 姿勢分析、GPT 摘要與計畫、快取。
- **RAG 資料串接**：  
  - **來源**：`hf_chunks.json` 的結構化規則。  
  - **條件**：`user_condition` + `risk_assessment`（以及影片分析時的 posture / region / weight_bearing）。  
  - **檢索與排序**：`ACSMRagEngine.retrieve_rules` + `RuleController.process`。  
  - **使用處**：(1) main 的 **decide_exercise**（影片安全判斷與 CAUTION 理由）；(2) **gpt_summary** 內每支影片再查一次 RAG，規則文字送入 GPT prompt。  

這樣撰寫進度報告時，可以清楚說明：**RAG 的資料從哪裡來、如何被選取、如何接到決策與生成**，並強調「同一套知識庫、兩處使用（決策 + GPT）、可解釋」的設計。
