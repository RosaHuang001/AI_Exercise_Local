# 檔案名稱：app.py
# 後端以 api_server.py + stream_engine 為準，此檔僅作為統一啟動入口。
# 分析與跟練：POST /api/analyze、GET /video_feed 均由 api_server 提供。

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

if __name__ == '__main__':
    import api_server
    print("啟動 Flask 本地 API 伺服器 (api_server)...")
    api_server.app.run(host='127.0.0.1', port=5000, debug=True)
