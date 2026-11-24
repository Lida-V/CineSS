---
name: ciness
description: CineSS（映像サラウンドMixMaster＆バイノーラル出力Webアプリ）の環境確認・セットアップ・Webサーバー起動・エージェント待機実行を行う。「CineSSを立ち上げて」「この映像をバイノーラル/Dolby Atmos/Ambisonicsにして」「開始ボタンを押したら処理して」と頼まれたら使う。
---

# CineSS をエージェントから操作・連携する

CineSS は映像を解析し、有音動画の「3分離＋話者別声分け＋3D位置再現」および無音動画の「HunyuanVideo-Foley生成＋空間配置」を行い、完全フリー構成の Dolby Atmos 互換（ITU-R BS.2076 ADM BWF + 7.1.4）および Ambisonics 1〜7次（AmbiX 4ch/16ch/36ch/64ch）マスターとヘッドホン用バイノーラル出力を作るローカル Web アプリです。

## 基本フロー（リポジトリを読んだら / 起動を頼まれたら）

1. **導入状況の確認**:
   ```powershell
   $root = "."
   $py   = "$root\.venv\Scripts\python.exe"
   $cli  = "$root\tools\ciness_agent.py"
   Test-Path $py
   ```
   - `.venv\Scripts\python.exe` が **未導入** の場合:
     - `uv` と `ffmpeg` の存在を確認し、問題なければ `setup.bat` の手順（`uv venv --python 3.11 .venv` → CUDA 12.4 版 PyTorch → `uv pip install --python .venv -r requirements.txt`）を実行する。
     - **絶対禁止**: `audio-separator` は入れない（CUDA版PyTorchをCPU版に上書きしようとする）。
   - **導入済み** の場合:
     - 環境診断と稼働確認を実行:
       ```powershell
       & $py -X utf8 $cli status
       ```

2. **Web アプリを起動してブラウザを開き、エージェント待機（watch）を開始する**:
   - サーバー起動（既起動ならそのままURL返却）:
     ```powershell
     & $py -X utf8 $cli start-server --port 8780
     ```
   - 常駐エージェントランナーをバックグラウンド起動（Web UI に「🤖 エージェント接続中・待機OK」と表示され、ユーザーが「▶ 処理開始」を押すと自動で処理を開始する）:
     ```powershell
     & $py -X utf8 $cli watch --server http://127.0.0.1:8780
     ```
   - ユーザーへ **`http://127.0.0.1:8780`** とPC環境診断サマリー（CPU/GPU/RAM判定と推奨動画長）を案内し、「Web画面で動画をアップロードして **▶ 処理開始** ボタンを押せば処理を開始します」と伝える。

3. **MCP サーバー (`ciness`) が使える場合**:
   - `.mcp.json` で登録済みの `ciness` ツールを使って同じフローを実行できる:
     1. `ciness_status` — PC診断・サーバー稼働・モデル確認
     2. `ciness_start_server` — サーバー起動＆ブラウザ表示
     3. `ciness_wait_job` — ハートビートを送りながら「▶ 処理開始」ボタン押下を待機
     4. `ciness_claim_and_run_job` — ジョブを受注してバックグラウンド実行
     5. `ciness_job_status` — 完了まで進捗・ログを確認し、成果物とメトリクスをユーザーに報告
     6. `ciness_cancel_job` — 中断依頼時に安全キャンセル

## CLI コマンド一覧

```powershell
# 環境診断 + サーバー稼働 + モデル一覧 (JSON)
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py status

# サーバー起動（既起動ならURLを返す）
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py start-server [--port 8780] [--no-browser]

# Web UI の「処理開始」ボタンを待ち受けて自動実行（常駐）
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py watch --server http://127.0.0.1:8780 [--once]

# 特定のジョブIDを1本だけ実行
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py run-job --job-id <JOB_ID> --server http://127.0.0.1:8780
```

## してはいけないこと

- **`audio-separator` をインストールしない**（CUDA版 PyTorch が破壊される）。
- **推論時にAIモデルを暗黙ダウンロードしない**。未取得モデルが必要な場合は、ユーザーに Web UI の「AIモデル管理」からの取得を案内する（隣接 `SoundSeparator` の取得済みモデルは自動共有される）。
- **VRAM要件を根拠なく誇張・断定しない**。`status` / `check_system()` の診断結果と推奨動画長に沿って案内する。
- **聴感で確認していない内容を「確認した」と書かない**。実測メトリクス（LUFS / True Peak / 左右バランス / 使用モデル）をそのまま報告する。
