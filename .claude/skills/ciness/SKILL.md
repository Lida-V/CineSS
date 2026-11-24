---
name: ciness
description: CineSS（映像サラウンドMixMaster＆バイノーラル出力Webアプリ）の環境確認・セットアップ・Webサーバー起動・エージェント待機実行・AIダビング監督リミックスを行う。「CineSSを立ち上げて」「この映像をバイノーラル/Dolby Atmos/Ambisonicsにして」「開始ボタンを押したら処理して」「監督演出を調整して」と頼まれたら使う。
---

# CineSS をエージェントから操作・連携する

CineSS は映像を解析し、有音動画の「3分離＋映画方式 SFX 4層分解（`sfx_sub`, `sfx_impact` L/C/R, `ambience`）＋話者別声分け＋シーン温度連動 3D位置再現」および無音動画の「HunyuanVideo-Foley 生成＋空間配置」を行い、完全フリー構成の Dolby Atmos 互換（ITU-R BS.2076 ADM BWF + 7.1.4）および Ambisonics 1〜7次（AmbiX 4ch/16ch/36ch/64ch）マスターとヘッドホン用バイノーラル出力を作るローカル Web アプリ（Windows / macOS Apple Silicon MPS・Intel / Linux 対応）です。

## 基本フロー（リポジトリを読んだら / 起動を頼まれたら）

1. **導入状況の確認（クロスプラットフォーム）**:
   - **Windows**:
     ```powershell
     $py = ".\.venv\Scripts\python.exe"
     $cli = ".\tools\ciness_agent.py"
     Test-Path $py
     ```
   - **macOS / Linux**:
     ```bash
     PY=".venv/bin/python"
     [ -x "$PY" ] || PY=".venv/bin/python3"
     CLI="tools/ciness_agent.py"
     [ -x "$PY" ]
     ```
   - 仮想環境が **未導入** の場合:
     - `uv` と `ffmpeg` の存在を確認し、問題なければ Windows は `setup.bat`、macOS / Linux は `./setup.sh`（`chmod +x setup.sh run.sh`）の手順を実行する。
     - **絶対禁止**: `audio-separator` は入れない（既存 PyTorch を CPU 版に上書きしようとする）。
   - **導入済み** の場合:
     - 環境診断と稼働確認を実行:
       ```powershell
       # Windows (.venv\Scripts\python.exe) / macOS・Linux (.venv/bin/python)
       <venv_python> -X utf8 tools/ciness_agent.py status
       ```

2. **Web アプリを起動してブラウザを開き、エージェント待機（watch）を開始する**:
   - macOS（Apple Silicon MPS）では `PYTORCH_ENABLE_MPS_FALLBACK=1` を有効にする（`run.sh` および `ciness_agent.py start-server` では自動付与）。
   - サーバー起動（既起動ならそのままURL返却）:
     ```powershell
     <venv_python> -X utf8 tools/ciness_agent.py start-server --port 8780
     ```
   - 常駐エージェントランナーをバックグラウンド起動（Web UI に「🤖 エージェント接続中・待機OK」と表示され、ユーザーが「▶ 処理開始」を押すと自動で処理を開始する）:
     ```powershell
     <venv_python> -X utf8 tools/ciness_agent.py watch --server http://127.0.0.1:8780
     ```
   - ユーザーへ **`http://127.0.0.1:8780`** とPC環境診断サマリー（CPU / CUDA GPU / Apple Silicon MPS ユニファイドメモリ / 非NVIDIA GPU 判定と推奨動画長）を案内し、「Web画面で動画をアップロードして **▶ 処理開始** ボタンを押せば処理を開始します」と伝える。

3. **AIダビング監督連携（完了ジョブの演出調整）**:
   - ジョブ完了後、`director-context`（CLI）または `ciness_get_director_context`（MCP）で以下を確認できる：
     - `intensity_track` / `scene_segments`（シーン温度カーブ：`静寂`/`通常`/`高揚`/`クライマックス`）
     - `sfx_layers` / `sfx_events`（`sfx_sub`, `sfx_impact` L/C/R, `ambience` の状態と衝撃時刻）
     - `director_sheet_url` / `director_sheet_local_path`（最大8コマのコンタクトシート JPEG）
     - `director_plan`（現在適用されている時間変化キーフレームと演出メモ）
   - コンタクトシート画像とシーン温度を踏まえて演出を微調整したい場合は、`apply-director`（CLI）または `ciness_apply_director_plan`（MCP）でカスタム `director_plan` を送信する。再分離・再映像解析なしで空間レンダリング＋マスタリングのみ数秒で再実行される（何度適用しても冪等）。

4. **MCP サーバー (`ciness`) が使える場合（計8ツール）**:
   - **注意（macOS / Linux）**: 同梱 `.mcp.json` は既定で Windows 用 `".venv/Scripts/python.exe"` になっているため、macOS / Linux で MCP クライアントを使う場合は `".venv/bin/python"` に読み替えて設定する。
   - 利用可能なツール一覧:
     1. `ciness_status` — PC診断（CUDA / MPS / 非NVIDIA含む）・サーバー稼働・モデル確認
     2. `ciness_start_server` — サーバー起動＆ブラウザ表示
     3. `ciness_wait_job` — ハートビートを送りながら「▶ 処理開始」ボタン押下を待機
     4. `ciness_claim_and_run_job` — ジョブを受注してバックグラウンド実行
     5. `ciness_job_status` — 完了まで進捗・ログを確認し、成果物とメトリクスをユーザーに報告
     6. `ciness_cancel_job` — 中断依頼時に安全キャンセル
     7. `ciness_get_director_context` — シーン温度・SFX 4層・コンタクトシート情報を取得
     8. `ciness_apply_director_plan` — AIダビング監督プランを適用し高速空間リミックスを実行

## CLI コマンド一覧

Windows では `.venv\Scripts\python.exe`、macOS / Linux では `.venv/bin/python` を使用する。

```powershell
# 環境診断 + サーバー稼働 + モデル一覧 (JSON)
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py status

# サーバー起動（既起動ならURLを返す）
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py start-server [--port 8780] [--no-browser]

# Web UI の「処理開始」ボタンを待ち受けて自動実行（常駐）
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py watch --server http://127.0.0.1:8780 [--once]

# 特定のジョブIDを1本だけ実行
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py run-job --job-id <JOB_ID> --server http://127.0.0.1:8780

# 映像シーン温度・SFX 4層・監督プラン・コンタクトシート情報を取得
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py director-context --job-id <JOB_ID> --server http://127.0.0.1:8780

# AIダビング監督プランを適用して再分離なしで高速空間リミックス
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py apply-director --job-id <JOB_ID> --plan '{"notes":"クライマックス強化","keyframes":[]}' --server http://127.0.0.1:8780
```

## してはいけないこと

- **`audio-separator` をインストールしない**（CUDA / MPS 用 PyTorch 環境が破壊される）。
- **推論時にAIモデルを暗黙ダウンロードしない**。未取得モデルが必要な場合は、ユーザーに Web UI の「AIモデル管理」からの取得を案内する（隣接 `SoundSeparator` や `~/AI/HunyuanVideo-Foley` の取得済みモデルは自動検出・共有される）。
- **VRAM・ユニファイドメモリ要件を根拠なく誇張・断定しない**。`status` / `check_system()` の診断結果と推奨動画長（特に Mac MPS での Foley 省メモリ注意や非NVIDIA GPU 時の CPU 軽量フォールバック）に沿って案内する。
- **聴感で確認していない内容を「確認した」と書かない**。実測メトリクス（LUFS / True Peak / 相関 / 使用モデル）をそのまま報告する。
- **`*.bat` の CRLF と `*.sh` の LF を壊さない**（`.gitattributes` 準拠）。
