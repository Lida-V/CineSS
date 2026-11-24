# CineSS

映像サラウンド MixMaster ＆ バイノーラル出力 Web アプリ。
開発詳細・構造の入口は [CLAUDE.md](CLAUDE.md)、エージェント操作スキルは [.claude/skills/ciness/SKILL.md](.claude/skills/ciness/SKILL.md)。
UI・コメント・ドキュメント・対話はすべて日本語で行う。

## リポジトリを読んだ AI エージェントへの起動・連携手順（最優先）

ユーザーが「このリポジトリを読んで」「CineSS を立ち上げて」「動画をサラウンド／バイノーラルにして」と指示したら、以下の手順で迷わず起動・連携する。

1. **導入状態の判定**
   - Windows: `.venv\Scripts\python.exe`、macOS / Linux: `.venv/bin/python`（または `.venv/bin/python3`）の有無を確認する（`app/config.py` の `find_venv_python()` でクロスプラットフォーム検出可能）。
   - **未導入の場合**:
     - `uv` と `ffmpeg` が PATH にあることを確認する。
     - **Windows**: `setup.bat` と同等手順で環境構築する（`uv venv --python 3.11 .venv` → CUDA 12.4 版 `torch==2.6.0 torchaudio==2.6.0` → `uv pip install --python .venv -r requirements.txt`）。非 NVIDIA PC では CPU 版 PyTorch を導入する。
     - **macOS / Linux**: `chmod +x setup.sh run.sh && ./setup.sh` と同等手順で環境構築する（Apple Silicon / Intel Mac は標準 PyPI 版 PyTorch で MPS/CPU 対応、Linux CUDA 環境は CUDA 12.4 版、Linux 非 GPU は CPU 版）。
   - **導入済みの場合**:
     - そのまま次へ進む（再セットアップしない）。

2. **Web アプリサーバー起動 ＆ エージェント待機開始**
   - macOS（Apple Silicon MPS）環境では未対応演算の CPU フォールバック用に `PYTORCH_ENABLE_MPS_FALLBACK=1` を有効にする（`run.sh` および `ciness_agent.py start-server` / `ciness_mcp.py` では自動設定）。
   - 環境診断＋稼働確認:
     ```powershell
     # Windows
     .venv\Scripts\python.exe -X utf8 tools/ciness_agent.py status
     # macOS / Linux
     .venv/bin/python -X utf8 tools/ciness_agent.py status
     ```
   - サーバー起動（既起動ならそのまま URL を返す）:
     ```powershell
     # Windows
     .venv\Scripts\python.exe -X utf8 tools/ciness_agent.py start-server
     # macOS / Linux
     .venv/bin/python -X utf8 tools/ciness_agent.py start-server
     ```
   - Web UI の「▶ 処理開始」ボタン押下を受注するエージェントランナーを起動（MCP `ciness` が有効な場合は `ciness_wait_job` → `ciness_claim_and_run_job` でも可）:
     ```powershell
     # Windows
     .venv\Scripts\python.exe -X utf8 tools/ciness_agent.py watch --server http://127.0.0.1:8780
     # macOS / Linux
     .venv/bin/python -X utf8 tools/ciness_agent.py watch --server http://127.0.0.1:8780
     ```
   - **`.mcp.json` の Mac / Linux 読み替え注意**: 同梱の `.mcp.json` は既定で Windows 用 `"command": ".venv/Scripts/python.exe"` を指定している。macOS / Linux の MCP クライアントで使用する場合はクライアント側またはローカル設定で `".venv/bin/python"` に読み替えること。

3. **ユーザーへの案内**
   - Web UI の URL（既定 `http://127.0.0.1:8780`）と PC 動作環境診断（CPU / GPU・MPS / RAM・ユニファイドメモリの判定と推奨動画長）を簡潔に伝える。
   - ユーザーがブラウザ上で動画をアップロードし **「▶ 処理開始」** を押すと、`watch` ランナーが `queued_for_agent` ジョブを自動検知して `claim` → パイプライン実行 → 進捗＆思考ログ送信 → 完了まで自動実行する。
   - 完了後は `director-context`（CLI）または `ciness_get_director_context`（MCP）で映像シーン温度・SFX 4層・コンタクトシート画像（`director_sheet.jpg`）を確認し、`apply-director` / `ciness_apply_director_plan` で再分離なしの高速監督リミックスを提案・実行できる。

## 厳守ルール（SoundSeparator 継承）

- **既存の CUDA 版 PyTorch（2.6.0+cu124）を保護し、`audio-separator` は絶対に入れない**（CPU版 torch への置換を防ぐ）。torch 依存ライブラリを追加する場合は `--no-deps` で入れて純 Python 依存を手動管理する。
- **推論時の暗黙モデルダウンロードを禁止する**。モデル取得は Web UI の「AIモデル管理」または明示的な `download_model()` 呼び出し時のみ行う。隣接 SoundSeparator フォルダの取得済みモデルは自動検出しコピーせず共有する。
- **VRAM・ユニファイドメモリ目安は誇張・断定しない**。提供元推奨・暫定目安・未確定を区別して表示する。
- **音声・映像はすべてローカルで処理する**。
- **改行コードを厳守する**: `*.bat` ファイルは Windows CRLF を実バイトで保持し（`*.bat -text`）、`*.sh` ファイルは Unix LF（CR なし、`*.sh text eol=lf`）を維持する（`.gitattributes` 定義）。

## 既知の観測メモ

- 2026-11-24: 初回構築。Python 3.11 (`.venv`) + Windows NVIDIA 環境では `torch==2.6.0+cu124` (CUDA有効) を採用。既定ポートは `8780`（環境変数 `CINESS_PORT` で変更可、SoundSeparator の `8765` と衝突しない）。
- 2026-11-24: 空間座標規約は `azimuth_deg` ∈ `[-180, 180]`（`0°`=正面、**負=左**、**正=右**）、`elevation_deg` ∈ `[-90, 90]`（正=上）、`distance_m` ∈ `[0.5, 15.0]`、サンプルレート `48000` Hz で統一。
- 2026-11-24: エージェント連携は CLI (`tools/ciness_agent.py`) と stdio MCP (`tools/ciness_mcp.py`, `.mcp.json` は相対パス・登録名 `ciness`・9ツール) の二系統を用意。`watch` モードは別スレッドで2秒間隔ハートビートを送信し、長時間ジョブ実行中も Web UI 上の接続状態が途切れない設計。
- 2026-11-24: サラウンド規格は完全フリーの **Dolby Atmos 互換（ITU-R BS.2076 ADM BWF + 7.1.4 + 直接バイノーラル、DaVinci Resolve / EAR / Reaper 互換・`.ec3 JOC` は外部ツール要）** および **Ambisonics 1〜7次 AmbiX**（`ambisonics_foa` 1次4ch / `ambisonics_hoa` 3次16ch既定 / `ambisonics_hoa5` 5次36ch / `ambisonics_hoa7` 7次64ch、3.5GB超はRF64自動切替）に統一。`hardware_advisor` が `max_recommended_ambisonics_order` とWAV容量見積りを返す。
- 2026-11-24: 映画方式 SFX 4層厳密相補分解（`sfx_sub`, `sfx_impact` L/C/R, `ambience`）・映像シーン温度解析（`intensity_track`, `scene_segments`, `director_sheet.jpg`）・AIダビング監督（`app/director.py` + `apply_director_plan_and_remix` + Web API `/director-context`, `/media/director_sheet`, `/director-plan`）および macOS Apple Silicon MPS / 非NVIDIA GPU フォールバック（`setup.sh` / `run.sh` / `config.get_torch_device()`）を追加。
- 2026-11-24: 360°ワールド空間モデリング（カメラ外360°環境 `world_profile` ＆画面外通過 `flyby_events` 解析・2段オールパス無相関7.1.4/AmbiX包囲・監督の `azimuth_deg`/`elevation_deg`/`world_360` 制御解禁・事前解析 `inspect-world` / MCP `ciness_inspect_video_world` / `POST /api/videos/inspect-world`・360°環境＋物体ヒントFoley強化）を追加。
- 2026-11-24: Windows ジョブオブジェクト環境（エージェントの非デーモンコマンド終了時に子プロセスツリーが自動終了する環境）では、`tools/ciness_agent.py start-server` ではなく `.venv\Scripts\python.exe -X utf8 -m uvicorn app.server:app --host 127.0.0.1 --port 8780` を直接デーモン（`IsDaemon=true`）として常駐起動する。
