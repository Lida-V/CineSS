# CineSS

映像サラウンド MixMaster ＆ バイノーラル出力 Web アプリ。
開発詳細・構造の入口は [CLAUDE.md](CLAUDE.md)、エージェント操作スキルは [.claude/skills/ciness/SKILL.md](.claude/skills/ciness/SKILL.md)。
UI・コメント・ドキュメント・対話はすべて日本語で行う。

## リポジトリを読んだ AI エージェントへの起動・連携手順（最優先）

ユーザーが「このリポジトリを読んで」「CineSS を立ち上げて」「動画をサラウンド／バイノーラルにして」と指示したら、以下の手順で迷わず起動・連携する。

1. **導入状態の判定**
   - `.venv\Scripts\python.exe` の有無を確認する。
   - **未導入の場合**:
     - `uv` と `ffmpeg` が PATH にあることを確認する。
     - `setup.bat` と同等手順で環境構築する（`uv venv --python 3.11 .venv` → CUDA 12.4 版 `torch==2.6.0 torchaudio==2.6.0` → `uv pip install --python .venv -r requirements.txt`）。GPU非搭載PCでは CPU 版 PyTorch を導入する。
   - **導入済みの場合**:
     - そのまま次へ進む（再セットアップしない）。

2. **Web アプリサーバー起動 ＆ エージェント待機開始**
   - 環境診断＋稼働確認:
     ```powershell
     .venv\Scripts\python.exe -X utf8 tools/ciness_agent.py status
     ```
   - サーバー起動（既起動ならそのまま URL を返す）:
     ```powershell
     .venv\Scripts\python.exe -X utf8 tools/ciness_agent.py start-server
     ```
   - Web UI の「▶ 処理開始」ボタン押下を受注するエージェントランナーを起動（MCP `ciness` が有効な場合は `ciness_wait_job` → `ciness_claim_and_run_job` でも可）:
     ```powershell
     .venv\Scripts\python.exe -X utf8 tools/ciness_agent.py watch --server http://127.0.0.1:8780
     ```

3. **ユーザーへの案内**
   - Web UI の URL（既定 `http://127.0.0.1:8780`）と PC 動作環境診断（CPU / GPU / RAM の判定と推奨動画長）を簡潔に伝える。
   - ユーザーがブラウザ上で動画をアップロードし **「▶ 処理開始」** を押すと、`watch` ランナーが `queued_for_agent` ジョブを自動検知して `claim` → パイプライン実行 → 進捗＆思考ログ送信 → 完了まで自動実行する。

## 厳守ルール（SoundSeparator 継承）

- **既存の CUDA 版 PyTorch（2.6.0+cu124）を保護し、`audio-separator` は絶対に入れない**（CPU版 torch への置換を防ぐ）。torch 依存ライブラリを追加する場合は `--no-deps` で入れて純 Python 依存を手動管理する。
- **推論時の暗黙モデルダウンロードを禁止する**。モデル取得は Web UI の「AIモデル管理」または明示的な `download_model()` 呼び出し時のみ行う。隣接 SoundSeparator フォルダの取得済みモデルは自動検出しコピーせず共有する。
- **VRAM 目安は誇張・断定しない**。提供元推奨・暫定目安・未確定を区別して表示する。
- **音声・映像はすべてローカルで処理する**。
- **`*.bat` ファイルは Windows CRLF を実バイトで保持する**（`.gitattributes` に `*.bat -text` を定義）。

## 既知の観測メモ

- 2026-11-24: 初回構築。Python 3.11.9 (`.venv`) + `torch==2.6.0+cu124` (CUDA有効) を採用。既定ポートは `8780`（環境変数 `CINESS_PORT` で変更可、SoundSeparator の `8765` と衝突しない）。
- 2026-11-24: 空間座標規約は `azimuth_deg` ∈ `[-180, 180]`（`0°`=正面、**負=左**、**正=右**）、`elevation_deg` ∈ `[-90, 90]`（正=上）、`distance_m` ∈ `[0.5, 15.0]`、サンプルレート `48000` Hz で統一。
- 2026-11-24: エージェント連携は CLI (`tools/ciness_agent.py`) と stdio MCP (`tools/ciness_mcp.py`, `.mcp.json` は相対パス・登録名 `ciness`) の二系統を用意。`watch` モードは別スレッドで2秒間隔ハートビートを送信し、長時間ジョブ実行中も Web UI 上の接続状態が途切れない設計。
- 2026-11-24: サラウンド規格は完全フリーの **Dolby Atmos 互換（ITU-R BS.2076 ADM BWF + 7.1.4 + 直接バイノーラル、DaVinci Resolve / EAR / Reaper 互換・`.ec3 JOC` は外部ツール要）** および **Ambisonics 1〜7次 AmbiX**（`ambisonics_foa` 1次4ch / `ambisonics_hoa` 3次16ch既定 / `ambisonics_hoa5` 5次36ch / `ambisonics_hoa7` 7次64ch、3.5GB超はRF64自動切替）に統一。`hardware_advisor` が `max_recommended_ambisonics_order` とWAV容量見積りを返す。
- 2026-11-24: Windows ジョブオブジェクト環境（エージェントの非デーモンコマンド終了時に子プロセスツリーが自動終了する環境）では、`tools/ciness_agent.py start-server` ではなく `.venv\Scripts\python.exe -X utf8 -m uvicorn app.server:app --host 127.0.0.1 --port 8780` を直接デーモン（`IsDaemon=true`）として常駐起動する。
