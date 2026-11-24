# CineSS 開発・運用ガイド

映像サラウンド MixMaster ＆ バイノーラル出力 Web アプリ。UI・コメント・ドキュメント・対話はすべて日本語。

## 起動とエージェント連携の入口

```text
通常起動:
  Windows:       run.bat  (→ .venv\Scripts\python.exe -X utf8 -m uvicorn app.server:app --host 127.0.0.1 --port 8780)
  macOS / Linux: ./run.sh (→ PYTORCH_ENABLE_MPS_FALLBACK=1 + .venv/bin/python -X utf8 -m uvicorn app.server:app ...)

エージェント CLI:
  # Windows は .venv\Scripts\python.exe、macOS / Linux は .venv/bin/python を使用
  1. <venv_python> -X utf8 tools/ciness_agent.py status
  2. <venv_python> -X utf8 tools/ciness_agent.py start-server
  3. <venv_python> -X utf8 tools/ciness_agent.py watch --server http://127.0.0.1:8780
  4. <venv_python> -X utf8 tools/ciness_agent.py inspect-world --video <VIDEO_PATH>
  5. <venv_python> -X utf8 tools/ciness_agent.py director-context --job-id <JOB_ID>
  6. <venv_python> -X utf8 tools/ciness_agent.py apply-director --job-id <JOB_ID> --plan <JSON_OR_PATH>
```

- **未導入環境**:
  - Windows: `.venv\Scripts\python.exe` が無ければ `setup.bat`（Python 3.11 + CUDA 12.4 / CPU PyTorch + `requirements.txt`）を実行する。
  - macOS / Linux: `.venv/bin/python` が無ければ `chmod +x setup.sh run.sh && ./setup.sh`（Apple Silicon MPS / Intel Mac 標準 PyPI PyTorch、または Linux CUDA 12.4 / CPU + `requirements.txt`）を実行する。
- **導入済み環境**: `start-server` と `watch` を実行し、ユーザーに `http://127.0.0.1:8780` を案内する。ユーザーが画面上で **「▶ 処理開始」** を押すと、`watch` が `queued_for_agent` を検知して自動受注・実行する。
- **MCP 登録**: リポジトリ直下の `.mcp.json`（登録名 `ciness`・相対パス）から `tools/ciness_mcp.py`（計9ツール）を起動できる。
  - **Mac / Linux 読み替え注意**: `.mcp.json` の既定 `"command"` は Windows 用 `".venv/Scripts/python.exe"` のため、macOS / Linux 環境の MCP クライアントで利用する場合は `".venv/bin/python"` に読み替えて設定すること。

## アーキテクチャ

| ファイル | 役割 |
|---|---|
| `app/config.py` | ルート解決・出力/ジョブ/モデルパス・既定ポート(`8780`)・`48000`Hz・`find_venv_python()`・`get_torch_device()` (`cuda -> mps -> cpu`)・`find_soundseparator_dir()`・`find_hunyuan_foley_dir()`（macOS/Linux `~/AI/HunyuanVideo-Foley` 対応） |
| `app/hardware_advisor.py` | Windows / macOS (`sysctl`) / Linux の CPU/GPU/MPS/ユニファイドメモリ/非NVIDIA GPU/RAM/FFmpeg/Disk/SoundSeparator 計測、✅/⚠️/✕ 判定、有音/無音の推奨動画長・モデル選定 (`check_system`, `evaluate_video_plan`) |
| `app/model_manager.py` | SoundSeparator共有モデル＋CineSS追加モデルのカタログ・状態検出・明示的ダウンロード (`list_models`, `download_model`)。Mac/非NVIDIA環境では CUDA 専用 wheel を避け標準 PyPI / CPU wheel を選択 |
| `app/vision/analyzer.py` | 動画検査 (`inspect_video`)・OpenCV顔/口元モーション/動き重心追跡・SFX距離計算（アクション近接 1.3〜2.2m / 静寂遠景 3.2〜5.2m）・シーン温度カーブ (`intensity_track`, `scene_segments`)・360°世界プロファイル (`world_profile`)＆画面外通過検出 (`flyby_events`)・コンタクトシート生成 (`export_director_contact_sheet`) |
| `app/audio/separator_bridge.py` | SoundSeparator CLI 流用 DME 3分離 / 残響除去 + ローカルDSP 3分離フォールバック |
| `app/audio/sfx_splitter.py` | 映画方式 4層厳密相補（ロスレス）SFX 分解 (`split_sfx_layers`: `sfx_sub` 140Hz LPF, `sfx_impact` L/C/R・-62dBFS未満の無音側チャンネル自動畳み込み, `ambience`) |
| `app/audio/speaker_splitter.py` | Nemotron-3-Diarization / 音響クラスタリングによる話者別声分け・重なり分離・口元モーション自動紐付け（`immersive` 時画面外話者の 360°側面〜後方配置・`cuda -> mps -> cpu` 対応） |
| `app/audio/foley_engine.py` | HunyuanVideo-Foley XL/XXL ラッパー（360°環境＋物体ヒント対応プロンプト強化・チャンク生成・オフロード・MPSフォールバック対応）＋テスト用フォールバック合成 |
| `app/director.py` | 映像シーン温度・360°世界・通過イベント・衝撃イベントに基づく AIダビング監督プラン自動構築 (`build_auto_director_plan`)・検証 (`validate_director_plan`: `±180°/±90°`・`world_360`)・冪等キーフレーム合成 (`apply_director_plan_to_scene`) |
| `app/spatial/` | 空間音響エンジン: `hrtf.py` (SOFA + 内蔵物理HRTF・2段オールパス無相関化 `decorrelate_allpass`・時間変化キーフレームゲイン補間・`sfx_impact`/`sfx_sub`/`ambience` 距離モデル), `ambisonics.py` (AmbiX 1〜7次 FOA/HOA・10方向360°無相関ベッド・`sfx_sub` ACN0・RF64対応), `atmos.py` (Dolby Atmos 互換 7.1.4 360°無相関Bed + Dynamic Objects・`sfx_sub` LFE / `sfx_impact` aux), `adm_bwf.py` (ITU-R BS.2076 ADM BWF), `mastering.py` (LUFS/TruePeak・スマートインパクト免除ダッキング) |
| `app/pipeline.py` | 全自動統合パイプライン (`run_job`: `enable_sfx_multilayer`, `enable_director`, `enable_world_360`, `scene_objects_hint` 対応)、3D配置再Mix (`remix_job`)、再分離不要の高速監督リミックス (`apply_director_plan_and_remix`)、360°事前解析 (`inspect_video_world`) |
| `app/server.py` | FastAPI バックエンド・ジョブ管理・エージェント連携ブリッジ (`/api/agent/*`)・監督/事前解析 API (`/api/videos/inspect-world`, `/api/jobs/{id}/director-context`, `/api/jobs/{id}/media/director_sheet`, `/api/jobs/{id}/director-plan`) |
| `app/static/` | バニラ HTML/CSS/JS のシネマティック空間音響スタジオUI（A/B比較・3D空間レーダー360°ガイド＆仰角表示・シーン温度＆360°＆AIダビング監督カード） |
| `tools/ciness_agent.py` | エージェント向け CLI / 常駐ランナー (`status`, `start-server`, `watch`, `run-job`, `inspect-world`, `director-context`, `apply-director`) |
| `tools/ciness_mcp.py` | stdio MCP サーバー（登録名 `ciness`、9ツール提供） |

## 厳守すべき規約・注意点

1. **仮想環境と PyTorch の保護**:
   - Python は `uv venv --python 3.11 .venv` を使用する（システム Python 3.13 等は使わない）。
   - Windows NVIDIA 環境の CUDA 版 PyTorch（`torch==2.6.0+cu124`）を維持する。**`audio-separator` は絶対に入れない**。
   - macOS Apple Silicon 環境では `PYTORCH_ENABLE_MPS_FALLBACK=1` を有効化して未対応演算の CPU フォールバックを保証する。
   - 追加で torch 依存パッケージを入れる場合は `--no-deps` を付け、純 Python 依存を手動で追加する。
2. **暗黙ダウンロード禁止**:
   - 推論処理や `status` / `check_system` の中でモデルを自動ダウンロードしない。未取得モデルはエラーまたはモデル管理モーダルへの案内で明示する。
   - 隣接 `../SoundSeparator`（または `SOUNDSEPARATOR_DIR`）が存在する場合、BandIt Plus / DBS-TIGER-DnR / MelBand Dereverb / Nemotron-3-Diarization / Nemotron RE-USE をコピーせずそのまま共有参照する。
3. **空間座標系（全モジュール共通）**:
   - サンプルレート: `48000` Hz
   - `azimuth_deg` ∈ `[-180, 180]`: `0°`=正面、**負=左 (`-90°`=真左)**、**正=右 (`+90°`=真右)**
   - `elevation_deg` ∈ `[-90, 90]`: 正=上
   - `distance_m` ∈ `[0.5, 15.0]`: 近いほどドライ＆大音量、遠いほど距離減衰＋空気吸収LPF＋リバーブ比増加
4. **FastAPI の非同期安全性**:
   - 重い同期処理・推論・FFmpeg 呼び出しは `def` またはバックグラウンドスレッドにし、`async def` 内でイベントループをブロックしない。
5. **改行コードと文字化け対策**:
   - `setup.bat` / `run.bat` は Windows CRLF（`.gitattributes` で `*.bat -text`）。
   - `setup.sh` / `run.sh` は Unix LF（CR なし、`.gitattributes` で `*.sh text eol=lf`）。
   - PowerShell / シェルから Python を呼ぶときは `-X utf8`（または `PYTHONIOENCODING=utf-8`）を指定する。

## テストコマンド

```powershell
# Windows: 全ユニットテスト
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v

# macOS / Linux: 全ユニットテスト
.venv/bin/python -X utf8 -m unittest discover -s tests -v

# ハードウェア診断・クロスプラットフォーム・MCPテストのみ
.venv\Scripts\python.exe -X utf8 -m unittest tests/test_hardware_advisor.py -v
```
