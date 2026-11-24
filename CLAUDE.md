# CineSS 開発・運用ガイド

映像サラウンド MixMaster ＆ バイノーラル出力 Web アプリ。UI・コメント・ドキュメント・対話はすべて日本語。

## 起動とエージェント連携の入口

```
通常起動:   run.bat → .venv\Scripts\python.exe -X utf8 -m uvicorn app.server:app --host 127.0.0.1 --port 8780
エージェント:
  1. .venv\Scripts\python.exe -X utf8 tools/ciness_agent.py status
  2. .venv\Scripts\python.exe -X utf8 tools/ciness_agent.py start-server
  3. .venv\Scripts\python.exe -X utf8 tools/ciness_agent.py watch --server http://127.0.0.1:8780
```

- **未導入環境**: `.venv\Scripts\python.exe` が無ければ `setup.bat`（Python 3.11 + CUDA 12.4 PyTorch + `requirements.txt`）を実行する。
- **導入済み環境**: `start-server` と `watch` を実行し、ユーザーに `http://127.0.0.1:8780` を案内する。ユーザーが画面上で **「▶ 処理開始」** を押すと、`watch` が `queued_for_agent` を検知して自動受注・実行する。
- **MCP 登録**: リポジトリ直下の `.mcp.json`（登録名 `ciness`・相対パス）から `tools/ciness_mcp.py` を起動できる。

## アーキテクチャ

| ファイル | 役割 |
|---|---|
| `app/config.py` | ルート解決・出力/ジョブ/モデルパス・既定ポート(`8780`)・`48000`Hz・`find_soundseparator_dir()` |
| `app/hardware_advisor.py` | 実PCの CPU/GPU/RAM/FFmpeg/Disk/SoundSeparator 計測、✅/⚠️/✕ 判定、有音/無音の推奨動画長・モデル選定 (`check_system`, `evaluate_video_plan`) |
| `app/model_manager.py` | SoundSeparator共有モデル＋CineSS追加モデルのカタログ・状態検出・明示的ダウンロード (`list_models`, `download_model`) |
| `app/vision/analyzer.py` | 動画検査 (`inspect_video`)・OpenCV顔/口元モーション/動き重心追跡 (`analyze_video_spatial`) |
| `app/audio/separator_bridge.py` | SoundSeparator CLI 流用 DME 3分離 / 残響除去 + ローカルDSP 3分離フォールバック |
| `app/audio/speaker_splitter.py` | Nemotron-3-Diarization / 音響クラスタリングによる話者別声分け・重なり分離・口元モーション自動紐付け |
| `app/audio/foley_engine.py` | HunyuanVideo-Foley XL/XXL ラッパー（チャンク生成・オフロード対応）＋テスト用フォールバック合成 |
| `app/spatial/` | 空間音響エンジン: `hrtf.py` (SOFA + 内蔵物理HRTF), `ambisonics.py` (AmbiX 1〜7次 FOA/HOA・RF64対応), `atmos.py` (Dolby Atmos 互換 7.1.4 Bed + Dynamic Objects・完全フリー), `adm_bwf.py` (ITU-R BS.2076 ADM BWF), `mastering.py` (LUFS/TruePeak) |
| `app/pipeline.py` | 全自動統合パイプライン (`run_job`) および 3D配置変更からの高速再Mix (`remix_job`) |
| `app/server.py` | FastAPI バックエンド・ジョブ管理・エージェント連携ブリッジ API (`/api/agent/*`) |
| `app/static/` | バニラ HTML/CSS/JS のシネマティック空間音響スタジオUI（A/B比較・3Dトップダウン空間レーダー） |
| `tools/ciness_agent.py` | エージェント向け CLI / 常駐ジョブ待機ランナー (`status`, `start-server`, `watch`, `run-job`) |
| `tools/ciness_mcp.py` | stdio MCP サーバー（登録名 `ciness`、6ツール提供） |

## 厳守すべき規約・注意点

1. **仮想環境と PyTorch の保護**:
   - Python は `uv venv --python 3.11 .venv` を使用する（システム Python 3.13 等は使わない）。
   - CUDA 版 PyTorch（`torch==2.6.0+cu124`）を維持する。**`audio-separator` は絶対に入れない**。
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
   - PowerShell から Python を呼ぶときは `-X utf8` または `$env:PYTHONIOENCODING="utf-8"` を指定する。

## テストコマンド

```powershell
# 全ユニットテスト
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v

# pytest での実行
.venv\Scripts\python.exe -X utf8 -m pytest tests -v

# ハードウェア診断のみ
.venv\Scripts\python.exe -X utf8 -m unittest tests/test_hardware_advisor.py -v
```
