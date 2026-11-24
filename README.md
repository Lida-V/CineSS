# CineSS — 映像サラウンド MixMaster ＆ バイノーラル出力スタジオ

**CineSS** は、手元の Windows PC 上で映像の音声を立体音響（**Dolby Atmos 互換 ADM BWF [完全フリー構成]** / **Ambisonics 1〜7次 AmbiX**）へ再構築し、**ヘッドホンで映画館のような3D定位を体感できるバイノーラル映像（MP4）** を出力するローカル Web アプリケーションです。

AI エージェント（Claude Code / Antigravity / MCP 対応クライアント）とのネイティブ連携を備えており、「リポジトリを読ませるだけで環境確認→Web アプリ起動→ブラウザ上の『▶ 処理開始』ボタン押下をトリガーにエージェントが処理開始」という体験を実現します。

---

## 主な機能

1. **PC動作環境チェック ＆ スペック別おすすめガイド**
   - 実機の **CPU / GPU（CUDA・VRAM空き/総容量）/ RAM / FFmpeg / 空きディスク** を自動計測し、**✅ 快適 / ⚠️ 警告 / ✕ 不足** で分かりやすく表示します。
   - PC スペックに応じて **Tier S / A / B / C** を自動判定し、「有音動画リミックス」「無音動画 Foley 生成」それぞれについて **推奨動画長（快適目安・上限目安）・選定モデル・期待クオリティ** を案内します。
   - 隣接フォルダに `SoundSeparator`（または環境変数 `SOUNDSEPARATOR_DIR`）がある場合は、取得済みの分離・話者分離モデルをコピーせず自動共有します。
2. **有音動画リミックス（3分離 ＋ 人物別声分け ＋ 3D位置再現）**
   - 映像音声を **話し声（Dialog）/ 効果音（SFX）/ BGM（Music）** へ3分離（BandIt Plus / DBS-TIGER-DnR、単体環境やCPU環境ではローカルDSPフォールバック）。
   - 複数話者の発話を人物別に声分けし、映像の顔・口元モーション活性度と自動紐付けして「画面左の人は左から、右の人は右から、遠くの人は奥行きをもって」定位させます。
3. **無音動画 Foley 生成（HunyuanVideo-Foley ＋ 空間配置）**
   - 無音映像から動き・カット・シーン特徴を解析し、HunyuanVideo-Foley（XXL / XL、VRAMに応じたオフロード対応）による 48kHz 効果音・環境音生成と3D空間配置を行います（オフライン検証用のモーション合成プレビューフォールバックにも対応）。
4. **プロ仕様サラウンド MixMaster ＆ バイノーラル出力（完全フリー構成）**
   - **Dolby Atmos 互換（完全フリー・国際標準）**: 商用有償エンコーダ（E-AC-3 JOC）に依存せず、国際標準規格 **ITU-R BS.2076 準拠 ADM BWF `.wav`**（7.1.4 Bed + Dynamic Objects）＋ 7.1.4 確認用 WAV ＋ 内蔵 HRTF 直接バイノーラルを出力します。DaVinci Resolve / EAR / Reaper 等と互換があります（民生 TV 向け `.ec3 (E-AC-3 JOC)` 配信変換のみ外部対応ツールが必要です）。
   - **Ambisonics 1〜7次対応（AmbiX: ACN + SN3D + Max-rE デコード）**:
     - `ambisonics_foa`（**1次 FOA / 4ch**、約 44 MB/分・48kHz 32bit float）
     - `ambisonics_hoa`（**3次 HOA / 16ch・既定**、約 176 MB/分）
     - `ambisonics_hoa5`（**5次 HOA / 36ch**、約 396 MB/分）
     - `ambisonics_hoa7`（**7次 HOA / 64ch・IEM Plug-in Suite / SPARTA 準拠上限**、約 703 MB/分）
     - 推定出力サイズが 3.5 GB を超える長尺・高次 WAV は自動で **RF64（64bit拡張WAV）** 形式で書き出します。
     - PC 診断（RAM 16GB 以下や低スペックでは 3次以下推奨・5次/7次は ⚠️ 警告、RAM 32GB 以上＆8コア以上の高スペックでは 7次 ✅ 快適）と連動して容量目安と警告を表示します。
   - **HRTF バイノーラル**: 実測 MIT KEMAR SOFA HRTF（任意取得）および内蔵物理ベース HRTF（Woodworth-Schlosberg ITD・球体回折 ILD・ピンナフィルタ・距離空気吸収＆初期反射/リバーブ）。
   - **ラウドネスマスタリング**: Cinema (-24 LKFS) / Atmos (-18 LKFS) / Web (-14 LKFS) ＆ True Peak ≤ -1.0 dBTP 保護。
5. **インタラクティブ 3D 空間ステージビューアー ＆ A/B 比較**
   - ヘッドホン試聴向けのワンクリック **元音声 ⇔ バイノーラル** 切替プレイヤー。
   - 3D トップダウン空間レーダー上で音源ノードをドラッグ調整し、分離済みステムを再利用して数秒で再 Mix できます。

---

## 動作要件

- **OS**: Windows 10 / 11（64bit）
- **必須ツール**: [uv](https://docs.astral.sh/uv/)、FFmpeg（`ffmpeg` / `ffprobe` が PATH に通っていること）
- **Python**: Python 3.11（`setup.bat` が `uv venv --python 3.11 .venv` で自動構築）
- **ハードウェア目安**:
  - **Tier S（快適・フル機能）**: 8コア以上 CPU / CUDA GPU VRAM 16GB以上 / RAM 32GB以上（有音 30〜90分・無音Foley 60〜180秒目安）
  - **Tier A（高品質）**: CUDA GPU VRAM 12GB前後 / RAM 16GB以上（有音 15〜45分・無音Foley XLオフロード併用 30〜90秒目安）
  - **Tier B（標準）**: CUDA GPU VRAM 8GB前後 / RAM 16GB（有音 5〜15分・無音Foley XLオフロード必須 15〜30秒限定）
  - **Tier C（CPU軽量）**: GPUなし / RAM 8GB〜（ローカルDSP 3分離＋物理HRTFによる 1〜3分の有音短尺リミックス向け。無音AI Foley本番生成は非対応）

> **注意**: VRAM目安は提供元推奨と暫定目安に基づく案内であり、他アプリのメモリ使用状況や動画の複雑さによって変動します。

---

## クイックスタート

### 1. セットアップ（初回のみ）

```bat
setup.bat
```

- `.venv`（Python 3.11）を作成し、NVIDIA GPU がある場合は **CUDA 12.4 版 PyTorch (`torch==2.6.0+cu124`)** を優先導入します。
- **重要**: `audio-separator` は PyTorch を CPU 版に上書きするため導入しません。
- セットアップ時に AI モデルは自動ダウンロードしません。

### 2. Web アプリ起動

```bat
run.bat
```

ブラウザで **`http://127.0.0.1:8780`** が自動的に開きます（ポートは環境変数 `CINESS_PORT` で変更可能）。
起動後、右上の **「AIモデル管理」** から必要なモデルを取得してください（隣接 `SoundSeparator` がある場合は BandIt Plus / DBS-TIGER-DnR / MelBand Dereverb / Nemotron-3-Diarization 等が自動で緑バッジ「SoundSeparator共有」になります）。

---

## AI エージェント連携（「開始ボタン」でエージェント実行）

### 方法 A: CLI 常駐ランナー（おすすめ）

```powershell
# 1. 環境診断とモデル確認
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py status

# 2. Web アプリ起動＆ブラウザ表示
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py start-server

# 3. Web UI の「▶ 処理開始」ボタンを待ち受けて自動実行
.venv\Scripts\python.exe -X utf8 tools/ciness_agent.py watch --server http://127.0.0.1:8780
```

`watch` を起動すると、Web UI ヘッダーが **「🤖 エージェント接続中・待機OK」** に変わり、画面上で **「▶ 処理開始」** を押すだけでエージェントがジョブを受注して実行・ログ出力・完了まで自動で進行します。

### 方法 B: MCP サーバー（`.mcp.json` 同梱）

リポジトリ直下の `.mcp.json`（登録名 `ciness`）を利用すると、MCP 対応クライアントから以下の6ツールを直接呼び出せます。

| ツール名 | 機能 |
|---|---|
| `ciness_status` | PC環境診断・サーバー稼働・エージェント接続・モデル一覧の確認 |
| `ciness_start_server` | FastAPI Web サーバーの起動確認＆ブラウザ起動 |
| `ciness_wait_job` | ハートビートを送りながら Web UI からのジョブ投入を待機 |
| `ciness_claim_and_run_job` | キューのジョブを受注してバックグラウンド（または同期）で実行 |
| `ciness_job_status` | ジョブの進捗・ログ・成果物パスの確認 |
| `ciness_cancel_job` | ジョブの安全キャンセル |

---

## テスト

```powershell
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

---

## ライセンス

本リポジトリのソースコードは [MIT License](LICENSE) のもとで公開されています。
なお、各 AI モデルおよび外部ライブラリのライセンス・利用規約はそれぞれの配布元に従います（AI モデルの重みファイル自体は本リポジトリに含まれておらず、利用者がモデル管理画面などから明示的に取得して使用する設計です）。
