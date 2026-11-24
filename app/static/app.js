/* ============================================================================
   CineSS — 映像サラウンドMixMaster & バイノーラルスタジオ
   フロントエンド制御・3D空間レーダー・Web Audio シームレスA/B比較エンジン
============================================================================ */
(() => {
  "use strict";

  // --------------------------------------------------------------------------
  // アプリケーション状態
  // --------------------------------------------------------------------------
  const state = {
    systemInfo: null,
    upload: null, // { upload_id, filename, preview_url, inspection, plan }
    currentStep: 1,
    activeJobId: null,
    activeJob: null,
    jobPollTimer: null,
    elapsedTimer: null,
    healthTimer: null,
    modelsPollTimer: null,

    // Step 3: 3D シーン＆ミキサー
    originalScene: null,
    scene: null,
    selectedObjectId: null,
    sceneDirty: false,
    radarDraggingId: null,
    radarHoverId: null,
    radarAnimFrame: null,
    directorContext: null,
    initialDirectorPlan: null,

    // Step 3: Web Audio A/B 比較試聴
    abSource: "binaural", // "original" | "binaural"
    audioCtx: null,
    bufOriginal: null,
    bufBinaural: null,
    srcOriginal: null,
    srcBinaural: null,
    gainOriginal: null,
    gainBinaural: null,
    analyserL: null,
    analyserR: null,
    webAudioReady: false,
    webAudioPlaying: false,
    audioStartCtxTime: 0,
    audioStartOffset: 0,
    stemPreviewAudio: null,
  };

  // 役割別パレット（話者が複数いる場合はシアン〜ブルー〜ミント系で視認区別）
  const DIALOG_COLORS = ["#22d3ee", "#38bdf8", "#34d399", "#2dd4bf", "#60a5fa", "#a3e635"];
  const ROLE_COLORS = {
    sfx_sub: "#f43f5e",
    sfx_impact: "#f97316",
    sfx: "#f59e0b",
    foley: "#fbbf24",
    music: "#818cf8",
    ambience: "#c084fc",
  };
  const ROLE_LABELS_JA = {
    dialog: "話者/セリフ",
    speaker: "話者/セリフ",
    voice: "話者/セリフ",
    sfx_sub: "SFX重低音 (Sub)",
    sfx_impact: "SFX衝撃 (Impact)",
    sfx_impact_left: "SFX衝撃 (左)",
    sfx_impact_center: "SFX衝撃 (中央)",
    sfx_impact_right: "SFX衝撃 (右)",
    sfx_ambience: "環境音 (Ambience)",
    sfx: "効果音 (SFX)",
    foley: "Foley効果音",
    music: "BGM",
    ambience: "環境音 (Ambience)",
  };

  function formatRoleLabelJa(role) {
    const r = String(role || "audio").toLowerCase();
    return ROLE_LABELS_JA[r] || role || "audio";
  }

  // --------------------------------------------------------------------------
  // DOM ユーティリティ
  // --------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
  }

  function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
  }

  function formatTimecode(sec) {
    const s = Math.max(0, Number(sec) || 0);
    const mins = Math.floor(s / 60);
    const rem = s - mins * 60;
    return `${String(mins).padStart(2, "0")}:${rem.toFixed(1).padStart(4, "0")}`;
  }

  function formatDurationLabel(sec) {
    const s = Math.max(0, Number(sec) || 0);
    if (s < 60) return `${s.toFixed(1)}秒`;
    const m = Math.floor(s / 60);
    const rem = Math.round(s - m * 60);
    return rem ? `${m}分${rem}秒` : `${m}分`;
  }

  function statusToBadgeInfo(status) {
    if (status === "ok") return { cls: "is-ok", text: "✅ 快適" };
    if (status === "warn") return { cls: "is-warn", text: "⚠️ 警告" };
    return { cls: "is-error", text: "✕ 不足" };
  }

  function showToast(message, type = "info", durationMs = 4200) {
    const box = $("toast-container");
    if (!box) return;
    const el = document.createElement("div");
    el.className = `toast ${type === "error" ? "is-error" : type === "ok" ? "is-ok" : ""}`;
    el.textContent = message;
    box.appendChild(el);
    window.setTimeout(() => {
      el.style.opacity = "0";
      el.style.transition = "opacity 0.25s ease";
      window.setTimeout(() => el.remove(), 260);
    }, durationMs);
  }

  async function apiJson(url, options = {}) {
    const resp = await fetch(url, options);
    let data = null;
    try {
      data = await resp.json();
    } catch (_) {
      data = null;
    }
    if (!resp.ok) {
      const detail = (data && (data.detail || data.error || data.message)) || `HTTP ${resp.status}`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  // --------------------------------------------------------------------------
  // ステップ切替
  // --------------------------------------------------------------------------
  function setStep(stepNum) {
    state.currentStep = stepNum;
    const sections = [null, $("step1-section"), $("step2-section"), $("step3-section")];
    for (let i = 1; i <= 3; i++) {
      if (sections[i]) sections[i].classList.toggle("hidden", i !== stepNum);
      const tab = $(`step-tab-${i}`);
      if (tab) {
        tab.classList.toggle("is-active", i === stepNum);
      }
    }
    if (stepNum === 3) {
      resizeRadarCanvas();
      startRadarLoop();
    } else {
      stopRadarLoop();
      pauseStudioPlayback();
    }
  }

  // --------------------------------------------------------------------------
  // PC動作環境診断＆エージェント連携ステータス
  // --------------------------------------------------------------------------
  function updateAgentBadge(info) {
    const badge = $("agent-status-badge");
    const text = $("agent-status-text");
    if (!badge || !text) return;

    const connected = Boolean(info && info.agent_connected);
    const agentStatus = (info && info.agent_status) || (connected ? "idle" : "offline");
    const agentName = (info && info.agent_name) || "ciness-agent";

    badge.classList.remove("is-connected", "is-working", "is-offline");
    if (connected && agentStatus === "working") {
      badge.classList.add("is-working");
      text.textContent = `エージェント処理中（${agentName}）`;
    } else if (connected) {
      badge.classList.add("is-connected");
      text.textContent = `エージェント接続中・待機OK（${agentName}）`;
    } else {
      badge.classList.add("is-offline");
      text.textContent = "エージェント未接続（ローカル自動実行可）";
    }
    updateExecutionModeHint();
  }

  function applyCardStatus(cardEl, badgeEl, status, customText = null) {
    if (cardEl) {
      cardEl.classList.remove("status-ok", "status-warn", "status-error");
      if (status) cardEl.classList.add(`status-${status}`);
    }
    if (badgeEl) {
      const info = statusToBadgeInfo(status);
      badgeEl.className = `verdict-badge ${info.cls}`;
      badgeEl.textContent = customText || info.text;
    }
  }

  function renderSystemCheck(data) {
    state.systemInfo = data;
    updateAgentBadge(data);

    // 総合ステータス & Tier
    const overallPill = $("overall-status-pill");
    const ov = data.overall_status || "ok";
    const ovMap = {
      ok: { cls: "is-ok", text: "✅ 快適動作OK" },
      warn: { cls: "is-warn", text: "⚠️ 一部制限・注意あり" },
      error: { cls: "is-error", text: "✕ スペック・構成要確認" },
    };
    const ovInfo = ovMap[ov] || ovMap.warn;
    overallPill.className = `status-pill ${ovInfo.cls}`;
    overallPill.textContent = ovInfo.text;

    const recs = data.recommendations || {};
    const tierPill = $("tier-pill");
    if (recs.tier) {
      tierPill.classList.remove("hidden");
      tierPill.textContent = `TIER ${recs.tier}`;
    }
    $("tier-label-text").textContent = recs.tier_label || "";

    // CPU
    const cpu = data.cpu || {};
    applyCardStatus($("hw-card-cpu"), $("cpu-status-badge"), cpu.status);
    $("cpu-value").textContent = cpu.name || "不明なCPU";
    $("cpu-meta").textContent = `${cpu.cores ?? "--"} 物理コア / ${cpu.threads ?? "--"} 論理スレッド`;
    $("cpu-note").textContent = cpu.note || "";

    // GPU（NVIDIA CUDA / Apple Silicon MPS / 非NVIDIA GPU / CPU）
    const gpu = data.gpu || {};
    const mem = data.memory || {};
    const isMps = Boolean(gpu.mps || String(gpu.backend || "").toLowerCase() === "mps");
    const isCuda = Boolean(gpu.cuda);
    const isUnified = Boolean(gpu.unified_memory || mem.unified_memory);
    const isNonNvidia = Boolean(gpu.non_nvidia_gpu);

    applyCardStatus($("hw-card-gpu"), $("gpu-status-badge"), gpu.status);
    $("gpu-value").textContent = gpu.name || (isMps ? "Apple Silicon GPU (MPS)" : "未検出（CPUモード）");
    if (isCuda || (!isMps && gpu.vram_total_gb && gpu.vram_total_gb > 0)) {
      const drv = gpu.driver ? ` · Driver ${gpu.driver}` : "";
      $("gpu-meta").textContent =
        `CUDA ${isCuda ? "有効" : "無効"} · VRAM 空き ${Number(gpu.vram_free_gb || 0).toFixed(1)} / 総容量 ${Number(gpu.vram_total_gb || 0).toFixed(1)} GB${drv}`;
    } else if (isMps) {
      const memTotal = Number(gpu.unified_memory_gb || mem.total_gb || gpu.vram_total_gb || 0).toFixed(1);
      $("gpu-meta").textContent = isUnified
        ? `MPS 有効 · ユニファイドメモリ ${memTotal} GB（CPU/GPU共有）`
        : "Apple Silicon MPS 有効";
    } else if (isNonNvidia) {
      $("gpu-meta").textContent = "非NVIDIA GPU（CUDA非対応 · CPU/DSP推論モード）";
    } else {
      $("gpu-meta").textContent = "CUDA / MPS 非対応 / VRAM 0.0 GB";
    }

    const gpuBadges = $("gpu-accel-badges");
    if (gpuBadges) {
      const pills = [];
      if (isCuda) pills.push(`<span class="accel-pill is-cuda">⚡ NVIDIA CUDA</span>`);
      if (isMps) pills.push(`<span class="accel-pill is-mps">🍎 Apple Silicon MPS</span>`);
      if (isUnified) pills.push(`<span class="accel-pill is-unified">🧠 ユニファイドメモリ共有</span>`);
      if (isNonNvidia && !isCuda && !isMps) {
        pills.push(`<span class="accel-pill is-non-nvidia">🧩 非NVIDIA GPU（AMD / Intel等）</span>`);
        pills.push(`<span class="accel-pill is-cpu">💻 CPUフォールバック</span>`);
      } else if (!isCuda && !isMps) {
        pills.push(`<span class="accel-pill is-cpu">💻 CPU動作モード</span>`);
      }
      if (pills.length) {
        gpuBadges.innerHTML = pills.join("");
        gpuBadges.classList.remove("hidden");
      } else {
        gpuBadges.innerHTML = "";
        gpuBadges.classList.add("hidden");
      }
    }
    $("gpu-note").textContent = gpu.note || "";

    const platformBanner = $("gpu-platform-note");
    if (platformBanner) {
      if (isMps) {
        platformBanner.className = "hw-platform-banner";
        platformBanner.textContent =
          "Apple Silicon MPS 環境です。PYTORCH_ENABLE_MPS_FALLBACK=1 により未対応演算を補完しつつ実行します（高負荷時はユニファイドメモリ残量にご注意ください）。";
      } else if (isUnified && !isMps && !isCuda) {
        platformBanner.className = "hw-platform-banner is-warn";
        platformBanner.textContent =
          "Apple Silicon 環境ですが PyTorch MPS が未有効のため CPU / DSP フォールバックモードで動作します。";
      } else if (isNonNvidia && !isCuda) {
        platformBanner.className = "hw-platform-banner is-warn";
        platformBanner.textContent =
          "非NVIDIA GPU 環境のため CUDA は使用できません。有音リミックスは CPU / DSP 分離で動作し、無音AI Foley生成はテスト用フォールバックのみ利用できます。";
      } else {
        platformBanner.className = "hw-platform-banner hidden";
        platformBanner.textContent = "";
      }
    }

    // RAM
    applyCardStatus($("hw-card-ram"), $("ram-status-badge"), mem.status);
    $("ram-value").textContent = isUnified
      ? `ユニファイドメモリ ${Number(mem.total_gb || 0).toFixed(1)} GB`
      : `総容量 ${Number(mem.total_gb || 0).toFixed(1)} GB`;
    $("ram-meta").textContent = `利用可能 ${Number(mem.available_gb || 0).toFixed(1)} GB / 総容量 ${Number(mem.total_gb || 0).toFixed(1)} GB`;
    const ramBadges = $("ram-accel-badges");
    if (ramBadges) {
      if (isUnified) {
        ramBadges.innerHTML = `<span class="accel-pill is-unified">🧠 CPU / GPU 共有ユニファイドメモリ</span>`;
        ramBadges.classList.remove("hidden");
      } else {
        ramBadges.innerHTML = "";
        ramBadges.classList.add("hidden");
      }
    }
    $("ram-note").textContent = mem.note || "";

    // FFmpeg / ディスク / SoundSeparator バッジ
    const ff = data.ffmpeg || {};
    const ffBadge = $("badge-ffmpeg");
    ffBadge.className = `aux-badge is-${ff.status || "error"}`;
    $("ffmpeg-detail").textContent = ff.available ? `利用可能 (${ff.version || "検出済み"})` : "未検出（要導入）";

    const disk = data.disk || {};
    const diskBadge = $("badge-disk");
    diskBadge.className = `aux-badge is-${disk.status || "warn"}`;
    $("disk-detail").textContent = `空き ${Number(disk.free_gb || 0).toFixed(1)} GB`;

    const ss = data.soundseparator || {};
    const ssBadge = $("badge-soundsep");
    if (ss.detected) {
      const cnt = Array.isArray(ss.shared_models) ? ss.shared_models.length : 0;
      ssBadge.className = "aux-badge is-ok";
      $("soundsep-detail").textContent = `連携済み（共有モデル ${cnt} 件）`;
    } else {
      ssBadge.className = "aux-badge is-warn";
      $("soundsep-detail").textContent = "未検出（CineSS単体モード可）";
    }

    // Ambisonics 推奨上限次数バッジ（Tier / RAM連携）
    const maxAmbi = getMaxRecommendedAmbiOrder(data);
    const maxAmbiCh = (maxAmbi + 1) * (maxAmbi + 1);
    const ambiBadge = $("badge-ambi-limit");
    if (ambiBadge) {
      ambiBadge.className = `aux-badge ${maxAmbi >= 5 ? "is-ok" : "is-warn"}`;
      $("ambi-limit-detail").textContent = `${maxAmbi}次 (${maxAmbiCh}ch) まで快適`;
    }
    updateAmbisonicsCapacityUI();

    // 有音動画ガイド
    const v = recs.voiced || {};
    applyCardStatus(
      null,
      $("voiced-status-badge"),
      v.status,
      v.supported === false ? "✕ 非対応" : statusToBadgeInfo(v.status).text
    );
    const vGrade = $("voiced-grade-badge");
    vGrade.className = `grade-badge grade-${v.quality_grade || "B"}`;
    vGrade.textContent = `品質 ${v.quality_grade || "-"}ランク`;
    $("voiced-comfortable").textContent = v.comfortable_minutes ? `〜${v.comfortable_minutes} 分` : "短尺のみ";
    $("voiced-max").textContent = v.max_recommended_minutes ? `最大 ${v.max_recommended_minutes} 分` : "--";
    $("voiced-quality-label").textContent = (v.quality_label || "").split("：")[0] || `${v.quality_grade || "-"}ランク`;
    const vm = v.models || {};
    $("voiced-model-sep").textContent = vm.separator || "--";
    $("voiced-model-diar").textContent = vm.diarization || "--";
    $("voiced-model-overlap").textContent = vm.overlap || "--";
    $("voiced-model-vision").textContent = vm.vision || "--";
    $("voiced-reason").textContent = v.reason || "";

    const vAlert = $("voiced-alert");
    if (v.status === "warn" || v.status === "error") {
      vAlert.className = `guide-alert is-${v.status}`;
      vAlert.textContent =
        v.status === "error"
          ? `✕ ${v.reason || "現在の環境では有音リミックスの実行に制限があります。"}`
          : `⚠️ ${v.quality_label || "スペックに応じた標準/軽量設定が適用されます。長尺動画はお試し区間をご活用ください。"}`;
      vAlert.classList.remove("hidden");
    } else {
      vAlert.classList.add("hidden");
    }

    // 無音動画ガイド
    const s = recs.silent || {};
    applyCardStatus(
      null,
      $("silent-status-badge"),
      s.status,
      s.supported === false ? "✕ 非対応" : statusToBadgeInfo(s.status).text
    );
    const sGrade = $("silent-grade-badge");
    sGrade.className = `grade-badge grade-${s.quality_grade || "C"}`;
    sGrade.textContent = s.supported === false ? "GPU必須" : `品質 ${s.quality_grade || "-"}ランク`;
    $("silent-comfortable").textContent = s.comfortable_seconds ? `〜${s.comfortable_seconds} 秒` : "非対応";
    $("silent-max").textContent = s.max_recommended_seconds ? `最大 ${s.max_recommended_seconds} 秒` : "テスト合成のみ";
    $("silent-quality-label").textContent = (s.quality_label || "").split("：")[0] || "--";
    const sm = s.models || {};
    $("silent-model-foley").textContent = sm.foley || "--";
    $("silent-model-offload").textContent = sm.offload ? "ON（CPUオフロード有効）" : "OFF（VRAM常駐・高速）";
    $("silent-model-vision").textContent = sm.vision || "--";
    $("silent-reason").textContent = s.reason || "";

    const sAlert = $("silent-alert");
    if (s.status === "warn" || s.status === "error") {
      sAlert.className = `guide-alert is-${s.status}`;
      sAlert.textContent =
        s.status === "error"
          ? `✕ ${s.reason || "無音動画のAI Foley生成にはCUDA GPUが必要です。"}`
          : `⚠️ ${s.reason || "VRAM節約のためオフロード併用・短尺動画での実行を推奨します。"}`;
      sAlert.classList.remove("hidden");
    } else {
      sAlert.classList.add("hidden");
    }
  }

  async function loadSystemCheck(showNotify = false) {
    const btn = $("btn-recheck-system");
    if (btn) btn.disabled = true;
    try {
      const data = await apiJson("/api/system/check");
      renderSystemCheck(data);
      if (showNotify) showToast("PC動作環境とエージェント接続状態を再診断しました。", "ok");
    } catch (err) {
      showToast(`環境診断の取得に失敗しました: ${err.message}`, "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function pollLightHealth() {
    try {
      const data = await apiJson("/api/health");
      if (state.systemInfo) {
        state.systemInfo.agent_connected = data.agent_connected;
        state.systemInfo.agent_status = data.agent_status;
        if (data.agent_name) state.systemInfo.agent_name = data.agent_name;
        if ("active_job_id" in data) state.systemInfo.active_job_id = data.active_job_id;
      }
      updateAgentBadge({
        agent_connected: data.agent_connected,
        agent_status: data.agent_status,
        agent_name: data.agent_name || (state.systemInfo && state.systemInfo.agent_name),
        active_job_id: data.active_job_id,
      });
    } catch (_) {
      // サーバー一時応答遅延は無視して次周期で再試行
    }
  }

  // --------------------------------------------------------------------------
  // ステップ1: 動画アップロード・検査・プラン再評価・ジョブ開始
  // --------------------------------------------------------------------------
  function getRadioValue(name, fallback) {
    const checked = document.querySelector(`input[name="${name}"]:checked`);
    return checked ? checked.value : fallback;
  }

  function getMaxRecommendedAmbiOrder(sysInfo = state.systemInfo) {
    const recs = (sysInfo && sysInfo.recommendations) || {};
    if (typeof recs.max_recommended_ambisonics_order === "number") {
      return recs.max_recommended_ambisonics_order;
    }
    const tier = String(recs.tier || "B").toUpperCase();
    const ramGb = Number((sysInfo && sysInfo.memory && sysInfo.memory.total_gb) || 0);
    if (tier === "S" && ramGb >= 30) return 7;
    if (tier === "A" || (tier === "S" && ramGb >= 16)) return 5;
    return 3;
  }

  function getSpatialSelection() {
    const family = getRadioValue("spatial_family", "atmos");
    if (family === "atmos") {
      return { spatial_format: "atmos", ambisonics_order: null };
    }
    const fmt = getRadioValue("spatial_format", "ambisonics_hoa");
    const orderMap = {
      ambisonics_foa: 1,
      ambisonics_hoa: 3,
      ambisonics_hoa5: 5,
      ambisonics_hoa7: 7,
    };
    return { spatial_format: fmt, ambisonics_order: orderMap[fmt] || 3 };
  }

  function formatBytesJa(bytes) {
    const b = Math.max(0, Number(bytes) || 0);
    if (b >= 1024 ** 3) return `${(b / 1024 ** 3).toFixed(2)} GB`;
    return `${(b / 1024 ** 2).toFixed(1)} MB`;
  }

  function updateAmbisonicsCapacityUI() {
    const family = getRadioValue("spatial_family", "atmos");
    const panel = $("ambisonics-order-panel");
    if (!panel) return;
    panel.classList.toggle("hidden", family !== "ambisonics");
    if (family !== "ambisonics") return;

    const { ambisonics_order: order } = getSpatialSelection();
    const ch = (order + 1) * (order + 1);
    const maxOrder = getMaxRecommendedAmbiOrder();

    // 対象動画長（未アップロード時は1分あたりの目安を表示）
    let durSec = 60.0;
    let durNote = "1分あたり目安";
    if (state.upload && state.upload.inspection) {
      const rawDur = Number(state.upload.inspection.duration_sec) || 0;
      const trimOn = $("chk-preview-trim") && $("chk-preview-trim").checked;
      const trimDur = Number($("input-preview-duration").value) || 15;
      durSec = trimOn ? Math.min(rawDur || trimDur, trimDur) : Math.max(1.0, rawDur);
      durNote = `対象 ${formatDurationLabel(durSec)}`;
    }

    // 48kHz * 32bit float (4 bytes) * ch
    const estBytes = durSec * 48000 * 4 * ch;
    const estGb = estBytes / 1024 ** 3;
    const diskFreeGb = Number((state.systemInfo && state.systemInfo.disk && state.systemInfo.disk.free_gb) || 999);
    const ramGb = Number((state.systemInfo && state.systemInfo.memory && state.systemInfo.memory.total_gb) || 0);

    const verdict = $("ambi-order-verdict");
    const capBar = $("ambi-capacity-bar");
    const capText = $("ambi-capacity-text");

    // hardware_advisor の orders["1"|"3"|"5"|"7"] 情報があれば優先活用
    const recs = (state.systemInfo && state.systemInfo.recommendations) || {};
    const orderSpec = recs.ambisonics && recs.ambisonics.orders && recs.ambisonics.orders[String(order)];

    let level = (orderSpec && orderSpec.status) || "ok";
    let msg = `${order}次 HOA (${ch}ch AmbiX) · 想定WAV容量（${durNote} / 48kHz 32bit float）: 約 ${formatBytesJa(estBytes)}`;
    if (estGb > 3.5) {
      msg += "（3.5GB超のため自動でRF64書き出し）";
    }

    if (estGb > diskFreeGb * 0.8) {
      level = "error";
      msg += ` — ✕ ディスク空き容量（${diskFreeGb.toFixed(1)} GB）が不足する恐れがあります。`;
    } else if (order > maxOrder || level === "warn") {
      level = "warn";
      msg += orderSpec && orderSpec.note
        ? ` — ⚠️ ${orderSpec.note}`
        : ` — ⚠️ このPC（RAM ${ramGb ? ramGb.toFixed(0) + "GB" : "計測中"}）では ${maxOrder}次以下が快適目安です。処理時間・メモリ消費が増加します。`;
    } else if (order >= 5) {
      level = "ok";
      msg += orderSpec && orderSpec.note
        ? ` — ✅ ${orderSpec.note}`
        : " — ✅ このPCで高次Ambisonics処理に対応可能です。";
    }

    const badgeInfo = statusToBadgeInfo(level);
    verdict.className = `verdict-badge ${badgeInfo.cls}`;
    verdict.textContent = level === "ok" ? `✅ ${order}次 OK` : level === "warn" ? `⚠️ ${order}次 高負荷注意` : "✕ 容量不足";
    capBar.className = `ambi-capacity-bar is-${level}`;
    capText.textContent = msg;
  }

  function updateExecutionModeHint() {
    const mode = getRadioValue("execution_mode", "auto");
    const hint = $("exec-mode-hint");
    if (!hint) return;
    const connected = Boolean(state.systemInfo && state.systemInfo.agent_connected);
    if (mode === "agent") {
      hint.textContent = connected
        ? "接続中のAIエージェントにジョブを依頼し、リアルタイムに進捗・判断ログを表示します。"
        : "エージェント待機キューに登録します（未接続でも後からローカル即時実行へ切替可能）。";
    } else if (mode === "local") {
      hint.textContent = "サーバー内蔵のローカルパイプラインで即座に処理を開始します。";
    } else {
      hint.textContent = connected
        ? "現在エージェント接続中のため、自動でエージェントへ依頼します。"
        : "現在エージェント未接続のため、ローカル即時実行で自動的に開始します。";
    }
  }

  function renderVideoPlan(plan) {
    const banner = $("video-plan-banner");
    if (!banner || !plan) return;
    banner.classList.remove("hidden", "status-ok", "status-warn", "status-error");
    banner.classList.add(`status-${plan.status || "ok"}`);

    const badge = $("plan-status-badge");
    const bInfo = statusToBadgeInfo(plan.status || "ok");
    badge.className = `verdict-badge ${bInfo.cls}`;
    badge.textContent = bInfo.text;

    const modeLabel = plan.effective_mode === "silent_foley" ? "無音Foley生成" : "有音リミックス";
    $("plan-headline").textContent = `${modeLabel} · 品質 ${plan.quality_grade || "-"}ランク`;
    $("plan-eta").textContent = plan.estimated_time_label ? `目安 ${plan.estimated_time_label}` : "";

    const warnList = $("plan-warnings");
    warnList.innerHTML = "";
    const warnings = Array.isArray(plan.warnings) ? plan.warnings : [];
    if (warnings.length) {
      warnList.classList.remove("hidden");
      warnings.forEach((w) => {
        const li = document.createElement("li");
        li.textContent = w;
        warnList.appendChild(li);
      });
    } else {
      warnList.classList.add("hidden");
    }

    const wavNote =
      plan.estimated_master_wav_mb != null
        ? `（マスターWAV目安: 約 ${
            plan.estimated_master_wav_gb >= 1
              ? Number(plan.estimated_master_wav_gb).toFixed(2) + " GB"
              : Number(plan.estimated_master_wav_mb).toFixed(1) + " MB"
          }${plan.requires_rf64 ? "・自動RF64" : ""}）`
        : "";
    $("plan-summary").textContent = (plan.quality_label || plan.quality_explanation || "") + (wavNote ? " " + wavNote : "");

    // 推奨上限超過時はお試し区間コントロールを自動提案
    if (plan.recommended_preview_seconds) {
      const durInput = $("input-preview-duration");
      if (durInput && !$("chk-preview-trim").checked) {
        durInput.value = String(plan.recommended_preview_seconds);
      }
      $("preview-trim-box").style.borderColor = "rgba(245, 158, 11, 0.58)";
    } else {
      $("preview-trim-box").style.borderColor = "";
    }
  }

  async function refreshVideoPlan() {
    if (!state.upload || !state.upload.inspection) return;
    const insp = state.upload.inspection;
    const trimEnabled = $("chk-preview-trim").checked;
    const trimDur = Number($("input-preview-duration").value) || 15;
    const rawDur = Number(insp.duration_sec) || 0;
    const targetDur = trimEnabled ? Math.min(rawDur || trimDur, trimDur) : rawDur;
    const hasAudio = !Boolean(insp.is_silent);

    const { spatial_format, ambisonics_order } = getSpatialSelection();
    updateAmbisonicsCapacityUI();

    try {
      const plan = await apiJson("/api/system/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          duration_sec: targetDur,
          has_audio: hasAudio,
          mode: getRadioValue("proc_mode", "auto"),
          quality_preset: $("select-quality").value,
          spatial_format,
          ambisonics_order,
        }),
      });
      state.upload.plan = plan;
      renderVideoPlan(plan);
    } catch (_) {
      // プラン更新失敗時は既存表示を維持
    }
  }

  async function handleVideoUpload(file) {
    if (!file) return;
    $("dropzone").classList.add("hidden");
    $("video-preview-card").classList.add("hidden");
    $("upload-progress-box").classList.remove("hidden");
    $("upload-progress-text").textContent = `「${file.name}」をアップロードして映像・音声を自動検査中…`;
    $("btn-start-job").disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await apiJson("/api/videos/upload", {
        method: "POST",
        body: formData,
      });
      state.upload = res;
      const insp = res.inspection || {};

      $("upload-progress-box").classList.add("hidden");
      $("video-preview-card").classList.remove("hidden");

      const previewVideo = $("upload-preview-video");
      previewVideo.src = res.preview_url;
      previewVideo.load();

      $("preview-filename").textContent = res.filename || file.name;
      const isSilent = Boolean(insp.is_silent);
      const audioBadge = $("preview-audio-badge");
      audioBadge.className = `meta-badge ${isSilent ? "is-silent" : "is-voiced"}`;
      audioBadge.textContent = isSilent ? "🔇 無音判定 → Foley生成推奨" : "🎙️ 有音判定 → リミックス推奨";

      $("meta-duration").textContent = `⏱ ${formatDurationLabel(insp.duration_sec)}`;
      $("meta-resolution").textContent = `📐 ${insp.width || "?"}×${insp.height || "?"}`;
      $("meta-fps").textContent = `🎞 ${Number(insp.fps || 0).toFixed(1)} fps`;
      const rms = insp.audio_rms_db;
      $("meta-rms").textContent = insp.has_audio_stream
        ? `🔊 RMS ${Number(rms ?? -120).toFixed(1)} dBFS`
        : "🔇 音声トラックなし";

      if (res.plan) renderVideoPlan(res.plan);
      updateAmbisonicsCapacityUI();
      $("btn-start-job").disabled = false;
      showToast(`動画を読み込みました（${isSilent ? "無音動画" : "有音動画"}として自動判定）`, "ok");
    } catch (err) {
      $("upload-progress-box").classList.add("hidden");
      $("dropzone").classList.remove("hidden");
      showToast(`動画アップロードに失敗しました: ${err.message}`, "error", 6000);
    }
  }

  async function startNewJob() {
    if (!state.upload || !state.upload.upload_id) {
      showToast("先に動画ファイルをアップロードしてください。", "error");
      return;
    }

    const trimEnabled = $("chk-preview-trim").checked;
    const { spatial_format, ambisonics_order } = getSpatialSelection();
    const rawLufs = $("select-lufs").value;
    const isMatchOriginal = rawLufs === "match_original";
    const dialogPanMode = $("select-dialog-pan-mode").value === "immersive" ? "immersive" : "cinema";
    const enableDialogDucking = $("chk-dialog-ducking").checked;

    const payload = {
      upload_id: state.upload.upload_id,
      mode: getRadioValue("proc_mode", "auto"),
      spatial_format,
      ambisonics_order,
      quality_preset: $("select-quality").value,
      loudness_mode: isMatchOriginal ? "match_original" : "fixed",
      target_lufs: isMatchOriginal ? null : Number(rawLufs),
      dialog_pan_mode: dialogPanMode,
      enable_dialog_ducking: enableDialogDucking,
      foley_prompt: $("input-foley-prompt").value.trim(),
      allow_foley_fallback: $("chk-foley-fallback").checked,
      preview_start: trimEnabled ? Math.max(0, Number($("input-preview-start").value) || 0) : 0.0,
      preview_duration: trimEnabled ? Math.max(2, Number($("input-preview-duration").value) || 15) : null,
      enable_dereverb: $("chk-dereverb").checked,
      enable_sfx_multilayer: $("chk-sfx-multilayer") ? $("chk-sfx-multilayer").checked : true,
      enable_director: $("chk-enable-director") ? $("chk-enable-director").checked : true,
      enable_world_360: $("chk-world-360") ? $("chk-world-360").checked : true,
      execution_mode: getRadioValue("execution_mode", "auto"),
    };

    const btn = $("btn-start-job");
    btn.disabled = true;
    try {
      const job = await apiJson("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      openJobMonitor(job);
      loadRecentJobs();
    } catch (err) {
      showToast(`ジョブ開始に失敗しました: ${err.message}`, "error", 6000);
      btn.disabled = false;
    }
  }

  async function loadRecentJobs() {
    try {
      const data = await apiJson("/api/jobs?limit=8");
      const jobs = data.jobs || [];
      const wrap = $("recent-jobs-wrap");
      const list = $("recent-jobs-list");
      if (!jobs.length) {
        wrap.classList.add("hidden");
        return;
      }
      wrap.classList.remove("hidden");
      list.innerHTML = "";
      jobs.forEach((j) => {
        const card = document.createElement("div");
        card.className = "recent-job-card";
        const stMap = {
          done: { cls: "is-ok", label: "完了" },
          queued_for_agent: { cls: "is-warn", label: "エージェント待機" },
          agent_running: { cls: "is-ok", label: "エージェント処理中" },
          local_running: { cls: "is-ok", label: "ローカル処理中" },
          remixing: { cls: "is-warn", label: "再Mix中" },
          error: { cls: "is-error", label: "エラー" },
          cancelled: { cls: "is-loading", label: "キャンセル" },
        };
        const stInfo = stMap[j.status] || { cls: "is-loading", label: j.status || "-" };
        card.innerHTML = `
          <div class="rj-info">
            <span class="rj-name">${escapeHtml(j.filename || j.job_id)}</span>
            <span class="rj-meta">${escapeHtml(formatSpatialName(j.spatial_format))} · ${Math.round(j.progress || 0)}%</span>
          </div>
          <span class="verdict-badge ${stInfo.cls}">${escapeHtml(stInfo.label)}</span>
        `;
        card.addEventListener("click", () => loadExistingJob(j.job_id));
        list.appendChild(card);
      });
    } catch (_) {
      // 履歴取得エラーは無視
    }
  }

  async function loadExistingJob(jobId) {
    try {
      const job = await apiJson(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (job.status === "done") {
        state.activeJobId = job.job_id;
        state.activeJob = job;
        renderMonitor(job);
        await enterStudioView(job);
      } else {
        openJobMonitor(job);
      }
    } catch (err) {
      showToast(`ジョブの読み込みに失敗しました: ${err.message}`, "error");
    }
  }

  // --------------------------------------------------------------------------
  // ステップ2: リアルタイム処理モニター
  // --------------------------------------------------------------------------
  function stopJobPolling() {
    if (state.jobPollTimer) {
      clearInterval(state.jobPollTimer);
      state.jobPollTimer = null;
    }
    if (state.elapsedTimer) {
      clearInterval(state.elapsedTimer);
      state.elapsedTimer = null;
    }
  }

  function openJobMonitor(job) {
    stopJobPolling();
    state.activeJobId = job.job_id;
    state.activeJob = job;
    setStep(2);
    renderMonitor(job);

    if (isJobInProgress(job.status)) {
      state.jobPollTimer = window.setInterval(pollActiveJob, 900);
      state.elapsedTimer = window.setInterval(updateMonitorElapsed, 300);
    }
  }

  function isJobInProgress(status) {
    return ["queued_for_agent", "agent_running", "local_running", "remixing"].includes(status);
  }

  function mapStageToIndex(stage, progress) {
    const s = String(stage || "").toLowerCase();
    if (s === "done") return 6;
    if (s === "queued") return -1;
    if (["inspect", "vision", "agent_plan", "preview", "init"].includes(s)) return 0;
    if (["separate", "separation", "dereverb", "foley", "audio", "sfx_split", "sfx"].includes(s)) return 1;
    if (["speakers", "diarization", "split", "speaker_split"].includes(s)) return 2;
    if (["mapping", "scene", "match", "tracking", "director", "director_plan"].includes(s)) return 3;
    if (["spatial", "atmos", "ambisonics", "master", "mastering", "remix", "director_remix"].includes(s)) return 4;
    if (["mux", "export", "package", "zip", "video"].includes(s)) return 5;
    const p = Number(progress) || 0;
    if (p >= 92) return 5;
    if (p >= 72) return 4;
    if (p >= 56) return 3;
    if (p >= 38) return 2;
    if (p >= 18) return 1;
    return p > 0 ? 0 : -1;
  }

  function updateMonitorElapsed() {
    const job = state.activeJob;
    if (!job) return;
    if (job.elapsed_sec != null && !isJobInProgress(job.status)) {
      $("monitor-elapsed").textContent = `所要 ${Number(job.elapsed_sec).toFixed(1)}秒`;
      return;
    }
    const baseTs = job.started_at || job.created_at;
    if (baseTs) {
      const sec = Math.max(0, Date.now() / 1000 - Number(baseTs));
      $("monitor-elapsed").textContent = `経過 ${sec.toFixed(1)}秒`;
    }
  }

  function renderMonitor(job) {
    state.activeJob = job;
    $("monitor-job-filename").textContent = job.filename || job.job_id;
    $("monitor-job-id").textContent = `#${job.job_id}`;

    const pct = clamp(Number(job.progress) || 0, 0, 100);
    $("monitor-percent").textContent = `${Math.round(pct)}%`;
    $("monitor-progress-bar").style.width = `${pct}%`;
    const track = $("monitor-progress-bar").parentElement;
    if (track) track.setAttribute("aria-valuenow", String(Math.round(pct)));
    $("monitor-current-message").textContent = job.message || "処理中…";
    updateMonitorElapsed();

    // 実行主体バッジ
    const pill = $("job-actor-pill");
    const statusLabels = {
      queued_for_agent: { cls: "is-warn", text: "🤖 エージェント受注待ち" },
      agent_running: { cls: "is-ok", text: `🤖 エージェント実行中 (${job.actor || "agent"})` },
      local_running: { cls: "is-ok", text: "💻 ローカル実行中" },
      remixing: { cls: "is-warn", text: "🔄 再MixMaster中" },
      done: { cls: "is-ok", text: "✅ 完了" },
      error: { cls: "is-error", text: "✕ エラー" },
      cancelled: { cls: "is-loading", text: "⏹ キャンセル済み" },
    };
    const pInfo = statusLabels[job.status] || { cls: "is-loading", text: job.status || "-" };
    pill.className = `status-pill ${pInfo.cls}`;
    pill.textContent = pInfo.text;

    // 待機バナー＆エラーバナー＆キャンセルボタン
    $("agent-wait-banner").classList.toggle("hidden", job.status !== "queued_for_agent");
    $("btn-cancel-job").disabled = !isJobInProgress(job.status);

    const errBanner = $("job-error-banner");
    if (job.status === "error") {
      errBanner.classList.remove("hidden");
      $("job-error-message").textContent = job.error || job.message || "不明なエラー";
    } else {
      errBanner.classList.add("hidden");
    }

    // 6段階ランプ更新
    const activeIdx = mapStageToIndex(job.stage, pct);
    const nodes = document.querySelectorAll("#pipeline-stages .stage-node");
    nodes.forEach((node, idx) => {
      node.classList.remove("is-active", "is-done", "is-error");
      if (job.status === "done" || idx < activeIdx) {
        node.classList.add("is-done");
      } else if (idx === activeIdx) {
        node.classList.add(job.status === "error" ? "is-error" : "is-active");
      }
    });

    // ログストリーム描画
    const logs = Array.isArray(job.logs) ? job.logs : [];
    $("log-count").textContent = `${logs.length} 件`;
    const stream = $("monitor-log-stream");
    const nearBottom = stream.scrollHeight - stream.scrollTop - stream.clientHeight < 48;
    stream.innerHTML = "";
    logs.forEach((entry) => {
      const line = document.createElement("div");
      line.className = "log-line";
      const actor = String(entry.actor || "system");
      const actorCls = actor.includes("local")
        ? "actor-local"
        : actor !== "server" && actor !== "system" && actor !== "user"
        ? "actor-agent"
        : "";
      line.innerHTML = `
        <span class="log-ts">${escapeHtml(entry.time_str || "")}</span>
        <span class="log-actor ${actorCls}">${escapeHtml(actor)}</span>
        <span class="log-msg">${escapeHtml(entry.message || "")}</span>
      `;
      stream.appendChild(line);
    });
    if (nearBottom) {
      stream.scrollTop = stream.scrollHeight;
    }
  }

  async function pollActiveJob() {
    if (!state.activeJobId) return;
    try {
      const job = await apiJson(`/api/jobs/${encodeURIComponent(state.activeJobId)}`);
      renderMonitor(job);
      if (!isJobInProgress(job.status)) {
        stopJobPolling();
        $("btn-start-job").disabled = !state.upload;
        pollLightHealth();
        loadRecentJobs();
        if (job.status === "done") {
          showToast("サラウンドMixMaster＆バイノーラル出力が完了しました！", "ok", 5000);
          $("step-tab-2").classList.add("is-done");
          await enterStudioView(job);
        } else if (job.status === "error") {
          showToast(`処理エラー: ${job.error || job.message}`, "error", 7000);
        }
      }
    } catch (_) {
      // 一時通信エラー時は次回ポーリングで再取得
    }
  }

  async function switchQueuedJobToLocal() {
    if (!state.activeJobId) return;
    const btn = $("btn-switch-local");
    btn.disabled = true;
    try {
      const job = await apiJson(`/api/jobs/${encodeURIComponent(state.activeJobId)}/start-local`, {
        method: "POST",
      });
      renderMonitor(job);
      showToast("ローカル即時実行へ切り替えました。", "ok");
    } catch (err) {
      showToast(`ローカル切替に失敗しました: ${err.message}`, "error");
    } finally {
      btn.disabled = false;
    }
  }

  async function cancelCurrentJob() {
    if (!state.activeJobId) return;
    try {
      const job = await apiJson(`/api/jobs/${encodeURIComponent(state.activeJobId)}/cancel`, {
        method: "POST",
      });
      renderMonitor(job);
      showToast("キャンセル要求を送信しました。", "info");
    } catch (err) {
      showToast(`キャンセルに失敗しました: ${err.message}`, "error");
    }
  }

  // --------------------------------------------------------------------------
  // ステップ3: 3D空間ステージビューアー・A/B比較試聴・再Mix・書き出し
  // --------------------------------------------------------------------------
  function formatSpatialName(fmt) {
    const f = String(fmt || "atmos").toLowerCase();
    if (f === "atmos") return "Dolby Atmos 互換 (ADM BWF + 7.1.4)";
    if (f === "ambisonics_foa") return "Ambisonics 1次 FOA (4ch AmbiX)";
    if (f === "ambisonics_hoa") return "Ambisonics 3次 HOA (16ch AmbiX)";
    if (f === "ambisonics_hoa5") return "Ambisonics 5次 HOA (36ch AmbiX)";
    if (f === "ambisonics_hoa7") return "Ambisonics 7次 HOA (64ch AmbiX)";
    return fmt || "Dolby Atmos 互換";
  }

  function objectColor(obj, idx) {
    if (obj && typeof obj.color === "string" && /^#[0-9a-fA-F]{3,8}$/.test(obj.color.trim())) {
      return obj.color.trim();
    }
    const role = String(obj.role || "dialog").toLowerCase();
    if (role === "dialog" || role === "speaker" || role === "voice") {
      return DIALOG_COLORS[idx % DIALOG_COLORS.length];
    }
    return ROLE_COLORS[role] || "#38bdf8";
  }

  function formatHrtfLabel(raw, metrics = null) {
    const s = String(raw || "").trim();
    if (!s) return "物理HRTF (Woodworth)";
    const low = s.toLowerCase();
    const hasDfEq = Boolean(
      (metrics && metrics.diffuse_field_eq) ||
        low.includes("df-eq") ||
        low.includes("diffuse-field") ||
        low.includes("diffuse_field")
    );
    if (low.includes("woodworth") || low.includes("physical") || low.includes("physics") || low.includes("内蔵物理")) {
      return "物理HRTF (Woodworth)";
    }
    if (low.includes("kemar")) {
      return hasDfEq ? "SOFA (KEMAR·DF-EQ)" : "SOFA (MIT KEMAR)";
    }
    if (low.includes("sofa")) {
      const base = (s.split(/[\\/]/).pop() || s).replace(/\.sofa$/i, "");
      const shortBase = base.length > 14 ? `${base.slice(0, 12)}…` : base;
      return hasDfEq ? `SOFA (${shortBase}·DF-EQ)` : `SOFA (${shortBase})`;
    }
    return s;
  }

  function syncRemixControlsFromJob(job) {
    const req = (job && job.request) || {};
    const mix = (state.scene && state.scene.mix_settings) || {};

    const fmtVal = req.spatial_format || mix.spatial_format;
    if (fmtVal && $("remix-format")) {
      $("remix-format").value = fmtVal;
    }

    const lufsEl = $("remix-lufs");
    if (lufsEl) {
      const lMode = mix.loudness_mode || req.loudness_mode;
      const rawLufsVal = req.target_lufs != null ? req.target_lufs : mix.target_lufs;
      if (lMode === "match_original" || (lMode == null && rawLufsVal == null)) {
        lufsEl.value = "match_original";
      } else if (rawLufsVal != null && isFinite(Number(rawLufsVal))) {
        const targetNum = Number(rawLufsVal);
        const numericOpts = Array.from(lufsEl.options).filter((o) => isFinite(Number(o.value)));
        const exact = numericOpts.find((o) => o.value === String(Math.round(targetNum)));
        if (exact) {
          lufsEl.value = exact.value;
        } else if (numericOpts.length) {
          // 既定選択肢外の数値には最も近いプリセットへ合わせる
          numericOpts.sort(
            (a, b) => Math.abs(Number(a.value) - targetNum) - Math.abs(Number(b.value) - targetNum)
          );
          lufsEl.value = numericOpts[0].value;
        }
      } else {
        lufsEl.value = "match_original";
      }
    }

    const panEl = $("remix-dialog-pan-mode");
    if (panEl) {
      const panMode = mix.dialog_pan_mode || req.dialog_pan_mode || "cinema";
      panEl.value = panMode === "immersive" ? "immersive" : "cinema";
    }

    const duckEl = $("chk-remix-ducking");
    if (duckEl) {
      const duckVal =
        mix.enable_dialog_ducking !== undefined
          ? mix.enable_dialog_ducking
          : req.enable_dialog_ducking !== undefined
          ? req.enable_dialog_ducking
          : true;
      duckEl.checked = Boolean(duckVal);
    }
  }

  async function enterStudioView(job) {
    state.activeJobId = job.job_id;
    state.activeJob = job;
    state.directorContext = null;
    state.initialDirectorPlan = null;
    setStep(3);
    pollLightHealth();

    // シーン初期化
    const scene = job.scene || { duration_sec: 10, sample_rate: 48000, objects: [] };
    state.originalScene = deepClone(scene);
    state.scene = deepClone(scene);
    state.sceneDirty = false;
    $("scene-dirty-badge").classList.add("hidden");

    const objects = state.scene.objects || [];
    state.selectedObjectId = objects.length ? objects[0].id : null;

    // 再Mix設定欄をジョブの現在値へ同期
    syncRemixControlsFromJob(job);

    renderMetricsAndDownloads(job);

    let dctx = job.director_context || null;
    if (!dctx && job.has_director_context && job.job_id) {
      try {
        dctx = await apiJson(`/api/jobs/${encodeURIComponent(job.job_id)}/director-context`);
      } catch (_) {
        dctx = null;
      }
    }
    const fallbackPlan =
      (scene && scene.director_plan) || (job.result && job.result.director_plan) || null;
    renderDirectorCard(job, dctx, fallbackPlan);

    renderTrackList();
    syncInspectorToSelected();
    renderIndividualStems(job.stems || []);

    await loadStudioMedia(job);
    resizeRadarCanvas();
    drawSpatialRadar();
  }

  function seekStudioTo(sec) {
    const video = $("studio-video");
    if (!video) return;
    const maxDur = Number(video.duration) || Number(state.scene && state.scene.duration_sec) || 9999;
    const target = clamp(Number(sec) || 0, 0, maxDur);
    video.currentTime = target;
    $("time-current").textContent = formatTimecode(target);
    if (state.webAudioPlaying) startWebAudioAt(target);
    drawSpatialRadar();
    syncInspectorToSelected();
  }

  function classifyTempLevel(temp, label) {
    const lbl = String(label || "").toLowerCase();
    if (lbl.includes("クライマックス") || lbl.includes("climax")) return "level-climax";
    if (lbl.includes("高揚") || lbl.includes("high") || lbl.includes("rising") || lbl.includes("action")) {
      return "level-high";
    }
    if (lbl.includes("静寂") || lbl.includes("quiet") || lbl.includes("calm")) return "level-quiet";
    const t = Number(temp);
    if (isFinite(t)) {
      if (t >= 0.75) return "level-climax";
      if (t >= 0.5) return "level-high";
      if (t <= 0.25) return "level-quiet";
    }
    return "level-normal";
  }

  function collectDirectorNotes(plan, dctx) {
    const raw = [];
    const pushVal = (v) => {
      if (!v) return;
      if (typeof v === "string") {
        if (v.trim()) raw.push(v.trim());
      } else if (Array.isArray(v)) {
        v.forEach(pushVal);
      } else if (typeof v === "object") {
        const text = String(v.message || v.note || v.comment || v.reason || v.summary || "").trim();
        if (text) {
          const hasTimeBracket = text.startsWith("[");
          const prefix = !hasTimeBracket && v.t_sec != null ? `[${Number(v.t_sec).toFixed(1)}s] ` : "";
          raw.push(`${prefix}${text}`);
        }
      }
    };
    if (plan) {
      pushVal(plan.summary_ja);
      pushVal(plan.notes);
      pushVal(plan.director_notes);
      pushVal(plan.comments);
      pushVal(plan.summary);
    }
    if (dctx) {
      pushVal(dctx.notes);
      pushVal(dctx.director_notes);
    }
    return Array.from(new Set(raw));
  }

  function collectDirectorKeyframeSummaries(plan, scene) {
    const rows = [];
    const pushKfRow = (target, kf, defaultExempt = false) => {
      rows.push({
        target,
        t_sec: kf.t_sec ?? kf.t,
        end_sec: kf.end_sec,
        gain_db: kf.gain_db,
        distance_m: kf.distance_m,
        azimuth_deg: kf.azimuth_deg,
        azimuth_offset_deg: kf.azimuth_offset_deg,
        elevation_deg: kf.elevation_deg,
        elevation_offset_deg: kf.elevation_offset_deg,
        duck_exempt: Boolean(kf.duck_exempt || defaultExempt),
        note: kf.note || kf.reason || "",
      });
    };
    if (plan && Array.isArray(plan.keyframes)) {
      plan.keyframes.forEach((kf) => {
        pushKfRow(kf.target || kf.object_id || kf.role || "演出対象", kf);
      });
    }
    const trackMaps = [
      plan && plan.tracks,
      plan && plan.object_overrides,
      plan && plan.overrides,
      plan && plan.roles,
    ];
    trackMaps.forEach((m) => {
      if (!m || typeof m !== "object" || Array.isArray(m)) return;
      Object.entries(m).forEach(([key, val]) => {
        if (Array.isArray(val)) {
          // 各トラックから特徴的なキーフレーム（理由付き/強調/免除/360°移動）を優先して最大4件ずつ抽出
          const prioritized = val.filter(
            (kf) =>
              kf &&
              (kf.reason ||
                kf.duck_exempt ||
                Math.abs(Number(kf.gain_db || 0)) >= 0.5 ||
                kf.distance_m != null ||
                kf.azimuth_deg != null ||
                Math.abs(Number(kf.azimuth_offset_deg || 0)) >= 1 ||
                kf.elevation_deg != null ||
                Math.abs(Number(kf.elevation_offset_deg || 0)) >= 1)
          );
          const picked = (prioritized.length ? prioritized : val).slice(0, 4);
          picked.forEach((kf) => pushKfRow(formatRoleLabelJa(key), kf));
        } else if (val && typeof val === "object") {
          const kfs = (Array.isArray(val.keyframes) ? val.keyframes : [val]).slice(0, 4);
          kfs.forEach((kf) =>
            pushKfRow(formatRoleLabelJa(val.role || key), kf, Boolean(val.duck_exempt))
          );
        }
      });
    });
    if (!rows.length && scene && Array.isArray(scene.objects)) {
      scene.objects.forEach((obj) => {
        (obj.keyframes || []).forEach((kf) => {
          if (kf.gain_db != null || kf.duck_exempt || Math.abs(Number(kf.azimuth_deg || 0)) > 60) {
            pushKfRow(obj.label || formatRoleLabelJa(obj.role), kf);
          }
        });
      });
    }
    return rows;
  }

  function renderDirectorCard(job, dctx, fallbackPlan = null) {
    const card = $("director-plan-card");
    if (!card) return;

    const plan = (dctx && dctx.director_plan) || fallbackPlan || null;
    if (!dctx && !plan) {
      state.directorContext = null;
      state.initialDirectorPlan = null;
      card.classList.add("hidden");
      return;
    }

    state.directorContext = dctx || null;
    if (plan && !state.initialDirectorPlan) {
      state.initialDirectorPlan = deepClone(plan);
    }
    card.classList.remove("hidden");

    const statusBadge = $("director-status-badge");
    const enabled = plan ? plan.enabled !== false : true;
    statusBadge.className = `status-pill ${enabled ? "is-ok" : "is-warn"}`;
    statusBadge.textContent = enabled ? "✅ 演出適用済み" : "⏸ 演出オフ";

    // コンタクトシート
    const sheetWrap = $("director-sheet-wrap");
    const sheetImg = $("director-sheet-img");
    const sheetLink = $("director-sheet-link");
    const sheetImgLink = $("director-sheet-img-link");
    const hasSheetMedia = Boolean(
      (dctx && dctx.director_sheet_url) ||
        (job && job.available_media && job.available_media.director_sheet && job.media_urls && job.media_urls.director_sheet)
    );
    const rawSheetUrl =
      (dctx && dctx.director_sheet_url) ||
      (hasSheetMedia && job.media_urls && job.media_urls.director_sheet) ||
      "";
    if (rawSheetUrl) {
      sheetImg.onerror = () => sheetWrap.classList.add("hidden");
      sheetImg.src = rawSheetUrl;
      sheetLink.href = rawSheetUrl;
      sheetImgLink.href = rawSheetUrl;
      sheetWrap.classList.remove("hidden");
    } else {
      sheetImg.removeAttribute("src");
      sheetWrap.classList.add("hidden");
    }

    // シーン温度サマリー ＆ 区間
    const segments = Array.isArray(dctx && dctx.scene_segments) ? dctx.scene_segments : [];
    const track = Array.isArray(dctx && dctx.intensity_track) ? dctx.intensity_track : [];
    const totalDur =
      Number((dctx && dctx.duration_sec) || (state.scene && state.scene.duration_sec)) || 15.0;

    let peakTemp = 0;
    track.forEach((pt) => {
      const val = typeof pt === "number" ? pt : Number(pt.temperature ?? pt.intensity ?? 0);
      if (isFinite(val) && val > peakTemp) peakTemp = val;
    });
    segments.forEach((seg) => {
      const val = Number(seg.peak_temperature ?? seg.mean_temperature ?? seg.temperature ?? 0);
      if (isFinite(val) && val > peakTemp) peakTemp = val;
    });

    $("director-temp-peak-badge").textContent = `最高温度 ${(peakTemp * 100).toFixed(0)}%`;
    $("director-segments-badge").textContent = `区間 ${segments.length}件`;

    const timelineEl = $("director-temp-timeline");
    const segListEl = $("director-segments-list");
    timelineEl.innerHTML = "";
    segListEl.innerHTML = "";

    if (segments.length) {
      segments.forEach((seg, idx) => {
        const sTime = Number(seg.start_sec ?? seg.start ?? 0);
        const eTime = Number(seg.end_sec ?? seg.end ?? sTime);
        const temp = Number(seg.mean_temperature ?? seg.temperature ?? 0);
        const label = seg.label || "通常";
        const span = Math.max(0.1, eTime - sTime);
        const widthPct = Math.max(2, (span / Math.max(0.1, totalDur)) * 100);
        const lvlCls = classifyTempLevel(temp, seg.label_code || label);

        const bar = document.createElement("div");
        bar.className = `temp-seg-bar ${lvlCls}`;
        bar.style.width = `${widthPct.toFixed(2)}%`;
        bar.title = `${sTime.toFixed(1)}s〜${eTime.toFixed(1)}s · ${label} (温度 ${(temp * 100).toFixed(0)}%)`;
        bar.addEventListener("click", () => seekStudioTo(sTime));
        timelineEl.appendChild(bar);

        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = `director-seg-chip ${lvlCls}`;
        chip.textContent = `${sTime.toFixed(1)}-${eTime.toFixed(1)}s ${label} (${(temp * 100).toFixed(0)}%)`;
        chip.title = "クリックして再生位置を移動";
        chip.addEventListener("click", () => seekStudioTo(sTime));
        segListEl.appendChild(chip);

        if (idx === 0 && !timelineEl.childElementCount) {
          timelineEl.appendChild(bar);
        }
      });
    } else if (track.length) {
      const stepW = 100 / track.length;
      track.forEach((pt) => {
        const t = typeof pt === "number" ? 0 : Number(pt.t_sec ?? pt.t ?? 0);
        const val = typeof pt === "number" ? pt : Number(pt.temperature ?? pt.intensity ?? 0);
        const bar = document.createElement("div");
        bar.className = `temp-seg-bar ${classifyTempLevel(val, "")}`;
        bar.style.width = `${stepW.toFixed(2)}%`;
        bar.title = `${t.toFixed(1)}s · 温度 ${(val * 100).toFixed(0)}%`;
        bar.addEventListener("click", () => seekStudioTo(t));
        timelineEl.appendChild(bar);
      });
      segListEl.innerHTML = `<span class="meta-chip">温度サンプル ${track.length}点</span>`;
    } else {
      segListEl.innerHTML = `<span class="meta-chip">温度区間情報なし</span>`;
    }

    // SFX 4層分解＆衝撃イベント
    const sfxEvents =
      Array.isArray(dctx && dctx.sfx_events) && dctx.sfx_events.length
        ? dctx.sfx_events
        : Array.isArray(plan && plan.events)
        ? plan.events
        : [];
    const evBadge = $("director-sfx-events-badge");
    if (sfxEvents.length) {
      evBadge.textContent = `衝撃イベント ${sfxEvents.length}件`;
      evBadge.classList.remove("hidden");
    } else {
      evBadge.classList.add("hidden");
    }

    const layersRow = $("director-sfx-layers-row");
    const rawLayers = dctx && dctx.sfx_layers;
    let layerItems = [];
    if (Array.isArray(rawLayers)) {
      layerItems = rawLayers;
    } else if (rawLayers && typeof rawLayers === "object") {
      layerItems = Object.entries(rawLayers).map(([k, v]) =>
        typeof v === "object" && v !== null ? { name: k, ...v } : { name: k }
      );
    }
    layersRow.innerHTML = "";
    if (layerItems.length) {
      layersRow.classList.remove("hidden");
      layerItems.forEach((ly) => {
        const role = ly.role || ly.name || "sfx";
        const color = ROLE_COLORS[String(role).toLowerCase()] || "#f59e0b";
        const pill = document.createElement("span");
        pill.className = "sfx-layer-pill";
        const rmsStr = ly.rms_db != null && isFinite(ly.rms_db) ? ` (${Number(ly.rms_db).toFixed(1)} dB)` : "";
        pill.innerHTML = `<i class="track-swatch" style="background:${color};width:8px;height:8px"></i><span>${escapeHtml(ly.label || formatRoleLabelJa(role) || ly.name)}${escapeHtml(rmsStr)}</span>`;
        layersRow.appendChild(pill);
      });
    } else {
      layersRow.classList.add("hidden");
    }

    // 360° ワールド空間拡張ピル（レガシージョブ等データ無し時は自動非表示）
    const w360Row = $("director-world360-row");
    if (w360Row) {
      const wp = (dctx && dctx.world_profile) || (job.result && job.result.world_profile) || null;
      const w360 =
        (plan && plan.world_360) ||
        (dctx && dctx.world_360) ||
        (state.scene && state.scene.world_360) ||
        null;
      const flybys =
        (Array.isArray(dctx && dctx.flyby_events) && dctx.flyby_events) ||
        (Array.isArray(plan && plan.flyby_events) && plan.flyby_events) ||
        [];
      w360Row.innerHTML = "";
      if ((wp && Object.keys(wp).length) || (w360 && Object.keys(w360).length) || flybys.length) {
        w360Row.classList.remove("hidden");
        const isOn = !w360 || w360.enabled !== false;
        const envLabel =
          (w360 && w360.environment_label_ja) ||
          (wp && wp.environment_label_ja) ||
          "360°想定空間";
        const pills = [`🌐 ${envLabel}`];
        if (isOn) {
          if (w360 && isFinite(w360.rear_level)) {
            pills.push(`🔊 後方包囲 ${Number(w360.rear_level).toFixed(2)}x`);
          }
          if (w360 && isFinite(w360.height_level)) {
            pills.push(`⬆️ 頭上ハイト ${Number(w360.height_level).toFixed(2)}x`);
          }
          pills.push(`🛫 画面外通過 ${flybys.length}件`);
        } else {
          pills.push("⏸ 360°拡張オフ（前方基準）");
        }
        pills.forEach((txt) => {
          const sp = document.createElement("span");
          sp.className = `world360-pill ${isOn ? "" : "is-off"}`;
          sp.textContent = txt;
          w360Row.appendChild(sp);
        });
      } else {
        w360Row.classList.add("hidden");
      }
    }

    // 監督コメント
    const notesList = $("director-notes-list");
    const notes = collectDirectorNotes(plan, dctx);
    notesList.innerHTML = "";
    if (notes.length) {
      notes.forEach((txt) => {
        const div = document.createElement("div");
        div.className = "director-note-item";
        div.textContent = txt;
        notesList.appendChild(div);
      });
    } else {
      notesList.innerHTML = `<div class="director-note-item">映像の温度変化に応じたダビング演出プランが設定されています。</div>`;
    }

    // 演出キーフレーム概要
    const kfBox = $("director-keyframes-summary");
    const kfRows = collectDirectorKeyframeSummaries(plan, state.scene);
    kfBox.innerHTML = "";
    if (kfRows.length) {
      kfRows.slice(0, 24).forEach((kf) => {
        const row = document.createElement("div");
        row.className = "director-kf-row";
        const timePart =
          kf.t_sec != null
            ? kf.end_sec != null
              ? `${Number(kf.t_sec).toFixed(1)}〜${Number(kf.end_sec).toFixed(1)}s`
              : `${Number(kf.t_sec).toFixed(1)}s`
            : "全区間";
        const parts = [timePart];
        if (kf.gain_db != null && isFinite(kf.gain_db)) {
          const g = Number(kf.gain_db);
          parts.push(`${g >= 0 ? "+" : ""}${g.toFixed(1)} dB`);
        }
        if (kf.distance_m != null && isFinite(kf.distance_m)) {
          parts.push(`${Number(kf.distance_m).toFixed(1)} m`);
        }
        if (kf.azimuth_deg != null && isFinite(kf.azimuth_deg)) {
          const azv = Math.round(Number(kf.azimuth_deg));
          const zone = Math.abs(azv) >= 115 ? "(後方)" : Math.abs(azv) >= 65 ? "(側面)" : "";
          parts.push(`${azv >= 0 ? "+" : ""}${azv}°${zone}`);
        } else if (kf.azimuth_offset_deg != null && Math.abs(Number(kf.azimuth_offset_deg)) >= 1) {
          const off = Math.round(Number(kf.azimuth_offset_deg));
          parts.push(`方位${off >= 0 ? "+" : ""}${off}°`);
        }
        if (kf.elevation_deg != null && isFinite(kf.elevation_deg)) {
          const elv = Math.round(Number(kf.elevation_deg));
          const ez = elv >= 45 ? "(頭上)" : "";
          parts.push(`仰角${elv >= 0 ? "+" : ""}${elv}°${ez}`);
        } else if (kf.elevation_offset_deg != null && Math.abs(Number(kf.elevation_offset_deg)) >= 1) {
          const eoff = Math.round(Number(kf.elevation_offset_deg));
          parts.push(`高さ${eoff >= 0 ? "+" : ""}${eoff}°`);
        }
        const exemptHtml = kf.duck_exempt
          ? `<span class="director-exempt-tag">ダッキング免除</span>`
          : "";
        row.innerHTML = `
          <span class="director-kf-target">${escapeHtml(kf.target)}</span>
          <span class="director-kf-meta">${escapeHtml(parts.join(" · "))} ${exemptHtml}</span>
        `;
        kfBox.appendChild(row);
      });
    } else {
      kfBox.innerHTML = `<div class="director-kf-row"><span class="director-kf-target">キーフレーム調整</span><span class="director-kf-meta">自動標準バランス</span></div>`;
    }

    // JSON エディタへ現在プランを反映
    const jsonArea = $("director-plan-json");
    if (jsonArea) {
      jsonArea.value = JSON.stringify(plan || {}, null, 2);
    }
  }

  async function applyDirectorPlan() {
    if (!state.activeJobId) return;
    const jsonArea = $("director-plan-json");
    let parsedPlan;
    try {
      parsedPlan = JSON.parse((jsonArea && jsonArea.value) || "{}");
      if (!parsedPlan || typeof parsedPlan !== "object" || Array.isArray(parsedPlan)) {
        throw new Error("ルートは JSON オブジェクト ({...}) である必要があります。");
      }
    } catch (err) {
      showToast(`監督プランJSONの構文エラー: ${err.message}`, "error", 5500);
      if (jsonArea) jsonArea.focus();
      return;
    }

    const btn = $("btn-director-reapply");
    btn.disabled = true;
    const origHtml = btn.innerHTML;
    btn.innerHTML = `<span class="spinner-sm"></span><span>監督演出リミックス中…</span>`;

    const video = $("studio-video");
    const resumeTime = video ? Number(video.currentTime) || 0 : 0;
    pauseStudioPlayback();

    try {
      const resp = await apiJson(
        `/api/jobs/${encodeURIComponent(state.activeJobId)}/director-plan`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ director_plan: parsedPlan }),
        }
      );

      const updatedJob = resp.state || state.activeJob || { job_id: state.activeJobId };
      if (resp.media_urls) updatedJob.media_urls = resp.media_urls;
      if (resp.result) updatedJob.result = resp.result;
      if (resp.director_context) updatedJob.director_context = resp.director_context;
      if (resp.scene) {
        updatedJob.scene = resp.scene;
        state.scene = deepClone(resp.scene);
        state.originalScene = deepClone(resp.scene);
      }
      state.activeJob = updatedJob;
      state.sceneDirty = false;
      $("scene-dirty-badge").classList.add("hidden");

      syncRemixControlsFromJob(updatedJob);
      renderMetricsAndDownloads(updatedJob);
      renderDirectorCard(
        updatedJob,
        resp.director_context || updatedJob.director_context,
        resp.director_plan || parsedPlan
      );
      renderTrackList();
      syncInspectorToSelected();
      drawSpatialRadar();
      await loadStudioMedia(updatedJob);
      if (video && resumeTime > 0) {
        video.addEventListener(
          "loadedmetadata",
          () => {
            video.currentTime = Math.min(resumeTime, video.duration || resumeTime);
          },
          { once: true }
        );
      }
      showToast(
        `監督演出プランを適用して再MixMasterが完了しました（${resp.elapsed_sec ?? ""}秒）！`,
        "ok",
        4500
      );
    } catch (err) {
      showToast(`監督演出の再適用に失敗しました: ${err.message}`, "error", 6500);
    } finally {
      btn.disabled = false;
      btn.innerHTML = origHtml;
    }
  }

  function renderMetricsAndDownloads(job) {
    const res = job.result || {};
    const metrics = res.metrics || (res.spatial && res.spatial.metrics) || {};
    const req = job.request || {};
    const fmt = metrics.format || res.spatial_format || req.spatial_format || "atmos";

    $("result-format-badge").textContent = formatSpatialName(fmt);

    const lufs = metrics.integrated_lufs;
    $("metric-lufs").textContent = lufs != null && isFinite(lufs) ? `${Number(lufs).toFixed(1)} LKFS` : "-- LKFS";

    const peak = metrics.true_peak_dbtp;
    $("metric-peak").textContent = peak != null && isFinite(peak) ? `${Number(peak).toFixed(2)} dBTP` : "-- dBTP";

    const bal = metrics.binaural_lr_balance_db ?? metrics.lr_balance_db;
    if (bal != null && isFinite(bal)) {
      const sign = Number(bal) >= 0 ? "+" : "";
      $("metric-balance").textContent = `${sign}${Number(bal).toFixed(2)} dB`;
    } else {
      $("metric-balance").textContent = "-- dB";
    }

    const rawHrtf = metrics.hrtf_source || "内蔵物理HRTF";
    const hrtfEl = $("metric-hrtf");
    hrtfEl.textContent = formatHrtfLabel(rawHrtf, metrics);
    const dfNote = metrics.diffuse_field_eq ? " [Diffuse-Field EQ 適用済み]" : "";
    hrtfEl.title = `${rawHrtf}${dfNote}`;

    const chText = metrics.channels ? `${metrics.channels} ch` : fmt === "atmos" ? "7.1.4 + Obj" : "--";
    const chEl = $("metric-channels");
    chEl.textContent = chText;
    chEl.title = chText;
    $("metric-elapsed").textContent = job.elapsed_sec != null ? `${Number(job.elapsed_sec).toFixed(1)} 秒` : "--";

    // ダウンロードリンク反映
    const urls = job.media_urls || {};
    const avail = job.available_media || {};

    function wireDownload(elId, kind) {
      const a = $(elId);
      if (!a) return;
      if (avail[kind] && urls[kind]) {
        const sep = urls[kind].includes("?") ? "&" : "?";
        a.href = `${urls[kind]}${sep}download=1`;
        a.classList.remove("is-disabled");
      } else {
        a.href = "#";
        a.classList.add("is-disabled");
      }
    }

    wireDownload("dl-binaural-video", "binaural_video");
    wireDownload("dl-binaural-audio", "binaural_audio");
    wireDownload("dl-master-spatial", "master_spatial");
    wireDownload("dl-stems-zip", "stems_zip");

    const masterLabel = $("dl-master-spatial-label");
    const masterSub = $("dl-master-spatial-sub");
    const fmtStr = String(fmt).toLowerCase();
    if (fmtStr.includes("atmos")) {
      masterLabel.textContent = "Dolby Atmos 互換 ADM BWF";
      masterSub.textContent = "ITU-R BS.2076 完全フリー .wav";
    } else if (fmtStr.includes("hoa7")) {
      masterLabel.textContent = "Ambisonics 7次 HOA (64ch)";
      masterSub.textContent = "AmbiX ACN/SN3D .wav (自動RF64)";
    } else if (fmtStr.includes("hoa5")) {
      masterLabel.textContent = "Ambisonics 5次 HOA (36ch)";
      masterSub.textContent = "AmbiX ACN/SN3D .wav (自動RF64)";
    } else if (fmtStr.includes("hoa")) {
      masterLabel.textContent = "Ambisonics 3次 HOA (16ch)";
      masterSub.textContent = "AmbiX ACN/SN3D .wav";
    } else {
      masterLabel.textContent = "Ambisonics 1次 FOA (4ch)";
      masterSub.textContent = "AmbiX ACN/SN3D .wav";
    }

    const dl714 = $("dl-surround-714");
    if (avail.surround_714 && urls.surround_714) {
      dl714.classList.remove("hidden");
      wireDownload("dl-surround-714", "surround_714");
    } else {
      dl714.classList.add("hidden");
    }
  }

  function renderIndividualStems(stems) {
    const box = $("individual-stems-box");
    const list = $("individual-stems-list");
    if (!Array.isArray(stems) || !stems.length) {
      box.classList.add("hidden");
      return;
    }
    box.classList.remove("hidden");
    list.innerHTML = "";
    stems.forEach((st) => {
      const chip = document.createElement("div");
      chip.className = "stem-chip";
      chip.innerHTML = `
        <span>${escapeHtml(st.label || st.name)}</span>
        <button type="button" title="このステムを単体試聴">▶</button>
        <a href="${escapeHtml(st.url)}?download=1" download title="WAVをダウンロード">⬇</a>
      `;
      const playBtn = chip.querySelector("button");
      playBtn.addEventListener("click", () => toggleStemPreview(st.url, playBtn));
      list.appendChild(chip);
    });
  }

  function toggleStemPreview(url, btnEl) {
    if (state.stemPreviewAudio) {
      const same = state.stemPreviewAudio.datasetUrl === url && !state.stemPreviewAudio.paused;
      state.stemPreviewAudio.pause();
      document.querySelectorAll(".stem-chip button").forEach((b) => (b.textContent = "▶"));
      state.stemPreviewAudio = null;
      if (same) return;
    }
    pauseStudioPlayback();
    const audio = new Audio(url);
    audio.datasetUrl = url;
    audio.addEventListener("ended", () => {
      btnEl.textContent = "▶";
      state.stemPreviewAudio = null;
    });
    audio.play().catch(() => {});
    btnEl.textContent = "⏹";
    state.stemPreviewAudio = audio;
  }

  // --------------------------------------------------------------------------
  // Web Audio API シームレス A/B 比較試聴プレイヤー
  // --------------------------------------------------------------------------
  function ensureAudioContext() {
    if (!state.audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) state.audioCtx = new AC();
    }
    return state.audioCtx;
  }

  function stopWebAudioNodes() {
    for (const key of ["srcOriginal", "srcBinaural"]) {
      if (state[key]) {
        try {
          state[key].stop();
          state[key].disconnect();
        } catch (_) {}
        state[key] = null;
      }
    }
    state.webAudioPlaying = false;
  }

  function pauseStudioPlayback() {
    const video = $("studio-video");
    if (video && !video.paused) video.pause();
    stopWebAudioNodes();
    const btn = $("btn-play-toggle");
    if (btn) btn.textContent = "▶";
  }

  async function loadStudioMedia(job) {
    pauseStudioPlayback();
    state.bufOriginal = null;
    state.bufBinaural = null;
    state.webAudioReady = false;

    const video = $("studio-video");
    const urls = job.media_urls || {};
    const videoUrl = urls.binaural_video || urls.original_video || "";
    video.src = videoUrl;
    video.load();

    $("ab-engine-status").textContent = "A/B比較用の48kHz音声バッファを読み込んでいます…";

    try {
      const ctx = ensureAudioContext();
      if (!ctx) throw new Error("Web Audio API 非対応ブラウザです");

      const tasks = [];
      if (urls.binaural_audio) {
        tasks.push(
          fetch(urls.binaural_audio)
            .then((r) => (r.ok ? r.arrayBuffer() : null))
            .then((ab) => (ab ? ctx.decodeAudioData(ab) : null))
            .then((buf) => {
              state.bufBinaural = buf;
            })
            .catch(() => {
              state.bufBinaural = null;
            })
        );
      }
      if (urls.original_audio) {
        tasks.push(
          fetch(urls.original_audio)
            .then((r) => (r.ok ? r.arrayBuffer() : null))
            .then((ab) => (ab ? ctx.decodeAudioData(ab) : null))
            .then((buf) => {
              state.bufOriginal = buf;
            })
            .catch(() => {
              state.bufOriginal = null;
            })
        );
      }
      await Promise.all(tasks);

      if (state.bufBinaural) {
        state.webAudioReady = true;
        video.muted = true; // 音声は Web Audio のシームレス A/B グラフから出力
        $("ab-engine-status").textContent = state.bufOriginal
          ? "シームレスA/B比較レディ（再生中に無音切れなく瞬時切替できます）"
          : "シームレスA/B比較レディ（元映像は無音です：無音 ⇔ バイノーラルを比較可能）";
      } else {
        video.muted = false;
        $("ab-engine-status").textContent = "動画トラック音声で再生します（バイノーラルWAV未生成）";
      }
    } catch (_) {
      video.muted = false;
      $("ab-engine-status").textContent = "動画プレイヤー標準音声で再生します";
    }

    setABSource("binaural");
  }

  function startWebAudioAt(offsetSec) {
    if (!state.webAudioReady || !state.bufBinaural) return;
    const ctx = ensureAudioContext();
    if (!ctx) return;
    if (ctx.state === "suspended") ctx.resume().catch(() => {});

    stopWebAudioNodes();

    const masterGain = ctx.createGain();
    masterGain.gain.value = 1.0;

    const splitter = ctx.createChannelSplitter(2);
    const anL = ctx.createAnalyser();
    const anR = ctx.createAnalyser();
    anL.fftSize = 256;
    anR.fftSize = 256;
    state.analyserL = anL;
    state.analyserR = anR;

    masterGain.connect(splitter);
    splitter.connect(anL, 0);
    splitter.connect(anR, 1);
    masterGain.connect(ctx.destination);

    const gBin = ctx.createGain();
    const gOrig = ctx.createGain();
    gBin.gain.value = state.abSource === "binaural" ? 1.0 : 0.0;
    gOrig.gain.value = state.abSource === "original" ? 1.0 : 0.0;
    gBin.connect(masterGain);
    gOrig.connect(masterGain);
    state.gainBinaural = gBin;
    state.gainOriginal = gOrig;

    const safeOffset = Math.max(0, Number(offsetSec) || 0);
    const startAt = ctx.currentTime + 0.015;

    if (safeOffset < state.bufBinaural.duration) {
      const srcB = ctx.createBufferSource();
      srcB.buffer = state.bufBinaural;
      srcB.connect(gBin);
      srcB.start(startAt, safeOffset);
      state.srcBinaural = srcB;
    }

    if (state.bufOriginal && safeOffset < state.bufOriginal.duration) {
      const srcO = ctx.createBufferSource();
      srcO.buffer = state.bufOriginal;
      srcO.connect(gOrig);
      srcO.start(startAt, safeOffset);
      state.srcOriginal = srcO;
    }

    state.webAudioPlaying = true;
    state.audioStartCtxTime = startAt;
    state.audioStartOffset = safeOffset;
  }

  function setABSource(source) {
    state.abSource = source === "original" ? "original" : "binaural";

    $("btn-ab-original").classList.toggle("is-active", state.abSource === "original");
    $("btn-ab-binaural").classList.toggle("is-active", state.abSource === "binaural");

    const badge = $("ab-overlay-badge");
    if (state.abSource === "binaural") {
      badge.className = "ab-overlay-badge is-binaural";
      badge.textContent = "🎧 バイノーラル立体音響";
    } else {
      badge.className = "ab-overlay-badge is-original";
      badge.textContent = state.bufOriginal ? "🔈 元音声（比較用）" : "🔇 元映像（無音）";
    }

    // 25ms 滑らかクロスフェード（クリックノイズゼロ）
    if (state.audioCtx && state.gainBinaural && state.gainOriginal) {
      const now = state.audioCtx.currentTime;
      const targetBin = state.abSource === "binaural" ? 1.0 : 0.0;
      const targetOrig = state.abSource === "original" ? 1.0 : 0.0;
      state.gainBinaural.gain.cancelScheduledValues(now);
      state.gainOriginal.gain.cancelScheduledValues(now);
      state.gainBinaural.gain.setValueAtTime(state.gainBinaural.gain.value, now);
      state.gainOriginal.gain.setValueAtTime(state.gainOriginal.gain.value, now);
      state.gainBinaural.gain.linearRampToValueAtTime(targetBin, now + 0.025);
      state.gainOriginal.gain.linearRampToValueAtTime(targetOrig, now + 0.025);
    } else if (!state.webAudioReady && state.activeJob && state.activeJob.media_urls) {
      // フォールバック時は動画ソースを切替
      const video = $("studio-video");
      const urls = state.activeJob.media_urls;
      const desired = state.abSource === "original" ? urls.original_video : urls.binaural_video;
      if (desired && video.src && !video.src.endsWith(desired)) {
        const curT = video.currentTime;
        const wasPlaying = !video.paused;
        video.src = desired;
        video.addEventListener(
          "loadedmetadata",
          () => {
            video.currentTime = curT;
            if (wasPlaying) video.play().catch(() => {});
          },
          { once: true }
        );
      }
    }
  }

  function toggleStudioPlayback() {
    const video = $("studio-video");
    if (!video) return;
    if (video.paused) {
      if (state.stemPreviewAudio) {
        state.stemPreviewAudio.pause();
        document.querySelectorAll(".stem-chip button").forEach((b) => (b.textContent = "▶"));
      }
      if (video.ended || (video.duration && video.currentTime >= video.duration - 0.05)) {
        video.currentTime = 0;
      }
      if (state.webAudioReady) {
        video.muted = true;
        startWebAudioAt(video.currentTime);
      }
      video.play().catch(() => {});
      $("btn-play-toggle").textContent = "⏸";
    } else {
      pauseStudioPlayback();
    }
  }

  function updateLevelMeters() {
    const fillL = $("meter-l-fill");
    const fillR = $("meter-r-fill");
    if (!fillL || !fillR) return;
    if (!state.webAudioPlaying || !state.analyserL || !state.analyserR) {
      fillL.style.width = "0%";
      fillR.style.width = "0%";
      return;
    }

    function calcPct(analyser) {
      const arr = new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(arr);
      let sum = 0;
      for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i];
      const rms = Math.sqrt(sum / arr.length);
      const db = rms > 1e-6 ? 20 * Math.log10(rms) : -60;
      return clamp(((db + 54) / 54) * 100, 0, 100);
    }

    fillL.style.width = `${calcPct(state.analyserL).toFixed(1)}%`;
    fillR.style.width = `${calcPct(state.analyserR).toFixed(1)}%`;
  }

  // --------------------------------------------------------------------------
  // 3D キーフレーム補間 ＆ 座標変換
  // --------------------------------------------------------------------------
  function interpolateObjectPose(obj, tSec) {
    const kfs = Array.isArray(obj.keyframes) && obj.keyframes.length
      ? obj.keyframes
      : [{ t: 0, azimuth_deg: 0, elevation_deg: 0, distance_m: 2.0 }];

    const kfTime = (kf) => Number(kf.t ?? kf.t_sec ?? 0);
    if (kfs.length === 1 || tSec <= kfTime(kfs[0])) {
      return {
        azimuth_deg: Number(kfs[0].azimuth_deg ?? 0),
        elevation_deg: Number(kfs[0].elevation_deg ?? 0),
        distance_m: Number(kfs[0].distance_m ?? 2.0),
      };
    }
    const last = kfs[kfs.length - 1];
    if (tSec >= kfTime(last)) {
      return {
        azimuth_deg: Number(last.azimuth_deg ?? 0),
        elevation_deg: Number(last.elevation_deg ?? 0),
        distance_m: Number(last.distance_m ?? 2.0),
      };
    }

    for (let i = 0; i < kfs.length - 1; i++) {
      const a = kfs[i];
      const b = kfs[i + 1];
      const ta = kfTime(a);
      const tb = kfTime(b);
      if (tSec >= ta && tSec <= tb) {
        const span = Math.max(1e-6, tb - ta);
        const alpha = (tSec - ta) / span;
        // 方位角は最短弧補間
        const azA = Number(a.azimuth_deg ?? 0);
        let dAz = Number(b.azimuth_deg ?? 0) - azA;
        if (dAz > 180) dAz -= 360;
        if (dAz < -180) dAz += 360;
        let az = azA + dAz * alpha;
        if (az > 180) az -= 360;
        if (az < -180) az += 360;
        return {
          azimuth_deg: az,
          elevation_deg: Number(a.elevation_deg ?? 0) + (Number(b.elevation_deg ?? 0) - Number(a.elevation_deg ?? 0)) * alpha,
          distance_m: Number(a.distance_m ?? 2.0) + (Number(b.distance_m ?? 2.0) - Number(a.distance_m ?? 2.0)) * alpha,
        };
      }
    }
    return { azimuth_deg: 0, elevation_deg: 0, distance_m: 2.0 };
  }

  // 距離 (0.5m..12m) <-> 正規化半径 (0..1)
  function distToNormRadius(distM) {
    const d = clamp(Number(distM) || 2.0, 0.5, 12.0);
    return Math.pow((d - 0.3) / 11.7, 0.72);
  }

  function normRadiusToDist(rNorm) {
    const r = clamp(rNorm, 0.04, 1.0);
    return clamp(0.3 + Math.pow(r, 1 / 0.72) * 11.7, 0.5, 12.0);
  }

  // CineSS座標（0°=上, -90°=左, +90°=右）-> Canvas座標
  function polarToCanvas(azDeg, distM, cx, cy, maxR) {
    const rad = (azDeg * Math.PI) / 180;
    const r = distToNormRadius(distM) * maxR;
    return {
      x: cx + Math.sin(rad) * r,
      y: cy - Math.cos(rad) * r,
    };
  }

  function canvasToPolar(x, y, cx, cy, maxR) {
    const dx = x - cx;
    const dy = y - cy;
    const rPx = Math.hypot(dx, dy);
    const azDeg = (Math.atan2(dx, -dy) * 180) / Math.PI;
    const distM = normRadiusToDist(rPx / Math.max(1, maxR));
    return {
      azimuth_deg: Math.round(clamp(azDeg, -180, 180)),
      distance_m: Math.round(distM * 10) / 10,
    };
  }

  // --------------------------------------------------------------------------
  // インタラクティブ 3D トップダウン空間レーダー（Canvas）
  // --------------------------------------------------------------------------
  function resizeRadarCanvas() {
    const canvas = $("spatial-radar-canvas");
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(rect.width * dpr);
    const h = Math.round(rect.height * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function getRadarGeometry(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    const cx = w / 2;
    const cy = h / 2 + 6;
    const maxR = Math.min(w, h) * 0.41;
    return { dpr, w, h, cx, cy, maxR };
  }

  function drawSpatialRadar() {
    const canvas = $("spatial-radar-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { dpr, w, h, cx, cy, maxR } = getRadarGeometry(canvas);
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    // 1. 前方スクリーン FOV 90° (-45° 〜 +45°) ガイドウェッジ ＆ 360°想定空間ガイド
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, maxR, -Math.PI / 2 - Math.PI / 4, -Math.PI / 2 + Math.PI / 4);
    ctx.closePath();
    const fovGrad = ctx.createRadialGradient(cx, cy, 8, cx, cy, maxR);
    fovGrad.addColorStop(0, "rgba(34, 211, 238, 0.14)");
    fovGrad.addColorStop(1, "rgba(34, 211, 238, 0.03)");
    ctx.fillStyle = fovGrad;
    ctx.fill();
    ctx.fillStyle = "rgba(103, 232, 249, 0.45)";
    ctx.font = "600 9.5px 'Segoe UI', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("前方視野 (FOV)", cx, cy - maxR * 0.68);
    ctx.fillStyle = "rgba(148, 163, 184, 0.34)";
    ctx.fillText("360° 想定ワールド（側面・後方）", cx, cy + maxR * 0.68);
    ctx.restore();

    // 上部スクリーンバー
    const screenW = maxR * 1.15;
    const screenY = cy - maxR - 14;
    ctx.strokeStyle = "rgba(34, 211, 238, 0.7)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx - screenW / 2, screenY);
    ctx.lineTo(cx + screenW / 2, screenY);
    ctx.stroke();
    ctx.fillStyle = "#67e8f9";
    ctx.font = "700 10.5px 'Segoe UI', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("SCREEN（映像スクリーン 0°）", cx, screenY - 6);

    // 2. 距離同心円 (1m, 3m, 6m, 10m)
    const rings = [1, 3, 6, 10];
    ctx.font = "10.5px Consolas, monospace";
    ctx.textAlign = "left";
    rings.forEach((m) => {
      const r = distToNormRadius(m) * maxR;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = m === 10 ? "rgba(148, 163, 184, 0.28)" : "rgba(148, 163, 184, 0.14)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "rgba(148, 163, 184, 0.72)";
      ctx.fillText(`${m}m`, cx + 4, cy - r + 11);
    });

    // 3. 方位角ガイド線
    for (let deg = -150; deg <= 180; deg += 30) {
      const rad = (deg * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.sin(rad) * maxR, cy - Math.cos(rad) * maxR);
      ctx.strokeStyle = deg % 90 === 0 ? "rgba(148, 163, 184, 0.24)" : "rgba(148, 163, 184, 0.08)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // L / R / 後方ラベル
    ctx.fillStyle = "#94a3b8";
    ctx.font = "700 11px Consolas, monospace";
    ctx.textAlign = "right";
    ctx.fillText("-90° L", cx - maxR - 8, cy + 4);
    ctx.textAlign = "left";
    ctx.fillText("+90° R", cx + maxR + 8, cy + 4);
    ctx.textAlign = "center";
    ctx.fillText("180° 後方", cx, cy + maxR + 15);

    // 4. 中央リスナー頭部アイコン（上向き＝画面方向）
    ctx.save();
    ctx.translate(cx, cy);
    // 両耳
    ctx.fillStyle = "#334155";
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.ellipse(-15, 0, 4, 7, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(15, 0, 4, 7, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    // 頭部＋鼻先（正面インジケータ）
    ctx.beginPath();
    ctx.moveTo(0, -18);
    ctx.lineTo(-4.5, -12);
    ctx.lineTo(4.5, -12);
    ctx.closePath();
    ctx.fillStyle = "#22d3ee";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(0, 0, 13, 0, Math.PI * 2);
    ctx.fillStyle = "#0f172a";
    ctx.fill();
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#e2e8f0";
    ctx.font = "700 9px 'Segoe UI', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("YOU", 0, 3);
    ctx.restore();

    // 5. 音源オブジェクトの軌跡＆現在位置ノードを描画（ラベルは後段で重なり回避配置）
    const video = $("studio-video");
    const curT = video ? Number(video.currentTime) || 0 : 0;
    const objects = (state.scene && state.scene.objects) || [];
    const pendingLabels = [];

    objects.forEach((obj, idx) => {
      const color = objectColor(obj, idx);
      const isSelected = obj.id === state.selectedObjectId;
      const isHover = obj.id === state.radarHoverId;
      const kfs = Array.isArray(obj.keyframes) ? obj.keyframes : [];

      // 軌跡ライン（2キーフレーム以上）
      if (kfs.length > 1) {
        ctx.save();
        ctx.beginPath();
        kfs.forEach((kf, kIdx) => {
          const pt = polarToCanvas(Number(kf.azimuth_deg ?? 0), Number(kf.distance_m ?? 2.0), cx, cy, maxR);
          if (kIdx === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.strokeStyle = color;
        ctx.globalAlpha = isSelected ? 0.65 : 0.26;
        ctx.lineWidth = isSelected ? 2 : 1.2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.restore();
      }

      // 現在時刻での補間座標
      const pose = interpolateObjectPose(obj, curT);
      const pos = polarToCanvas(pose.azimuth_deg, pose.distance_m, cx, cy, maxR);

      // Bed音源なら広がりリングを描画
      if (obj.is_bed) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.35;
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();
      }

      // 選択ハロー
      if (isSelected || isHover) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, isSelected ? 18 : 15, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = isSelected ? 0.24 : 0.14;
        ctx.fill();
        ctx.restore();
      }

      // ノード本体 ＆ 仰角(頭上/下方)インジケータリング
      const elev = Number(pose.elevation_deg ?? 0);
      const nodeR = isSelected ? 11 : 9;
      if (Math.abs(elev) >= 5) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR + 4.5, 0, Math.PI * 2);
        ctx.strokeStyle = elev > 0 ? "#67e8f9" : "#fda4af";
        ctx.globalAlpha = 0.72;
        ctx.lineWidth = 1.6;
        if (elev > 0) ctx.setLineDash([2.5, 2.5]);
        ctx.stroke();
        ctx.restore();
      }
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = isSelected ? 2.5 : 1.5;
      ctx.strokeStyle = isSelected ? "#ffffff" : "#060a12";
      ctx.stroke();

      pendingLabels.push({ obj, idx, color, isSelected, pose, pos, nodeR });
    });

    // 6. ラベルタグ描画（左半球は左寄せ・右半球は右寄せ＋衝突回避オフセット）
    const placedBoxes = [];
    const boxH = 28;
    const boxesOverlap = (a, b) =>
      !(a.x + a.w + 4 <= b.x || b.x + b.w + 4 <= a.x || a.y + a.h + 4 <= b.y || b.y + b.h + 4 <= a.y);

    pendingLabels.forEach(({ obj, idx, color, isSelected, pose, pos, nodeR }) => {
      const label = obj.label || obj.id || `Obj ${idx + 1}`;
      const elevNum = Math.round(Number(pose.elevation_deg ?? 0));
      const elevStr = Math.abs(elevNum) >= 5 ? (elevNum > 0 ? ` ↑+${elevNum}°` : ` ↓${elevNum}°`) : "";
      const coordTxt = `${Math.round(pose.azimuth_deg)}° · ${pose.distance_m.toFixed(1)}m${elevStr}`;
      ctx.font = "700 11.5px 'Segoe UI', sans-serif";
      const labelW = ctx.measureText(label).width;
      ctx.font = "10px Consolas, monospace";
      const coordW = ctx.measureText(coordTxt).width;
      const textW = Math.min(w - 16, Math.max(labelW, coordW) + 14);

      const dxFromCenter = pos.x - cx;
      let boxX;
      let boxY;
      let align = "center";
      if (Math.abs(dxFromCenter) < 16) {
        // 中央付近は上下配置
        boxX = clamp(pos.x - textW / 2, 6, w - textW - 6);
        boxY = pos.y >= cy ? pos.y + nodeR + 8 : pos.y - nodeR - boxH - 8;
        align = "center";
      } else if (dxFromCenter < 0) {
        // 左半球：ノードの左側へ右端揃えで配置（左右の近距離話者同士が重ならない）
        boxX = clamp(pos.x - nodeR - 8 - textW, 6, w - textW - 6);
        boxY = clamp(pos.y - boxH / 2, 6, h - boxH - 6);
        align = "right";
      } else {
        // 右半球：ノードの右側へ左端揃えで配置
        boxX = clamp(pos.x + nodeR + 8, 6, w - textW - 6);
        boxY = clamp(pos.y - boxH / 2, 6, h - boxH - 6);
        align = "left";
      }

      // 既存ラベルとの衝突回避（上下方向へ段階的にオフセット）
      const cand = { x: boxX, y: boxY, w: textW, h: boxH };
      const origY = boxY;
      const offsets = [0, -34, 34, -68, 68, -102, 102];
      for (const dy of offsets) {
        const tryY = clamp(origY + dy, 6, h - boxH - 6);
        const testBox = { x: cand.x, y: tryY, w: cand.w, h: cand.h };
        if (!placedBoxes.some((pb) => boxesOverlap(testBox, pb))) {
          cand.y = tryY;
          break;
        }
      }
      placedBoxes.push(cand);

      // ノードとラベルを結ぶ細い引き出し線
      const anchorX = align === "right" ? cand.x + cand.w : align === "left" ? cand.x : cand.x + cand.w / 2;
      const anchorY = cand.y + cand.h / 2;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(pos.x, pos.y);
      ctx.lineTo(anchorX, anchorY);
      ctx.strokeStyle = isSelected ? color : "rgba(148, 163, 184, 0.45)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = "rgba(6, 10, 18, 0.90)";
      ctx.strokeStyle = isSelected ? color : "rgba(148, 163, 184, 0.38)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(cand.x, cand.y, cand.w, cand.h, 6);
      else ctx.rect(cand.x, cand.y, cand.w, cand.h);
      ctx.fill();
      ctx.stroke();

      const textX = align === "right" ? cand.x + cand.w - 7 : align === "left" ? cand.x + 7 : cand.x + cand.w / 2;
      ctx.textAlign = align;
      ctx.fillStyle = "#f8fafc";
      ctx.font = "700 11.5px 'Segoe UI', sans-serif";
      ctx.fillText(label, textX, cand.y + 12);
      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px Consolas, monospace";
      ctx.fillText(coordTxt, textX, cand.y + 23);
    });

    ctx.restore();
  }

  function startRadarLoop() {
    stopRadarLoop();
    const tick = () => {
      if (state.currentStep !== 3) return;
      drawSpatialRadar();
      updateLevelMeters();
      syncVideoTimeUI();
      state.radarAnimFrame = window.requestAnimationFrame(tick);
    };
    state.radarAnimFrame = window.requestAnimationFrame(tick);
  }

  function stopRadarLoop() {
    if (state.radarAnimFrame) {
      window.cancelAnimationFrame(state.radarAnimFrame);
      state.radarAnimFrame = null;
    }
  }

  function syncVideoTimeUI() {
    const video = $("studio-video");
    if (!video) return;
    const cur = Number(video.currentTime) || 0;
    const dur = Number(video.duration) || 0;
    $("time-current").textContent = formatTimecode(cur);
    if (dur > 0) {
      $("time-duration").textContent = formatTimecode(dur);
      const seek = $("studio-seek");
      if (document.activeElement !== seek) {
        seek.max = String(dur);
        seek.value = String(cur);
      }
    }
    // 再生中の軽微なドリフト監視
    if (state.webAudioPlaying && state.audioCtx && !video.paused) {
      const audioPos = state.audioStartOffset + (state.audioCtx.currentTime - state.audioStartCtxTime);
      if (Math.abs(audioPos - cur) > 0.16) {
        startWebAudioAt(cur);
      }
    }
  }

  function hitTestRadarNode(clientX, clientY) {
    const canvas = $("spatial-radar-canvas");
    if (!canvas || !state.scene) return null;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const { cx, cy, maxR } = getRadarGeometry(canvas);
    const curT = Number($("studio-video").currentTime) || 0;

    let best = null;
    let bestDist = 24;
    (state.scene.objects || []).forEach((obj) => {
      const pose = interpolateObjectPose(obj, curT);
      const pt = polarToCanvas(pose.azimuth_deg, pose.distance_m, cx, cy, maxR);
      const d = Math.hypot(x - pt.x, y - pt.y);
      if (d < bestDist) {
        bestDist = d;
        best = obj.id;
      }
    });
    return best;
  }

  // --------------------------------------------------------------------------
  // トラックミキサー ＆ シーン編集
  // --------------------------------------------------------------------------
  function getSelectedObject() {
    if (!state.scene || !Array.isArray(state.scene.objects)) return null;
    return state.scene.objects.find((o) => o.id === state.selectedObjectId) || null;
  }

  function markSceneDirty() {
    state.sceneDirty = true;
    $("scene-dirty-badge").classList.remove("hidden");
  }

  function updateObjectSpatialCoords(obj, newAz, newDist, newElev) {
    if (!obj) return;
    if (!Array.isArray(obj.keyframes) || !obj.keyframes.length) {
      obj.keyframes = [{ t: 0.0, azimuth_deg: 0.0, elevation_deg: 0.0, distance_m: 2.0 }];
    }
    const curT = Number($("studio-video").currentTime) || 0;
    const curPose = interpolateObjectPose(obj, curT);
    const applyAll = $("chk-apply-all-kf").checked;

    if (applyAll || obj.keyframes.length === 1) {
      const dAz = newAz != null ? newAz - curPose.azimuth_deg : 0;
      const dDist = newDist != null ? newDist - curPose.distance_m : 0;
      const dElev = newElev != null ? newElev - curPose.elevation_deg : 0;
      obj.keyframes.forEach((kf) => {
        if (newAz != null) {
          let az = Number(kf.azimuth_deg ?? 0) + dAz;
          if (az > 180) az -= 360;
          if (az < -180) az += 360;
          kf.azimuth_deg = Math.round(clamp(az, -180, 180) * 10) / 10;
        }
        if (newDist != null) {
          kf.distance_m = Math.round(clamp(Number(kf.distance_m ?? 2.0) + dDist, 0.5, 12.0) * 10) / 10;
        }
        if (newElev != null) {
          kf.elevation_deg = Math.round(clamp(Number(kf.elevation_deg ?? 0) + dElev, -90, 90) * 10) / 10;
        }
      });
    } else {
      // 現在時刻に最も近いキーフレームを更新（0.35秒以内）、なければ挿入
      let nearest = null;
      let nearestDiff = Infinity;
      obj.keyframes.forEach((kf) => {
        const diff = Math.abs(Number(kf.t ?? kf.t_sec ?? 0) - curT);
        if (diff < nearestDiff) {
          nearestDiff = diff;
          nearest = kf;
        }
      });
      if (nearest && nearestDiff <= 0.35) {
        if (newAz != null) nearest.azimuth_deg = Math.round(clamp(newAz, -180, 180) * 10) / 10;
        if (newDist != null) nearest.distance_m = Math.round(clamp(newDist, 0.5, 12.0) * 10) / 10;
        if (newElev != null) nearest.elevation_deg = Math.round(clamp(newElev, -90, 90) * 10) / 10;
      } else {
        obj.keyframes.push({
          t: Math.round(curT * 100) / 100,
          azimuth_deg: Math.round(clamp(newAz ?? curPose.azimuth_deg, -180, 180) * 10) / 10,
          elevation_deg: Math.round(clamp(newElev ?? curPose.elevation_deg, -90, 90) * 10) / 10,
          distance_m: Math.round(clamp(newDist ?? curPose.distance_m, 0.5, 12.0) * 10) / 10,
        });
        obj.keyframes.sort((a, b) => Number(a.t ?? a.t_sec ?? 0) - Number(b.t ?? b.t_sec ?? 0));
      }
    }

    markSceneDirty();
    renderTrackList();
    syncInspectorToSelected();
    drawSpatialRadar();
  }

  function renderTrackList() {
    const list = $("track-list");
    if (!list) return;
    const objects = (state.scene && state.scene.objects) || [];
    const curT = Number($("studio-video").currentTime) || 0;
    list.innerHTML = "";

    if (!objects.length) {
      list.innerHTML = `<div class="loading-placeholder">音源オブジェクト情報がありません</div>`;
      return;
    }

    objects.forEach((obj, idx) => {
      const pose = interpolateObjectPose(obj, curT);
      const color = objectColor(obj, idx);
      const item = document.createElement("div");
      item.className = `track-item ${obj.id === state.selectedObjectId ? "is-selected" : ""}`;
      const gain = Number(obj.gain_db ?? 0);
      const gainStr = `${gain >= 0 ? "+" : ""}${gain.toFixed(1)}dB`;
      const typeStr = obj.is_bed ? "Bed" : "Object";
      item.innerHTML = `
        <div class="track-left">
          <span class="track-swatch" style="background:${color}"></span>
          <div>
            <div class="track-title">${escapeHtml(obj.label || obj.id)}</div>
            <div class="track-sub">${escapeHtml(formatRoleLabelJa(obj.role))} · ${typeStr}</div>
          </div>
        </div>
        <div class="track-right">${Math.round(pose.azimuth_deg)}° / ${pose.distance_m.toFixed(1)}m / ${gainStr}</div>
      `;
      item.addEventListener("click", () => {
        state.selectedObjectId = obj.id;
        renderTrackList();
        syncInspectorToSelected();
        drawSpatialRadar();
      });
      list.appendChild(item);
    });
  }

  function syncInspectorToSelected() {
    const obj = getSelectedObject();
    const sliders = ["slider-gain", "slider-azimuth", "slider-distance", "slider-elevation"];
    if (!obj) {
      $("insp-title").textContent = "音源を選択してください";
      $("insp-role-badge").textContent = "--";
      sliders.forEach((id) => ($(id).disabled = true));
      return;
    }
    sliders.forEach((id) => ($(id).disabled = false));
    const curT = Number($("studio-video").currentTime) || 0;
    const pose = interpolateObjectPose(obj, curT);
    const gain = Number(obj.gain_db ?? 0);

    $("insp-title").textContent = obj.label || obj.id;
    $("insp-role-badge").textContent = `${formatRoleLabelJa(obj.role)} · ${obj.is_bed ? "Bed" : "Dynamic Object"}`;

    $("slider-gain").value = String(clamp(gain, -24, 12));
    $("val-gain").textContent = `${gain >= 0 ? "+" : ""}${gain.toFixed(1)} dB`;

    $("slider-azimuth").value = String(Math.round(clamp(pose.azimuth_deg, -180, 180)));
    $("val-azimuth").textContent = `${Math.round(pose.azimuth_deg)}°`;

    $("slider-distance").value = String(clamp(pose.distance_m, 0.5, 12.0).toFixed(1));
    $("val-distance").textContent = `${pose.distance_m.toFixed(1)} m`;

    $("slider-elevation").value = String(Math.round(clamp(pose.elevation_deg, -90, 90)));
    $("val-elevation").textContent = `${Math.round(pose.elevation_deg)}°`;
  }

  function handleRemixDialogPanModeChange() {
    if (!state.scene || !Array.isArray(state.scene.objects)) return;
    const mode = $("remix-dialog-pan-mode").value === "immersive" ? "immersive" : "cinema";
    let hasRawKeyframes = false;

    state.scene.objects.forEach((obj) => {
      if (String(obj.role || "").toLowerCase() !== "dialog") return;
      if (Array.isArray(obj.raw_keyframes) && obj.raw_keyframes.length) {
        hasRawKeyframes = true;
        obj.keyframes = obj.raw_keyframes.map((rkf) => {
          const rawAz = Number(rkf.azimuth_deg ?? 0);
          const mappedAz =
            mode === "immersive"
              ? Math.round(rawAz * 100) / 100
              : Math.round(22 * Math.tanh(rawAz / 22) * 100) / 100;
          return { ...rkf, azimuth_deg: mappedAz };
        });
      }
    });

    if (hasRawKeyframes) {
      state.scene.recompute_dialog_pan = false;
      renderTrackList();
      syncInspectorToSelected();
      drawSpatialRadar();
    } else {
      state.scene.recompute_dialog_pan = true;
    }

    if (!state.scene.mix_settings || typeof state.scene.mix_settings !== "object") {
      state.scene.mix_settings = {};
    }
    state.scene.mix_settings.dialog_pan_mode = mode;
    markSceneDirty();
  }

  async function applyRemix() {
    if (!state.activeJobId || !state.scene) return;
    const btn = $("btn-remix-apply");
    btn.disabled = true;
    const origText = btn.innerHTML;
    btn.innerHTML = `<span class="spinner-sm"></span><span>再MixMaster中…</span>`;

    const video = $("studio-video");
    const resumeTime = video ? Number(video.currentTime) || 0 : 0;
    pauseStudioPlayback();

    const remixFmt = $("remix-format").value;
    const remixOrderMap = {
      ambisonics_foa: 1,
      ambisonics_hoa: 3,
      ambisonics_hoa5: 5,
      ambisonics_hoa7: 7,
    };
    const rawRemixLufs = $("remix-lufs").value;
    const isMatchOriginal = rawRemixLufs === "match_original";
    const dialogPanMode =
      $("remix-dialog-pan-mode").value === "immersive" ? "immersive" : "cinema";
    const enableDialogDucking = $("chk-remix-ducking").checked;

    try {
      const resp = await apiJson(`/api/jobs/${encodeURIComponent(state.activeJobId)}/remix`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene: state.scene,
          spatial_format: remixFmt,
          ambisonics_order: remixOrderMap[remixFmt] || null,
          loudness_mode: isMatchOriginal ? "match_original" : "fixed",
          target_lufs: isMatchOriginal ? null : Number(rawRemixLufs),
          dialog_pan_mode: dialogPanMode,
          enable_dialog_ducking: enableDialogDucking,
        }),
      });

      const updatedJob = resp.state || state.activeJob;
      if (resp.media_urls) updatedJob.media_urls = resp.media_urls;
      if (resp.result) updatedJob.result = resp.result;
      if (resp.scene) {
        updatedJob.scene = resp.scene;
        state.scene = deepClone(resp.scene);
      }
      state.activeJob = updatedJob;
      state.sceneDirty = false;
      $("scene-dirty-badge").classList.add("hidden");

      syncRemixControlsFromJob(updatedJob);
      renderMetricsAndDownloads(updatedJob);
      renderTrackList();
      syncInspectorToSelected();
      drawSpatialRadar();
      await loadStudioMedia(updatedJob);
      if (video && resumeTime > 0) {
        video.addEventListener(
          "loadedmetadata",
          () => {
            video.currentTime = Math.min(resumeTime, video.duration || resumeTime);
          },
          { once: true }
        );
      }
      showToast(`新しい3D配置で再MixMasterが完了しました（${resp.elapsed_sec ?? ""}秒）！`, "ok", 4500);
    } catch (err) {
      showToast(`再MixMasterに失敗しました: ${err.message}`, "error", 6500);
    } finally {
      btn.disabled = false;
      btn.innerHTML = origText;
    }
  }

  // --------------------------------------------------------------------------
  // AIモデル管理モーダル
  // --------------------------------------------------------------------------
  function openModelsModal() {
    $("models-modal").classList.remove("hidden");
    loadModelsCatalog();
  }

  function closeModelsModal() {
    $("models-modal").classList.add("hidden");
    if (state.modelsPollTimer) {
      clearInterval(state.modelsPollTimer);
      state.modelsPollTimer = null;
    }
  }

  function formatModelSize(mb) {
    const n = Number(mb) || 0;
    if (n >= 1024) return `${(n / 1024).toFixed(1)} GB`;
    return `${n} MB`;
  }

  function renderModelsModal(models) {
    const body = $("models-modal-body");
    if (!body) return;
    if (!Array.isArray(models) || !models.length) {
      body.innerHTML = `<div class="loading-placeholder">モデルカタログが見つかりません</div>`;
      return;
    }

    body.innerHTML = "";
    let anyDownloading = false;

    models.forEach((m) => {
      const row = document.createElement("div");
      row.className = "model-row";

      const st = m.status || "missing";
      const dl = m.download_state || null;
      const isDownloading = dl && dl.status === "downloading";
      if (isDownloading) anyDownloading = true;

      let badgeHtml = `<span class="model-badge is-missing">未取得</span>`;
      if (st === "shared_soundsep") {
        badgeHtml = `<span class="model-badge is-shared">✅ SoundSeparator共有</span>`;
      } else if (st === "installed") {
        badgeHtml = `<span class="model-badge is-installed">✅ 導入済み</span>`;
      }

      const sizeText = m.download_mb ? `容量目安: ${formatModelSize(m.download_mb)}` : "";
      const catText = m.category_label || m.category || "";

      let actionHtml = "";
      if (isDownloading) {
        const pct = Math.round(Number(dl.progress) || 0);
        actionHtml = `
          <div class="dl-mini-progress"><div class="dl-mini-fill" style="width:${pct}%"></div></div>
          <div class="dl-mini-msg">${pct}% · ${escapeHtml(dl.message || "取得中…")}</div>
        `;
      } else if (st === "shared_soundsep") {
        actionHtml = `<span class="verdict-badge is-ok">共有利用OK</span>`;
      } else if (st === "installed") {
        actionHtml = `<span class="verdict-badge is-ok">利用可能</span>`;
      } else if (m.downloadable !== false) {
        actionHtml = `<button class="btn btn-outline-cyan btn-sm btn-dl-model" data-model-id="${escapeHtml(m.id)}" type="button">⬇ ダウンロード</button>`;
        if (dl && dl.status === "error") {
          actionHtml += `<div class="dl-mini-msg" style="color:#fda4af">${escapeHtml(dl.error || "失敗")}</div>`;
        }
      } else {
        actionHtml = `<span class="verdict-badge is-warn">手動配置</span>`;
      }

      row.innerHTML = `
        <div class="model-row-info">
          <div class="model-row-title">
            <strong>${escapeHtml(m.label || m.id)}</strong>
            ${badgeHtml}
            ${catText ? `<span class="meta-chip">${escapeHtml(catText)}</span>` : ""}
          </div>
          <p class="model-row-desc">${escapeHtml(m.description || "")}</p>
          <div class="model-row-meta">
            <span>${escapeHtml(m.vram_summary || "")}</span>
            ${sizeText ? `<span>${escapeHtml(sizeText)}</span>` : ""}
            ${m.source_url ? `<a href="${escapeHtml(m.source_url)}" target="_blank" rel="noopener noreferrer" style="color:var(--cyan)">配布元 ↗</a>` : ""}
          </div>
        </div>
        <div class="model-row-actions">${actionHtml}</div>
      `;

      const dlBtn = row.querySelector(".btn-dl-model");
      if (dlBtn) {
        dlBtn.addEventListener("click", () => startModelDownload(m.id, m.label));
      }
      body.appendChild(row);
    });

    if (anyDownloading && !state.modelsPollTimer) {
      state.modelsPollTimer = window.setInterval(loadModelsCatalog, 1300);
    } else if (!anyDownloading && state.modelsPollTimer) {
      clearInterval(state.modelsPollTimer);
      state.modelsPollTimer = null;
    }
  }

  async function loadModelsCatalog() {
    try {
      const data = await apiJson("/api/models");
      renderModelsModal(data.models || []);
    } catch (err) {
      $("models-modal-body").innerHTML = `<div class="loading-placeholder">読み込みエラー: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function startModelDownload(modelId, label) {
    try {
      await apiJson(`/api/models/${encodeURIComponent(modelId)}/download`, { method: "POST" });
      showToast(`「${label || modelId}」のダウンロードを開始しました。`, "info");
      loadModelsCatalog();
    } catch (err) {
      showToast(`ダウンロード開始に失敗しました: ${err.message}`, "error", 6000);
    }
  }

  function stopAllTimers() {
    [
      "jobPollTimer",
      "elapsedTimer",
      "healthTimer",
      "modelsPollTimer",
    ].forEach((key) => {
      if (state[key]) {
        clearInterval(state[key]);
        state[key] = null;
      }
    });
    if (state.radarAnimFrame) {
      cancelAnimationFrame(state.radarAnimFrame);
      state.radarAnimFrame = null;
    }
  }

  async function shutdownApp() {
    if (!window.confirm("CineSS サーバーと待機エージェントを停止しますか？")) {
      return;
    }
    const btn = $("btn-shutdown-app");
    if (btn) btn.disabled = true;

    const sendShutdown = async (forceFlag) => {
      const resp = await fetch("/api/system/shutdown", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: Boolean(forceFlag) }),
      });
      let data = null;
      try {
        data = await resp.json();
      } catch (_) {
        data = null;
      }
      return { resp, data };
    };

    try {
      let { resp, data } = await sendShutdown(false);
      if (resp.status === 409) {
        const msg = (data && data.detail) || "処理中のジョブがあります。強制停止しますか？";
        if (!window.confirm(`${msg}\n強制的に停止してよろしいですか？`)) {
          if (btn) btn.disabled = false;
          return;
        }
        ({ resp, data } = await sendShutdown(true));
      }
      if (!resp.ok) {
        const detail = (data && (data.detail || data.error || data.message)) || `HTTP ${resp.status}`;
        throw new Error(detail);
      }
      stopAllTimers();
      pauseStudioPlayback();
      stopWebAudioNodes();
      const badge = $("agent-status-badge");
      const badgeText = $("agent-status-text");
      if (badge && badgeText) {
        badge.classList.remove("is-connected", "is-working");
        badge.classList.add("is-offline");
        badgeText.textContent = "CineSS 停止済み";
      }
      const overlay = $("shutdown-overlay");
      if (overlay) overlay.classList.remove("hidden");
    } catch (err) {
      if (btn) btn.disabled = false;
      showToast(`アプリの停止に失敗しました: ${err.message}`, "error");
    }
  }

  // --------------------------------------------------------------------------
  // イベント結線・初期化
  // --------------------------------------------------------------------------
  function bindEvents() {
    // ヘッダー
    $("btn-recheck-system").addEventListener("click", () => loadSystemCheck(true));
    $("btn-open-models").addEventListener("click", openModelsModal);
    $("btn-shutdown-app").addEventListener("click", shutdownApp);
    $("btn-close-models").addEventListener("click", closeModelsModal);
    $("btn-refresh-models").addEventListener("click", loadModelsCatalog);
    $("models-modal").addEventListener("click", (e) => {
      if (e.target === $("models-modal")) closeModelsModal();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !$("models-modal").classList.contains("hidden")) {
        closeModelsModal();
      }
    });

    // ステップタブ
    $("step-tab-1").addEventListener("click", () => setStep(1));
    $("step-tab-2").addEventListener("click", () => {
      if (state.activeJob) setStep(2);
      else showToast("実行中または過去のジョブがありません。", "info");
    });
    $("step-tab-3").addEventListener("click", () => {
      if (state.activeJob && state.activeJob.status === "done") setStep(3);
      else showToast("完了したジョブがある場合に3D空間ステージを開けます。", "info");
    });

    // ドロップゾーン＆動画選択
    const dropzone = $("dropzone");
    const fileInput = $("video-file-input");
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-dragover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) handleVideoUpload(file);
    });
    fileInput.addEventListener("change", () => {
      const file = fileInput.files && fileInput.files[0];
      if (file) handleVideoUpload(file);
      fileInput.value = "";
    });
    $("btn-change-video").addEventListener("click", () => fileInput.click());

    // お試し区間切替＆プラン再計算
    $("chk-preview-trim").addEventListener("change", (e) => {
      $("preview-trim-fields").classList.toggle("hidden", !e.target.checked);
      updateAmbisonicsCapacityUI();
      refreshVideoPlan();
    });
    $("input-preview-duration").addEventListener("change", () => {
      updateAmbisonicsCapacityUI();
      refreshVideoPlan();
    });
    document.querySelectorAll('input[name="proc_mode"], input[name="spatial_family"], input[name="spatial_format"]').forEach((el) => {
      el.addEventListener("change", () => {
        updateAmbisonicsCapacityUI();
        refreshVideoPlan();
      });
    });
    $("select-quality").addEventListener("change", refreshVideoPlan);
    document.querySelectorAll('input[name="execution_mode"]').forEach((el) => {
      el.addEventListener("change", updateExecutionModeHint);
    });

    // 処理開始＆履歴更新
    $("btn-start-job").addEventListener("click", startNewJob);
    $("btn-refresh-jobs").addEventListener("click", loadRecentJobs);

    // ステップ2 モニター操作
    $("btn-switch-local").addEventListener("click", switchQueuedJobToLocal);
    $("btn-cancel-job").addEventListener("click", cancelCurrentJob);
    $("btn-back-step1").addEventListener("click", () => setStep(1));

    // ステップ3 プレイヤー＆A/B比較
    $("btn-new-project").addEventListener("click", () => {
      pauseStudioPlayback();
      setStep(1);
    });
    $("btn-play-toggle").addEventListener("click", toggleStudioPlayback);
    $("btn-ab-original").addEventListener("click", () => setABSource("original"));
    $("btn-ab-binaural").addEventListener("click", () => setABSource("binaural"));

    const studioVideo = $("studio-video");
    studioVideo.addEventListener("click", toggleStudioPlayback);
    studioVideo.addEventListener("ended", () => {
      stopWebAudioNodes();
      $("btn-play-toggle").textContent = "▶";
    });
    studioVideo.addEventListener("loadedmetadata", () => {
      $("time-duration").textContent = formatTimecode(studioVideo.duration);
      $("studio-seek").max = String(studioVideo.duration || 100);
    });

    $("studio-seek").addEventListener("input", (e) => {
      const t = Number(e.target.value) || 0;
      studioVideo.currentTime = t;
      $("time-current").textContent = formatTimecode(t);
      if (state.webAudioPlaying) startWebAudioAt(t);
      drawSpatialRadar();
      syncInspectorToSelected();
    });

    // 3D レーダー Canvas ドラッグ操作
    const canvas = $("spatial-radar-canvas");
    canvas.addEventListener("pointerdown", (e) => {
      const hitId = hitTestRadarNode(e.clientX, e.clientY);
      if (!hitId) return;
      state.selectedObjectId = hitId;
      state.radarDraggingId = hitId;
      canvas.classList.add("is-dragging");
      canvas.setPointerCapture(e.pointerId);
      renderTrackList();
      syncInspectorToSelected();
      drawSpatialRadar();
    });

    canvas.addEventListener("pointermove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      if (state.radarDraggingId) {
        const { cx, cy, maxR } = getRadarGeometry(canvas);
        const polar = canvasToPolar(x, y, cx, cy, maxR);
        const obj = getSelectedObject();
        updateObjectSpatialCoords(obj, polar.azimuth_deg, polar.distance_m, null);
        const tip = $("radar-hover-tip");
        tip.textContent = `${obj ? obj.label || obj.id : ""}: ${polar.azimuth_deg}° / ${polar.distance_m.toFixed(1)}m`;
        tip.style.left = `${clamp(x + 14, 8, rect.width - 150)}px`;
        tip.style.top = `${clamp(y - 26, 8, rect.height - 32)}px`;
        tip.classList.remove("hidden");
      } else {
        const hitId = hitTestRadarNode(e.clientX, e.clientY);
        state.radarHoverId = hitId;
        canvas.style.cursor = hitId ? "grab" : "default";
        if (!hitId) $("radar-hover-tip").classList.add("hidden");
      }
    });

    const endDrag = (e) => {
      if (!state.radarDraggingId) return;
      state.radarDraggingId = null;
      canvas.classList.remove("is-dragging");
      $("radar-hover-tip").classList.add("hidden");
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch (_) {}
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    window.addEventListener("resize", () => {
      if (state.currentStep === 3) {
        resizeRadarCanvas();
        drawSpatialRadar();
      }
    });

    // インスペクタスライダー
    $("slider-gain").addEventListener("input", (e) => {
      const obj = getSelectedObject();
      if (!obj) return;
      const val = Number(e.target.value);
      obj.gain_db = val;
      $("val-gain").textContent = `${val >= 0 ? "+" : ""}${val.toFixed(1)} dB`;
      markSceneDirty();
      renderTrackList();
    });
    $("slider-azimuth").addEventListener("input", (e) => {
      updateObjectSpatialCoords(getSelectedObject(), Number(e.target.value), null, null);
    });
    $("slider-distance").addEventListener("input", (e) => {
      updateObjectSpatialCoords(getSelectedObject(), null, Number(e.target.value), null);
    });
    $("slider-elevation").addEventListener("input", (e) => {
      updateObjectSpatialCoords(getSelectedObject(), null, null, Number(e.target.value));
    });

    $("btn-reset-scene").addEventListener("click", () => {
      if (!state.originalScene) return;
      state.scene = deepClone(state.originalScene);
      state.sceneDirty = true;
      $("scene-dirty-badge").classList.remove("hidden");
      syncRemixControlsFromJob(state.activeJob);
      renderTrackList();
      syncInspectorToSelected();
      drawSpatialRadar();
      showToast("3D音源配置を初期状態に戻しました（「この配置で再Mix」で反映できます）。", "info");
    });

    $("remix-dialog-pan-mode").addEventListener("change", handleRemixDialogPanModeChange);
    $("chk-remix-ducking").addEventListener("change", () => markSceneDirty());
    $("btn-remix-apply").addEventListener("click", applyRemix);

    if ($("btn-director-reapply")) {
      $("btn-director-reapply").addEventListener("click", applyDirectorPlan);
    }
    if ($("btn-director-format-json")) {
      $("btn-director-format-json").addEventListener("click", () => {
        const el = $("director-plan-json");
        if (!el) return;
        try {
          el.value = JSON.stringify(JSON.parse(el.value || "{}"), null, 2);
        } catch (err) {
          showToast(`JSON整形エラー: ${err.message}`, "error");
        }
      });
    }
    if ($("btn-director-reset-json")) {
      $("btn-director-reset-json").addEventListener("click", () => {
        const el = $("director-plan-json");
        if (!el || !state.initialDirectorPlan) return;
        el.value = JSON.stringify(state.initialDirectorPlan, null, 2);
        showToast("監督演出プランを初期の自動生成内容に戻しました。", "info");
      });
    }
  }

  // --------------------------------------------------------------------------
  // 起動シーケンス
  // --------------------------------------------------------------------------
  function init() {
    bindEvents();
    loadSystemCheck(false);
    loadRecentJobs();
    state.healthTimer = window.setInterval(pollLightHealth, 4500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
