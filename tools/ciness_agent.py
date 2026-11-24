# coding: utf-8
"""CineSS エージェント連携 CLI / 常駐ランナー。

主な用途:
  .venv/Scripts/python.exe -X utf8 tools/ciness_agent.py status
  .venv/Scripts/python.exe -X utf8 tools/ciness_agent.py start-server
  .venv/Scripts/python.exe -X utf8 tools/ciness_agent.py watch --server http://127.0.0.1:8780
  .venv/Scripts/python.exe -X utf8 tools/ciness_agent.py run-job --job-id <id>

設計方針:
- リポジトリ位置はこのファイルから相対解決し、絶対パスを持たない。
- 推論時の暗黙モデルダウンロードは行わない。
- watch モードでは別スレッドで継続的にハートビートを送信し、Web UI 上の
  「▶ 処理開始」ボタン押下（queued_for_agent）を検知して即座に受注・実行する。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.config import (  # noqa: E402
    AGENT_STATE_PATH,
    DEFAULT_PORT,
    JOBS_DIR,
    OUTPUT_DIR,
    SHUTDOWN_FLAG_PATH,
    ensure_directories,
)
from app.hardware_advisor import check_system  # noqa: E402
from app.model_manager import list_models  # noqa: E402


class AgentError(Exception):
    """エージェントCLI実行時の明示的エラー。"""


def _default_server_url(port: int | None = None) -> str:
    return f"http://127.0.0.1:{port or DEFAULT_PORT}"


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json", "User-Agent": "CineSS-Agent/1.0"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        if not raw.strip():
            return {}
        return json.loads(raw)


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_server(server_url: str, timeout: float = 2.5) -> dict[str, Any] | None:
    base = server_url.rstrip("/")
    try:
        return _http_json("GET", f"{base}/api/system/check", timeout=timeout)
    except Exception:
        return None


def _write_local_agent_state(
    agent_name: str,
    status: str,
    active_job_id: str | None = None,
) -> None:
    """サーバー未応答時のフォールバックとしても参照できるよう状態ファイルを原子更新する。"""
    ensure_directories()
    payload = {
        "agent_name": agent_name,
        "pid": os.getpid(),
        "status": status,
        "active_job_id": active_job_id,
        "last_seen": time.time(),
    }
    tmp = AGENT_STATE_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, AGENT_STATE_PATH)
    except OSError:
        tmp.unlink(missing_ok=True)


def _send_heartbeat(
    server_url: str,
    agent_name: str,
    status: str,
    active_job_id: str | None = None,
) -> bool:
    base = server_url.rstrip("/")
    payload = {
        "agent_name": agent_name,
        "pid": os.getpid(),
        "status": status,
        "active_job_id": active_job_id,
    }
    try:
        _http_json("POST", f"{base}/api/agent/heartbeat", payload, timeout=3.0)
        return True
    except Exception:
        _write_local_agent_state(agent_name, status, active_job_id)
        return False


# ---------------------------------------------------------------------------
# サブコマンド実装
# ---------------------------------------------------------------------------


def cmd_status(server_url: str | None = None) -> dict[str, Any]:
    """環境診断・サーバー稼働確認・モデル導入状況をまとめて返す。"""
    url = (server_url or _default_server_url()).rstrip("/")
    system_info = check_system()
    models = list_models()
    server_resp = _probe_server(url)
    server_running = server_resp is not None

    agent_connected = False
    agent_info: dict[str, Any] = {}
    if server_resp is not None:
        agent_connected = bool(server_resp.get("agent_connected", False))
        agent_info = {
            "agent_connected": agent_connected,
            "agent_name": server_resp.get("agent_name"),
            "agent_last_seen": server_resp.get("agent_last_seen"),
            "active_job_id": server_resp.get("active_job_id"),
        }
    elif AGENT_STATE_PATH.is_file():
        try:
            st = json.loads(AGENT_STATE_PATH.read_text(encoding="utf-8"))
            age = time.time() - float(st.get("last_seen", 0))
            agent_connected = age < 15.0 and st.get("status") not in ("stopped", "offline")
            agent_info = {
                "agent_connected": agent_connected,
                "agent_name": st.get("agent_name"),
                "agent_last_seen": st.get("last_seen"),
                "active_job_id": st.get("active_job_id"),
            }
        except (OSError, ValueError):
            pass

    return {
        "app_root": str(APP_ROOT),
        "python": sys.executable,
        "venv_ready": (APP_ROOT / ".venv" / "Scripts" / "python.exe").is_file()
        or (APP_ROOT / ".venv" / "bin" / "python").is_file(),
        "server_url": url,
        "server_running": server_running,
        "agent": agent_info,
        "system": system_info,
        "models": [
            {
                "id": m["id"],
                "label": m["label"],
                "category": m["category"],
                "status": m["status"],
                "path": m["path"],
                "vram_summary": m["vram_summary"],
            }
            for m in models
        ],
    }


def cmd_start_server(
    host: str = "127.0.0.1",
    port: int | None = None,
    *,
    open_browser: bool = True,
    wait_sec: float = 30.0,
) -> dict[str, Any]:
    """FastAPI サーバーを起動して応答確認し、必要ならブラウザで開く。"""
    target_port = int(port or DEFAULT_PORT)
    url = f"http://{host}:{target_port}"

    if _probe_server(url) is not None or _port_open(host, target_port):
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return {
            "status": "already_running",
            "url": url,
            "host": host,
            "port": target_port,
            "pid": None,
            "message": f"CineSS サーバーは既に {url} で稼働中です。",
        }

    server_py = APP_ROOT / "app" / "server.py"
    if not server_py.is_file():
        raise AgentError(f"サーバーファイルが見つかりません: {server_py}")

    ensure_directories()
    log_path = OUTPUT_DIR / "_server.log"

    venv_py = APP_ROOT / ".venv" / "Scripts" / "python.exe"
    python_exe = str(venv_py) if venv_py.is_file() else sys.executable

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["CINESS_PORT"] = str(target_port)

    command = [
        python_exe,
        "-X",
        "utf8",
        "-m",
        "uvicorn",
        "app.server:app",
        "--host",
        host,
        "--port",
        str(target_port),
    ]

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=str(APP_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

    deadline = time.time() + max(3.0, float(wait_sec))
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
            except OSError:
                pass
            raise AgentError(f"サーバーが起動直後に終了しました (exit={proc.returncode}):\n{tail}")
        if _probe_server(url, timeout=1.0) is not None:
            ready = True
            break
        time.sleep(0.35)

    if not ready:
        raise AgentError(f"サーバー起動がタイムアウトしました ({wait_sec}秒): ログ {log_path} を確認してください。")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return {
        "status": "started",
        "url": url,
        "host": host,
        "port": target_port,
        "pid": proc.pid,
        "log_path": str(log_path),
        "message": f"CineSS サーバーを {url} で起動しました。",
    }


def _extract_job_payload(resp: dict[str, Any]) -> dict[str, Any] | None:
    """GET /api/agent/next のレスポンス形式の差異を吸収する。"""
    if not isinstance(resp, dict):
        return None
    if "job" in resp:
        job = resp.get("job")
        return job if isinstance(job, dict) and job.get("job_id") else None
    if resp.get("job_id"):
        return resp
    return None


def _resolve_job_dir_and_request(
    job_id: str,
    server_url: str | None = None,
    job_info: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    job_dir = Path(job_info["job_dir"]) if job_info and job_info.get("job_dir") else (JOBS_DIR / job_id)
    request_data: dict[str, Any] | None = None

    if job_info and isinstance(job_info.get("request"), dict):
        request_data = dict(job_info["request"])

    req_file = job_dir / "request.json"
    if request_data is None and req_file.is_file():
        request_data = json.loads(req_file.read_text(encoding="utf-8"))

    if request_data is None and server_url:
        try:
            remote = _http_json("GET", f"{server_url.rstrip('/')}/api/jobs/{job_id}", timeout=5.0)
            if isinstance(remote.get("request"), dict):
                request_data = dict(remote["request"])
            if remote.get("job_dir"):
                job_dir = Path(remote["job_dir"])
        except Exception:
            pass

    if request_data is None:
        raise AgentError(f"ジョブ {job_id} の request.json が見つかりません（{job_dir}）。")

    return job_dir, request_data


def _execute_pipeline_job(
    job_id: str,
    job_dir: Path,
    request_data: dict[str, Any],
    *,
    server_url: str | None = None,
    agent_name: str = "ciness-agent",
    sync_server: bool = True,
) -> dict[str, Any]:
    """app.pipeline.run_job を実行し、進捗と完了状態をサーバーへ逐次送信する。"""
    base = server_url.rstrip("/") if server_url else None
    state = {"progress": 0, "stage": "init", "last_sent": 0.0}

    def _post_progress(
        progress: int | float,
        stage: str,
        message: str,
        log_text: str | None = None,
        actor: str = agent_name,
        force: bool = False,
    ) -> None:
        pct = max(0, min(100, int(round(float(progress)))))
        now = time.time()
        if not force and log_text is None and pct == state["progress"] and stage == state["stage"] and (now - state["last_sent"]) < 0.4:
            return
        state["progress"] = pct
        state["stage"] = stage
        state["last_sent"] = now
        print(f"[{job_id}][{pct:3d}%][{stage}] {message}", file=sys.stderr, flush=True)
        if sync_server and base:
            payload = {
                "progress": pct,
                "stage": stage,
                "message": message,
                "log_text": log_text if log_text is not None else message,
                "actor": actor,
            }
            try:
                _http_json("POST", f"{base}/api/agent/jobs/{job_id}/progress", payload, timeout=5.0)
            except Exception as exc:
                print(f"[warn] 進捗送信に失敗 ({job_id}): {exc}", file=sys.stderr, flush=True)

    def progress_cb(*args: Any, **kwargs: Any) -> None:
        """progress_cb(pct, stage, message) / progress_cb(pct, message) の両形式に対応。"""
        pct = kwargs.get("progress", args[0] if len(args) >= 1 else state["progress"])
        if len(args) >= 3:
            stage = str(args[1])
            message = str(args[2])
        elif len(args) == 2:
            stage = str(kwargs.get("stage", state["stage"]))
            message = str(args[1])
        else:
            stage = str(kwargs.get("stage", state["stage"]))
            message = str(kwargs.get("message", ""))
        _post_progress(pct, stage, message)

    def log_cb(*args: Any, **kwargs: Any) -> None:
        """log_cb(message) / log_cb(actor, message) / log_cb(message, actor=...) の各形式に対応。"""
        actor = str(kwargs.get("actor", agent_name))
        if len(args) >= 2:
            actor = str(args[0])
            text = str(args[1])
        elif len(args) == 1:
            text = str(args[0])
        else:
            text = str(kwargs.get("message") or kwargs.get("log_text") or "")
        if not text:
            return
        _post_progress(state["progress"], state["stage"], text, log_text=text, actor=actor, force=True)

    # 受注（claim）
    if sync_server and base:
        try:
            _http_json(
                "POST",
                f"{base}/api/agent/jobs/{job_id}/claim",
                {"agent_name": agent_name, "pid": os.getpid()},
                timeout=5.0,
            )
        except urllib.error.HTTPError as exc:
            # 既に claim 済みならそのまま実行継続、404/409 は明示
            if exc.code not in (409,):
                print(f"[warn] claim 応答エラー ({exc.code}): {exc}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[warn] claim 送信失敗: {exc}", file=sys.stderr, flush=True)

    # エージェントによる初期判断ログを送信
    try:
        hw = check_system()
        tier = hw["recommendations"]["tier"]
        gpu_name = hw["gpu"]["name"]
        mode_req = request_data.get("mode", "auto")
        fmt_req = request_data.get("spatial_format", "atmos")
        _post_progress(
            2,
            "agent_plan",
            f"エージェント({agent_name})がジョブを受注しました",
            log_text=(
                f"受注完了: PC診断 Tier {tier}（{gpu_name}） / モード={mode_req} / 出力規格={fmt_req}。"
                "パイプラインを実行します。"
            ),
            actor=agent_name,
            force=True,
        )
    except Exception:
        _post_progress(2, "agent_plan", f"エージェント({agent_name})がジョブを受注しました", force=True)

    try:
        from app.pipeline import run_job  # noqa: PLC0415
    except ImportError as exc:
        err_msg = f"app.pipeline.run_job の読み込みに失敗しました: {exc}"
        if sync_server and base:
            try:
                _http_json(
                    "POST",
                    f"{base}/api/agent/jobs/{job_id}/complete",
                    {"status": "error", "result": None, "error": err_msg},
                    timeout=5.0,
                )
            except Exception:
                pass
        raise AgentError(err_msg) from exc

    try:
        result = run_job(job_dir, request_data, progress_cb=progress_cb, log_cb=log_cb)
        _post_progress(
            100,
            "done",
            "サラウンドMixMasterおよびバイノーラル出力が完了しました",
            log_text="全パイプライン工程が完了しました。Web UI の3Dステージビューアーで試聴・微調整できます。",
            actor=agent_name,
            force=True,
        )
        if sync_server and base:
            _http_json(
                "POST",
                f"{base}/api/agent/jobs/{job_id}/complete",
                {"status": "done", "result": result, "error": None},
                timeout=10.0,
            )
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[error] ジョブ {job_id} 実行中に例外発生:\n{tb}", file=sys.stderr, flush=True)
        err_text = f"{type(exc).__name__}: {exc}"
        if sync_server and base:
            try:
                _http_json(
                    "POST",
                    f"{base}/api/agent/jobs/{job_id}/complete",
                    {"status": "error", "result": None, "error": err_text},
                    timeout=10.0,
                )
            except Exception:
                pass
        raise AgentError(err_text) from exc


def cmd_run_job(
    job_id: str,
    *,
    server_url: str | None = None,
    agent_name: str = "ciness-agent",
    sync_server: bool = True,
) -> dict[str, Any]:
    """指定された job_id を1本実行する。"""
    url = (server_url or _default_server_url()).rstrip("/")
    job_dir, request_data = _resolve_job_dir_and_request(job_id, url if sync_server else None)
    if sync_server:
        _send_heartbeat(url, agent_name, "running", active_job_id=job_id)
    try:
        return _execute_pipeline_job(
            job_id,
            job_dir,
            request_data,
            server_url=url,
            agent_name=agent_name,
            sync_server=sync_server,
        )
    finally:
        if sync_server:
            _send_heartbeat(url, agent_name, "idle", active_job_id=None)


def cmd_watch(
    server_url: str | None = None,
    *,
    agent_name: str = "ciness-agent",
    poll_interval: float = 1.0,
    once: bool = False,
) -> dict[str, Any]:
    """サーバーへハートビートを送りながら queued_for_agent ジョブを常駐待機・自動実行する。"""
    url = (server_url or _default_server_url()).rstrip("/")
    stop_event = threading.Event()
    status_lock = threading.Lock()
    current = {"status": "idle", "active_job_id": None}

    def _handle_signal(_signum: int, _frame: Any) -> None:
        stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handle_signal)
            except (ValueError, OSError):
                pass

    def _heartbeat_loop() -> None:
        while not stop_event.is_set():
            with status_lock:
                st = current["status"]
                jid = current["active_job_id"]
            _send_heartbeat(url, agent_name, st, active_job_id=jid)
            stop_event.wait(2.0)

    # 起動直後に即時ハートビートを送って UI に接続中を反映させる
    _send_heartbeat(url, agent_name, "idle", active_job_id=None)
    hb_thread = threading.Thread(target=_heartbeat_loop, name="ciness-heartbeat", daemon=True)
    hb_thread.start()

    print(
        f"[ciness-agent] 待機を開始しました: server={url} agent={agent_name} pid={os.getpid()}",
        file=sys.stderr,
        flush=True,
    )

    last_result: dict[str, Any] = {"status": "stopped", "processed_jobs": 0}
    processed = 0
    consecutive_failures = 0

    try:
        while not stop_event.is_set():
            if SHUTDOWN_FLAG_PATH.is_file():
                print(
                    "[ciness-agent] アプリ停止要求を検知したため待機ランナーを終了します。",
                    file=sys.stderr,
                    flush=True,
                )
                last_result = {"status": "shutdown", "processed_jobs": processed}
                break

            try:
                resp = _http_json("GET", f"{url}/api/agent/next", timeout=5.0)
                consecutive_failures = 0
                if isinstance(resp, dict) and resp.get("shutdown"):
                    print(
                        "[ciness-agent] サーバーから停止シグナルを受信したため待機ランナーを終了します。",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_result = {"status": "shutdown", "processed_jobs": processed}
                    break
                job = _extract_job_payload(resp)
            except Exception:
                job = None
                consecutive_failures += 1
                if consecutive_failures >= 6:
                    print(
                        "[ciness-agent] サーバー停止（連続接続失敗）を検知したため待機ランナーを終了します。",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_result = {"status": "server_gone", "processed_jobs": processed}
                    break

            if job is not None:
                job_id = str(job["job_id"])
                print(f"[ciness-agent] キューからジョブを検知: {job_id}", file=sys.stderr, flush=True)
                with status_lock:
                    current["status"] = "running"
                    current["active_job_id"] = job_id
                _send_heartbeat(url, agent_name, "running", active_job_id=job_id)

                try:
                    job_dir, req_data = _resolve_job_dir_and_request(job_id, url, job)
                    result = _execute_pipeline_job(
                        job_id,
                        job_dir,
                        req_data,
                        server_url=url,
                        agent_name=agent_name,
                        sync_server=True,
                    )
                    processed += 1
                    last_result = {"status": "done", "job_id": job_id, "processed_jobs": processed, "result": result}
                except AgentError as exc:
                    processed += 1
                    last_result = {"status": "error", "job_id": job_id, "processed_jobs": processed, "error": str(exc)}
                finally:
                    with status_lock:
                        current["status"] = "idle"
                        current["active_job_id"] = None
                    _send_heartbeat(url, agent_name, "idle", active_job_id=None)

                if once:
                    break
                continue

            stop_event.wait(max(0.25, float(poll_interval)))
    finally:
        stop_event.set()
        _send_heartbeat(url, agent_name, "offline", active_job_id=None)

    return last_result


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CineSS エージェント連携 CLI（環境診断・サーバー起動・ジョブ待機/実行）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="環境診断・サーバー稼働・モデル一覧をJSON出力")
    p_status.add_argument("--server", default=_default_server_url(), help="CineSS サーバーURL")

    p_start = sub.add_parser("start-server", help="FastAPI サーバーを起動してブラウザを開く")
    p_start.add_argument("--host", default="127.0.0.1")
    p_start.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_start.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    p_start.add_argument("--wait-sec", type=float, default=30.0, help="起動待機タイムアウト秒")

    p_watch = sub.add_parser("watch", help="Web UIの開始ボタン押下を待機し、受注・自動実行する常駐ランナー")
    p_watch.add_argument("--server", default=_default_server_url(), help="CineSS サーバーURL")
    p_watch.add_argument("--agent-name", default="ciness-agent", help="Web UI に表示するエージェント名")
    p_watch.add_argument("--poll-interval", type=float, default=1.0, help="ポーリング間隔（秒）")
    p_watch.add_argument("--once", action="store_true", help="1ジョブ処理したら終了する")

    p_run = sub.add_parser("run-job", help="特定のジョブIDを1本実行する")
    p_run.add_argument("--job-id", required=True, help="実行対象の job_id")
    p_run.add_argument("--server", default=_default_server_url(), help="CineSS サーバーURL")
    p_run.add_argument("--agent-name", default="ciness-agent", help="進捗ログに記録するエージェント名")
    p_run.add_argument("--no-server-sync", action="store_true", help="サーバーAPIへの進捗送信を行わない")

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    # 処理中の標準出力混入を防ぎ、最終結果JSONのみをstdoutへ出力する
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        if args.command == "status":
            result = cmd_status(server_url=args.server)
        elif args.command == "start-server":
            result = cmd_start_server(
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
                wait_sec=args.wait_sec,
            )
        elif args.command == "watch":
            result = cmd_watch(
                server_url=args.server,
                agent_name=args.agent_name,
                poll_interval=args.poll_interval,
                once=args.once,
            )
        else:
            result = cmd_run_job(
                args.job_id,
                server_url=args.server,
                agent_name=args.agent_name,
                sync_server=not args.no_server_sync,
            )
    except AgentError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=real_stdout)
        return 2
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=real_stdout)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), file=real_stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
