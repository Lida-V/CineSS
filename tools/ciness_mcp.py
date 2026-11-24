# coding: utf-8
"""CineSS stdio MCP サーバー（JSON-RPC 2.0 / 1行1メッセージ）。登録名: ciness。

リポジトリ直下の `.mcp.json` から `.venv` の Python で起動する。
提供ツール:
- ciness_status: 環境診断・サーバー稼働状態・モデル一覧の確認
- ciness_start_server: FastAPI サーバーの起動確認＆ブラウザ起動
- ciness_wait_job: ハートビートを送りながら Web UI「▶ 処理開始」ボタンの押下を待機
- ciness_claim_and_run_job: キューにあるジョブ（または指定ID）を受注してパイプラインを実行
- ciness_job_status: ジョブの進捗・ログ・成果物パスを取得
- ciness_cancel_job: 実行中/待機中ジョブのキャンセル
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
APP_ROOT = TOOLS_DIR.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from app.config import DEFAULT_PORT, JOBS_DIR, OUTPUT_DIR, ensure_directories  # noqa: E402
from ciness_agent import (  # noqa: E402
    AgentError,
    _default_server_url,
    _extract_job_payload,
    _http_json,
    _send_heartbeat,
    cmd_run_job,
    cmd_start_server,
    cmd_status,
)

PROTOCOLS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
AGENT_CLI = TOOLS_DIR / "ciness_agent.py"
MCP_PROCS_DIR = OUTPUT_DIR / "_mcp_procs"


def _schema(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


TOOLS: list[dict[str, Any]] = [
    {
        "name": "ciness_status",
        "description": "CineSS のPC動作環境診断（CPU/GPU/RAM/FFmpeg/SoundSeparator共有）、Webサーバー稼働状態、エージェント接続状態、AIモデル導入状況を返す。ダウンロードや推論は行わない。",
        "inputSchema": _schema(
            {
                "server_url": {"type": "string", "description": "省略時は http://127.0.0.1:8780"},
            }
        ),
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "ciness_start_server",
        "description": "CineSS の FastAPI Webアプリサーバーを起動して応答待機し、URLを返す。既に起動済みならそのままURLを返す。",
        "inputSchema": _schema(
            {
                "host": {"type": "string", "default": "127.0.0.1"},
                "port": {"type": "integer", "default": DEFAULT_PORT, "minimum": 1, "maximum": 65535},
                "open_browser": {"type": "boolean", "default": True, "description": "起動後に既定ブラウザでWeb UIを開く"},
                "wait_sec": {"type": "number", "default": 30.0, "minimum": 3.0, "maximum": 120.0},
            }
        ),
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    },
    {
        "name": "ciness_wait_job",
        "description": "サーバーへエージェント接続ハートビートを送りながら、Web UI 上でユーザーが「▶ 処理開始」ボタンを押して queued_for_agent ジョブが投入されるのを待つ。",
        "inputSchema": _schema(
            {
                "server_url": {"type": "string", "description": "省略時は http://127.0.0.1:8780"},
                "timeout_sec": {"type": "number", "default": 60.0, "minimum": 1.0, "maximum": 600.0},
                "poll_interval": {"type": "number", "default": 1.0, "minimum": 0.25, "maximum": 10.0},
                "agent_name": {"type": "string", "default": "ciness-mcp"},
            }
        ),
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "ciness_claim_and_run_job",
        "description": "指定 job_id（省略時はキュー先頭の queued_for_agent ジョブ）を受注して CineSS パイプラインを実行する。background=true（既定）なら別プロセスで開始し即座に job_id を返すため、ciness_job_status で完了まで監視できる。",
        "inputSchema": _schema(
            {
                "job_id": {"type": "string", "description": "省略時は /api/agent/next の先頭ジョブを受注する"},
                "server_url": {"type": "string", "description": "省略時は http://127.0.0.1:8780"},
                "agent_name": {"type": "string", "default": "ciness-mcp"},
                "background": {"type": "boolean", "default": True, "description": "true で非同期サブプロセス起動、false で完了まで同期実行"},
            }
        ),
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    },
    {
        "name": "ciness_job_status",
        "description": "指定ジョブ（または job_id 省略時は全ジョブ一覧）の状態・進捗・ログ末尾・出力結果を読む。",
        "inputSchema": _schema(
            {
                "job_id": {"type": "string", "description": "省略時はジョブ一覧を返す"},
                "server_url": {"type": "string", "description": "省略時は http://127.0.0.1:8780"},
            }
        ),
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "ciness_cancel_job",
        "description": "指定ジョブをキャンセルする。MCPから起動した実行プロセスが稼働中であれば安全に終了させる。",
        "inputSchema": _schema(
            {
                "job_id": {"type": "string"},
                "server_url": {"type": "string", "description": "省略時は http://127.0.0.1:8780"},
            },
            ["job_id"],
        ),
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    },
]


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
            check=False,
        )
        return f'"{int(pid)}"' in proc.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not job_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in job_id):
        raise AgentError("job_id の形式が正しくありません。")
    return job_id


class CineSSMcpServer:
    def status(self, server_url: str | None = None) -> dict[str, Any]:
        return cmd_status(server_url=server_url)

    def start_server(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        open_browser: bool = True,
        wait_sec: float = 30.0,
    ) -> dict[str, Any]:
        return cmd_start_server(host=host, port=port, open_browser=open_browser, wait_sec=wait_sec)

    def wait_job(
        self,
        server_url: str | None = None,
        timeout_sec: float = 60.0,
        poll_interval: float = 1.0,
        agent_name: str = "ciness-mcp",
    ) -> dict[str, Any]:
        url = (server_url or _default_server_url()).rstrip("/")
        deadline = time.time() + max(1.0, float(timeout_sec))
        last_hb = 0.0
        while True:
            now = time.time()
            if now - last_hb >= 2.0:
                _send_heartbeat(url, agent_name, "idle", active_job_id=None)
                last_hb = now
            try:
                resp = _http_json("GET", f"{url}/api/agent/next", timeout=4.0)
                job = _extract_job_payload(resp)
                if job is not None:
                    return {"found": True, "job": job}
            except Exception as exc:
                if time.time() >= deadline:
                    raise AgentError(f"サーバー {url} に接続できませんでした: {exc}") from exc
            if time.time() >= deadline:
                return {"found": False, "job": None, "message": f"{timeout_sec:.0f}秒以内に新規ジョブは投入されませんでした。"}
            time.sleep(max(0.25, float(poll_interval)))

    def claim_and_run_job(
        self,
        job_id: str | None = None,
        server_url: str | None = None,
        agent_name: str = "ciness-mcp",
        background: bool = True,
    ) -> dict[str, Any]:
        url = (server_url or _default_server_url()).rstrip("/")
        target_id = job_id
        if not target_id:
            resp = _http_json("GET", f"{url}/api/agent/next", timeout=5.0)
            job = _extract_job_payload(resp)
            if job is None:
                return {"started": False, "message": "待機中（queued_for_agent）のジョブはありません。"}
            target_id = str(job["job_id"])

        target_id = _validate_job_id(target_id)
        if not background:
            result = cmd_run_job(target_id, server_url=url, agent_name=agent_name, sync_server=True)
            return {"started": True, "job_id": target_id, "status": "done", "result": result}

        ensure_directories()
        MCP_PROCS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = MCP_PROCS_DIR / f"{target_id}.log"
        pid_path = MCP_PROCS_DIR / f"{target_id}.pid"

        # 先にサーバーへ claim して二重受注を防ぐ
        try:
            _http_json(
                "POST",
                f"{url}/api/agent/jobs/{target_id}/claim",
                {"agent_name": agent_name, "pid": os.getpid()},
                timeout=5.0,
            )
        except Exception:
            pass

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        cmd = [
            sys.executable,
            "-X",
            "utf8",
            str(AGENT_CLI),
            "run-job",
            "--job-id",
            target_id,
            "--server",
            url,
            "--agent-name",
            agent_name,
        ]
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

        log_fp = log_path.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(APP_ROOT),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        finally:
            log_fp.close()

        pid_path.write_text(str(proc.pid), encoding="ascii")
        return {
            "started": True,
            "job_id": target_id,
            "status": "agent_running",
            "pid": proc.pid,
            "log_path": str(log_path),
            "message": f"ジョブ {target_id} をバックグラウンドで開始しました。ciness_job_status で進捗を確認できます。",
        }

    def job_status(self, job_id: str | None = None, server_url: str | None = None) -> dict[str, Any]:
        url = (server_url or _default_server_url()).rstrip("/")
        if job_id:
            target_id = _validate_job_id(job_id)
            data: dict[str, Any] | None = None
            try:
                data = _http_json("GET", f"{url}/api/jobs/{target_id}", timeout=5.0)
            except Exception:
                state_file = JOBS_DIR / target_id / "state.json"
                if state_file.is_file():
                    data = json.loads(state_file.read_text(encoding="utf-8"))
            if data is None:
                raise AgentError(f"ジョブ {target_id} が見つかりません。")

            log_path = MCP_PROCS_DIR / f"{target_id}.log"
            if log_path.is_file():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
                data["agent_process_log_tail"] = lines
            return data

        try:
            return _http_json("GET", f"{url}/api/jobs", timeout=5.0)
        except Exception:
            jobs = []
            if JOBS_DIR.is_dir():
                for st_file in sorted(JOBS_DIR.glob("*/state.json")):
                    try:
                        jobs.append(json.loads(st_file.read_text(encoding="utf-8")))
                    except (OSError, ValueError):
                        continue
            return {"jobs": jobs}

    def cancel_job(self, job_id: str, server_url: str | None = None) -> dict[str, Any]:
        target_id = _validate_job_id(job_id)
        url = (server_url or _default_server_url()).rstrip("/")
        server_result: dict[str, Any] | None = None
        try:
            server_result = _http_json("POST", f"{url}/api/jobs/{target_id}/cancel", {}, timeout=5.0)
        except Exception as exc:
            server_result = {"warning": f"サーバーキャンセル通知に失敗: {exc}"}

        killed = False
        pid_path = MCP_PROCS_DIR / f"{target_id}.pid"
        if pid_path.is_file():
            try:
                pid = int(pid_path.read_text(encoding="ascii").strip())
                if _pid_alive(pid):
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False)
                    else:
                        os.kill(pid, 15)
                    killed = True
            except (OSError, ValueError):
                pass

        return {
            "job_id": target_id,
            "cancelled": True,
            "process_terminated": killed,
            "server_response": server_result,
        }


class ProtocolError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def dispatch(request: Any, server: CineSSMcpServer) -> dict[str, Any] | None:
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
        raise ProtocolError(-32600, "Invalid JSON-RPC request")
    method = request["method"]
    params = request.get("params", {})
    if not isinstance(params, dict):
        raise ProtocolError(-32602, "params must be an object")
    if method.startswith("notifications/"):
        return None
    if method == "initialize":
        requested = params.get("protocolVersion")
        return {
            "protocolVersion": requested if requested in PROTOCOLS else "2025-03-26",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "ciness", "version": "1.0.0"},
            "instructions": (
                "ローカル映像サラウンドMixMaster＆バイノーラル出力アプリ CineSS。"
                "まず ciness_status で環境確認し、ciness_start_server でWeb UIを起動する。"
                "ユーザーがWeb UIで「▶ 処理開始」を押したら ciness_wait_job → ciness_claim_and_run_job → ciness_job_status で完了まで伴走する。"
                "推論時の暗黙モデルダウンロードは行わない。"
            ),
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method != "tools/call":
        raise ProtocolError(-32601, "Method not found")

    functions = {
        "ciness_status": server.status,
        "ciness_start_server": server.start_server,
        "ciness_wait_job": server.wait_job,
        "ciness_claim_and_run_job": server.claim_and_run_job,
        "ciness_job_status": server.job_status,
        "ciness_cancel_job": server.cancel_job,
    }
    name = params.get("name")
    args = params.get("arguments", {})
    if name not in functions:
        raise ProtocolError(-32602, f"Unknown tool: {name}")
    if not isinstance(args, dict):
        raise ProtocolError(-32602, "arguments must be an object")

    try:
        value = functions[name](**args)
        return {
            "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)}],
            "isError": False,
        }
    except (AgentError, OSError, ValueError, KeyError, TypeError) as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}


def serve(server: CineSSMcpServer) -> None:
    for line in sys.stdin:
        request_id: Any = None
        has_id = True
        try:
            if len(line) > 1024 * 1024:
                raise ProtocolError(-32600, "Request exceeds 1 MiB")
            try:
                request = json.loads(line)
            except ValueError as exc:
                raise ProtocolError(-32700, "Invalid JSON") from exc
            if isinstance(request, dict):
                request_id = request.get("id")
                has_id = "id" in request
            result = dispatch(request, server)
            if not has_id:
                continue
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        except ProtocolError as exc:
            if not has_id:
                continue
            response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": exc.code, "message": exc.message}}
        except Exception as exc:
            if not has_id:
                continue
            response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
        print(json.dumps(response, ensure_ascii=False, allow_nan=False, default=str), flush=True)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    serve(CineSSMcpServer())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
