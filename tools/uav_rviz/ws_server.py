"""asyncio 静态托管 + 原生 RFC6455 WebSocket 服务端（零第三方依赖）。

- ``GET /``、``/assets/*``：托管 ``web/dist`` 离线静态产物（防目录穿越，JS MIME 显式声明）；
- ``GET /ws``：WebSocket 升级。文本帧 = JSON 控制 / 状态；二进制帧 = 紧凑遥测 / RGB 帧（见 ``protocol``）；
- ``GET /api/status``：JSON 状态（脚本 / 无头校验用）；
- 慢客户端不会拖累采集：每个客户端仅保留"最新一帧"槽位，发送协程按自己的速度取最新帧（天然丢帧合并）。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlsplit

LOG = logging.getLogger("uav_rviz.server")

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA

MAX_INCOMING_MESSAGE = 1 << 20  # 控制帧上限 1 MiB
MAX_HEADER_BYTES = 16 * 1024

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".map": "application/json",
    ".txt": "text/plain; charset=utf-8",
}

_STATUS_TEXT = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
                426: "Upgrade Required", 503: "Service Unavailable"}


# ---------------------------------------------------------------------------
# RFC6455 编解码
# ---------------------------------------------------------------------------


def compute_accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key.strip() + WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(opcode: int, payload: bytes = b"", *, fin: bool = True, mask_key: Optional[bytes] = None) -> bytes:
    """编码一个 WebSocket 帧。服务端发送不加掩码；``mask_key`` 仅供测试模拟客户端帧。"""
    first = (0x80 if fin else 0) | (opcode & 0x0F)
    length = len(payload)
    mask_bit = 0x80 if mask_key else 0
    if length < 126:
        header = struct.pack("!BB", first, mask_bit | length)
    elif length < (1 << 16):
        header = struct.pack("!BBH", first, mask_bit | 126, length)
    else:
        header = struct.pack("!BBQ", first, mask_bit | 127, length)
    if mask_key:
        masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return header + mask_key + masked
    return header + payload


def _unmask(data: bytes, key: bytes) -> bytes:
    if not data:
        return data
    # 大载荷用整数异或加速
    n = len(data)
    mask = (key * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "little") ^ int.from_bytes(mask, "little")).to_bytes(n, "little")


class ProtocolError(Exception):
    pass


async def read_frame(reader: asyncio.StreamReader, *, require_mask: bool = True) -> tuple[bool, int, bytes]:
    """读取一个帧，返回 ``(fin, opcode, payload)``。客户端帧必须带掩码（RFC6455 §5.1）。"""
    b1, b2 = await reader.readexactly(2)
    fin = bool(b1 & 0x80)
    if b1 & 0x70:
        raise ProtocolError("RSV 位必须为 0")
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if require_mask and not masked:
        raise ProtocolError("客户端帧必须掩码")
    if length == 126:
        (length,) = struct.unpack("!H", await reader.readexactly(2))
    elif length == 127:
        (length,) = struct.unpack("!Q", await reader.readexactly(8))
    if opcode >= 0x8 and (length > 125 or not fin):
        raise ProtocolError("控制帧非法")
    if length > MAX_INCOMING_MESSAGE:
        raise ProtocolError("消息过大")
    key = await reader.readexactly(4) if masked else b""
    payload = await reader.readexactly(length) if length else b""
    if masked:
        payload = _unmask(payload, key)
    return fin, opcode, payload


# ---------------------------------------------------------------------------
# HTTP 解析
# ---------------------------------------------------------------------------


@dataclass
class HttpRequest:
    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)


async def read_http_request(reader: asyncio.StreamReader) -> Optional[HttpRequest]:
    try:
        raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ConnectionError):
        return None
    if len(raw) > MAX_HEADER_BYTES:
        return None
    lines = raw.decode("latin-1").split("\r\n")
    try:
        method, target, _version = lines[0].split(" ", 2)
    except ValueError:
        return None
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
    return HttpRequest(method=method.upper(), path=target, headers=headers)


def http_response(status: int, body: bytes = b"", content_type: str = "text/plain; charset=utf-8",
                  extra: Optional[dict[str, str]] = None) -> bytes:
    headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
        "Connection": "close",
        "Cache-Control": "no-cache",
        "X-Content-Type-Options": "nosniff",
    }
    headers.update(extra or {})
    head = f"HTTP/1.1 {status} {_STATUS_TEXT.get(status, 'OK')}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    return head.encode("latin-1") + b"\r\n" + body


def resolve_static_path(root: Path, url_path: str) -> Optional[Path]:
    """URL 路径 → ``root`` 下的文件；任何越界（``..``、绝对路径、盘符）一律返回 None。"""
    path = unquote(urlsplit(url_path).path)
    if "\x00" in path or "\\" in path:
        return None
    rel = path.lstrip("/") or "index.html"
    root_resolved = root.resolve()
    candidate = (root_resolved / rel).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    if candidate.is_dir():
        candidate = candidate / "index.html"
    return candidate if candidate.is_file() else None


PLACEHOLDER_HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>UAV RViz</title>
<body style="background:#0b0f14;color:#c9d4e0;font:14px/1.7 'Microsoft YaHei UI',sans-serif;padding:48px">
<h2>UAV RViz 前端尚未构建</h2>
<p>请先构建离线静态产物：<code>cd tools/uav_rviz/web &amp;&amp; npm install &amp;&amp; npm run build</code></p>
<p>WebSocket 与 <a style="color:#5eead4" href="/api/status">/api/status</a> 已可用。</p></body></html>"""


# ---------------------------------------------------------------------------
# 服务器
# ---------------------------------------------------------------------------


class _Client:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self.slots: dict[str, bytes] = {}
        self.ctrl: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self.wake = asyncio.Event()
        self.sent_frames = 0
        self.dropped_frames = 0


class RvizServer:
    """HTTP + WebSocket 服务器。``publish_*`` 方法线程安全，可由采集线程直接调用。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        static_dir: Optional[Path] = None,
        command_handler: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        status_provider: Optional[Callable[[], dict[str, Any]]] = None,
        on_clients_changed: Optional[Callable[[int], None]] = None,
        initial_messages: Optional[Callable[[], list[dict[str, Any]]]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.static_dir = static_dir
        self.command_handler = command_handler
        self.status_provider = status_provider
        self.on_clients_changed = on_clients_changed
        self.initial_messages = initial_messages
        self._clients: set[_Client] = set()
        self._server: Optional[asyncio.AbstractServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def start(self) -> int:
        self._loop = asyncio.get_running_loop()
        self._server = await asyncio.start_server(self._handle, self.host, self.port, limit=MAX_HEADER_BYTES * 4)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def stop(self) -> None:
        for client in list(self._clients):
            client.writer.close()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    # -- 线程安全发布 -----------------------------------------------------
    def publish_binary(self, data: bytes, kind: str = "telemetry") -> None:
        if self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._put_slot, kind, data)

    def publish_json(self, message: dict[str, Any]) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        text = json.dumps(message, separators=(",", ":"), allow_nan=False)
        self._loop.call_soon_threadsafe(self._put_ctrl, text)

    def _put_slot(self, kind: str, data: bytes) -> None:
        for client in self._clients:
            if kind in client.slots:
                client.dropped_frames += 1  # 上一帧还没发出去，被新帧覆盖
            client.slots[kind] = data
            client.wake.set()

    def _put_ctrl(self, text: str) -> None:
        for client in self._clients:
            self._queue_ctrl(client, text)

    @staticmethod
    def _queue_ctrl(client: _Client, text: str) -> None:
        try:
            client.ctrl.put_nowait(text)
        except asyncio.QueueFull:
            pass
        client.wake.set()

    # -- 连接处理 -----------------------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            req = await read_http_request(reader)
            if req is None:
                return
            route = urlsplit(req.path).path
            if route == "/ws":
                await self._handle_ws(req, reader, writer)
            else:
                await self._handle_http(req, writer, route)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception:
            LOG.exception("连接处理异常")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_http(self, req: HttpRequest, writer: asyncio.StreamWriter, route: str) -> None:
        if req.method not in ("GET", "HEAD"):
            writer.write(http_response(405))
            await writer.drain()
            return
        if route == "/api/status":
            body = json.dumps(self._status(), ensure_ascii=False).encode("utf-8")
            writer.write(http_response(200, body, "application/json; charset=utf-8"))
        elif route == "/healthz":
            writer.write(http_response(200, b"ok"))
        else:
            target = resolve_static_path(self.static_dir, req.path) if self.static_dir and self.static_dir.exists() else None
            if target is not None:
                ctype = _MIME.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                body = target.read_bytes()
                writer.write(http_response(200, b"" if req.method == "HEAD" else body, ctype,
                                           {"Content-Length": str(len(body))}))
            elif self.static_dir is None or not self.static_dir.exists():
                writer.write(http_response(503, PLACEHOLDER_HTML.encode("utf-8"), "text/html; charset=utf-8"))
            else:
                writer.write(http_response(404, b"Not Found"))
        await writer.drain()

    def _status(self) -> dict[str, Any]:
        status = dict(self.status_provider()) if self.status_provider else {}
        status["web_clients"] = self.client_count
        return status

    def _origin_allowed(self, origin: Optional[str]) -> bool:
        """仅允许同源（loopback 主机名 + 本服务端口）或无 Origin（脚本客户端）。"""
        if not origin:
            return True
        netloc = urlsplit(origin).netloc.lower()
        return netloc in {f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}",
                          f"{self.host}:{self.port}"}

    async def _handle_ws(self, req: HttpRequest, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        h = req.headers
        if req.method != "GET" or "websocket" not in h.get("upgrade", "").lower() \
                or "upgrade" not in h.get("connection", "").lower() or not h.get("sec-websocket-key"):
            writer.write(http_response(426, b"Upgrade Required", extra={"Upgrade": "websocket"}))
            await writer.drain()
            return
        if h.get("sec-websocket-version") != "13":
            writer.write(http_response(400, b"Unsupported WebSocket version", extra={"Sec-WebSocket-Version": "13"}))
            await writer.drain()
            return
        if not self._origin_allowed(h.get("origin")):
            writer.write(http_response(403, b"Forbidden origin"))
            await writer.drain()
            return
        accept = compute_accept_key(h["sec-websocket-key"])
        writer.write(
            ("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
             f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode("ascii")
        )
        await writer.drain()

        client = _Client(writer)
        self._clients.add(client)
        if self.on_clients_changed:
            self.on_clients_changed(len(self._clients))
        self._queue_ctrl(client, json.dumps({**self._status(), "type": "hello", "protocol": 1},
                                            separators=(",", ":"), allow_nan=False))
        for message in (self.initial_messages() if self.initial_messages else []):
            self._queue_ctrl(client, json.dumps(message, separators=(",", ":"), allow_nan=False))
        sender = asyncio.create_task(self._sender(client))
        try:
            await self._receiver(client, reader)
        finally:
            sender.cancel()
            self._clients.discard(client)
            if self.on_clients_changed:
                self.on_clients_changed(len(self._clients))

    async def _sender(self, client: _Client) -> None:
        writer = client.writer
        try:
            while True:
                await client.wake.wait()
                client.wake.clear()
                while not client.ctrl.empty():
                    writer.write(encode_frame(OP_TEXT, client.ctrl.get_nowait().encode("utf-8")))
                slots, client.slots = client.slots, {}
                for data in slots.values():
                    writer.write(encode_frame(OP_BINARY, data))
                    client.sent_frames += 1
                await writer.drain()
        except (asyncio.CancelledError, ConnectionError):
            pass

    async def _receiver(self, client: _Client, reader: asyncio.StreamReader) -> None:
        buffer = bytearray()
        buffer_op = OP_TEXT
        try:
            while True:
                fin, opcode, payload = await read_frame(reader)
                if opcode == OP_CLOSE:
                    client.writer.write(encode_frame(OP_CLOSE, payload[:2]))
                    await client.writer.drain()
                    return
                if opcode == OP_PING:
                    client.writer.write(encode_frame(OP_PONG, payload))
                    continue
                if opcode == OP_PONG:
                    continue
                if opcode in (OP_TEXT, OP_BINARY):
                    buffer, buffer_op = bytearray(payload), opcode
                elif opcode == OP_CONT:
                    buffer.extend(payload)
                    if len(buffer) > MAX_INCOMING_MESSAGE:
                        raise ProtocolError("消息过大")
                else:
                    raise ProtocolError(f"未知 opcode {opcode}")
                if fin and buffer_op == OP_TEXT:
                    self._on_text(client, bytes(buffer))
                    buffer = bytearray()
        except (asyncio.IncompleteReadError, ConnectionError):
            return
        except ProtocolError as exc:
            LOG.warning("WebSocket 协议错误: %s", exc)
            try:
                client.writer.write(encode_frame(OP_CLOSE, struct.pack("!H", 1002)))
                await client.writer.drain()
            except ConnectionError:
                pass

    def _on_text(self, client: _Client, raw: bytes) -> None:
        try:
            message = json.loads(raw.decode("utf-8"))
            if not isinstance(message, dict) or "cmd" not in message:
                raise ValueError("控制消息必须是含 cmd 的 JSON 对象")
            reply = self.command_handler(message) if self.command_handler else {"ok": False, "error": "无命令处理器"}
        except Exception as exc:
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            message = {}
        ack = {"type": "ack", "cmd": message.get("cmd"), "id": message.get("id"), **reply}
        self._queue_ctrl(client, json.dumps(ack, separators=(",", ":"), allow_nan=False, default=str))


# ---------------------------------------------------------------------------
# 控制指令路由
# ---------------------------------------------------------------------------


class CommandRouter:
    """把前端 JSON 控制指令映射到采集器（全部是只读的显示 / 采集配置，不涉及任何仿真写操作）。"""

    COLOR_MODES = ("distance", "height")

    def __init__(self, tap: Any, web_port_getter: Callable[[], int],
                 broadcast: Optional[Callable[[dict[str, Any]], None]] = None) -> None:
        self.tap = tap
        self.web_port_getter = web_port_getter
        self.broadcast = broadcast
        self.color_mode = "distance"

    def __call__(self, message: dict[str, Any]) -> dict[str, Any]:
        cmd = message.get("cmd")
        if cmd == "set_port":
            port = message.get("port")
            if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                return {"ok": False, "error": "端口必须是 1-65535 的整数"}
            if port == self.web_port_getter():
                return {"ok": False, "error": "不能把 AirSim 端口设为本监控台自身的 Web 端口"}
            self.tap.set_port(port)
        elif cmd == "set_mode":
            from .airsim_tap import PROFILES

            if message.get("mode") not in PROFILES:
                return {"ok": False, "error": f"mode 必须是 {sorted(PROFILES)}"}
            self.tap.set_mode(message["mode"])
        elif cmd == "set_rgb":
            enabled = message.get("enabled")
            if not isinstance(enabled, bool):
                return {"ok": False, "error": "enabled 必须为布尔值"}
            self.tap.set_rgb(enabled)
        elif cmd == "set_color_mode":
            if message.get("value") not in self.COLOR_MODES:
                return {"ok": False, "error": f"value 必须是 {list(self.COLOR_MODES)}"}
            self.color_mode = message["value"]
            if self.broadcast:
                self.broadcast({"type": "event", "event": "color_mode", "value": self.color_mode})
        elif cmd == "clear_trail":
            self.tap.clear_trail()
        elif cmd == "refresh_scene":
            self.tap.refresh_scene()
        elif cmd == "get_status":
            pass
        else:
            return {"ok": False, "error": f"未知指令 {cmd!r}"}
        return {"ok": True, "status": self.tap.status(), "color_mode": self.color_mode}

