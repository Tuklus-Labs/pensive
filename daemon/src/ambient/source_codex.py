"""Codex CLI JSONL source adapter for the ambient distiller.

Codex session logs are newline-delimited JSON records under ``~/.codex/sessions``.
This adapter is deliberately pure: it reads one log file and returns the
``transcriptSource`` shape consumed by :func:`ambient.distiller.distill`. It never
calls the distiller and never touches the store.

Malformed/torn JSONL lines are skipped. Offsets are the 1-based JSONL line number
of the record that produced the delta, which is a genuine stable position in the
log file and lets the distiller's exact-reference idempotency path engage.
"""
import json
from pathlib import Path

__all__ = ["codexSource"]

SOURCE_CODEX = "codex"
DISTILL_POLICY = "distill"

_MESSAGE_TYPES = {"message"}
_ASSISTANT_EVENT_TYPES = {"agent_message"}
_USER_EVENT_TYPES = {"user_message"}
_TOOL_CALL_TYPES = {
    "function_call",
    "custom_tool_call",
    "web_search_call",
}
_TOOL_RESULT_TYPES = {
    "function_call_output",
    "custom_tool_call_output",
    "patch_apply_end",
    "web_search_end",
}


def codexSource(logPath):
    """Adapt one Codex CLI session JSONL log to a distiller transcript source."""
    path = Path(logPath)
    sessionId = path.stem
    records = []

    try:
        lines = path.open("rb")
    except OSError:
        raise

    with lines:
        for lineNumber, lineBytes in enumerate(lines, 1):
            try:
                line = lineBytes.decode("utf-8")
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(record, dict):
                continue

            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue

            if record.get("type") == "session_meta":
                candidate = _sessionId(payload)
                if candidate is not None:
                    sessionId = candidate
                continue

            records.append((lineNumber, record, payload))

    pending = []
    responseItemAssistantEvents = []
    eventMessageAssistantEvents = []
    for lineNumber, record, payload in records:
        event = _eventFromRecord(record, payload)
        if event is None:
            continue
        delta = {
            "sessionId": sessionId,
            "offset": lineNumber,
            "events": [event],
        }
        if _isResponseItemAssistantMessage(record, payload, event):
            responseItemAssistantEvents.append(delta)
        elif _isEventMessageAssistant(record, payload, event):
            eventMessageAssistantEvents.append(delta)
        else:
            pending.append(delta)

    assistantEvents = (
        responseItemAssistantEvents
        if responseItemAssistantEvents
        else eventMessageAssistantEvents
    )
    pending.extend(assistantEvents)
    pending.sort(key=lambda delta: delta["offset"])

    for delta in pending:
        delta["sessionId"] = sessionId

    return {
        "sessionId": sessionId,
        "source": SOURCE_CODEX,
        "policy": DISTILL_POLICY,
        "deltas": pending,
    }


def _sessionId(payload):
    for key in ("session_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _eventFromRecord(record, payload):
    recordType = record.get("type")
    payloadType = payload.get("type")

    if recordType == "response_item" and payloadType in _MESSAGE_TYPES:
        return _messageEvent(payload)
    if recordType == "event_msg" and payloadType in _ASSISTANT_EVENT_TYPES:
        return _plainEvent("assistant", payload.get("message"), output=True)
    if recordType == "event_msg" and payloadType in _USER_EVENT_TYPES:
        return _plainEvent("user", payload.get("message"), output=False)
    if recordType == "response_item" and payloadType == "reasoning":
        # 2026-07 scan: Codex reasoning records had encrypted content and empty
        # summaries in 809/809 cases, leaving no plaintext transcript to map.
        # If future Codex writes non-empty summary texts, map each summary text
        # as assistant text and let distiller heuristics decide what to keep.
        return None
    if recordType == "response_item" and payloadType in _TOOL_CALL_TYPES:
        return _toolUseEvent(payload)
    if (
        payloadType in _TOOL_RESULT_TYPES
        or (recordType == "event_msg" and payloadType in _TOOL_RESULT_TYPES)
    ):
        return _toolResultEvent(payload)
    return None


def _isResponseItemAssistantMessage(record, payload, event):
    return (
        record.get("type") == "response_item"
        and payload.get("type") in _MESSAGE_TYPES
        and event.get("role") == "assistant"
    )


def _isEventMessageAssistant(record, payload, event):
    return (
        record.get("type") == "event_msg"
        and payload.get("type") in _ASSISTANT_EVENT_TYPES
        and event.get("role") == "assistant"
    )


def _messageEvent(payload):
    role = _role(payload.get("role"))
    if role is None:
        return None
    blocks = _textBlocks(payload.get("content"), output=(role == "assistant"))
    if not blocks:
        return None
    return {"role": role, "content": blocks}


def _plainEvent(role, message, output):
    if not isinstance(message, str) or not message:
        return None
    blockType = "output_text" if output else "input_text"
    return {
        "role": role,
        "content": [_textBlock({"type": blockType, "text": message}, output=output)],
    }


def _role(value):
    if value == "assistant":
        return "assistant"
    if value in {"user", "developer", "system"}:
        return "user"
    return None


def _textBlocks(content, output):
    if isinstance(content, str):
        text = content
        return [{"type": "text", "text": text}] if text else []
    if not isinstance(content, list):
        return []

    blocks = []
    for block in content:
        if not isinstance(block, dict):
            continue
        mapped = _textBlock(block, output=output)
        if mapped is not None:
            blocks.append(mapped)
    return blocks


def _textBlock(block, output):
    blockType = block.get("type")
    expected = "output_text" if output else "input_text"
    if blockType not in {expected, "text"}:
        return None
    text = block.get("text")
    if not isinstance(text, str) or text == "":
        return None
    return {"type": "text", "text": text}


def _toolUseEvent(payload):
    name = payload.get("name") or payload.get("action")
    if not isinstance(name, str) or not name:
        name = str(payload.get("type") or "codex_tool_call")
    block = {
        "type": "tool_use",
        "name": name,
        "input": _toolInput(payload),
    }
    callId = payload.get("call_id") or payload.get("id")
    if isinstance(callId, str) and callId:
        block["id"] = callId
    return {"role": "assistant", "content": [block]}


def _toolInput(payload):
    if "arguments" in payload:
        return _jsonObject(payload.get("arguments"))
    if "input" in payload:
        value = payload.get("input")
        return _jsonObject(value) if isinstance(value, str) else _dictOrValue(value)
    data = {}
    for key in ("action", "status"):
        value = payload.get(key)
        if value is not None:
            data[key] = value
    return data


def _jsonObject(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or value == "":
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}
    return _dictOrValue(decoded)


def _dictOrValue(value):
    if isinstance(value, dict):
        return value
    return {"value": value}


def _toolResultEvent(payload):
    block = {"type": "tool_result"}
    callId = payload.get("call_id") or payload.get("id")
    if isinstance(callId, str) and callId:
        block["tool_use_id"] = callId
    output = _toolOutput(payload)
    if output is not None:
        block["content"] = output
    return {"role": "assistant", "content": [block]}


def _toolOutput(payload):
    if "output" in payload:
        return payload.get("output")
    data = {}
    for key in ("status", "success", "stdout", "stderr", "action", "query"):
        value = payload.get(key)
        if value is not None:
            data[key] = value
    return data or None
