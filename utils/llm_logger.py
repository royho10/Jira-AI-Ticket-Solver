import os
from datetime import datetime
from threading import Lock

_log_file = None
_log_lock = Lock()
_enabled = True
_run_total_duration_ms = 0
_run_total_input_tokens = 0
_run_total_output_tokens = 0
_run_start_time = None


def disable():
    """Stop writing prompts and responses to disk, for the rest of the process.

    The Remote MCP Server calls this at start-up: its prompts carry raw,
    unsanitized ticket text belonging to whichever user made the request, and a
    shared server must not persist that. Local runs (Streamlit, the indexer)
    leave logging on -- there the operator owns the data already.
    """
    global _enabled, _log_file
    with _log_lock:
        _enabled = False
        if _log_file is not None:
            _log_file.close()
            _log_file = None


def is_enabled():
    """Whether LLM call logging is writing to disk."""
    return _enabled


def _estimate_tokens(text):
    """Rough token estimate: ~4 chars per token for English text."""
    if text is None:
        return 0
    return max(1, len(str(text)) // 4)


def _estimate_input_tokens(input_messages):
    """Estimate total input tokens from messages."""
    total = 0
    if isinstance(input_messages, list):
        for msg in input_messages:
            if hasattr(msg, "content"):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                total += _estimate_tokens(content)
            elif isinstance(msg, tuple) and len(msg) == 2:
                total += _estimate_tokens(msg[1])
            else:
                total += _estimate_tokens(msg)
    else:
        total = _estimate_tokens(input_messages)
    return total


def _ensure_log_file():
    global _log_file, _run_start_time
    if _log_file is not None:
        return
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "llm_calls")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(log_dir, f"llm_calls_{timestamp}.log")
    _log_file = open(path, "a", encoding="utf-8")
    _run_start_time = datetime.now()


def log_llm_call(call_name, model, input_messages, output, duration_ms=None):
    """Log an LLM call's input and output to the run-specific log file.

    Args:
        call_name: Purpose of the call (e.g. "image_analysis", "reranking").
        model: Model/deployment name used.
        input_messages: List of (role, content) tuples or raw string.
        output: Response text or object (will be str()'d).
        duration_ms: Optional duration in milliseconds.
    """
    global _run_total_duration_ms, _run_total_input_tokens, _run_total_output_tokens

    if not _enabled:
        return

    input_tokens = _estimate_input_tokens(input_messages)
    output_tokens = _estimate_tokens(output)

    with _log_lock:
        _ensure_log_file()

        _run_total_input_tokens += input_tokens
        _run_total_output_tokens += output_tokens
        if duration_ms is not None:
            _run_total_duration_ms += duration_ms

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 80

        _log_file.write(f"\n{sep}\n")
        _log_file.write(f"[{now}] CALL: {call_name} | MODEL: {model}\n")
        _log_file.write(f"{sep}\n\n")

        _log_file.write("--- INPUT ---\n")
        if isinstance(input_messages, list):
            for msg in input_messages:
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    role = msg.type.capitalize()
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    _log_file.write(f"[{role}]: {content}\n\n")
                elif isinstance(msg, tuple) and len(msg) == 2:
                    _log_file.write(f"[{msg[0]}]: {msg[1]}\n\n")
                else:
                    _log_file.write(f"{msg}\n\n")
        else:
            _log_file.write(f"{input_messages}\n\n")

        _log_file.write("--- OUTPUT ---\n")
        _log_file.write(f"{output}\n\n")

        if duration_ms is not None:
            _log_file.write(f"--- Duration: {duration_ms}ms ---\n")
        _log_file.write(f"--- Estimated tokens — input: ~{input_tokens}, output: ~{output_tokens} ---\n")

        _log_file.write(f"{sep}\n")
        _log_file.flush()


def log_run_summary():
    """Write a summary of all LLM calls in this run (total time and tokens)."""
    if not _enabled:
        return
    with _log_lock:
        _ensure_log_file()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "#" * 80

        _log_file.write(f"\n{sep}\n")
        _log_file.write(f"[{now}] RUN SUMMARY\n")
        _log_file.write(f"{sep}\n\n")
        _log_file.write(f"Total LLM call duration: {_run_total_duration_ms}ms "
                        f"({_run_total_duration_ms / 1000:.1f}s)\n")
        _log_file.write(f"Total estimated input tokens:  ~{_run_total_input_tokens}\n")
        _log_file.write(f"Total estimated output tokens: ~{_run_total_output_tokens}\n")
        if _run_start_time:
            wall_time = (datetime.now() - _run_start_time).total_seconds()
            _log_file.write(f"Wall time since first log call: {wall_time:.1f}s\n")
        _log_file.write(f"{sep}\n")
        _log_file.flush()


def reset_run_stats():
    """Reset all run-level accumulators. Call before a new test session."""
    global _run_total_duration_ms, _run_total_input_tokens, _run_total_output_tokens
    global _run_start_time, _log_file
    with _log_lock:
        _run_total_duration_ms = 0
        _run_total_input_tokens = 0
        _run_total_output_tokens = 0
        _run_start_time = None
        if _log_file is not None:
            _log_file.close()
            _log_file = None


def get_run_stats() -> dict:
    """Return current run stats as a dict."""
    with _log_lock:
        wall_time = None
        if _run_start_time:
            wall_time = (datetime.now() - _run_start_time).total_seconds()
        return {
            "total_duration_ms": _run_total_duration_ms,
            "estimated_input_tokens": _run_total_input_tokens,
            "estimated_output_tokens": _run_total_output_tokens,
            "estimated_total_tokens": _run_total_input_tokens + _run_total_output_tokens,
            "wall_time_seconds": wall_time,
        }
