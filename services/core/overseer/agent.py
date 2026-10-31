"""The overseer's traffic agent: reads a deployment's logged traffic and ships it to the buffer.

One file, standard library only (plus the curl binary on a VM), so it can be uploaded to a sandbox VM
or run by hand inside a private network:

    FORGE_SOURCE_SECRET=... FORGE_INGEST_TOKEN=... python3 agent.py <directory with config.json>

It works out where the records are and which fields hold the prompt and the answer, keeps a cursor,
and posts batches to the sink. The cursor only moves after the sink has accepted a page, and the sink
ignores duplicates, so a crash repeats work but never loses a trace.

Everything it reads is untrusted production text and it holds credentials, so it is deliberately
not a model: no text it reads can change what it does. Its state file holds counters and field
names, never trace text. It writes that file for the lab to read and takes no instructions back.
"""

import base64
import collections
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

BATCH = 200
MAX_BODY = 3_500_000
MAX_READ = 32 * 1024 * 1024
MAX_PAGES = 1000
MAX_TEXT = 32000
MAX_BACKOFF = 900
HEARTBEAT = 30
SAMPLE = 25
COVERAGE = 0.8
SEEN = 20000
RECORD_KEYS = ("data", "records", "items", "results", "traces", "rows", "logs", "generations", "observations")
CANDIDATES = {
    "id": ("id", "trace_id", "traceId", "request_id", "requestId", "log_id", "uuid", "_id"),
    "prompt": (
        "prompt",
        "question",
        "query",
        "input",
        "messages",
        "request.messages",
        "request.prompt",
        "request.input",
        "request_body.messages",
        "requestBody.messages",
        "inputs",
    ),
    "output": (
        "output",
        "completion",
        "answer",
        "response.choices.0.message.content",
        "choices.0.message.content",
        "response_body.choices.0.message.content",
        "responseBody.choices.0.message.content",
        "response.output",
        "response.content",
        "response.text",
        "response",
        "outputs",
    ),
    "finish_reason": (
        "finish_reason",
        "finishReason",
        "stop_reason",
        "choices.0.finish_reason",
        "response.choices.0.finish_reason",
        "response_body.choices.0.finish_reason",
    ),
    "model_version": ("model_version", "modelVersion", "model", "response.model", "request.model"),
    "latency_ms": ("latency_ms", "latencyMs", "duration_ms", "durationMs", "response_time_ms"),
    "timestamp": ("timestamp", "created_at", "createdAt", "startTime", "time", "ts"),
    "context": ("context", "retrieved_context", "retrievedContext"),
    "deployment": ("deployment", "environment"),
    "rating": ("feedback.rating", "rating"),
    "correction": ("feedback.correction", "correction"),
    "prompt_tokens": (
        "tokens.prompt",
        "usage.prompt_tokens",
        "usage.input_tokens",
        "usageDetails.input",
        "response.usage.prompt_tokens",
        "response_body.usage.prompt_tokens",
    ),
    "completion_tokens": (
        "tokens.completion",
        "usage.completion_tokens",
        "usage.output_tokens",
        "usageDetails.output",
        "response.usage.completion_tokens",
        "response_body.usage.completion_tokens",
    ),
    "metadata": ("metadata",),
}
CURSOR_FIELDS = ("created_at", "createdAt", "timestamp", "startTime", "time", "ts", "id")


class Hold(Exception):
    """Stop this poll without moving the cursor. Carries the status to report."""

    def __init__(self, status, detail=""):
        super().__init__(status)
        self.status, self.detail = status, detail


class NoRedirect(urllib.request.HTTPRedirectHandler):
    # urllib forwards Authorization across hosts on a redirect; a credential goes only where it was sent.
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def now():
    return datetime.now(timezone.utc).isoformat()


def dig(record, path):
    """Follow a dotted path through objects and lists, opening JSON held in strings on the way."""
    value = record
    for part in path.split("."):
        if isinstance(value, str) and value[:1] in "[{":
            value = _loads(value)
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def _loads(text):
    try:
        return json.loads(text)
    except ValueError:
        return None


def _is_message_shape(parsed):
    if isinstance(parsed, dict):
        return any(key in parsed for key in ("role", "messages", "choices"))
    if isinstance(parsed, list):
        return any(isinstance(m, dict) and ("role" in m or m.get("type") == "text") for m in parsed)
    return False


def text_of(value, role, depth=0):
    """The text a logged prompt or answer holds, in the shapes model APIs and tracers log them."""
    if depth > 6 or value is None:
        return ""
    if isinstance(value, str):
        # Tracers log messages as JSON text. Any other JSON here is the real text: an extraction
        # task's answer is itself JSON, and must not be unwrapped or lost.
        if value.lstrip()[:1] in ("[", "{"):
            parsed = _loads(value)
            if _is_message_shape(parsed):
                return text_of(parsed, role, depth + 1)
        return value
    if isinstance(value, list):
        spoken = [m for m in value if isinstance(m, dict) and "role" in m]
        if spoken:
            matching = [m for m in spoken if m.get("role") == role]
            return text_of(matching[-1], role, depth + 1) if matching else ""
        parts = [text_of(item, role, depth + 1) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("content", "text", "messages", "choices", "message"):
            if key in value:
                found = text_of(value[key], role, depth + 1)
                if found:
                    return found
        for key in ("prompt", "question", "query", "input", "output", "completion", "answer"):
            if key in value:
                found = text_of(value[key], role, depth + 1)
                if found:
                    return found
    return ""


def find_records(body, path=None):
    """The list of records in a reply: named, the reply itself, or under a usual key."""
    if path:
        found = dig(body, path)
        return [r for r in found if isinstance(r, dict)] if isinstance(found, list) else []
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    if isinstance(body, dict):
        for key in RECORD_KEYS:
            if isinstance(body.get(key), list):
                return [r for r in body[key] if isinstance(r, dict)]
    return []


def field_names(records):
    """Field names only, never values: enough for a person to write a mapping."""
    names = set()
    for record in records[:SAMPLE]:
        for key, value in record.items():
            names.add(str(key)[:64])
            if isinstance(value, dict):
                names.update(f"{key}.{inner}"[:64] for inner in value)
    return sorted(names)[:80]


def discover(records, given=None):
    """A field mapping from a sample, or None when the prompt and answer can't be found reliably."""
    given = dict(given or {})
    sample = records[:SAMPLE]
    needed = max(1, int(len(sample) * COVERAGE))
    mapping = {}
    for field, paths in CANDIDATES.items():
        if field in given:
            mapping[field] = given[field]
            continue
        for path in paths:
            if field in ("prompt", "output"):
                role = "user" if field == "prompt" else "assistant"
                hits = sum(1 for r in sample if text_of(dig(r, path), role).strip())
            else:
                hits = sum(1 for r in sample if dig(r, path) is not None)
            if hits >= (needed if field in ("prompt", "output") else 1):
                mapping[field] = path
                break
    for field, path in given.items():
        mapping.setdefault(field, path)
    return mapping if "prompt" in mapping and "output" in mapping else None


def to_trace(record, mapping, given=None):
    """One trace in the overseer's schema, or None when the record has no prompt or answer."""
    given = given or {}

    def text(field, role):
        value = dig(record, mapping[field])
        found = text_of(value, role).strip()
        # A path a person named is taken at its word, even when it holds a structure.
        if not found and field in given and value is not None:
            return json.dumps(value)
        return found

    prompt, output = text("prompt", "user"), text("output", "assistant")
    if not prompt or not output:
        return None
    raw_id = dig(record, mapping["id"]) if "id" in mapping else None
    if raw_id is None or raw_id == "":
        stamp = dig(record, mapping["timestamp"]) if "timestamp" in mapping else ""
        raw_id = hashlib.sha256(f"{stamp}\n{prompt}\n{output}".encode()).hexdigest()
    trace_id = str(raw_id)
    if len(trace_id) > 200:
        trace_id = hashlib.sha256(trace_id.encode()).hexdigest()
    trace = {"id": trace_id, "prompt": prompt, "output": output}

    def value_of(field):
        return dig(record, mapping[field]) if field in mapping else None

    for field in ("finish_reason", "model_version", "deployment", "timestamp", "context"):
        value = value_of(field)
        if isinstance(value, str) and value:
            trace[field] = value
        elif field == "timestamp" and isinstance(value, (int, float)):
            trace[field] = str(value)
    latency = value_of("latency_ms")
    if isinstance(latency, (int, float)) and not isinstance(latency, bool):
        trace["latency_ms"] = float(latency)
    elif "latency_seconds" in mapping:
        seconds = dig(record, mapping["latency_seconds"])
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            trace["latency_ms"] = float(seconds) * 1000
    tokens = {}
    for name, field in (("prompt", "prompt_tokens"), ("completion", "completion_tokens")):
        value = value_of(field)
        if isinstance(value, int) and not isinstance(value, bool):
            tokens[name] = value
    if tokens:
        trace["tokens"] = tokens
    feedback = {}
    rating = value_of("rating")
    if isinstance(rating, (int, float)) and not isinstance(rating, bool):
        feedback["rating"] = int(rating)
    correction = value_of("correction")
    if isinstance(correction, str) and correction.strip():
        feedback["correction"] = correction
    if feedback:
        trace["feedback"] = feedback
    metadata = value_of("metadata")
    if isinstance(metadata, dict):
        scalars = [
            (str(k)[:64], v)
            for k, v in metadata.items()
            if v is None or isinstance(v, (str, int, float, bool))
        ]
        trace["metadata"] = dict(scalars[:16])
    return trace


def _is_number(value):
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def later(a, b):
    """The later of two cursor values: numerically when both are numbers, else as text."""
    if a is None or b is None:
        return b if a is None else a
    try:
        return a if float(a) >= float(b) else b
    except (TypeError, ValueError):
        return a if str(a) >= str(b) else b


def rewind(cursor, seconds):
    """Read a little behind a time cursor: records can appear late, and duplicates are filtered."""
    if not seconds or not isinstance(cursor, str):
        return cursor
    try:
        moment = datetime.fromisoformat(cursor.replace("Z", "+00:00"))
    except ValueError:
        return cursor
    return (moment - timedelta(seconds=seconds)).isoformat()


def source_request(config, cursor, page):
    url = config["url"]
    cursor = rewind(cursor, config.get("cursor_overlap_seconds"))
    params = {}
    if "{cursor}" in url:
        url = url.replace("{cursor}", urllib.parse.quote(str(cursor if cursor is not None else "")))
    elif config.get("cursor_param") and cursor is not None:
        params[config["cursor_param"]] = cursor
    if page is not None:
        params[config["page_cursor_param"]] = page
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    headers = {"Accept": "application/json", "User-Agent": "forge-overseer-agent"}
    auth = config.get("auth") or {"type": "none"}
    secret = os.environ.get("FORGE_SOURCE_SECRET", "")
    if auth["type"] == "bearer":
        headers["Authorization"] = f"Bearer {secret}"
    elif auth["type"] == "api-key":
        headers[auth["header"]] = auth.get("prefix", "") + secret
    elif auth["type"] == "basic":
        pair = f"{auth.get('username', '')}:{secret}".encode()
        headers["Authorization"] = "Basic " + base64.b64encode(pair).decode()
    return urllib.request.Request(url, headers=headers)


def _curl_quote(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "") + '"'


def _call_with_curl(request):
    # The URL and headers go over a private pipe: an argument would be visible to every process.
    lines = [f"url = {_curl_quote(request.full_url)}", 'header = "Expect:"']
    lines += [f"header = {_curl_quote(f'{key}: {value}')}" for key, value in request.header_items()]
    read_end, write_end = os.pipe()
    try:
        os.write(write_end, "\n".join(lines).encode())
        os.close(write_end)
        command = [
            "curl",
            "--silent",
            "--max-time",
            "60",
            "--max-filesize",
            str(MAX_READ),
            "--write-out",
            "\n%{http_code}",
            "--config",
            f"/dev/fd/{read_end}",
        ]
        if request.data is not None:
            command += ["--data-binary", "@-"]
        try:
            done = subprocess.run(
                command,
                input=request.data,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                pass_fds=(read_end,),
                timeout=70,
            )
        except subprocess.TimeoutExpired as error:
            raise OSError("curl timed out") from error
    finally:
        os.close(read_end)
    if done.returncode:
        raise OSError(f"curl exited with {done.returncode}")
    body, _, status = done.stdout.rpartition(b"\n")
    return int(status), body


def http_call(request):
    """The status and body of a request. Raises OSError when there is no answer at all.

    A sandbox's secrets work through an https:// proxy, which is TLS to the proxy and TLS again inside
    it. urllib can't speak that (the proxy resets it), so curl carries those requests. Neither follows
    a redirect.
    """
    if (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "").startswith("https://"):
        return _call_with_curl(request)
    try:
        with OPENER.open(request, timeout=60) as reply:
            return reply.status, reply.read(MAX_READ + 1)
    except urllib.error.HTTPError as error:
        return error.code, b""


def read_source(config, cursor, page):
    try:
        status, raw = http_call(source_request(config, cursor, page))
    except OSError as error:
        raise Hold("source_error", f"Could not reach the source ({type(error).__name__})") from error
    if status in (401, 403):
        raise Hold("source_auth_failed", f"The source answered {status}")
    if not 200 <= status < 300:
        raise Hold("source_error", f"The source answered {status}")
    if len(raw) > MAX_READ:
        raise Hold("source_error", "One reply was over 32 MB; ask for fewer records per page")
    text = raw.decode("utf-8", "replace")
    parsed = _loads(text)
    if parsed is not None:
        return parsed
    rows = [_loads(line) for line in text.splitlines() if line.strip()]
    if rows and all(isinstance(row, dict) for row in rows):
        return rows
    raise Hold("source_error", "The source's reply is not JSON or JSON lines")


def post_batch(config, batch):
    """Post one batch. Returns the sink's accepted, duplicates and rejected counts."""
    body = json.dumps(
        {"source": config["source"], "task_pack": config["task_pack"], "traces": batch}
    ).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "forge-overseer-agent"}
    token = os.environ.get("FORGE_INGEST_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(config["sink_url"], data=body, headers=headers)
    try:
        status, raw = http_call(request)
    except OSError as error:
        raise Hold("sink_unreachable", f"Could not reach the sink ({type(error).__name__})") from error
    if status == 413 and len(batch) > 1:
        middle = len(batch) // 2
        first, second = post_batch(config, batch[:middle]), post_batch(config, batch[middle:])
        return {key: first[key] + second[key] for key in first}
    if status in (413, 422):
        return {"accepted": 0, "duplicates": 0, "rejected": len(batch)}
    if status == 429:
        raise Hold("capped", "The token's daily ingest cap is reached")
    if status in (401, 403):
        raise Hold("sink_auth_failed", f"The sink answered {status}")
    if not 200 <= status < 300:
        raise Hold("sink_unreachable", f"The sink answered {status}")
    result = _loads(raw.decode("utf-8", "replace"))
    if not isinstance(result, dict):
        raise Hold("sink_unreachable", "The sink's reply is not JSON")
    return {
        "accepted": int(result.get("accepted", 0)),
        "duplicates": int(result.get("duplicates", 0)),
        "rejected": len(result.get("rejected", [])),
    }


def batches(traces):
    current, size = [], 0
    for trace in traces:
        length = len(json.dumps(trace))
        if current and (len(current) >= BATCH or size + length > MAX_BODY):
            yield current
            current, size = [], 0
        current.append(trace)
        size += length
    if current:
        yield current


class Agent:
    def __init__(self, directory):
        self.directory = directory
        self.config_path = os.path.join(directory, "config.json")
        self.state_path = os.path.join(directory, "state.json")
        self.stopping = False
        self.seen = collections.OrderedDict()
        self.config = self._read_config()
        previous = {}
        if os.path.exists(self.state_path):
            with open(self.state_path) as handle:
                previous = _loads(handle.read()) or {}
        self.state = {
            "status": "starting",
            "detail": "",
            "cursor": previous.get("cursor", self.config.get("start_cursor")),
            "mapping": previous.get("mapping"),
            "fields_seen": [],
            "polls": previous.get("polls", 0),
            "fetched": previous.get("fetched", 0),
            "accepted": previous.get("accepted", 0),
            "duplicates": previous.get("duplicates", 0),
            "rejected": previous.get("rejected", 0),
            "skipped": previous.get("skipped", 0),
            "last_success_at": previous.get("last_success_at"),
            "heartbeat_at": now(),
        }

    def _read_config(self):
        self.config_stamp = os.stat(self.config_path).st_mtime_ns
        with open(self.config_path) as handle:
            return json.load(handle)

    def write(self, **changes):
        self.state.update(changes, heartbeat_at=now())
        temporary = f"{self.state_path}.{os.getpid()}"
        with open(temporary, "w") as handle:
            json.dump(self.state, handle)
        os.replace(temporary, self.state_path)

    def stop_requested(self):
        # A missing configuration means the lab removed this connection.
        return (
            self.stopping
            or os.path.exists(os.path.join(self.directory, "stop"))
            or not os.path.exists(self.config_path)
        )

    def wait(self, seconds):
        deadline, beat = time.monotonic() + seconds, time.monotonic()
        while time.monotonic() < deadline and not self.stop_requested():
            time.sleep(min(1, max(0, deadline - time.monotonic())))
            if self._config_changed():
                return  # A new mapping from the lab applies now, not after a long backoff.
            if time.monotonic() - beat >= HEARTBEAT:
                self.write()
                beat = time.monotonic()

    def _config_changed(self):
        try:
            return os.stat(self.config_path).st_mtime_ns != self.config_stamp
        except OSError:
            return False

    def mapping_for(self, records):
        given = self.config.get("mapping") or {}
        if self.state["mapping"] and all(self.state["mapping"].get(k) == v for k, v in given.items()):
            return self.state["mapping"]
        mapping = discover(records, given)
        if mapping is None:
            self.write(fields_seen=field_names(records))
            raise Hold("needs_mapping", "Could not tell which fields hold the prompt and the answer")
        self.write(mapping=mapping, fields_seen=[])
        return mapping

    def ship(self, records, mapping):
        """Send one page. Returns how many of its records were new."""
        given, traces, skipped = self.config.get("mapping") or {}, [], 0
        for record in records:
            trace = to_trace(record, mapping, given)
            if trace is None:
                skipped += 1
            elif trace["id"] not in self.seen:
                traces.append(trace)
        for batch in batches(traces):
            result = post_batch(self.config, batch)
            for trace in batch:
                self.seen[trace["id"]] = True
            while len(self.seen) > SEEN:
                self.seen.popitem(last=False)
            for key, value in result.items():
                self.state[key] += value
        self.state["fetched"] += len(records)
        self.state["skipped"] += skipped
        return len(traces)

    def cursor_field(self, records):
        named = self.config.get("cursor_field")
        if named:
            return named
        if not (self.config.get("cursor_param") or "{cursor}" in self.config["url"]):
            return None
        # A numeric cursor so far means the source filters on a number, so a time field won't do.
        numeric = _is_number(self.state["cursor"])
        for field in ("id",) if numeric else CURSOR_FIELDS:
            values = [dig(r, field) for r in records[:SAMPLE]]
            # Only an increasing id is a cursor: a UUID has no order, and would skip records.
            kinds = (int,) if field == "id" else (str, int, float)
            if values and all(isinstance(v, kinds) and not isinstance(v, bool) for v in values):
                return field
        return None

    def poll(self):
        """Read until nothing new is left. The cursor moves after a page, or a paged walk, is sent."""
        config, cursor, page = self.config, self.state["cursor"], None
        high = cursor
        paged = bool(config.get("page_cursor_param"))
        for _ in range(MAX_PAGES):
            if self.stop_requested():
                return
            body = read_source(config, cursor, page)
            records = find_records(body, config.get("records_path"))
            if not records:
                break
            mapping = self.mapping_for(records)
            fresh = self.ship(records, mapping)
            field = self.cursor_field(records)
            if field:
                for record in records:
                    value = dig(record, field)
                    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                        high = later(high, value)
            if paged:
                page = dig(body, config.get("page_cursor_path") or "next_cursor")
                self.write(status="syncing", detail="")
                if page in (None, ""):
                    break
            else:
                # Cursor filtering returns the oldest unread records, so each page can be committed.
                self.write(status="syncing", detail="", cursor=high)
                if not field or not fresh:
                    break
                cursor = high
        else:
            raise Hold("source_error", "Stopped after 1000 pages without reaching the end")
        self.write(status="syncing", detail="", cursor=high, last_success_at=now())

    def run(self):
        failures = 0
        self.write()
        while not self.stop_requested():
            self.config = self._read_config()
            self.state["polls"] += 1
            pause = self.config.get("poll_seconds", 60)
            try:
                self.poll()
                failures = 0
            except Hold as hold:
                failures += 1
                self.write(status=hold.status, detail=hold.detail)
                pause = min(MAX_BACKOFF, pause * 2 ** min(failures, 8))
            self.wait(pause)
        if os.path.exists(self.config_path):
            self.write(status="stopped", detail="")


def main(argv):
    directory = os.path.abspath(argv[1] if len(argv) > 1 else ".")
    lock = open(os.path.join(directory, "agent.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0  # Another agent already holds this connection, so starting twice is harmless.
    agent = Agent(directory)

    def stop(*_):
        agent.stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    agent.run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
