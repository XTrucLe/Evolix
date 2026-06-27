import os, json, re
import queue, threading

PATTERN = re.compile(
    r"step\s+(?P<step>\d+)\s+\| "
    r"loss (?P<loss>[0-9.]+)\s+\| "
    r"lr (?P<lr>[0-9.eE+-]+)"
    r"(?: \| (?P<tok_s>[0-9.]+)k tok/s \| (?P<ms>[0-9.]+)ms)?"
)


def write_log(msg, log_file="src/log/logs.jsonl"):
    if not (m := PATTERN.match(msg)):
        return

    d = m.groupdict()
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        json.dump(
            {
                "step": int(d["step"]),
                "loss": float(d["loss"]),
                "lr": float(d["lr"]),
                "tok/s": float(d["tok_s"]) * 1000 if d["tok_s"] else None,
                "ms": float(d["ms"]) if d["ms"] else None,
            },
            f,
        )
        f.write("\n")


task_queue = queue.Queue()


def _worker():
    while True:
        task = task_queue.get()
        if task is None:
            break

        try:
            if task["type"] == "log":
                write_log(task["data"])
                print(task["data"])
        except Exception as e:
            print(f"[logger] error: {e}")
        finally:
            task_queue.task_done()


threading.Thread(target=_worker, daemon=True).start()
