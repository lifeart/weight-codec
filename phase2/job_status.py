"""Print status + the last N log lines of an HF Job (non-interactive; the log stream of a running
job never ends, so it is read in a daemon thread for at most --secs seconds)."""
import os, sys, threading, time
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import HfApi
api = HfApi()
jid = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 20
j = api.inspect_job(job_id=jid)
print("status:", j.status.stage, getattr(j.status, "message", None) or "")
lines, err = [], []
def rd():
    try:
        for line in api.fetch_job_logs(job_id=jid):
            lines.append(line)
    except Exception as e:  # report, do not hide
        err.append(f"{type(e).__name__}: {str(e)[:200]}")
t = threading.Thread(target=rd, daemon=True); t.start(); t.join(secs)
if err:
    print("log error:", err[0])
print(f"[{len(lines)} log lines read in <= {secs:.0f}s]")
for l in lines[-n:]:
    print(l[:300])
sys.stdout.flush(); os._exit(0)
