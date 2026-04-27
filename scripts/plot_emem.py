"""Parse sglang server log and plot elastic memory timeline."""

import re
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

LOG_FILE = (
    sys.argv[1] if len(sys.argv) > 1 else "/tmp/sglang_server_20260427_052336.log"
)
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/emem_timeline.png"

# ── Parse ──────────────────────────────────────────────────────────────
ts_re = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

decode_re = re.compile(
    r"Decode batch, #running-req: (\d+), #full token: (\d+), "
    r"full token usage: ([\d.]+), #swa token: (\d+), swa token usage: ([\d.]+), "
    r"cuda graph: \w+, gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
)
prefill_re = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
    r"full token usage: ([\d.]+), swa token usage: ([\d.]+), "
    r"#running-req: (\d+), #queue-req: (\d+), #pending-token: (\d+), "
    r"cuda graph: \w+, input throughput \(token/s\): ([\d.]+)"
)
mark_set_re = re.compile(
    r"mark_unmap_candidate: set candidate, self\.size=(\d+), self\.candidate_size=(\d+)"
)
mark_reset_re = re.compile(r"mark_unmap_candidate: is_candidate=False")
do_resize_re = re.compile(r"ElasticMempoolOrchestrator do_resize")
post_free_re = re.compile(r"post_free: candidate_unmap_pages (\d+)->(\d+)/(\d+)")

# Storage
decode_ts, decode_throughput, decode_running_req = [], [], []
decode_full_usage, decode_swa_usage = [], []
prefill_ts, prefill_input_tps = [], []

events = []  # (ts, type, info_dict)
post_free_ts, post_free_cur, post_free_target = [], [], []


def parse_ts(line):
    m = ts_re.search(line)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return None


with open(LOG_FILE) as f:
    for line in f:
        if "TP0]" not in line:
            continue
        ts = parse_ts(line)
        if ts is None:
            continue

        m = decode_re.search(line)
        if m:
            decode_ts.append(ts)
            decode_running_req.append(int(m.group(1)))
            decode_full_usage.append(float(m.group(3)))
            decode_swa_usage.append(float(m.group(5)))
            decode_throughput.append(float(m.group(6)))
            continue

        m = prefill_re.search(line)
        if m:
            prefill_ts.append(ts)
            prefill_input_tps.append(float(m.group(9)))
            continue

        m = mark_set_re.search(line)
        if m:
            events.append(
                (
                    ts,
                    "mark_candidate",
                    {
                        "size": int(m.group(1)),
                        "candidate_size": int(m.group(2)),
                    },
                )
            )
            continue

        if mark_reset_re.search(line):
            events.append((ts, "reset_candidate", {}))
            continue

        if do_resize_re.search(line):
            events.append((ts, "do_resize", {}))
            continue

        m = post_free_re.search(line)
        if m:
            post_free_ts.append(ts)
            post_free_cur.append(int(m.group(2)))
            post_free_target.append(int(m.group(3)))
            continue

print(
    f"Parsed: {len(decode_ts)} decode batches, {len(prefill_ts)} prefill batches, "
    f"{len(events)} elastic events, {len(post_free_ts)} post_free updates"
)

if not decode_ts:
    print("No decode batch data found, exiting.")
    sys.exit(1)

# ── Plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)
fig.suptitle("ElasticMem Timeline (TP0)", fontsize=16, fontweight="bold")

colors = {
    "mark_candidate": "#e74c3c",
    "reset_candidate": "#3498db",
    "do_resize": "#2ecc71",
}
labels_added = set()


def add_event_lines(ax, events_list):
    for ts, etype, info in events_list:
        label = None
        if etype not in labels_added:
            label = etype
            labels_added.add(etype)
        c = colors.get(etype, "gray")
        ls = "--" if etype == "reset_candidate" else "-"
        lw = 2.5 if etype == "do_resize" else 1.5
        ax.axvline(ts, color=c, linestyle=ls, linewidth=lw, alpha=0.8, label=label)


# ── Panel 1: Gen Throughput (token/s) ──────────────────────────────────
ax1 = axes[0]
ax1.plot(
    decode_ts,
    decode_throughput,
    color="#2980b9",
    linewidth=1.2,
    marker=".",
    markersize=3,
)
ax1.set_ylabel("Gen Throughput\n(token/s)", fontsize=11)
ax1.grid(True, alpha=0.3)
add_event_lines(ax1, events)
ax1.legend(loc="upper right", fontsize=9)

# ── Panel 2: Running Requests (batch size) ─────────────────────────────
ax2 = axes[1]
ax2.plot(
    decode_ts,
    decode_running_req,
    color="#8e44ad",
    linewidth=1.2,
    marker=".",
    markersize=3,
)
ax2.set_ylabel("#Running Req\n(batch size)", fontsize=11)
ax2.grid(True, alpha=0.3)
add_event_lines(ax2, events)

# ── Panel 3: Token Usage (full vs swa) ─────────────────────────────────
ax3 = axes[2]
ax3.plot(
    decode_ts, decode_full_usage, color="#e74c3c", linewidth=1.2, label="full_usage"
)
ax3.plot(decode_ts, decode_swa_usage, color="#27ae60", linewidth=1.2, label="swa_usage")
ax3.axhline(0.7, color="#e74c3c", linestyle=":", alpha=0.5, label="CAN_MAP (0.7)")
ax3.axhline(0.5, color="#27ae60", linestyle=":", alpha=0.5, label="CAN_UNMAP (0.5)")
ax3.set_ylabel("Token Usage", fontsize=11)
ax3.set_ylim(-0.05, 1.05)
ax3.grid(True, alpha=0.3)
add_event_lines(ax3, events)
ax3.legend(loc="upper right", fontsize=9, ncol=2)

# ── Panel 4: Candidate Unmap Pages Progress ────────────────────────────
ax4 = axes[3]
if post_free_ts:
    ax4.plot(
        post_free_ts,
        post_free_cur,
        color="#e67e22",
        linewidth=1.2,
        marker=".",
        markersize=3,
        label="candidate_unmap_pages",
    )
    ax4.plot(
        post_free_ts,
        post_free_target,
        color="#e74c3c",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        label="target_unmap",
    )
    ax4.fill_between(post_free_ts, post_free_cur, alpha=0.2, color="#e67e22")
    ax4.legend(loc="upper left", fontsize=9)
ax4.set_ylabel("Candidate\nUnmap Pages", fontsize=11)
ax4.set_xlabel("Time", fontsize=11)
ax4.grid(True, alpha=0.3)
add_event_lines(ax4, events)

# Format x-axis
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
fig.autofmt_xdate(rotation=30)

# ── Annotate events ────────────────────────────────────────────────────
for ts, etype, info in events:
    if etype == "mark_candidate":
        for ax in axes:
            ax.annotate(
                f"mark_candidate\nsize={info['size']}→{info['candidate_size']}",
                xy=(ts, 0.95),
                xycoords=("data", "axes fraction"),
                fontsize=7,
                color="#e74c3c",
                fontweight="bold",
                ha="left",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffeaea", alpha=0.8),
            )
            break  # only annotate on first panel
    elif etype == "do_resize":
        for ax in axes:
            ax.annotate(
                "do_resize!",
                xy=(ts, 0.95),
                xycoords=("data", "axes fraction"),
                fontsize=8,
                color="#2ecc71",
                fontweight="bold",
                ha="left",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#eafff0", alpha=0.8),
            )
            break

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_FILE}")
