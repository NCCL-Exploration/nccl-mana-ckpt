#!/bin/bash
# Gather a campaign's results into the repo, as tables for the paper.
#
#   experiments/collect.sh                    # the newest campaign
#   experiments/collect.sh 20261030           # one campaign
#   NAME=final experiments/collect.sh 20261030 20261102
#                                             # merge campaigns (e.g. a rerun of
#                                             # failed or skipped runs) into one
#
# Writes experiments/results/<name>/ (name: the campaign, or NAME):
#   summary.md     every table, Markdown -- start here
#   *.csv          aggregated tables (overhead, micro_sizes, ckpt, torchsave,
#                  vs_torchsave, p2p, failures)
#   tex/*.tex      the main tables as LaTeX tabulars (\input them)
#   raw/           the campaign's per-run results.csv, ckpts.csv, torchsave.csv,
#                  campaign.txt, skipped.txt and sysinfo/
# Only small files are copied; logs and run directories stay on $SCRATCH (the
# failures table names them). Safe to run while jobs are still going: it
# summarizes what has finished so far. Runs on a login node, no modules.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

if [ $# -eq 0 ]; then
    newest=$(ls -1t "$RESULTS_ROOT" 2>/dev/null | grep -v "^bin$\|^build$" | head -n 1 || true)
    [ -n "$newest" ] || { echo "no campaigns in $RESULTS_ROOT" >&2; exit 1; }
    set -- "$newest"
fi
name=${NAME:-$1}
dest=$EXP_DIR/results/$name
raw=$dest/raw
rm -rf "$dest"
mkdir -p "$raw"

# merge <file> <campaign...>: that file of every campaign, one header.
merge() {
    local f=$1 c first=1
    shift
    for c in "$@"; do
        local p=$RESULTS_ROOT/$c/$f
        [ -s "$p" ] || continue
        if [ $first = 1 ]; then cat "$p"; first=0; else tail -n +2 "$p"; fi
    done
}
for f in results.csv ckpts.csv torchsave.csv; do
    merge "$f" "$@" >"$raw/$f"
    [ -s "$raw/$f" ] || rm -f "$raw/$f"
done
for c in "$@"; do
    d=$RESULTS_ROOT/$c
    [ -d "$d" ] || { echo "no campaign $d" >&2; exit 1; }
    sfx=
    [ $# -gt 1 ] && sfx=.$c
    [ -f "$d/campaign.txt" ] && cat "$d/campaign.txt" >>"$raw/campaign.txt"
    [ -f "$d/skipped.txt" ] && cp "$d/skipped.txt" "$raw/skipped$sfx.txt"
    [ -d "$d/sysinfo" ] && mkdir -p "$raw/sysinfo$sfx" && cp -r "$d/sysinfo/." "$raw/sysinfo$sfx/"
done
cat "$raw"/skipped*.txt >"$raw/skipped.txt.all" 2>/dev/null || true
[ -s "$raw/skipped.txt.all" ] && mv "$raw/skipped.txt.all" "$raw/skipped.txt" || rm -f "$raw/skipped.txt.all"

CAMPAIGN_NAME="$name ($*)" PT_WIDTH=$PT_WIDTH python3 "$LIB_DIR/summarize.py" "$raw" "$dest" >/dev/null
echo "campaign(s) $* -> $dest"
echo "  $(tail -n +2 "$raw/results.csv" 2>/dev/null | wc -l) runs," \
    "$(awk -F, 'NR > 1 && $8 == "ok"' "$raw/results.csv" 2>/dev/null | wc -l) ok;" \
    "summary: $dest/summary.md"
if command -v squeue >/dev/null; then
    q=$(squeue -h -u "$USER" -n nccl-ckpt-paper -o '%i %T %M/%l %D' 2>/dev/null)
    [ -n "$q" ] && echo "  still queued/running (jobid state elapsed/limit nodes):" && echo "$q" | sed 's/^/    /'
fi
exit 0
