#!/bin/bash
# Hand this Claude Code conversation off to mbph.
#
# Same shape as ../su3-diffusion/scripts/handoff_to_mbph.sh: there is no live
# migration, but the transcript is what `claude --resume` reads, so copying it
# (with the embedded paths rewritten) reconstitutes the session on the far
# side.
#
# Three things have to line up or the resume silently fails:
#
#   1. The project directory name is the ABSOLUTE PATH with '/' -> '-'. This
#      machine is /Users/samforeman/..., mbph is /Users/sam/..., so the
#      directory name differs and must be derived from the REMOTE path.
#   2. The `cwd` field inside every record carries the old absolute path and
#      is rewritten to match.
#   3. The repo itself must exist at that path on mbph.
#
# Re-runnable: the transcript is appended to live, so a later run captures
# more of the conversation. Refuses to clobber an existing remote transcript
# unless FORCE=1.
set -euo pipefail

REMOTE=${REMOTE:-mbph}
SESSION=${SESSION:-b09e4b10-ba26-4278-92b7-9cdc17634795}
LOCAL_ROOT=/Users/samforeman/projects/saforem2/torchtitan
SRC="$HOME/.claude/projects/-Users-samforeman-projects-saforem2-torchtitan/${SESSION}.jsonl"

[ -f "$SRC" ] || { echo "no transcript at $SRC" >&2; exit 1; }

echo "== resolving remote layout =="
REMOTE_HOME=$(ssh "$REMOTE" 'echo $HOME')
REMOTE_ROOT="$REMOTE_HOME/projects/saforem2/torchtitan"
SLUG=$(printf '%s' "$REMOTE_ROOT" | sed 's|/|-|g')
REMOTE_DIR="$REMOTE_HOME/.claude/projects/$SLUG"
echo "   remote repo: $REMOTE_ROOT"
echo "   remote slug: $SLUG"

echo "== ensuring the repo exists on $REMOTE =="
ssh "$REMOTE" "set -e
  mkdir -p '$REMOTE_HOME/projects/saforem2'
  if [ -d '$REMOTE_ROOT/.git' ]; then
    cd '$REMOTE_ROOT' && git fetch -q origin
  else
    git clone -q https://github.com/saforem2/torchtitan.git '$REMOTE_ROOT'
  fi
  cd '$REMOTE_ROOT' && git log --oneline -1"

echo "== checking for an existing transcript on $REMOTE =="
if ssh "$REMOTE" "test -f '$REMOTE_DIR/${SESSION}.jsonl'"; then
  if [ "${FORCE:-0}" != "1" ]; then
    echo "   a transcript for this session already exists on $REMOTE." >&2
    echo "   re-run with FORCE=1 to overwrite it with a fresher snapshot." >&2
    exit 2
  fi
  echo "   FORCE=1 -- overwriting, after backing the old one up"
  ssh "$REMOTE" "cp '$REMOTE_DIR/${SESSION}.jsonl' \
                    '$REMOTE_DIR/${SESSION}.jsonl.bak'"
fi

# Rewrite the embedded cwd paths while streaming, so the file is never
# duplicated on this disk. Anchored on the exact local root so unrelated text
# mentioning a similar path is untouched.
echo "== copying transcript (rewriting $LOCAL_ROOT -> $REMOTE_ROOT) =="
LINES=$(wc -l < "$SRC" | tr -d ' ')
echo "   $LINES records, $(du -h "$SRC" | cut -f1)"
ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"
sed "s|$LOCAL_ROOT|$REMOTE_ROOT|g" "$SRC" \
  | gzip -1 \
  | ssh "$REMOTE" "gunzip > '$REMOTE_DIR/${SESSION}.jsonl'"

echo "== verifying =="
ssh "$REMOTE" "set -e
  f='$REMOTE_DIR/${SESSION}.jsonl'
  n=\$(wc -l < \"\$f\" | tr -d ' ')
  echo \"   records: \$n (expected >= $LINES)\"
  # >= not ==: the transcript is appended to LIVE, so it grows while the copy
  # streams -- the handoff conversation is itself being recorded. An exact
  # match reported RECORD COUNT MISMATCH on a perfectly good 222,498-record
  # copy (expected 222,494) and aborted before the memory rsync. Fewer records
  # than expected is still a real truncation.
  [ \"\$n\" -ge '$LINES' ] || { echo '   TRUNCATED: fewer records than source' >&2; exit 1; }
  if grep -q '$LOCAL_ROOT' \"\$f\"; then
    echo '   STALE PATHS REMAIN' >&2; exit 1
  fi
  grep -q '$REMOTE_ROOT' \"\$f\" || { echo '   rewrite produced no hits' >&2; exit 1; }
  tail -1 \"\$f\" | python3 -c 'import json,sys; json.load(sys.stdin)' \
    && echo '   tail record parses as JSON'
  echo '   OK'"

echo "== copying project memory =="
# The memory dir is keyed by the same slug and is what MEMORY.md is loaded
# from at session start. Without it the resumed session loses every
# project_*/feedback_* note.
LOCAL_MEM="$HOME/.claude/projects/-Users-samforeman-projects-saforem2-torchtitan/memory"
if [ -d "$LOCAL_MEM" ]; then
  echo "   $(ls "$LOCAL_MEM" | wc -l | tr -d ' ') files"
  rsync -az --delete "$LOCAL_MEM/" "$REMOTE:$REMOTE_DIR/memory/"
  ssh "$REMOTE" "ls '$REMOTE_DIR/memory' | wc -l | tr -d ' ' | xargs -I{} echo '   remote now has {} files'"
else
  echo "   (no local memory dir -- skipping)"
fi

cat <<EOF

== done ==

On $REMOTE, resume with:

    cd $REMOTE_ROOT && claude --resume $SESSION

Before it can do anything useful on the cluster:

  - ALCF SSH from $REMOTE needs an MFA passcode -- non-interactive ssh is
    refused (Permission denied (keyboard-interactive,hostbased)). Log in
    once by hand so the ControlMaster/agent is warm, or the resumed session
    cannot run qstat.
  - $REMOTE's checkout was behind at the time of writing. \`git pull\` it;
    the grad-norm-guard fix (40de81570) matters for the live 20B chain.

Not transferred:
  - Background Monitors. The old session's watchers on job 7560196 and on
    the prod clone's HEAD do NOT survive; re-arm them if wanted.
  - PBS state. Jobs keep running on ALCF; query with qstat as usual.
  - Anything said after this snapshot. Re-run with FORCE=1 for a fresher one.
EOF
