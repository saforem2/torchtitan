#!/bin/bash
# Stage a dir that has: symlinked weights+config (7.9 GB, don't copy) + a COPY of
# the tokenizer with the SFT gemma chat_template injected. Never mutates the
# original SFT deliverable.
SRC=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf
DST=/home/foremans/rl-repro/run/agpt2b-ckpt900
mkdir -p "$DST"
# symlink the big/immutable model files
for f in model.safetensors config.json; do
    ln -sf "$SRC/$f" "$DST/$f"
done
# copy the tokenizer files (we will edit tokenizer_config.json)
for f in tokenizer.json tokenizer.model special_tokens_map.json tokenizer_config.json; do
    cp -f "$SRC/$f" "$DST/$f"
done
echo "staged files:"; ls -la "$DST"
# Restore the literal training-time template and Gemma generation IDs.  Do not
# reconstruct this here: the old staging template added BOS, used generic trim,
# mapped system roles differently, and omitted assistant-mask markers.
python3 "$(dirname "$0")/../repair_agpt_hf_artifact.py" "$DST"
