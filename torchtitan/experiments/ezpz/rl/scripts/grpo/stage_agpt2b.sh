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
# inject the gemma chat template into the COPIED tokenizer_config.json
python3 - <<'PYEOF'
import json
p = "/home/foremans/rl-repro/run/agpt2b-ckpt900/tokenizer_config.json"
d = json.load(open(p))
# Gemma chat template (matches the SFT _CHAT_TEMPLATE_GEMMA format:
# <bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n).
gemma = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + (message['role'] if message['role'] != 'assistant' else 'model') + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
)
d["chat_template"] = gemma
json.dump(d, open(p, "w"), indent=2)
print("injected gemma chat_template into", p)
print("chat_template len:", len(d["chat_template"]))
PYEOF
