# agpt-2b (Llama) port of the upstream GRPO+LoRA XPU smoke

Patches + scripts to run the alphabet_sort GRPO+LoRA smoke with the SFT
checkpoint-900 (AuroraGPT-2B, LlamaForCausalLM) instead of Qwen3-0.6B.

Apply the 3 patches to the songhappy/torchtitan@rl fork in ~/rl-repro/torchtitan-fork:
  01 -- llama3 agpt-2b flavor (exact ckpt-900 params: theta=50000, scaling=none, ffn=11008)
  02 -- parallelize_llama skip_dp (real fork gap; generator init needs it)
  03 -- _llama3_rl_model_registry + rl_grpo_lora_agpt_2b config

Then: stage_agpt2b.sh (symlink weights + inject gemma chat_template), s4_agpt2b.sh (2-tile launcher).

STATUS: port verified correct (weights/adapter/RoPE/model-math all match HF); a
small numerical residual under long greedy decode is unresolved. See
../grpo-lora-agpt2b-repro.md for the full diagnosis + next steps.
