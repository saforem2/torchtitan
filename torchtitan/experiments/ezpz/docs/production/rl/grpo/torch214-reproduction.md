# Torch 2.14 Monarch GRPO: AGPT-2B rollout diagnosis

> **Status (2026-09-23):** the Torch 2.14 Monarch/TorchStore/vLLM XPU path is
> operational. Robust-SFT-v3, trained from the broad MDS stage-3 base, passes
> both held-out gates: clean 8/8 and adversarial 8/8, with exact output framing
> and EOT termination. A three-step bounded GRPO canary completed technically,
> but did not establish improvement on its broader mixed FIRST/LAST task and is
> rejected as a policy advance. Robust-SFT-v3 remains the accepted artifact.

## Executive conclusion

The evidence separates the rejected historical artifact from the subsequently
trained stage-3-derived policies:

1. **Historical `checkpoint-900` is rejected.** Its reconstructed HF interface
   did not match SFT training: it added an unused BOS token, applied generic
   whitespace trimming, serialized system roles differently, omitted assistant
   generation markers, and inherited reversed BOS/EOS metadata. Repairing those
   export defects fixed serialization and stopping mismatches, but direct
   native-vLLM generation from that same checkpoint remained semantically
   malformed. The earlier `global_step138650` base was also a weak instruction
   follower under the recovered prompt contract.
2. **The genuine broad MDS stage-3 base is coherent but not instruction-tuned.**
   Direct generation from `stage3-mix/global_step154391` produced coherent base
   completions, making it the controlled starting point for new full-weight SFT
   rather than evidence that `checkpoint-900` was usable.
3. **Targeted SFT from the MDS base succeeds on the clean task.** The SFT-100
   candidate achieved 8/8 exact held-out clean generations with correct AGPT EOT
   termination. It also passed bounded automatic TorchTitan/TorchStore
   synchronization. This supersedes any broad claim that the current AGPT
   candidate is semantically malformed.
4. **The controlled robustness curriculum closed the remaining direct-generation
   gate.** The first targeted candidate scored 0/8 under conflicting distractor
   examples. Robust-SFT-v2 preserved 8/8 clean behavior and improved the
   adversarial suite to 5/8. Robust-SFT-v3 then reached 8/8 clean and 8/8
   adversarial, with all samples bounded and EOT-terminated. It qualifies for a
   small bounded GRPO canary; any resulting policy must still be re-evaluated on
   both suites before being accepted.
5. **The bounded GRPO canary is an integration pass, not a policy improvement.**
   All three optimizer and synchronization steps completed and every recorded
   completion was bounded, but exact post-validation correctness was 3/8 and
   mean validation reward changed from 0.758 to 0.698 on different prompts. The
   GRPO environment also mixed FIRST- and LAST-name tasks although robust-v3 SFT
   trained FIRST-name sorting, while its fuzzy character reward assigned high
   partial credit to semantically wrong outputs. The policy-version-3 checkpoint
   is rejected; no longer run should continue from it.

These conclusions do **not** rely on reward, loss, token accuracy, or throughput
alone. They follow from raw completion inspection across direct AGPT loading, a
known-good Qwen control, the genuine MDS stage-3 base, held-out clean and
adversarial SFT evaluations, and pre/post TorchStore synchronization.

## Artifacts and environment

| Item | Value |
|---|---|
| TorchTitan branch | `sync/upstream-0fd69d0b` |
| AGPT SFT source | `checkpoint-900`, 96 intact DCP shards |
| Reconstructed AGPT model SHA-256 | `001cccc6331bca9c6695cb559e923d54640b845a755fe5191667ea2a6925b58e` |
| Qwen3-0.6B model SHA-256 | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| MDS stage3-mix source SHA-256 | `619b5d2581ac798f8bfe7de0d152b19f3ca7805b8566ab5b1191239e150a9a5f` |
| MDS stage3-mix HF weights SHA-256 | `df4a7289d4eddb57c380d14ceb4c510d7ee6c55b5b79a7ef0f96a3a61e644fd1` |
| MDS source checkpoint | `/lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/optimizer-experiments/stage3-mix/Megatron-DeepSpeed/checkpoints/AuroraGPT-2B-ws3072-ds-stage0-nl12-hs2048-mb1-seq8192-gb6144-sp1-pp1-tp1-bf16-optsophiag-lr2.17e-5-lwf0.05_ntok7770B_tokHF_tmgoogle_gemma-7b_flash/global_step154391/mp_rank_00_model_states.pt` |
| MDS converted HF directory | `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/outputs/evals/agpt-2b-mds-7771T/stage3-mix/step-154391/hf` |
| Python | 3.12.12 |
| PyTorch | `2.14.0+xpu` |
| Triton XPU | 3.8.0 |
| vLLM | `0.29.1rc1.dev481+gdc479a2d8.xpu` |
| torchmonarch | 0.6.0 |
| ezpz | 0.27.3 from `git+https://github.com/saforem2/ezpz` |

The validated upstream integration still includes `aurora_full_sonic`, with
`activation_checkpoint_mode="none"`, on the current TorchTitan stack.

## Symptom: mechanically healthy training, invalid AGPT rollouts

The initial easy and shaped AGPT-2B runs reached step 4 with nonzero gradients
and approximately 1,800 trainer tokens/s. That reproduced the historical
infrastructure throughput, but not a valid learning result.

| Recipe | Rollouts inspected | Hit length limit | Copied demo names |
|---|---:|---:|---:|
| easy | 836 | 100% | 94.4% |
| shaped | 852 | 100% | 94.2% |

The malformed completions repeatedly copied the formatting example
`QuinnRivera/OmarSaito/PiaValdez`, opened multiple blocks without closing them,
and drifted into unrelated prose or code. Some still received high shaped
reward, showing that reward was exploitable and could not establish semantic
quality. Both long jobs were stopped rather than training on that signal.

Representative AGPT completion:

```text
<alphabetical_sorted>
QuinnRivera
OmarSaito
PavilionFalls
SamanthaJenkins
LiamRubin
...
<alphabetical_sorted>
OliviaBenitez
LiamBarreto
...
```

The requested name was `BethMillar`; the output copied the unrelated example
names, invented additional names, opened repeated tags, and reached the token
limit.

## Isolation 1: direct AGPT loading reproduces the failure

Native-vLLM job `12478432` loaded the reconstructed checkpoint-900 HF artifact
directly. This bypassed TorchTitan, Monarch, live state-dict conversion, and
TorchStore. All eight samples reached 128 tokens and reproduced the same
failure modes.

A representative direct completion was:

```text
I'll provide the alphabetical order by name with the given format:

QuinnRivera
OmarSaito
PiaValdez
Marina Marquez
</alphabetical_sorted>

However, if the names already in alphabetical order ...
```

Another generated unrelated code:

~~~~text
Here is the Python solution for the problem:

```python
def sort_names(names):
    return sorted(names, key=lambda x: x.startswith('Quinn', 'Omar', 'Pia'))
```
~~~~

This establishes that the bad output is already present in the reconstructed
HF artifact. It is not introduced by the RL synchronization path.

Native-vLLM job `12478434` then loaded the pre-SFT `global_step138650` weights
with the same tokenizer and prompt serialization. All four samples also reached
128 tokens. Examples included an irrelevant sorting tutorial and invalid code:

```text
def alphabetic_sorted(array):
    return sorted_array(array).sort(lambda x: x.servername)
```

Checkpoint-900 shifted the behavior toward copying the demonstration and using
the requested tags, but it did not corrupt a previously reliable instruction
model; the earlier base was already weak under this interaction contract.

## Isolation 2: Qwen proves the runtime path works

The official `Qwen/Qwen3-0.6B` checkpoint was tested with the same task to
separate an AGPT-specific problem from a platform-wide generation failure.

Direct-vLLM job `12478419` exited 0. Seven of eight outputs were exact; the
remaining answer had the correct sorted names but omitted only its closing tag.
A representative completion was:

```text
<alphabetical_sorted>
BethMillar
</alphabetical_sorted>
```

Job `12478425` then loaded Qwen through TorchTitan, synchronized it into the
vLLM generator, generated rollouts, ran backward and AdamW, saved a checkpoint,
and exited 0. All four recorded rollouts terminated normally and were coherent.
For example:

```text
<alphabetical_sorted>
RobertPeschanski
</alphabetical_sorted>
```

Two samples split a concatenated first/last name across adjacent lines, so task
quality was not perfect, but none copied demonstrations, looped, or produced
character soup. The control proves that the Torch 2.14 XPU generation and
TorchTitan synchronization path can preserve a known-good model.

## Concrete AGPT interface defects

The SFT job selected the Gemma tokenizer because `<start_of_turn>` and
`<end_of_turn>` are single tokens, IDs 106 and 107. Its exact template is in
[`train_grpo.py`](https://github.com/saforem2/torchtitan/blob/sync/upstream-0fd69d0b/torchtitan/experiments/ezpz/rl/train_grpo.py#L566-L600).
The training serialization had these properties:

- no explicit BOS prefix;
- `system` and `user` both serialize as a `user` turn;
- only leading newline characters are removed with `.lstrip("\n")`;
- trailing whitespace is preserved;
- the assistant header is `<start_of_turn>model\n`;
- assistant content, `<end_of_turn>`, and the trailing newline are enclosed by
  Jinja generation markers for `assistant_only_loss`.

The reconstructed staging template differed in every one of those areas: it
added `{{ bos_token }}`, used generic `trim`, emitted a `system` role header,
and omitted generation markers. Its metadata also inherited Llama defaults
that conflict with the Gemma tokenizer:

| Meaning | Correct token ID | Inherited model metadata |
|---|---:|---:|
| pad | 0 | not consistently set |
| EOS | 1 | 2 |
| BOS | 2 | 1 |
| start of turn | 106 | not a generation stop |
| end of turn | 107 | not a generation stop |

The repair is implemented in
[`repair_agpt_hf_artifact.py`](https://github.com/saforem2/torchtitan/blob/sync/upstream-0fd69d0b/torchtitan/experiments/ezpz/rl/scripts/repair_agpt_hf_artifact.py)
and integrated into consolidation and legacy staging by
[`0f98b248e`](https://github.com/saforem2/torchtitan/commit/0f98b248eb8b9686ac7ab831a121c520006259a3).
Fast-tokenizer compatibility was added by
[`002218169`](https://github.com/saforem2/torchtitan/commit/002218169680a3ce0ef0dbd9d234668335e7e2a1).

The repair installs the literal training template and sets:

```json
{
  "bos_token_id": 2,
  "eos_token_id": [1, 107],
  "pad_token_id": 0
}
```

Focused tests pass on Sunspot (`2 passed`). Direct-vLLM job `12478440` verified
the behavioral effect on an immutable repaired copy. One sample emitted token
107 and stopped after 36 tokens instead of running to 128:

```text
Here is the block of names sorted by the name with the first letter in the
alphabet first: QuinnRivera, OmarSaito, PiaValdez, BethMillar.
<end_of_turn>
```

It stopped correctly, but still copied all three demonstration names. The
other seven outputs remained malformed or reached the token limit. Therefore:

- template and token metadata were genuinely wrong;
- the repair fixes serialization and stopping behavior;
- those defects do not explain or repair the checkpoint's semantic weakness.

## Isolation 3: the genuine MDS stage-3 base is coherent

The label “stage-3 / 7.77T” had been used ambiguously. The frequently cited
`ntok7770B/global_step140352` artifact is actually the stage-2 terminus at
7.064T; `ntok7770B` records a training target, not consumed tokens.

The genuine stage-3 endpoints are `global_step154391` at exactly
7,770,753,466,368 tokens. Two checkpoints exist:

- broad `stage3-mix` finisher;
- narrow `nvidia-math1-code2` finisher.

The broad model is the preferred base because its recorded evaluation scores
are materially stronger:

| Metric | broad stage3-mix | narrow math/code |
|---|---:|---:|
| ARC-Easy | 0.7138 | 0.6216 |
| ARC-Challenge, 25-shot | 0.4164 | 0.3703 |
| HellaSwag | 0.5874 | 0.4215 |
| GSM8K, 5-shot | 0.0167 | 0.0303 |

The broad source checkpoint is 3,972,173,129 bytes with SHA-256
`619b5d2581ac798f8bfe7de0d152b19f3ca7805b8566ab5b1191239e150a9a5f`.
Its existing 256,000-vocabulary HF conversion has SHA-256
`df4a7289d4eddb57c380d14ceb4c510d7ee6c55b5b79a7ef0f96a3a61e644fd1`.

The exact source path is:

```text
/lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/optimizer-experiments/stage3-mix/Megatron-DeepSpeed/checkpoints/AuroraGPT-2B-ws3072-ds-stage0-nl12-hs2048-mb1-seq8192-gb6144-sp1-pp1-tp1-bf16-optsophiag-lr2.17e-5-lwf0.05_ntok7770B_tokHF_tmgoogle_gemma-7b_flash/global_step154391/mp_rank_00_model_states.pt
```

The converted HF directory used by direct generation is:

```text
/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/outputs/evals/agpt-2b-mds-7771T/stage3-mix/step-154391/hf
```

Aurora direct-generation job `8851384` loaded that exact converted endpoint and
exited 0. Representative greedy continuations were:

```text
Prompt: The capital of France is
Output: Paris.

Prompt: Water boils at a temperature of
Output: 100 degrees Celsius.

Prompt: 2 + 2 =
Output: 4

Prompt: def sort_list(xs):
Output:
    """
    Sorts a list of integers in ascending order using the bubble sort algorithm.
```

The factual answers subsequently repeated because this is a pretrained base
without instruction tuning, but the text was coherent and the first answers
were correct. Unlike checkpoint-900, it did not immediately copy an unrelated
few-shot demonstration or degrade into character soup. It is therefore the
cleanest available base for corrected instruction SFT.

A separate two-node rollout control then loaded the same hash-verified HF
artifact independently on one PVC tile per node. Sunspot job `12478450` exited
0 and wrote one JSONL corpus per host:

```text
rank 0: x1922c6s3b0n0
rank 1: x1922c6s4b0n0
model:  df4a7289d4eddb57c380d14ceb4c510d7ee6c55b5b79a7ef0f96a3a61e644fd1
```

Both nodes returned the same correct initial factual continuations (`Paris`,
`100 degrees Celsius`, and `4`) and coherent Python. The complete decoded
rollouts are shown below; every row used the model SHA-256 above and ended with
`finish_reason="length"` after 64 generated tokens.

#### Prompt: `The capital of France is`

Rank 0 on `x1922c6s3b0n0`:

```text
 Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is
```

Rank 1 on `x1922c6s4b0n0`:

```text
 Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is
```

#### Prompt: `Water boils at a temperature of`

Rank 0 on `x1922c6s3b0n0`:

```text
 100 degrees Celsius.

The boiling point of water is affected by the atmospheric pressure.

The boiling point of water is affected by the atmospheric pressure.

The boiling point of water is affected by the atmospheric pressure.

The boiling point of water is affected by the atmospheric pressure.

The boiling point of
```

Rank 1 on `x1922c6s4b0n0`:

```text
 100 degrees Celsius.

The boiling point of water is affected by the atmospheric pressure.

The boiling point of water is affected by the atmospheric pressure.

The boiling point of water is affected by the atmospheric pressure.

The boiling point of water is affected by the atmospheric pressure.

The boiling point of
```

#### Prompt: `2 + 2 =`

Rank 0 on `x1922c6s3b0n0`:

```text
 4

The number 2 + 2 = 4 is a fundamental mathematical expression that represents
 the sum of two and two. The expression is often used in various mathematical
 contexts, including algebra and calculus.

### Definition

The expression 2 + 2 = 4 can be interpreted in several ways:
```

Rank 1 on `x1922c6s4b0n0`:

```text
 4

The number 2 + 2 = 4 is a fundamental mathematical expression that represents
 the sum of two and two. The expression is often used in various mathematical
 contexts, including algebra and calculus.

### Definition

The expression 2 + 2 = 4 can be interpreted in several ways:
```

#### Prompt: `def sort_list(xs):`

Rank 0 on `x1922c6s3b0n0`:

```python
def sort_list(xs):
    """
    Sorts a list of integers in ascending order using the bubble sort algorithm.

    Args:
        xs (list): A list of integers to be sorted.

    Returns:
        list: A new list containing the sorted integers.

    Raises:
        TypeError: If the
```

Rank 1 on `x1922c6s4b0n0`:

```python
def sort_list(xs):
    """
    Sorts a list of integers in ascending order using the bubble sort algorithm.

    Args:
        xs (List[int]): The list of integers to be sorted.

    Returns:
        List[int]: A new list containing the sorted integers.

    Raises:
```

The first three deterministic token sequences matched exactly across nodes. The
Python docstring differed slightly (`list` versus `List[int]`) despite
`temperature=0`, which is a useful warning that accelerator execution need not
be bitwise deterministic across hosts. Both outputs remained semantically
equivalent. All generations reached the 64-token cap because this is a
pretrained completion model, not an instruction-tuned chat model. This
establishes multi-node loading and coherent rollout execution; it does not claim
multi-node tensor parallelism or instruction following.

## What is and is not wrong

### Proven

- The old checkpoint-900 export/staging interface was not training-equivalent.
- Its BOS/EOS metadata was inconsistent with its tokenizer.
- Correcting the interface improves stopping but does not restore task quality.
- Malformed checkpoint-900 generation occurs under direct native vLLM, before
  TorchTitan or live synchronization.
- The earlier `global_step138650` model is also a weak instruction follower for
  this prompt.
- Qwen works directly and after the complete TorchTitan synchronization path.
- The genuine broad MDS stage-3 base produces coherent factual and code
  continuations.

### Not proven

- The checkpoint-900 tensors are byte-identical to the now-missing historical
  HF export. They were regenerated from the surviving DCP shards.
- The MDS stage-3 base can perform alphabet-sort without instruction tuning. It
  is a pretrained causal LM, not a chat model.
- A corrected SFT run from MDS stage-3 will necessarily solve the task. It is
  the best-supported next experiment, not a completed result.

Static and runtime checks found no concrete QKV split-order or stacked-MLP
mapping defect in the path exercised here. The direct AGPT failure also occurs
without those TorchTitan live-sync conversions.

## Corrective path

1. Treat the canonical AGPT artifact repair as mandatory for every SFT export.
2. Use the broad MDS `stage3-mix/global_step154391` endpoint with the dedicated
   `2b-mds` flavor and 256,000-token Gemma vocabulary.
3. Run instruction SFT with the literal training template and supervised token
   107 turn termination.
4. Reject a candidate unless direct raw completions are coherent, bounded, and
   task-compliant.
5. Only then load it through TorchTitan, synchronize it into vLLM, compare
   logits/completions, and resume GRPO.
6. Do not infer semantic success from throughput, nonzero gradients, or reward.

The next acceptance gate is explicit: a stage-3-derived SFT candidate must pass
direct alphabet-sort generation before any longer RL or multi-node run.

## Reproducing the results

The commands below reproduce the software environment, artifact repair, and
control experiments used in this diagnosis. They were validated on Sunspot PVC
nodes unless the subsection explicitly says Aurora. Site account names,
filesystem roots, and queue names must be adapted outside ALCF.

### 1. Check out the exact TorchTitan revision

Use the published branch containing this reproduction bundle:

```bash
git clone https://github.com/saforem2/torchtitan.git
cd torchtitan
git fetch origin sync/upstream-0fd69d0b
git checkout --track origin/sync/upstream-0fd69d0b
```

For archival runs, record `git rev-parse HEAD` and pin that SHA in the job log;
the branch includes the post-`f3dd7074090e2a01d7691783bc72031b4260ff0b`
reproduction additions described below.

The repair implementation is in
[`repair_agpt_hf_artifact.py`](https://github.com/saforem2/torchtitan/blob/sync/upstream-0fd69d0b/torchtitan/experiments/ezpz/rl/scripts/repair_agpt_hf_artifact.py).
The focused regression tests are in
[`test_repair_agpt_hf_artifact.py`](https://github.com/saforem2/torchtitan/blob/sync/upstream-0fd69d0b/torchtitan/experiments/ezpz/tests/test_repair_agpt_hf_artifact.py).

### 2. Create the Python 3.12 environment

The validated environment used CPython 3.12.12 and was fully stored on
`/lus/tegu`; compute jobs did not depend on a `/home` Python executable.

```bash
export REPO=$PWD
export VENV=/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214
export SRC=/lus/tegu/projects/datascience/foremans/src

uv venv --clear --python 3.12.12 "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel cmake ninja pybind11 build \
  setuptools_scm 'numpy<2.5'
```

Install the protected accelerator stack first. Resolve all later packages with
constraints or `--no-deps`; otherwise dependency resolution may replace this
Torch/Triton pair with CUDA or incompatible XPU builds.

```bash
uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/xpu \
  'torch==2.14.0+xpu' 'triton-xpu==3.8.0'
uv pip install --python "$VENV/bin/python" --no-deps \
  --index-url https://download.pytorch.org/whl/nightly/xpu \
  'torchvision==0.30.0.dev20260921+xpu'

uv pip install --python "$VENV/bin/python" --no-deps 'torchmonarch==0.6.0'
uvi 'git+https://github.com/saforem2/ezpz@6b0c729ab61f0a8e6ab26d7d83d63d3039fd4feb'
```

`uvi` above is the required ezpz installation route. If `uvi` does not target
`$VENV`, activate the venv before invoking it and verify the import path below.

Install TorchStore from its validated revision:

```bash
git clone https://github.com/songhappy/torchstore.git "$SRC/torchstore"
git -C "$SRC/torchstore" checkout 03f588e53ff3bb69a6b42c542c6e03662b333c0f
uv pip install --python "$VENV/bin/python" --no-deps \
  --no-build-isolation "$SRC/torchstore"
```

### 3. Build vLLM and the XPU kernels

The validated vLLM source revision is
`dc479a2d839d5c474b36ac521dc5a72294bcb186`. The validated kernel revision is
`0bfb37287b335fccf94f005141774af99e3eef33`. Both were built against the
already-installed Torch 2.14 XPU wheel. Install the full remaining dependency
lock in the next subsection before importing vLLM; these two source installs
use `--no-deps` deliberately to protect Torch and Triton.

```bash
source /soft/compilers/oneapi/2026.1.0/oneapi/setvars.sh --force
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14

git clone https://github.com/vllm-project/vllm.git "$SRC/vllm-torch214"
git -C "$SRC/vllm-torch214" checkout dc479a2d839d5c474b36ac521dc5a72294bcb186
VLLM_TARGET_DEVICE=xpu uv pip install --python "$VENV/bin/python" \
  --no-deps --no-build-isolation "$SRC/vllm-torch214"

git clone https://github.com/vllm-project/vllm-xpu-kernels.git \
  "$SRC/vllm-xpu-kernels-torch214"
git -C "$SRC/vllm-xpu-kernels-torch214" checkout \
  0bfb37287b335fccf94f005141774af99e3eef33
git -C "$SRC/vllm-xpu-kernels-torch214" apply \
  "$REPO/torchtitan/experiments/ezpz/docs/production/rl/grpo/vllm-xpu-kernels-torch214.patch"
VLLM_TARGET_DEVICE=xpu uv pip install --python "$VENV/bin/python" \
  --no-deps --no-build-isolation "$SRC/vllm-xpu-kernels-torch214"
```

The kernel patch makes two build fixes used for the PVC build:

1. exposes the repository root so sources that include `csrc/...` resolve;
2. forwards `MHC_KERNELS_ENABLED` through `setup.py`.

Install the remaining Python packages without allowing them to replace Torch,
Triton, vLLM, or the native kernels. The complete captured environment is
[`torch214-environment.freeze.txt`](https://github.com/saforem2/torchtitan/blob/sync/upstream-0fd69d0b/torchtitan/experiments/ezpz/docs/production/rl/grpo/torch214-environment.freeze.txt),
whose SHA-256 is
`efc547a8bcde254f2e7a3db44fe729e3ea12eabf831c74a6b4742d0712607ddb`.
The key resolved versions were:

```text
torch==2.14.0+xpu
torchvision==0.30.0.dev20260921+xpu
triton-xpu==3.8.0
torchmonarch==0.6.0
vllm==0.29.1rc1.dev481+gdc479a2d8.xpu
vllm-xpu-kernels==0.1.15.dev30+g0bfb372.d20260922
torchstore==0.0.0.dev0
ezpz==0.27.3
renderers==0.1.11
transformers==5.17.0
trl==1.13.0
datasets==5.0.1
accelerate==1.15.0
safetensors==0.8.0
pytest==9.1.1
```

Do not blindly install the freeze file on a different host: it records local
editable source paths for vLLM, its kernels, and TorchStore. Build those three
at the revisions above, then install the remaining captured packages with the
filtered lock file while constraining the protected stack:

```bash
printf '%s\n' \
  'torch==2.14.0+xpu' \
  'torchvision==0.30.0.dev20260921+xpu' \
  'triton-xpu==3.8.0' \
  'torchmonarch==0.6.0' \
  'vllm==0.29.1rc1.dev481+gdc479a2d8.xpu' \
  'vllm-xpu-kernels==0.1.15.dev30+g0bfb372.d20260922' \
  > /tmp/torch214-protected.txt

uv pip install --python "$VENV/bin/python" \
  --constraint /tmp/torch214-protected.txt \
  -r torchtitan/experiments/ezpz/docs/production/rl/grpo/torch214-python-requirements.txt
```

The filtered requirements file is
[`torch214-python-requirements.txt`](https://github.com/saforem2/torchtitan/blob/sync/upstream-0fd69d0b/torchtitan/experiments/ezpz/docs/production/rl/grpo/torch214-python-requirements.txt).
It excludes the eight protected/source-built packages and can therefore be
used after those packages are installed. Re-run the environment verification
after dependency installation and fail if any protected version changed.

### 4. Verify the environment on a compute node

The compute wrapper must load oneAPI and use the venv's Python directly:

```bash
source /soft/compilers/oneapi/2026.1.0/oneapi/setvars.sh --force
export PATH="$VENV/bin:$PATH"
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ONEAPI_DEVICE_SELECTOR='level_zero:gpu'
export ZE_AFFINITY_MASK=0

"$VENV/bin/python" - <<'PY'
import torch, triton, vllm, vllm_xpu_kernels, monarch, torchstore, ezpz
print('torch', torch.__version__)
print('xpu', torch.xpu.is_available(), torch.xpu.device_count())
print('triton', triton.__version__)
print('vllm', vllm.__version__)
print('ezpz', ezpz.__version__)
PY
```

Expected versions are those listed above, with `torch.xpu.is_available()` true.

### 5. Prepare and validate the repaired AGPT SFT artifact

Never mutate the historical model in place. Copy or stage its metadata and run
the canonical repair against that directory:

```bash
export AGPT_REPAIRED=/path/to/immutable-copy-of-checkpoint-900-hf
"$VENV/bin/python" \
  torchtitan/experiments/ezpz/rl/scripts/repair_agpt_hf_artifact.py \
  "$AGPT_REPAIRED"

"$VENV/bin/python" -m pytest -q \
  torchtitan/experiments/ezpz/tests/test_repair_agpt_hf_artifact.py
```

Expected results are the message `repaired and validated` and `2 passed`.
Then reproduce the repaired checkpoint-900 semantic check:

```bash
export AGPT_OUT=/path/to/agpt-direct-control-fixed.jsonl
"$VENV/bin/python" \
  torchtitan/experiments/ezpz/rl/scripts/agpt_direct_control.py \
  "$AGPT_REPAIRED" "$AGPT_OUT"
```

The captured result was one EOT-terminated sample out of eight; the others
remained malformed or reached 128 tokens. Inspect the JSONL rather than treating
the successful process exit as model success.

If starting from an FSDP SFT checkpoint, use
[`consolidate_sft_ckpt.sh`](https://github.com/saforem2/torchtitan/blob/sync/upstream-0fd69d0b/torchtitan/experiments/ezpz/rl/scripts/consolidate_sft_ckpt.sh);
it merges the shards and invokes the same repair automatically.

### 6. Reproduce the Qwen control

Stage official `Qwen/Qwen3-0.6B` and verify
`model.safetensors` SHA-256
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
Run the checked-in direct harness on one PVC tile:

```bash
export QWEN=/path/to/Qwen3-0.6B-hf-control
export OUT=/path/to/qwen-direct-control.jsonl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$VENV/bin/python" \
  torchtitan/experiments/ezpz/rl/scripts/qwen_direct_control.py \
  "$QWEN" "$OUT"
```

This harness explicitly requests vLLM `TRITON_ATTN`; the Xe2 CUTLASS path is
not available on PVC. The expected semantic result is coherent, bounded XML:
seven of eight captured outputs were exact and the eighth omitted only its
closing tag.

The one-step TorchTitan/Monarch control is launched from a PBS compute node as:

```bash
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ENABLE_V1_MULTIPROCESSING=1 WANDB_MODE=disabled
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

set +e
ezpz launch --auto-retry --nproc 1 --nproc_per_node 1 --timeout 1800 -- \
  "$VENV/bin/python" -u -m torchtitan.experiments.ezpz.rl.train_upstream \
  --module torchtitan.rl.examples.alphabet_sort.config_registry \
  --config rl_grpo_qwen3_0_6b_flex \
  --hf_assets_path="$QWEN" \
  --async-loop.num-training-steps=1 \
  --async-loop.num-prompts-per-train-step=2 \
  --async-loop.num-samples-per-prompt=4 \
  --async-loop.target-offpolicy-steps=0 \
  --async-loop.validation.num-samples=0 \
  --async-loop.training-sample-builder.no-drop-zero-std-reward-groups \
  --generator.sampling.max-tokens=128 \
  --generator.parallelism.tensor-parallel-degree=1 \
  --trainer.parallelism.tensor-parallel-degree=1 \
  --trainer.checkpointer.interval=1 \
  --metrics.no-enable-wandb \
  --dump_folder=/path/to/qwen-control-output
rc=$?
set -e
exit "$rc"
```

The expected signature is coherent rollout text, nonzero loss/gradient,
successful backward and AdamW, a saved checkpoint, and process exit 0. Always
propagate the inner return code from a PBS wrapper with `exit "$rc"`.

### 7. Reproduce direct MDS stage-3 generation on Aurora

Use the exact source checkpoint—not the mislabeled stage-2 endpoint:

```bash
export MDS_SOURCE=/lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/optimizer-experiments/stage3-mix/Megatron-DeepSpeed/checkpoints/AuroraGPT-2B-ws3072-ds-stage0-nl12-hs2048-mb1-seq8192-gb6144-sp1-pp1-tp1-bf16-optsophiag-lr2.17e-5-lwf0.05_ntok7770B_tokHF_tmgoogle_gemma-7b_flash/global_step154391/mp_rank_00_model_states.pt
export MDS_HF=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/outputs/evals/agpt-2b-mds-7771T/stage3-mix/step-154391/hf
export MDS_OUT="$MDS_HF/direct-generation.jsonl"

sha256sum "$MDS_SOURCE"
sha256sum "$MDS_HF/model-00001-of-00001.safetensors"
python -u torchtitan/experiments/ezpz/rl/scripts/mds_stage3_direct.py \
  "$MDS_HF" "$MDS_OUT"
```

Expected hashes are the two MDS hashes in the artifact table. Expected initial
continuations include `Paris`, `100 degrees Celsius`, and `4`. Repetition after
the correct answer is expected from this pretrained, non-chat base; incoherent
character soup is not.

### 8. Inspect raw completions and decide pass/fail

Do not use throughput or aggregate reward as the acceptance criterion. Read
every JSONL record and report, at minimum:

- prompt and decoded continuation;
- token count and finish reason;
- whether EOS/EOT terminated generation;
- exact task correctness;
- copied demonstration names;
- repeated or unclosed tags;
- unrelated prose/code and repetition.

The reproduced diagnosis should satisfy all three checks:

1. repaired checkpoint-900 can stop on EOT but remains semantically poor;
2. Qwen produces coherent bounded alphabet-sort output on the same XPU stack;
3. MDS `stage3-mix/global_step154391` produces coherent base completions but is
   not yet instruction-tuned.

Only a stage-3-derived SFT candidate that passes direct semantic inspection
should proceed to TorchTitan synchronization and GRPO.

## Targeted stage-3 alphabet-sort SFT

A controlled full-weight SFT experiment started from the verified broad MDS
`stage3-mix/global_step154391` HF conversion rather than the rejected historical
`checkpoint-900`. The source weight SHA-256 was
`df4a7289d4eddb57c380d14ceb4c510d7ee6c55b5b79a7ef0f96a3a61e644fd1`.

The deterministic curriculum builder
[`build_alphabet_sort_curriculum.py`](https://github.com/saforem2/torchtitan/blob/sync/upstream-0fd69d0b/torchtitan/experiments/ezpz/rl/scripts/sft/build_alphabet_sort_curriculum.py)
created 8,192 examples containing one to eight names. It varied the instruction
wording, supervised only assistant tokens, serialized turns with the recovered
Gemma contract, and explicitly supervised token 107 (`<end_of_turn>`). The
held-out evaluation names (`BethMillar`, `ShahramKhosravi`, `AliceZimmer`, and
`BobYoung`) and distractor names (`QuinnRivera`, `OmarSaito`, and `PiaValdez`)
were excluded from curriculum generation.

After two fail-closed launcher checks caught an incorrect ancestry SHA and an
incorrect assertion about the newline following token 107, job `12478521`
completed 100 full-weight steps over 24 MPICH ranks on two Sunspot nodes. PBS
reported `Exit_status=0`. The run processed 446,300 tokens in 79.52 seconds,
reported aggregate training loss `0.06139`, and reached token accuracies of
`0.9961` to `1.0` in the final steps. Collective FSDP model saving completed.

The final artifact is:

```text
/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-mds154391-alphabet-sft100/final
model.safetensors SHA-256: cd8c185a84c0ef274d04b0be91ba72046b02e7533d9f0507f4948b146e0a5c6e
config.json SHA-256: 761b56c19a67201ca1204eddb50798cdf30d5a4ddb5fa5a8fb535e797a36ebfd
tokenizer_config.json SHA-256: 72fa042775bf2f16182acbccf1f4562734476c6ab121ab370e38242153141e97
safetensors tensors: 111
```

### Clean held-out direct generation

The first clean-control invocation accidentally reused the Qwen harness and did
not stop on AGPT token 107. Its first blocks were correct, but generation then
repeated those blocks until the token limit. The corrected AGPT harness used
`stop_token_ids=[1, 107]`; job `12478524` exited 0 with **8/8 exact answers and
8/8 proper stops**. All four samples for each prompt were identical:

```text
Prompt: Sort these names in alphabetical order by FIRST name: BethMillar
Output:
<alphabetical_sorted>
BethMillar
</alphabetical_sorted>
```

```text
Prompt: Sort these names in alphabetical order by FIRST name:
        ShahramKhosravi, AliceZimmer, BobYoung
Output:
<alphabetical_sorted>
AliceZimmer
BobYoung
ShahramKhosravi
</alphabetical_sorted>
```

This demonstrates held-out task generalization: those names were absent from the
curriculum, the requested order was not memorized, and all completions terminated
at the AGPT EOT token.

### Distractor-example stress test

Job `12478522` used the same model and stop IDs but appended a formatting example
with different names. All eight generations used one correctly closed output
block and stopped at EOT, but **0/8 contained only the requested names**. The raw
outputs were:

```text
# BethMillar samples 1, 2, and 4
<alphabetical_sorted>
OmarSaito
QuinnRivera
PiaValdez
BethMillar
</alphabetical_sorted>

# BethMillar sample 3
<alphabetical_sorted>
BethMillar
OmarSaito
QuinnRivera
PiaValdez
</alphabetical_sorted>

# Three-name sample 1
<alphabetical_sorted>
OmarSaito
QuinnRivera
PiaValdez
BobYoung
ShahramKhosravi
</alphabetical_sorted>

# Three-name samples 2 and 3
<alphabetical_sorted>
AliceZimmer
BobYoung
PiaValdez
QuinnRivera
</alphabetical_sorted>

# Three-name sample 4
<alphabetical_sorted>
AmirSaito
BobYoung
AliceZimmer
PiaValdez
QuinnRivera
</alphabetical_sorted>
```

The targeted SFT therefore learned clean alphabet sorting, exact output framing,
and EOT termination, but remains vulnerable to copying names from a conflicting
in-context demonstration. The current artifact qualifies for a synchronization
parity test on the clean prompt; it does not yet qualify as a robust final policy
or for longer GRPO. A follow-up curriculum must explicitly train contrastive
examples in which formatting demonstrations contain names that must be ignored.

## TorchTitan and TorchStore synchronization gate

The clean-passing SFT-100 artifact exposed two compatibility gaps before the
weight-transfer test could run:

1. TorchTitan's text-only Jinja renderer rejected Transformers' assistant-mask
   `{% generation %}...{% endgeneration %}` extension. Commit `9716b6540d`
   added a byte-preserving no-op extension. A real-artifact comparison then
   rendered exactly the same serialized prompt through Transformers and
   TorchTitan.
2. The generic `DefaultRenderer` returned only EOS token 1 as a generation stop.
   AGPT uses token 107 (`<end_of_turn>`) as its normal assistant boundary. Commits
   `4ed6f887bb` and `180f52edbf` added a reusable extra-stop-token renderer
   adapter and configured AGPT with stop IDs `[1, 107]`.

The failed setup attempts are retained as controlled evidence:

| Job | Exit | Finding |
|---|---:|---|
| `12478526` | 13 | Invalid wrapper preflight invoked `train_upstream --help` without required `--module`; no controller was launched. |
| `12478528` | 1 | Reached controller construction, then failed on the unsupported Jinja `generation` tag. |
| `12478531` | 0 | Forced-Gloo transport completed, but before the renderer fix every validation sample repeated after a correct block until 128 tokens. |

Job `12478536` reran the one-step integration after both renderer fixes, using
forced Gloo only as a weight-transfer isolation control. PBS reported
`Exit_status=0`. TorchStore published and pulled the 7.94 GB model twice:

```text
initial trainer put: 3.4063 s, 2.33 GB/s
initial generator get: 3.1322 s, 2.54 GB/s
post-step trainer put: completed
post-step generator get: completed
pre-validation mean response length: 23.62
post-validation mean response length: 23.12
```

The authoritative raw artifact is:

```text
/lus/tegu/projects/datascience/foremans/reproductions/
  agpt2b-sft100-torchstore-sync-12478536/rollout_samples.jsonl
```

All eight policy-version-0 validation rows were exact, contained one output
block, and ended at AGPT EOT:

```text
<alphabetical_sorted>
JonathanPritchard
SukanyaRandhawa
</alphabetical_sorted><end_of_turn>

<alphabetical_sorted>
GianmassimoTasinato
</alphabetical_sorted><end_of_turn>

<alphabetical_sorted>
CameronPratt
MarioSantos
ShyamUpadhyay
</alphabetical_sorted><end_of_turn>

<alphabetical_sorted>
JonathanBraithwaite
QingxingCao
</alphabetical_sorted><end_of_turn>

<alphabetical_sorted>
DhairyaMalhotra
JamesClark
YuHasegawa
</alphabetical_sorted><end_of_turn>

<alphabetical_sorted>
UriLerner
</alphabetical_sorted><end_of_turn>

<alphabetical_sorted>
AmitabhaBagchi
</alphabetical_sorted><end_of_turn>

<alphabetical_sorted>
DinushaVatsalan
MarlanScully
WeitaoDuan
</alphabetical_sorted><end_of_turn>
```

All eight policy-version-1 rows were likewise bounded, single-block, and
EOT-terminated. Four received exact reward 1.0; the remaining four received
partial rewards on a different post-validation prompt set. Because the built-in
validation iterator advances between passes and the logged gradient norm was
zero, these unmatched rows are evidence that version-1 weights reached vLLM and
remained bounded, not an apples-to-apples semantic regression measurement.

### Standard and explicit-XCCL controls

Gloo is not the target production transport. Historical job `12478403` first
showed that the standard `LocalRankStrategy` with `TransportType.Unset` could
complete both TorchStore publications/pulls, one optimizer step, and shutdown.

Job `12478538` repeated that automatic, no-override path with the SFT-100 AGPT
artifact and the corrected stop-token contract. No
`TORCHTITAN_TORCHSTORE_TRANSPORT`, `TORCHSTORE_GLOO_ENABLED`, or
`TORCHSTORE_XCCL_ENABLED` variable was present. PBS reported `Exit_status=0`:

```text
TorchStore strategy: TransportType.Unset
initial trainer put: 4.3504 s, 1.83 GB/s
initial generator get: 0.3648 s, 21.79 GB/s
post-step trainer put: 0.0725 s, 109.61 GB/s
post-step generator get: 0.2689 s, 29.56 GB/s
pre-validation mean response length: 23.62
post-validation mean response length: 23.12
```

The raw policy-version-0 validation set was 8/8 reward 1.0, 8/8 single-block,
and 8/8 EOT-terminated. Policy-version 1 was 8/8 single-block and 8/8
EOT-terminated; 4/8 received reward 1.0 on the different advancing validation
prompt set. The authoritative artifact is:

```text
/lus/tegu/projects/datascience/foremans/reproductions/
  agpt2b-sft100-torchstore-sync-12478538/rollout_samples.jsonl
```

This is the primary successful synchronization result. `TransportType.Unset`
means TorchStore selected its standard transport automatically; the report does
not relabel that selection as XCCL without direct evidence.

Job `12478537` therefore forced `TransportType.XCCL` explicitly while holding the
AGPT model, renderer, actor topology, and one-step recipe fixed. It initialized
both actors and entered the first trainer publication, then stopped after
`flatten` and `cast`; no batch publication completed, no rollout JSONL was
created, and the log remained unchanged for more than eleven minutes. The
oneCCL diagnostic was:

```text
CCL_PROCESS_LAUNCHER changed to none
CCL_ATL_TRANSPORT changed to ofi
could not get local_idx/count from environment variables, trying to get them from ATL
```

Repeated compute-node probes showed the controller and actor processes sleeping
in `epoll` with no log growth. The disposable canary was cancelled and reached
PBS `Exit_status=143`. Thus the current evidence is:

- automatic TorchStore selection: full AGPT integration and bounded-output pass
  (`12478538`; earlier control `12478403` also passed);
- forced Gloo weight transfer: AGPT integration and bounded-output pass
  (`12478536`);
- explicitly forced XCCL weight transfer under the current Monarch TCP-KVS actor
  bootstrap: first-publication hang (`12478537`).

This failure must not be described as a model-quality defect or as failure of
TorchTitan's ordinary XCCL training collectives. It is specific to forcing
TorchStore's weight-transfer transport to XCCL with the current actor bootstrap.

## Distractor-resistant SFT follow-up

A second 8,192-row deterministic curriculum retained 75% clean examples and
used 25% contrastive examples split evenly between two held-out-safe formatting
wrappers. Target and distractor names were disjoint, distractors appeared only
in the user prompt, and the completion contained only sorted target names.
Training restarted from the same MDS stage-3 base rather than continuing from
SFT-100, preserving a controlled curriculum comparison.

Sunspot job `12478527` completed 100/100 full-weight steps across 24 MPICH ranks
on two nodes and exited 0:

```text
train_runtime: 91.38 s
train_loss: 0.07072
final-step loss: 0.007229
final-step token accuracy: 0.9961
```

The collective save produced:

```text
/lus/tegu/projects/datascience/foremans/reproductions/
  agpt2b-mds154391-alphabet-robust-sft100/final/model.safetensors
SHA-256: 8e1af0b6a5da8cacfd1e57b772fe7f511385a68ae42cb1c9b555736f76cb7224
```

Direct clean evaluation job `12478529` passed **8/8 exact and 8/8 EOT**. The
outputs were unchanged from the accepted SFT-100 clean control.

Adversarial evaluation job `12478530` improved from 0/8 to **5/8 exact**, with
5/8 proper EOT stops and distractor leakage in 3/8. All four three-name samples
were exact. One of four single-name samples was exact; the other three began
with a correct block but continued generating repeated or hallucinated blocks
until the 128-token limit. Representative failures were:

```text
<alphabetical_sorted>
BethMillar
</alphabetical_sorted>
<alphabetical_sorted>
BethMillar
</alphabetical_sorted>
<alphabetical_sorted>
Dina
BethMillar
</alphabetical_sorted>
<alphabetical_sorted>
Elicia
Pamita
QuinnRivera
</alphabetical_sorted>
...
```

```text
<alphabetical_sorted>
BethMillar
</alphabetical_sorted>
<alphabetical_sorted>
BethMillar
</alphabetical_sorted>
<alphabetical_sorted>
Diego
BethMillar
</alphabetical_sorted>
<alphabetical_sorted>
Felix
OmarSaito
QuinnRivera
</alphabetical_sorted>
...
```

The v2 curriculum therefore materially improves distractor resistance without
regressing clean behavior, but it fails the mandatory 8/8 adversarial and EOT
gates. It is rejected for GRPO. A controlled v3 should target the observed
failure mode—single-name adversarial prompts and explicit EOT supervision—rather
than changing the base checkpoint, optimizer recipe, and curriculum together.

## Current decision

The stage-3-derived SFT-100 policy passes clean direct generation and the
TorchTitan/TorchStore integration gate through the standard automatic transport
path. Forced Gloo remains an isolation control; explicit TorchStore XCCL requires
a separate actor-bootstrap fix.
Neither SFT-100 nor robust-SFT-v2 passes the adversarial semantic release gate.
No AGPT policy in this report should advance to bounded GRPO yet.

## Robust-SFT-v3: controlled single-name/EOT follow-up

Robust-SFT-v2 isolated the remaining failure to three stochastic single-name
adversarial samples that began with a correct block and then leaked, repeated, or
failed to terminate cleanly. V3 changes only the deterministic curriculum mix:
50% clean rows, 50% adversarial rows, and exactly 25% of all rows as single-name
adversarial cases. The base checkpoint, 8,192-row size, seed 154391, recovered
serialization, assistant-only masks, EOT supervision, 100 steps, learning rate
`1e-5`, and global batch 48 remain fixed.

A protected-runtime preflight over 32 rows confirmed 16 adversarial rows, exactly
eight single-name adversarial rows, and canonical `[107, 108]` tails. The code is
in commits `870cb9278` and `be043ef62`.

Sunspot job `12478541` ran from commit `8e9049d4b` and completed with PBS
`Exit_status=0`. The full dataset preflight confirmed 8,192 rows, 4,096
adversarial rows, and exactly 2,048 single-name adversarial rows. The 100 steps
completed in 74.44 seconds with aggregate training loss `0.06451`; collective
saving succeeded. The artifact is:

```text
/lus/tegu/projects/datascience/foremans/reproductions/
  agpt2b-mds154391-alphabet-robust-v3-sft100/final
model.safetensors SHA-256: 2e075c04a8560a0a4f197a24e3638148f491eb507dbb0506ef8ef14f5c4aaf17
```

These training metrics establish technical completion only. The release gate was
then evaluated directly against the exact artifact hash.

Clean job `12478542` and adversarial job `12478543` both completed with PBS
`Exit_status=0`. Each produced eight generations: four stochastic samples for a
single held-out name and four for three held-out names. All 16 generations had
`finish_reason="stop"`, ended with token 107, contained exactly one closed output
block, and avoided repetition or truncation.

The clean suite was **8/8 exact and EOT-terminated**. Its two distinct outputs
were:

```text
<alphabetical_sorted>
BethMillar
</alphabetical_sorted>
```

```text
<alphabetical_sorted>
AliceZimmer
BobYoung
ShahramKhosravi
</alphabetical_sorted>
```

The adversarial suite appended the conflicting formatting demonstration with
`QuinnRivera`, `OmarSaito`, and `PiaValdez`. It produced the same two correct
output blocks above for all eight samples: **8/8 exact and EOT-terminated**, with
zero distractor-name leakage. The authoritative raw artifacts are:

```text
/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-robust-v3-clean-direct.jsonl
/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-robust-v3-adversarial-direct.jsonl
```

Robust-SFT-v3 is therefore the first AGPT candidate in this report to pass the
conjunctive clean and adversarial semantic release gate. It may proceed to a
small bounded GRPO experiment; this direct-generation result does not by itself
establish that GRPO improves the policy.

## Robust-SFT-v3 bounded GRPO canary

Commit `9d2200546` adds a dedicated fail-closed canary rather than reusing the
obsolete checkpoint-900 launcher. Its protected-runtime preflight confirmed the
exact robust-v3 model SHA, LoRA rank 8 with alpha 16 on `wqkv` and `wo`, three
optimizer steps, four prompt groups per step with eight samples each, one turn
with at most five names, learning rate `2e-6`, 128-token maximum generation,
checkpoint interval 1, eight validation samples, and no explicit TorchStore
transport override.

Sunspot job `12478544` ran from commit `9d2200546` and completed all three
optimizer steps with PBS `Exit_status=0`. It loaded the exact v3 artifact,
selected automatic TorchStore transport (`TransportType.Unset`), completed the
initial 7.94 GB publication/pull, published and pulled after every step, saved
DCP checkpoints at steps 1, 2, and 3, ran post-step validation, and shut down
cleanly. The output directory is:

```text
/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-robust-v3-grpo3-12478544
```

The three step summaries were:

| Step | rollout reward mean | gradient norm | loss | LR |
|---:|---:|---:|---:|---:|
| 1 | 0.86 | 0.0032 | 0.0000027 | 0.000001 |
| 2 | 0.95 | 0 | 0 | 0.000002 |
| 3 | 0.83 | 0.022 | -0.00006 | 0.000002 |

Validation reward mean changed from `0.758` before training to `0.698` after
training. The pre/post prompts were different, so this is not a paired estimate
of degradation, but it is also not evidence of improvement.

Raw inspection of all 40 rows in `rollout_samples.jsonl` found every completion
bounded, single-block, and EOT-terminated. Exact task correctness on the broader
random one-to-five-name FIRST/LAST distribution was:

| Partition / policy version | Exact | Rows | Mean recorded reward |
|---|---:|---:|---:|
| pre-validation / 0 | 0 | 8 | 0.758 validation summary |
| training / 0 | 6 | 8 | included with policy-0 mean below |
| all recorded / 0 | 6 | 16 | 0.812 |
| training / 1 | 6 | 8 | 0.950 |
| training / 2 | 3 | 8 | 0.784 |
| post-validation / 3 | 3 | 8 | 0.698 |

The failures included incorrect ordering, omitted names, truncated/mutated
names, and one hallucinated pair (`ChenQichen`, `DomenicChen`). Representative
post-policy failures were:

```text
Prompt: sort XiaobinChen, TimoEkholm by FIRST name
Expected: TimoEkholm, XiaobinChen
Generated: XiaobinChen, TimoEkholm

Prompt: sort ToshihikoFujimori, AndreaPetri, RuihongYue, OlgutaBuse,
        YoshitakaKoyama by FIRST name
Generated: AndreaPetri, Fujimori, YoshitakaKoyama, OlgutaBuse, RuihongYue
```

The canary therefore passes the bounded GRPO integration and nonzero-learning-
signal test, but it does **not** establish policy improvement and must not be
extended as-is. Robust-v3 remains the accepted pre-GRPO artifact. Any further RL
experiment needs a paired fixed validation suite and a reward/task design that
matches the desired FIRST-name behavior rather than mixing untrained LAST-name
cases.

The first scheduled notifier for this job was misconfigured to probe local
`qstat` and local paths rather than SSH to Sunspot. It failed after one probe,
leaving the terminal pane open until manual follow-up. The notifier and pane have
now been removed. Future remote-HPC monitors must pass a verified SSH-based
probe before being considered active.
