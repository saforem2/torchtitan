# Torch 2.14 Monarch GRPO: AGPT-2B rollout diagnosis

> **Status (2026-09-22):** the Torch 2.14 Monarch/TorchStore/vLLM XPU path is
> operational, but the tested AGPT-2B SFT checkpoint is not suitable for the
> alphabet-sort GRPO run. The malformed output originates before RL weight
> synchronization. A known-good Qwen control survives the complete path, while
> the genuine broad MDS stage-3 7.771T checkpoint generates coherent base-model
> text and is the preferred base for corrected SFT.

## Executive conclusion

The observed AGPT-2B failure is a combination of two distinct issues:

1. **The reconstructed HF interface did not match SFT training.** It added a
   BOS token that training did not use, applied generic whitespace trimming,
   serialized system roles differently, omitted assistant generation markers,
   and inherited reversed BOS/EOS metadata. These are concrete export defects
   and are now repaired.
2. **The checkpoint is still a poor model for this task after repairing that
   interface.** Direct native-vLLM generation from checkpoint-900 remains
   semantically malformed. The earlier `global_step138650` base is likewise a
   weak instruction follower under the same prompt contract. The interface
   defects explain bad stopping and some prompt mismatch, but not the poor
   content.

This conclusion does **not** rely on reward or throughput. It follows from raw
completion inspection and three controlled comparisons:

- AGPT direct load, bypassing TorchTitan and TorchStore;
- a known-good Qwen model through both direct and synchronized execution;
- direct generation from the genuine MDS stage-3 7.771T base.

## Artifacts and environment

| Item | Value |
|---|---|
| TorchTitan branch | `sync/upstream-0fd69d0b` |
| AGPT SFT source | `checkpoint-900`, 96 intact DCP shards |
| Reconstructed AGPT model SHA-256 | `001cccc6331bca9c6695cb559e923d54640b845a755fe5191667ea2a6925b58e` |
| Qwen3-0.6B model SHA-256 | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| MDS stage3-mix source SHA-256 | `619b5d2581ac798f8bfe7de0d152b19f3ca7805b8566ab5b1191239e150a9a5f` |
| MDS stage3-mix HF weights SHA-256 | `df4a7289d4eddb57c380d14ceb4c510d7ee6c55b5b79a7ef0f96a3a61e644fd1` |
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
[`002218169`](https://github.com/saforem2/torchtitan/commit/0022181696f4ef6490f2439017541f2a1cfca83e).

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
