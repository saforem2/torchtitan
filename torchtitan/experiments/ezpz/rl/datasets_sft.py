# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SFT dataset registry. Each entry is a (load_fn, description) pair;
# load_fn takes no args and returns an HF Dataset with columns:
#   - prompt: list[dict] of chat messages (user turn(s))
#   - completion: list[dict] of chat messages (assistant turn(s))
# SFTTrainer with assistant_only_loss=True will only compute loss on
# the assistant turn(s), which is the conventional SFT setup.

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Callable

from datasets import Dataset

log = logging.getLogger(__name__)

# Directory where pre-materialized interleaved-mix arrow files live.
# Picked /home (NFS-shared with compute nodes) so a build done on the
# login node is visible to every compute rank. Subpath structure:
#   ~/.cache/ezpz_sft_mixes/<sha256-of-recipe>/
# Each cache entry is a complete Dataset.save_to_disk() tree, loadable
# in <5s via Dataset.load_from_disk regardless of how slow the original
# interleave_datasets() call was. This is the same content-hashed cache
# pattern blendcorpus uses for its (data_prefix, num_samples, seq_len,
# seed) index files — see blendcorpus/data/gpt_dataset.py
# _build_index_mappings (desc_hash + _doc_idx.npy / _sample_idx.npy /
# _shuffle_idx.npy).
EZPZ_SFT_MIX_CACHE_DIR = os.environ.get(
    "EZPZ_SFT_MIX_CACHE_DIR",
    os.path.expanduser("~/.cache/ezpz_sft_mixes"),
)


def _interleave_all_exhausted_fast(datasets_list, probabilities, seed):
    """Vectorized, bit-identical drop-in for HF
    ``interleave_datasets(..., probabilities=..., stopping_strategy="all_exhausted")``.

    HF's ``_interleave_map_style_datasets`` builds the output index list in a
    pure-Python for-loop -- one iteration per output row. For a big mix
    (OpenMathInstruct-2's 14M rows -> ~93M interleaved rows) that loop takes
    ~90 min single-threaded, which at 384 ranks blew the XPU oneCCL barrier
    (jobs 12468348/371/398) and is why the big-mix path was avoided. The RNG is
    already batched; it is Python interpreter overhead, not compute. This
    reproduces HF's exact algorithm with numpy: ~5s for 93M rows.

    Bit-identical to stock for a fixed ``seed`` because it consumes the RNG
    exactly as HF does (``rng.choice(n, size=1000, p=probabilities)`` blocks)
    and applies the same rolling-window index mapping
    (``current_index`` wraps to 0 on exhaustion). Verified against stock across
    multiple (lengths, probabilities, seed) cases; a self-check in
    ``_materialized_mix_load_or_build`` guards against a future HF change to the
    RNG-consumption pattern.

    Only handles the ``all_exhausted`` + probabilities-given case (what the ezpz
    SFT mixes use). Callers must fall back to stock for anything else.
    """
    import numpy as np
    from datasets import concatenate_datasets

    lengths = [len(d) for d in datasets_list]
    n = len(lengths)
    offsets = np.cumsum([0] + lengths[:-1])
    L = np.asarray(lengths, dtype=np.int64)

    # Replay HF's iter_random_indices EXACTLY: default_rng(seed), drawn in
    # 1000-sized blocks. Keep drawing until every source has appeared at least
    # `length` times (all_exhausted). Track cumulative counts to know when.
    rng = np.random.default_rng(seed)
    blocks = []
    counts = np.zeros(n, dtype=np.int64)
    while not np.all(counts >= L):
        block = rng.choice(n, size=1000, p=probabilities)
        blocks.append(block)
        counts += np.bincount(block, minlength=n)
    draws = np.concatenate(blocks)

    # HF sets is_exhausted[s]=True right after source s's L-th draw, and breaks
    # the loop (before appending) once ALL are exhausted. So the last appended
    # draw is at position max_s(position of s's L-th occurrence).
    reach = np.full(n, -1, dtype=np.int64)
    for s in range(n):
        occ = np.flatnonzero(draws == s)
        if len(occ) >= lengths[s]:
            reach[s] = occ[lengths[s] - 1]
    use = draws[: reach.max() + 1]

    # Each source's k-th appearance maps to concatenated row
    # (k-1) % length + offset (rolling window on exhaustion).
    idx = np.empty(len(use), dtype=np.int64)
    for s in range(n):
        pos = np.flatnonzero(use == s)
        idx[pos] = (np.arange(len(pos)) % lengths[s]) + offsets[s]

    return concatenate_datasets(datasets_list).select(idx.tolist())


_FAST_INTERLEAVE_VERIFIED: bool | None = None


def _fast_interleave_matches_stock(seed: int = 42) -> bool:
    """One-time guard: confirm the vectorized fast path is bit-identical to
    stock ``interleave_datasets`` on a small synthetic case for the installed
    ``datasets`` version. Cached after the first call. If HF ever changes its
    RNG-consumption pattern, this returns False and callers fall back to stock
    (correctness over speed). Cheap (tiny datasets, sub-second).
    """
    global _FAST_INTERLEAVE_VERIFIED
    if _FAST_INTERLEAVE_VERIFIED is not None:
        return _FAST_INTERLEAVE_VERIFIED
    try:
        from datasets import interleave_datasets

        probs = [0.65, 0.15, 0.20]
        dsets = [
            Dataset.from_dict({"__v": [f"{i}_{j}" for j in range(L)]})
            for i, L in enumerate((97, 20, 43))
        ]
        want = interleave_datasets(
            dsets, probabilities=probs, seed=seed,
            stopping_strategy="all_exhausted",
        )["__v"]
        got = _interleave_all_exhausted_fast(dsets, probs, seed)["__v"]
        _FAST_INTERLEAVE_VERIFIED = want == got
        if not _FAST_INTERLEAVE_VERIFIED:
            log.warning(
                "[mix-cache] fast interleave DIVERGED from stock for this "
                "datasets version; falling back to the (slow) stock path."
            )
    except Exception as e:  # noqa: BLE001 -- never let the guard break a build
        log.warning(f"[mix-cache] fast-interleave self-check errored ({e}); "
                    "using stock path.")
        _FAST_INTERLEAVE_VERIFIED = False
    return _FAST_INTERLEAVE_VERIFIED


def _materialized_mix_load_or_build(
    component_names: list[str],
    weights: list[float],
    seed: int,
    stopping_strategy: str = "all_exhausted",
    caps: dict[str, int] | None = None,
) -> Dataset:
    """Load a pre-materialized interleaved mix from disk if available,
    otherwise build via ``interleave_datasets(...)`` and save to disk.

    Why: at production scale (384 ranks), the live ``interleave_datasets``
    setup over a multi-million-row mix runs >20 min on rank 0 even with
    every component's HF .map() cache warm (job 12468398 died here). The
    interleave's runtime per-row index resolution doesn't cache well in
    HF datasets today. By collapsing it into a single
    ``Dataset.save_to_disk(...)`` tree keyed by a content hash of the
    recipe, subsequent runs (this rank, any rank, any future job)
    ``load_from_disk()`` in <5 sec.

    Recipe hash: SHA-256 of the sorted component names + weights + seed
    + stopping_strategy. Pre-renormalize weights so mathematically-
    equivalent specs ('a:1,b:1' and 'a:0.5,b:0.5') hash identically.

    Safe to call from multiple ranks concurrently: we save to a tmp
    dir first, then atomically rename. If two ranks race, the second
    rename overwrites with the same content (atomic on POSIX) and both
    end up with a valid cache. ``load_from_disk`` is mmap-based so
    concurrent readers are fine.
    """
    from datasets import interleave_datasets

    # Pre-renormalize weights (in case caller passes unnormalized) so
    # the hash is canonical regardless of input scale.
    wsum = sum(weights)
    norm_weights = [w / wsum for w in weights] if wsum > 0 else weights

    recipe = {
        "components": list(component_names),
        "weights": [round(w, 8) for w in norm_weights],
        "seed": int(seed),
        "stopping": str(stopping_strategy),
        "caps": {k: int(v) for k, v in sorted((caps or {}).items())},
        # Bump if the on-disk arrow format changes incompatibly.
        # v2: filter empty/whitespace-only prompt+completion rows (see below).
        "v": 2,
    }
    recipe_str = repr(sorted(recipe.items()))
    recipe_hash = hashlib.sha256(recipe_str.encode("utf-8")).hexdigest()[:16]

    cache_dir = os.path.join(EZPZ_SFT_MIX_CACHE_DIR, recipe_hash)
    marker_file = os.path.join(cache_dir, "dataset_info.json")

    if os.path.isfile(marker_file):
        log.info(
            f"[mix-cache] hit {recipe_hash}: loading pre-materialized mix from {cache_dir}"
        )
        t0 = time.monotonic()
        ds = Dataset.load_from_disk(cache_dir)
        log.info(f"[mix-cache] loaded {len(ds):,} rows in {time.monotonic()-t0:.1f}s")
        return ds

    log.info(
        f"[mix-cache] miss {recipe_hash}: building + saving interleaved mix to {cache_dir}"
    )
    t0 = time.monotonic()
    components_built = []
    for n in component_names:
        comp = SFT_REGISTRY[n].build()
        cap = (caps or {}).get(n)
        if cap is not None and len(comp) > cap:
            # shuffle before capping so we sample across the source, not a
            # head slice; seed-tied for reproducibility + a stable cache key.
            comp = comp.shuffle(seed=seed).select(range(cap))
            log.info(f"[mix-cache] {n}: capped to {cap:,} rows")
        components_built.append(comp)

    # Drop empty / whitespace-only prompt or completion rows -- BEFORE interleave,
    # per source. Some upstream sources carry an empty assistant `completion`;
    # with packing=True + assistant_only_loss=True a packed window landing on such
    # a row has zero valid label tokens and the degenerate backward manifests as a
    # deterministic GPU illegal memory access (SFT jobs 12470254/258/262 crashed at
    # step 211 / rank 61 with `Segmentation fault from GPU ... NotPresent`,
    # reproducible across nodes -- NOT a bad node). Filtering here, per source, is
    # ~Nx cheaper than filtering the interleaved output: `all_exhausted` cycling
    # inflates the sources ~6x (e.g. ~15M source rows -> ~93M interleaved), and a
    # post-interleave filter over 93M rows takes hours (job 12470276). Empty rows
    # are empty regardless of cycling, so pre-filtering is identical + much faster.
    def _nonempty(ex: dict) -> bool:
        def _content(turns) -> str:
            if not turns:
                return ""
            c = turns[0].get("content", "")
            return c if isinstance(c, str) else ""

        return bool(_content(ex["prompt"]).strip()) and bool(
            _content(ex["completion"]).strip()
        )

    filtered_components = []
    for name, comp in zip(component_names, components_built):
        nb = len(comp)
        comp = comp.filter(_nonempty, num_proc=16)
        nd = nb - len(comp)
        if nd:
            log.info(
                f"[mix-cache] {name}: dropped {nd:,} empty rows "
                f"({100 * nd / max(nb, 1):.3f}%); {len(comp):,} remain"
            )
        filtered_components.append(comp)
    components_built = filtered_components

    # Fast path: for the probabilities-given all_exhausted case (all ezpz SFT
    # mixes), use the vectorized interleave -- HF's Python-loop index build is
    # ~90 min for a 93M-row mix; the vectorized version is ~5s and bit-identical.
    # Fall back to stock for any other config. A one-time self-check on tiny
    # synthetic datasets confirms bit-equivalence against stock for THIS
    # datasets version, so a future change to HF's RNG-consumption pattern fails
    # loudly here instead of silently producing a different (still-valid but
    # non-reproducible) mix.
    use_fast = stopping_strategy == "all_exhausted" and norm_weights is not None
    if use_fast and _fast_interleave_matches_stock(seed=seed):
        ds = _interleave_all_exhausted_fast(
            components_built, probabilities=norm_weights, seed=seed
        )
        log.info(
            f"[mix-cache] interleave built via FAST vectorized path "
            f"({len(ds):,} rows in {time.monotonic()-t0:.1f}s)"
        )
    else:
        ds = interleave_datasets(
            components_built,
            probabilities=norm_weights,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        log.info(
            f"[mix-cache] interleave built via stock path "
            f"({len(ds):,} rows in {time.monotonic()-t0:.1f}s)"
        )

    log.info(f"[mix-cache] writing {len(ds):,} rows to disk")

    # Save to tmp + atomic rename so a partial write from one rank
    # doesn't leave a corrupt cache that another rank picks up.
    os.makedirs(EZPZ_SFT_MIX_CACHE_DIR, exist_ok=True)
    tmp_dir = os.path.join(EZPZ_SFT_MIX_CACHE_DIR, f"{recipe_hash}.tmp.{os.getpid()}")
    t1 = time.monotonic()
    ds.save_to_disk(tmp_dir)
    # os.rename is atomic on POSIX (same filesystem)
    try:
        os.rename(tmp_dir, cache_dir)
    except OSError:
        # Another rank beat us to it — fine, clean up our tmp and use
        # theirs.
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    log.info(
        f"[mix-cache] saved + renamed in {time.monotonic()-t1:.1f}s "
        f"(total miss path: {time.monotonic()-t0:.1f}s)"
    )
    return Dataset.load_from_disk(cache_dir)


@dataclass
class SFTDataset:
    name: str
    build: Callable[..., Dataset]
    description: str = ""


SFT_REGISTRY: dict[str, SFTDataset] = {}


def register_sft_dataset(ds: SFTDataset) -> SFTDataset:
    SFT_REGISTRY[ds.name] = ds
    return ds


def get_sft_dataset(name: str) -> SFTDataset:
    """Resolve an SFT dataset by name OR by a CLI mix-spec string.

    Two accepted shapes:
      - A registered dataset name (e.g. 'gsm8k', 'tulu_math_uc_mix') —
        looked up in SFT_REGISTRY.
      - A mix spec like 'tulu-3-sft-mixture:0.5,gsm8k:0.5' — parsed
        and synthesized into an ad-hoc SFTDataset that interleaves
        the components. Weights are renormalized to sum to 1.

    The mix-spec form lets you A/B different ratios from the launch
    command without touching code.
    """
    if _is_mix_spec(name):
        return SFTDataset(
            name=name,
            build=lambda: _build_ad_hoc_mix(name),
            description=f"Ad-hoc mix from CLI spec: {name}",
        )
    if name not in SFT_REGISTRY:
        available = ", ".join(sorted(SFT_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown SFT dataset {name!r}. Available: {available}")
    return SFT_REGISTRY[name]


# ---------------------------------------------------------------------------
# gsm8k — grade-school math word problems with step-by-step solutions
# ---------------------------------------------------------------------------


def _build_gsm8k() -> Dataset:
    """Load gsm8k 'main' split (7473 train examples) and format as chat.

    Each gsm8k row has {question, answer} where answer is the chain-of-
    thought ending with '#### N'. We format the prompt as a user turn
    asking the question and the completion as an assistant turn
    containing the full CoT solution including the '#### N' answer
    marker. extract_answer() in tasks.common already matches this format.
    """
    from datasets import load_dataset

    raw = load_dataset("openai/gsm8k", "main", split="train")

    def _format(ex):
        return {
            "prompt": [{"role": "user", "content": ex["question"]}],
            "completion": [{"role": "assistant", "content": ex["answer"]}],
        }

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="gsm8k",
        build=_build_gsm8k,
        description=(
            "Grade-school math word problems with step-by-step CoT "
            "solutions (openai/gsm8k 'main' split, 7473 examples). "
            "Answers end with '#### N' format already handled by "
            "tasks.common.extract_answer."
        ),
    )
)


# ---------------------------------------------------------------------------
# gsm8k-r1cot -- gsm8k reformatted into the <think>/<answer> CoT envelope
# ---------------------------------------------------------------------------


def _build_gsm8k_r1cot() -> Dataset:
    """gsm8k reformatted to teach the R1-style reasoning envelope.

    Reuses gsm8k's own step-by-step rationale (the ``answer`` field, which
    already ends in ``#### N``) but wraps it as
    ``<think>{rationale}</think>\n<answer>\\boxed{{N}}</answer>`` so the model
    learns to EMIT a delimited reasoning trace + boxed final answer. This is the
    Stage 1 cold-start for the CoT plan
    (docs/live/chains/rl/plans/cot.md) -- teaches the FORMAT from gsm8k's own
    data, no teacher model required.

    Two details that matter:
      - the completion content starts directly with ``<think>`` (no leading
        newline); the gemma chat template lstrips a leading ``\n`` on the
        assistant turn, which otherwise trips TRL's prompt/completion
        tokenization-mismatch and silently breaks the assistant-only loss mask.
      - the user prompt suffix is byte-identical to the Stage 0 eval's
        (scripts/eval/eval_cot_gsm8k.py) so train and eval prompts match.
    """
    import re

    from datasets import load_dataset

    raw = load_dataset("openai/gsm8k", "main", split="train")
    calc_re = re.compile(r"<<[^>]*>>")  # gsm8k inline calculator annotations
    suffix = (
        "\nReason step by step inside <think></think>, then give the final "
        "answer inside <answer>\\boxed{}</answer>."
    )

    def _format(ex):
        answer = ex["answer"]
        if "####" in answer:
            cot, final = answer.split("####", 1)
        else:
            cot, final = answer, ""
        cot = calc_re.sub("", cot).strip()
        final = final.strip().replace(",", "")
        completion = f"<think>{cot}</think>\n<answer>\\boxed{{{final}}}</answer>"
        return {
            "prompt": [{"role": "user", "content": ex["question"] + suffix}],
            "completion": [{"role": "assistant", "content": completion}],
        }

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="gsm8k-r1cot",
        build=_build_gsm8k_r1cot,
        description=(
            "gsm8k rationales wrapped in the <think>/<answer> R1-style CoT "
            "envelope (openai/gsm8k 'main', 7473 examples). Stage 1 cold-start "
            "for teaching agpt-2b to emit reasoning traces; no teacher needed."
        ),
    )
)


# ---------------------------------------------------------------------------
# OpenR1-Math-220k -- rich R1-distilled long-form CoT, reformatted into our
# <think>/<answer>\boxed{} envelope (matches gsm8k-r1cot + the eval).
# ---------------------------------------------------------------------------

_OPENR1_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_OPENR1_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_OPENR1_SUFFIX = (
    "\nReason step by step inside <think></think>, then give the final "
    "answer inside <answer>\\boxed{}</answer>."
)

# Max <think> trace length (chars) kept from OpenR1. The B3 regression traced to
# long-form R1 run-on traces (mean gen_len 601, 26/200 never closed </answer>)
# that taught a verbose style diluting a 2B's short-arithmetic competence. Drop
# traces above this so only short, useful reasoning is trained. Env-tunable.
OPENR1_MAX_THINK_CHARS = int(os.environ.get("OPENR1_MAX_THINK_CHARS", "1200"))


def _openr1_format_row(ex):
    """Pick a verified-correct generation, extract its <think> trace + boxed
    answer, and reformat to our envelope. Returns a formatted row dict, or
    None if no correct generation with an extractable boxed answer exists
    (caller drops None rows so we never train a malformed envelope).

    OpenR1 generations already carry <think>...</think>; we keep that trace
    verbatim and normalize the tail to a single <answer>\\boxed{}</answer>.

    Missing or short correctness/completeness metadata is treated as
    NOT verified (default-exclude): a generation at an index beyond the
    metadata list's length is never accepted as unverified-but-allowed.
    """
    gens = ex.get("generations") or []
    correct = ex.get("correctness_math_verify") or []
    complete = ex.get("is_reasoning_complete") or []
    for i, gen in enumerate(gens):
        correct_i = correct[i] if i < len(correct) else False
        complete_i = complete[i] if i < len(complete) else False
        if not correct_i or not complete_i:
            continue
        if not gen:
            continue
        tm = _OPENR1_THINK_RE.search(gen)
        if not tm:
            continue
        trace = tm.group(1).strip()
        if len(trace) > OPENR1_MAX_THINK_CHARS:
            continue  # run-on trace: skip this generation (may fall through to None)
        # boxed answer: prefer the LAST \boxed{} anywhere in the generation
        boxes = _OPENR1_BOXED_RE.findall(gen)
        if not boxes:
            continue
        ans = boxes[-1].strip()
        if not trace or not ans:
            continue
        completion = f"<think>{trace}</think>\n<answer>\\boxed{{{ans}}}</answer>"
        return {
            "prompt": [{"role": "user", "content": ex["problem"] + _OPENR1_SUFFIX}],
            "completion": [{"role": "assistant", "content": completion}],
        }
    return None


def _openr1_map_row(ex):
    """map() fn for _build_openr1_math_cot: format, or emit a same-shaped
    empty-content sentinel row on failure (see _build_openr1_math_cot for
    why the sentinel must be same-shaped rather than empty-list). Factored
    out so tests can exercise the exact map+filter logic without
    load_dataset("open-r1/OpenR1-Math-220k").
    """
    out = _openr1_format_row(ex)
    if out is not None:
        return out
    return {
        "prompt": [{"role": "user", "content": ""}],
        "completion": [{"role": "assistant", "content": ""}],
    }


def _openr1_filter_row(ex):
    """filter() predicate for _build_openr1_math_cot: keep only rows with
    non-empty prompt/completion content, i.e. drop _openr1_map_row's
    sentinel rows. A real formatted row always has non-empty problem text
    and a non-empty envelope, so this never drops a genuine row.
    """
    return (
        bool(ex["prompt"])
        and bool(ex["completion"])
        and bool(ex["prompt"][0]["content"])
        and bool(ex["completion"][0]["content"])
    )


def _build_openr1_math_cot():
    """open-r1/OpenR1-Math-220k reformatted to the <think>/<answer> envelope.

    Selects a verified-correct (correctness_math_verify) + complete generation
    per problem, keeps its R1 reasoning trace, and normalizes the final answer
    to <answer>\\boxed{ans}</answer>. Rows with no correct+boxed generation
    are dropped: map() emits a same-shaped empty-content sentinel row (NOT an
    empty-list sentinel -- see _openr1_map_row) and filter() removes it by
    content truthiness. Needs the HF download cached first (see
    _pretokenize_b3_instruct_cot_mix_1n.sh).
    """
    from datasets import load_dataset

    raw = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    cols = raw.column_names

    mapped = raw.map(_openr1_map_row, remove_columns=cols)
    return mapped.filter(_openr1_filter_row)


register_sft_dataset(
    SFTDataset(
        name="OpenR1-Math-220k",
        build=_build_openr1_math_cot,
        description=(
            "open-r1/OpenR1-Math-220k R1-distilled long-form CoT, reformatted "
            "into the <think>/<answer>\\boxed{} envelope. Selects a verified- "
            "correct generation per problem; rows without a correct+boxed "
            "generation are dropped. The rich-reasoning half of b3_instruct_cot_mix."
        ),
    )
)


# ---------------------------------------------------------------------------
# metamathqa — augmented + rephrased GSM8K+MATH, ~400k examples
# ---------------------------------------------------------------------------


def _build_metamathqa() -> Dataset:
    """Load MetaMathQA (~395k augmented GSM8K+MATH solutions).

    Columns: query, response, type, original_question. We use query +
    response and ignore the rest. Much bigger than gsm8k — useful when
    you have the compute and want stronger generalization.
    """
    from datasets import load_dataset

    raw = load_dataset("meta-math/MetaMathQA", split="train")

    def _format(ex):
        return {
            "prompt": [{"role": "user", "content": ex["query"]}],
            "completion": [{"role": "assistant", "content": ex["response"]}],
        }

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="metamathqa",
        build=_build_metamathqa,
        description=(
            "Augmented + rephrased GSM8K+MATH solutions "
            "(meta-math/MetaMathQA, ~395k examples). Stronger "
            "generalization than gsm8k at higher cost."
        ),
    )
)


# ---------------------------------------------------------------------------
# alpaca — broad instruction-following (52k examples)
# ---------------------------------------------------------------------------


def _build_alpaca() -> Dataset:
    """Load tatsu-lab/alpaca (52k instruction-following examples).

    Each row has {instruction, input, output}. We concatenate
    instruction+input into the user turn (with a blank line between
    them when input is non-empty) and use output as the assistant
    turn. Broader-purpose than math-only datasets — useful for
    teaching the base AuroraGPT-2B model to follow instructions
    other than just "do the math".
    """
    from datasets import load_dataset

    raw = load_dataset("tatsu-lab/alpaca", split="train")

    def _format(ex):
        instruction = ex["instruction"]
        if ex.get("input"):
            user_content = f"{instruction}\n\n{ex['input']}"
        else:
            user_content = instruction
        return {
            "prompt": [{"role": "user", "content": user_content}],
            "completion": [{"role": "assistant", "content": ex["output"]}],
        }

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="alpaca",
        build=_build_alpaca,
        description=(
            "Broad instruction-following (tatsu-lab/alpaca, ~52k "
            "examples). Use to teach instruction-following beyond "
            "math; mix with gsm8k/metamathqa via the 'math_alpaca_mix' "
            "entry."
        ),
    )
)


# ---------------------------------------------------------------------------
# math_alpaca_mix — interleaves metamathqa + gsm8k + alpaca by weight
# ---------------------------------------------------------------------------


def _build_math_alpaca_mix(
    weights=(0.6, 0.1, 0.3),
    seed: int = 42,
) -> Dataset:
    """Interleave metamathqa + gsm8k + alpaca with given probabilities.

    Default weight 0.6/0.1/0.3 keeps the math signal dominant (where
    we have the reward functions to exercise it) while exposing the
    model to ~30% general instruction-following data — useful for
    tasks like word_sort that aren't pure arithmetic.

    Uses ``interleave_datasets(stopping_strategy='all_exhausted')`` so
    smaller datasets cycle until the largest is exhausted; total
    yielded examples will be roughly ``max_size / max_weight`` so the
    sampled proportions actually match the requested weights.
    """
    from datasets import interleave_datasets

    if len(weights) != 3:
        raise ValueError(
            f"math_alpaca_mix weights must be (w_metamath, w_gsm8k, w_alpaca); "
            f"got {weights!r}"
        )
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0; got {sum(weights)}")

    return _materialized_mix_load_or_build(
        component_names=["metamathqa", "gsm8k", "alpaca"],
        weights=list(weights),
        seed=seed,
    )


register_sft_dataset(
    SFTDataset(
        name="math_alpaca_mix",
        build=_build_math_alpaca_mix,
        description=(
            # %% — argparse %-formats help strings so literal % must be
            # escaped or it crashes with "unsupported format character"
            # (we hit this in job 12468232 — '60% m' parses as %m).
            "Weighted mix: 60%% metamathqa + 10%% gsm8k + 30%% alpaca. "
            "Broader than math-only; better for downstream tasks "
            "that aren't pure arithmetic (e.g. word_sort)."
        ),
    )
)


# ---------------------------------------------------------------------------
# Helpers for the multi-turn / chat-format datasets below
# ---------------------------------------------------------------------------


def _messages_to_prompt_completion(messages: list[dict]) -> dict:
    """Split a chat-format `messages` list into SFTTrainer's
    {prompt, completion} shape.

    Convention used by tulu-3-sft-mixture, ultrachat_200k, and most
    HF chat datasets: `messages` is an ordered list of
    `{role, content}` dicts ending with an assistant turn that's the
    target for SFT loss. We put everything except the last assistant
    turn in `prompt`, and the last assistant turn in `completion`.

    For datasets where the last turn isn't `assistant`, fall back to
    "everything is the prompt, completion is empty" (which `interleave`
    will tolerate; SFTTrainer skips empty completions).
    """
    if not messages or messages[-1].get("role") != "assistant":
        return {"prompt": messages, "completion": []}
    return {"prompt": messages[:-1], "completion": [messages[-1]]}


# ---------------------------------------------------------------------------
# tulu-3-sft-mixture — allenai's curated SFT mix, ~940k examples
# ---------------------------------------------------------------------------


def _build_tulu3_sft_mixture() -> Dataset:
    """Load allenai/tulu-3-sft-mixture (~939k chat-format examples).

    The canonical recipe behind Tulu-3 — blends math + reasoning + IF +
    code + safety at validated ratios. Single best "drop-in" SFT
    dataset for a small instruct model in 2026. Each row has
    {id, messages, source}; we split messages into prompt/completion.
    """
    from datasets import load_dataset

    raw = load_dataset("allenai/tulu-3-sft-mixture", split="train")

    def _format(ex):
        return _messages_to_prompt_completion(ex["messages"])

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="tulu-3-sft-mixture",
        build=_build_tulu3_sft_mixture,
        description=(
            "AllenAI's curated SFT mix from Tulu-3 (~939k examples). "
            "Math + reasoning + IF + code + safety at validated ratios. "
            "Single best drop-in choice for general SFT in 2026."
        ),
    )
)


# ---------------------------------------------------------------------------
# OpenMathInstruct-2 — NVIDIA's massive augmented math dataset
# ---------------------------------------------------------------------------


def _build_openmath_instruct2() -> Dataset:
    """Load nvidia/OpenMathInstruct-2 (~14M math instruction examples).

    NVIDIA's superset of GSM8K + MATH augmentations. Each row has
    {problem, generated_solution, expected_answer, problem_source}.
    Use when you want to push math capability hard; 14M is a lot of
    data — consider mixing with broader data unless you specifically
    want math depth.
    """
    from datasets import load_dataset

    raw = load_dataset("nvidia/OpenMathInstruct-2", split="train")

    def _format(ex):
        return {
            "prompt": [{"role": "user", "content": ex["problem"]}],
            "completion": [
                {"role": "assistant", "content": ex["generated_solution"]}
            ],
        }

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="OpenMathInstruct-2",
        build=_build_openmath_instruct2,
        description=(
            "NVIDIA OpenMathInstruct-2 (~14M math examples, superset of "
            "GSM8K + MATH augmentations). Use for math depth; mix with "
            "broader datasets unless you specifically want math-only."
        ),
    )
)


# ---------------------------------------------------------------------------
# ultrachat-200k — high-quality GPT-distilled multi-turn conversations
# ---------------------------------------------------------------------------


def _build_ultrachat_200k() -> Dataset:
    """Load HuggingFaceH4/ultrachat_200k (~208k multi-turn chats).

    Note: this dataset's train split is named 'train_sft' (not
    'train'). Common gotcha. Each row has {prompt, prompt_id,
    messages} where messages includes both the user prompt and
    assistant response(s); we use messages directly.
    """
    from datasets import load_dataset

    raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    def _format(ex):
        return _messages_to_prompt_completion(ex["messages"])

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="ultrachat-200k",
        build=_build_ultrachat_200k,
        description=(
            "HuggingFaceH4/ultrachat_200k (~208k high-quality GPT-distilled "
            "multi-turn conversations). Good IF/conversational complement "
            "to math-heavy SFT data."
        ),
    )
)


# ---------------------------------------------------------------------------
# tulu_math_uc_mix — canonical "blessed" 3-way mix for AuroraGPT-2B SFT
# ---------------------------------------------------------------------------


def _build_tulu_math_uc_mix(
    weights=(0.65, 0.15, 0.20),
    seed: int = 42,
) -> Dataset:
    """Interleave tulu-3-sft-mixture + OpenMathInstruct-2 + ultrachat-200k.

    Default 0.65 / 0.15 / 0.20 ratio comes from a 0.55 / 0.15 / 0.20
    plan with the dropped 0.10 olmo-mix replay absorbed into tulu
    (tulu-3 already contains some pretrain-flavored data like FLAN and
    OpenAssistant, so this partially approximates the replay effect).

    Uses 'all_exhausted' stopping so smaller datasets cycle until the
    largest is fully consumed — OpenMathInstruct-2 is ~14M, so the
    effective size is dominated by it at ~93M after scaling.
    """
    from datasets import interleave_datasets

    if len(weights) != 3:
        raise ValueError(
            f"tulu_math_uc_mix weights must be (w_tulu, w_math, w_uc); "
            f"got {weights!r}"
        )
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0; got {sum(weights)}")

    return _materialized_mix_load_or_build(
        component_names=[
            "tulu-3-sft-mixture", "OpenMathInstruct-2", "ultrachat-200k",
        ],
        weights=list(weights),
        seed=seed,
    )


register_sft_dataset(
    SFTDataset(
        name="tulu_math_uc_mix",
        build=_build_tulu_math_uc_mix,
        description=(
            # %% escapes for argparse — see note above on math_alpaca_mix.
            "Canonical SFT recipe: 65%% tulu-3-sft-mixture + "
            "15%% OpenMathInstruct-2 + 20%% ultrachat-200k. Tulu's "
            "validated broad mix + extra math depth + multi-turn IF."
        ),
    )
)


# ---------------------------------------------------------------------------
# b3_instruct_cot_mix -- balanced instruction + CoT mix for the B3 cold-start
# rebuild (docs/live/chains/sft/agpt/2b-mds/b3-instruct-cot-mix/design.md)
# ---------------------------------------------------------------------------


def _build_b3_instruct_cot_mix(seed: int = 42):
    """Balanced instruction + CoT SFT mix for the B3 cold-start rebuild
    (docs/live/chains/sft/agpt/2b-mds/b3-instruct-cot-mix/design.md):
      0.30 tulu-3-sft-mixture (general instruction-following)
      0.25 OpenR1-Math-220k   (rich long-form R1 CoT, our envelope)
      0.15 gsm8k-r1cot        (in-distribution CoT, eval-matching format)
      0.15 ultrachat-200k     (multi-turn chat)
      0.15 OpenMathInstruct-2 (math breadth)
    45%% instruction/chat, 55%% math/CoT. Folds the CoT envelope INTO one SFT
    (vs B2's two-stage tulu-math -> gsm8k-r1cot lineage). Uses the same
    materialized-mix cache + all_exhausted interleave as tulu_math_uc_mix.
    """
    return _materialized_mix_load_or_build(
        component_names=[
            "tulu-3-sft-mixture",
            "OpenR1-Math-220k",
            "gsm8k-r1cot",
            "ultrachat-200k",
            "OpenMathInstruct-2",
        ],
        weights=[0.30, 0.25, 0.15, 0.15, 0.15],
        seed=seed,
        # Cold-start SFT, not pretraining: cap OpenMathInstruct-2 (14M) so the
        # all_exhausted interleave maxes at a cold-start-appropriate size
        # (~10-13M rows, not 93M). Keeps tokenize + packed-dataset size
        # tractable under the tegu user quota; 2M math rows is ample here.
        caps={"OpenMathInstruct-2": 2_000_000},
    )


register_sft_dataset(
    SFTDataset(
        name="b3_instruct_cot_mix",
        build=_build_b3_instruct_cot_mix,
        description=(
            "B3 cold-start mix: 30%% tulu-3 + 25%% OpenR1-Math-220k (CoT) + "
            "15%% gsm8k-r1cot + 15%% ultrachat-200k + 15%% OpenMathInstruct-2. "
            "Combined instruction+CoT rebuild from gs138650."
        ),
    )
)


# ---------------------------------------------------------------------------
# b4_reweight_mix -- reweighted mix fixing the B3 dilution regression
# (docs/live/chains/sft/agpt/2b-mds/b4-finish-and-reweight/design.md)
# ---------------------------------------------------------------------------


def _build_b4_reweight_mix(seed: int = 42):
    """B4 reweighted mix -- fixes the B3 dilution regression
    (docs/live/chains/sft/agpt/2b-mds/b4-finish-and-reweight/design.md):
      0.40 gsm8k-r1cot        (was 0.15 in b3 -- restore in-distribution short CoT)
      0.15 OpenR1-Math-220k   (LENGTH-FILTERED via OPENR1_MAX_THINK_CHARS -- short
                               traces only; the run-on ones caused the regression)
      0.30 tulu-3-sft-mixture (general instruction-following)
      0.15 ultrachat-200k     (multi-turn chat)
    The b3 math-breadth component is dropped here (it added math breadth
    the 2B could not convert to accuracy). Same materialized-mix cache +
    all_exhausted interleave as b3.
    """
    return _materialized_mix_load_or_build(
        component_names=[
            "gsm8k-r1cot",
            "OpenR1-Math-220k",
            "tulu-3-sft-mixture",
            "ultrachat-200k",
        ],
        weights=[0.40, 0.15, 0.30, 0.15],
        seed=seed,
    )


register_sft_dataset(
    SFTDataset(
        name="b4_reweight_mix",
        build=_build_b4_reweight_mix,
        description=(
            "B4 reweighted mix: 40%% gsm8k-r1cot + 15%% length-filtered "
            "OpenR1-Math-220k + 30%% tulu-3 + 15%% ultrachat-200k "
            "(OpenMathInstruct-2 dropped). Fixes the B3 long-CoT dilution."
        ),
    )
)


# ---------------------------------------------------------------------------
# OpenThoughts-114k -- pre-distilled frontier reasoning traces (DeepSeek-R1),
# reformatted from the OpenThoughts <|begin_of_thought|>/<|begin_of_solution|>
# markers into our <think>/<answer>\boxed{} envelope. The breadth half of the
# reasoning-distillation cold-start (the distill_cot_mix recipe; no report
# written yet): OpenThoughts spans math + code + science reasoning, vs
# OpenR1-Math's math-only. Non-math math-only cold-starts were too narrow.
# ---------------------------------------------------------------------------

# OpenThoughts wraps its two spans with these literal markers (verified against
# the open-thoughts/OpenThoughts-114k 'default' config first-rows). We keep the
# thought verbatim and normalize the tail to a single <answer>\boxed{}</answer>.
_OT_THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL
)
_OT_SOLUTION_RE = re.compile(
    r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL
)

# Max <think> trace length (chars) kept from OpenThoughts. Frontier R1 traces
# are much longer than OpenR1-Math's, and the whole point of the distillation
# cold-start is to expose the 2B to that richer reasoning, so this defaults far
# more generous than OPENR1_MAX_THINK_CHARS (1200). It only drops pathological
# run-ons. Env-tunable: tighten it (e.g. 4000) at the GPU pre-tokenize step if
# the B4 verbose-dilution regression re-appears on this corpus.
DISTILL_MAX_THINK_CHARS = int(os.environ.get("DISTILL_MAX_THINK_CHARS", "16000"))


def _openthoughts_question(ex) -> str:
    """Extract the user problem text from an OpenThoughts 'conversations' row
    (list of {from, value}; the human turn is tagged from='user')."""
    for turn in ex.get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") in ("user", "human"):
            return str(turn.get("value", ""))
    return ""


def _openthoughts_format_row(ex):
    """Reformat one OpenThoughts row into our <think>/<answer> envelope, or
    return None if it can't be cleanly wrapped (caller drops None rows).

    Keeps the R1 reasoning trace (the <|begin_of_thought|> span) verbatim and
    normalizes the final answer to <answer>\\boxed{ans}</answer>, taking the
    LAST \\boxed{} anywhere in the assistant turn. Rows with no boxed answer are
    dropped: OpenThoughts' code-generation rows end in a code block, not a boxed
    scalar, so they don't fit an <answer>\\boxed{}</answer> tail and would teach
    a malformed envelope. This keeps the math/science-reasoning rows (which the
    GSM8K CoT eval rewards) and drops pure-code rows -- an intentional filter,
    not an accident.
    """
    question = _openthoughts_question(ex)
    if not question:
        return None
    assistant = ""
    for turn in ex.get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") == "assistant":
            assistant = str(turn.get("value", ""))
            break
    if not assistant:
        return None
    tm = _OT_THOUGHT_RE.search(assistant)
    trace = tm.group(1).strip() if tm else ""
    if not trace or len(trace) > DISTILL_MAX_THINK_CHARS:
        return None
    boxes = _OPENR1_BOXED_RE.findall(assistant)
    if not boxes:
        return None
    ans = boxes[-1].strip()
    if not ans:
        return None
    completion = f"<think>{trace}</think>\n<answer>\\boxed{{{ans}}}</answer>"
    return {
        "prompt": [{"role": "user", "content": question + _OPENR1_SUFFIX}],
        "completion": [{"role": "assistant", "content": completion}],
    }


def _openthoughts_map_row(ex):
    """map() fn for _build_openthoughts_cot: format or emit a same-shaped
    empty-content sentinel row (dropped by _openr1_filter_row). Mirrors
    _openr1_map_row so tests can exercise map+filter without load_dataset."""
    out = _openthoughts_format_row(ex)
    if out is not None:
        return out
    return {
        "prompt": [{"role": "user", "content": ""}],
        "completion": [{"role": "assistant", "content": ""}],
    }


def _build_openthoughts_cot():
    """open-thoughts/OpenThoughts-114k reformatted to the <think>/<answer>
    envelope. Selects rows with a reasoning trace + a boxed final answer;
    code-generation rows (no boxed scalar) are dropped. NOT decontaminated here
    -- decontam is applied per-mix (see _build_distill_cot_mix) so this raw
    formatted view stays reusable. Needs the HF download cached first (see
    scripts/sft/_pretokenize_distill_cot_mix_1n.sh)."""
    from datasets import load_dataset

    raw = load_dataset("open-thoughts/OpenThoughts-114k", "default", split="train")
    cols = raw.column_names
    mapped = raw.map(_openthoughts_map_row, remove_columns=cols)
    return mapped.filter(_openr1_filter_row)


register_sft_dataset(
    SFTDataset(
        name="OpenThoughts-114k",
        build=_build_openthoughts_cot,
        description=(
            "open-thoughts/OpenThoughts-114k R1-distilled reasoning traces "
            "(math + science + code), reformatted into the <think>/<answer>"
            "\\boxed{} envelope. Rows without a boxed final answer (pure-code "
            "generations) are dropped. The breadth half of distill_cot_mix."
        ),
    )
)


# ---------------------------------------------------------------------------
# distill_cot_mix -- reasoning-distillation cold-start mix
# (TODO: no report written yet -- the docs/ dir this named was never created)
# ---------------------------------------------------------------------------


def _decontaminate_against_gsm8k(ds, source: str, n: int = 13):
    """Drop rows whose question collides with the GSM8K TEST split (the eval).

    Runs the shared decontam detector (decontam_traces) over an already
    envelope-formatted {prompt, completion} dataset and logs the per-source
    drop count. The critical prereq for the distillation experiment: the
    distillation corpora are built from the same public math pools GSM8K was
    drawn from, so a copied test question would silently inflate the GSM8K CoT
    eval. See decontam_traces.py for the 13-gram + normalized-exact signals.
    """
    from torchtitan.experiments.ezpz.rl.decontam_traces import (
        filter_dataset,
        GSM8KDecontaminator,
        load_gsm8k_test_questions,
    )

    detector = GSM8KDecontaminator(load_gsm8k_test_questions(), n=n)
    kept, report = filter_dataset(ds, detector, num_proc=16)
    log.info(
        f"[decontam] {source}: dropped {report.dropped:,}/{report.total:,} "
        f"GSM8K-test-contaminated rows ({report.as_dict()['drop_rate']:.4%}); "
        f"by_reason={report.by_reason}; {report.kept:,} remain"
    )
    return kept


def _build_openthoughts_cot_decontam():
    """OpenThoughts-114k envelope-formatted + GSM8K-test-decontaminated."""
    return _decontaminate_against_gsm8k(_build_openthoughts_cot(), "OpenThoughts-114k")


def _build_openr1_math_cot_decontam():
    """OpenR1-Math-220k envelope-formatted + GSM8K-test-decontaminated."""
    return _decontaminate_against_gsm8k(_build_openr1_math_cot(), "OpenR1-Math-220k")


register_sft_dataset(
    SFTDataset(
        name="OpenThoughts-114k-decontam",
        build=_build_openthoughts_cot_decontam,
        description=(
            "OpenThoughts-114k in the <think>/<answer> envelope, with GSM8K-"
            "test-contaminated rows dropped (13-gram + exact-question match)."
        ),
    )
)

register_sft_dataset(
    SFTDataset(
        name="OpenR1-Math-220k-decontam",
        build=_build_openr1_math_cot_decontam,
        description=(
            "OpenR1-Math-220k in the <think>/<answer> envelope, with GSM8K-"
            "test-contaminated rows dropped (13-gram + exact-question match)."
        ),
    )
)


def _build_distill_cot_mix(
    weights=(0.5, 0.5),
    seed: int = 42,
):
    """Reasoning-distillation cold-start mix: broad frontier CoT traces
    (TODO: no report written yet for this recipe).

    Replaces the team's narrow math-only CoT cold-start with breadth from
    pre-distilled frontier reasoning traces:
      0.50 OpenThoughts-114k-decontam  (math + science + code R1 traces)
      0.50 OpenR1-Math-220k-decontam   (math-focused R1 traces)
    Both are re-wrapped into the <think>/<answer>\\boxed{} envelope (matching
    gsm8k-r1cot + the eval) and GSM8K-test-decontaminated BEFORE interleave, so
    a copied test question can never leak into training and inflate the GSM8K
    CoT eval. Uses the same materialized-mix cache + all_exhausted interleave as
    the other mixes; the decontam is baked into each component's registered
    build, so it is captured in the recipe hash (component names) -- a cache hit
    is guaranteed to be decontaminated.

    Optional additions (not wired in; add via a CLI mix-spec once cached):
    bespokelabs/Bespoke-Stratos-17k, NovaSky-AI/Sky-T1_data_17k.
    """
    if len(weights) != 2:
        raise ValueError(
            f"distill_cot_mix weights must be (w_openthoughts, w_openr1); "
            f"got {weights!r}"
        )
    return _materialized_mix_load_or_build(
        component_names=[
            "OpenThoughts-114k-decontam",
            "OpenR1-Math-220k-decontam",
        ],
        weights=list(weights),
        seed=seed,
    )


register_sft_dataset(
    SFTDataset(
        name="distill_cot_mix",
        build=_build_distill_cot_mix,
        description=(
            "Reasoning-distillation cold-start: 50%% OpenThoughts-114k + "
            "50%% OpenR1-Math-220k, both in the <think>/<answer>\\boxed{} "
            "envelope and GSM8K-test-decontaminated. Broad frontier CoT "
            "breadth replacing the math-only cold-start."
        ),
    )
)


# ---------------------------------------------------------------------------
# Generic mix-spec parser — `--sft_dataset 'a:0.5,b:0.3,c:0.2'`
# ---------------------------------------------------------------------------


def _is_mix_spec(name: str) -> bool:
    """Detect a CLI mix-spec string like 'tulu-3-sft-mixture:0.5,gsm8k:0.5'.

    A mix spec contains at least one ':' and at least one ',' (or a
    single entry like 'gsm8k:1.0' is also legal). A bare dataset name
    won't match because it has no ':'.
    """
    return ":" in name


def _parse_mix_spec(spec: str) -> tuple[list[str], list[float]]:
    """Parse 'a:0.5,b:0.3,c:0.2' into (['a','b','c'], [0.5,0.3,0.2])."""
    pairs = [p.strip() for p in spec.split(",") if p.strip()]
    names: list[str] = []
    weights: list[float] = []
    for pair in pairs:
        if ":" not in pair:
            raise ValueError(
                f"Mix-spec entry {pair!r} missing ':<weight>'. "
                f"Expected 'name:0.5' format; got {spec!r}"
            )
        n, w = pair.rsplit(":", 1)
        names.append(n.strip())
        try:
            wval = float(w.strip())
        except ValueError as e:
            raise ValueError(
                f"Mix-spec weight {w!r} (in {pair!r}) isn't a valid float"
            ) from e
        # float() accepts 'NaN' and 'inf'; reject both — they'd silently
        # poison interleave_datasets' probabilities.
        if not (wval == wval) or wval == float("inf") or wval == float("-inf"):
            raise ValueError(
                f"Mix-spec weight {w!r} (in {pair!r}) isn't finite"
            )
        if wval < 0:
            raise ValueError(
                f"Mix-spec weight {w!r} (in {pair!r}) is negative"
            )
        weights.append(wval)
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"Mix-spec weights sum to {total}; need > 0")
    # Renormalize so the user doesn't have to make them sum to 1
    weights = [w / total for w in weights]
    return names, weights


def _build_ad_hoc_mix(spec: str, seed: int = 42) -> Dataset:
    """Build a one-off interleaved mix from a CLI spec string.

    Each component name must already be in SFT_REGISTRY. Components
    are interleaved with their (renormalized) weights via
    `interleave_datasets(stopping_strategy='all_exhausted')`, same
    as the canonical hardcoded mixes.
    """
    from datasets import interleave_datasets

    names, weights = _parse_mix_spec(spec)
    for n in names:
        if n not in SFT_REGISTRY:
            available = ", ".join(sorted(SFT_REGISTRY))
            raise ValueError(
                f"Mix-spec references unknown dataset {n!r}. "
                f"Available: {available}"
            )
    return _materialized_mix_load_or_build(
        component_names=names,
        weights=weights,
        seed=seed,
    )
