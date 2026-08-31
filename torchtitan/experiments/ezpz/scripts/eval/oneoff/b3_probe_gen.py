"""B3 diagnostic: print RAW vLLM generations for a few GSM8K CoT prompts.
Reuses eval_cot_gsm8k.py exactly (same prompt suffix, chat template, extraction).
All vLLM work is inside main() (vLLM v1 uses mp spawn -> needs __main__ guard).
"""
import sys
sys.path.insert(0, "torchtitan/experiments/ezpz/scripts/eval")
from eval_cot_gsm8k import build_prompts, extract_cot_answer, gold_answer


def main():
    MODEL = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_dataset("gsm8k", "main", split="test").select(range(N))
    questions = [r["question"] for r in ds]
    golds = [gold_answer(r["answer"]) for r in ds]

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = build_prompts(tok, questions)
    print("=== tokenizer eos_token_id:", tok.eos_token_id, " bos:", tok.bos_token_id)

    llm = LLM(model=MODEL, dtype="float32", gpu_memory_utilization=0.70,
              enforce_eager=True, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=768,
                        stop=["</answer>"], include_stop_str_in_output=True)
    outs = llm.generate(prompts, sp)

    for i, o in enumerate(outs):
        text = o.outputs[0].text
        fmt_ok, ans = extract_cot_answer(text)
        fin = o.outputs[0].finish_reason
        print("\n" + "#" * 78)
        print("### EXAMPLE %d  gold=%s  pred=%s  format_ok=%s  correct=%s  finish=%s  gen_chars=%d ntok=%d"
              % (i, golds[i], ans, fmt_ok, (ans is not None and ans == golds[i]),
                 fin, len(text), len(o.outputs[0].token_ids)))
        print("### QUESTION:", questions[i][:400])
        print("### RAW GENERATION >>>")
        print(text)
        print("### <<< END RAW")


if __name__ == "__main__":
    main()
