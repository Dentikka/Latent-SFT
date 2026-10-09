"""Exit diagnostics of a Stage-2 latent model on training rows: is the `</think>` exit learned?

Two measurements on the same rows (a slice of the Stage-2 train set with its soft labels, see
h200/export_labels_slice.py):

  teacher-forced — the input is built exactly as in Stage-2 training (src/stage2/data.py, Qwen
      branch): prompt, `<think>`, the gold latents as top-K mixtures of the soft labels, then
      `</think>`. Read p(`</`) at every latent position; the last one is where training puts the
      exit. Arms: clean labels, and labels with the training Gumbel noise.
  free — the model runs its own latent phase from the same prompt (as in eval_forced_latent.py)
      and we record whether / where it exits, against the gold latent count.

Exit learned under teacher forcing but missed in the free run -> the model's own latents drift
away from the training states. Missed even under teacher forcing -> the exit is under-trained.

Usage: python h200/diag_exit_tf.py --model M [--lora A] --rows rows.jsonl --labels labels.pt
           --out DIR [--n 128] [--max_latent 4064] [--batch 32]
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_forced_latent import INSTR, latent_phase, prompt_ids  # noqa: E402

logger = logging.getLogger(__name__)


def train_prefix_ids(tok, problem: str) -> list:
    """Prompt + `<think>` exactly as src/stage2/data.py builds it for Qwen (separate tokenization)."""
    msgs = [{"role": "system", "content": INSTR.strip()}, {"role": "user", "content": problem}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"] + tok("<think>", add_special_tokens=False)["input_ids"]


def gumbel_safe(probs: torch.Tensor, idx: torch.Tensor, end_id: int, gen: torch.Generator) -> torch.Tensor:
    """Training noise (Stage2Dataset.apply_gumbel_noise_safe, scale 1, temperature 1): resample
    until no position has the exit token as top-1."""
    logp = torch.log(probs.float() + 1e-10)
    for _ in range(100):
        g = -torch.empty(logp.shape).exponential_(generator=gen).log()
        noisy = torch.softmax(logp + g.clamp(-1.5, 3.0), -1)
        if not (idx.gather(-1, noisy.argmax(-1, keepdim=True)) == end_id).any():
            break
    return noisy.to(probs.dtype)


@torch.no_grad()
def teacher_forced(model, emb, pre: list, probs, idx, end_ids: list) -> list:
    """(p(`</`), argmax == `</`) at every latent prediction position: logits[s-1+j] predicts
    latent j, and the logits at the last latent (s-1+k) predict `</`. Returns k + 1 pairs."""
    dev = emb.weight.device
    W = emb.weight
    lat = (F.embedding(idx.to(dev), W) * probs.to(dev, W.dtype).unsqueeze(-1)).sum(-2)
    x = torch.cat([emb(torch.tensor(pre, device=dev)), lat, emb(torch.tensor(end_ids, device=dev))])
    logits = model(inputs_embeds=x.unsqueeze(0)).logits[0].float()
    s, k = len(pre), lat.shape[0]
    lg = logits[s - 1:s + k]
    p = torch.softmax(lg, -1)[:, end_ids[0]].tolist()
    top = (lg.argmax(-1) == end_ids[0]).tolist()
    return list(zip(p, top))


def summarize(pt: list) -> dict:
    """pt has k+1 (p, is_argmax) pairs: k latent predictions, then the gold exit position."""
    before, (p_end, top_end) = pt[:-1], pt[-1]
    return {"p_end_gold": round(p_end, 4), "exit_at_gold": top_end,
            "max_p_before": round(max(v for v, _ in before), 4) if before else 0.0,
            "early_exit": any(t for _, t in before)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--rows", required=True, help="train rows (jsonl with problem, cot_answer)")
    ap.add_argument("--labels", required=True, help="torch list of (probs [k, K], indices [k, K]) per row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_latent", type=int, default=4064)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(dev).eval()
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora).merge_and_unload().eval()
    emb = model.get_input_embeddings()
    end_ids = tok("</think>", add_special_tokens=False)["input_ids"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = [json.loads(line) for line in open(args.rows, encoding="utf-8")][:args.n]
    labels = torch.load(args.labels, map_location="cpu")[:len(rows)]
    gen = torch.Generator().manual_seed(args.seed)

    recs, same_prompt = [], 0
    t0 = time.time()
    for i, (row, (probs, idx)) in enumerate(zip(rows, labels)):
        pre = train_prefix_ids(tok, row["problem"])
        same_prompt += pre == prompt_ids(tok, row["problem"], "system")
        rec = {"i": i, "k_gold": int(probs.shape[0])}
        rec["clean"] = summarize(teacher_forced(model, emb, pre, probs, idx, end_ids))
        rec["noisy"] = summarize(teacher_forced(model, emb, pre, gumbel_safe(probs, idx, end_ids[0], gen), idx, end_ids))
        recs.append(rec)
        if (i + 1) % 32 == 0:
            logger.info("teacher-forced %d/%d, %.0f s", i + 1, len(rows), time.time() - t0)
    logger.info("eval prompt (--prompt system) == training prompt for %d/%d rows", same_prompt, len(rows))

    for s in range(0, len(rows), args.batch):
        pids = [train_prefix_ids(tok, r["problem"]) for r in rows[s:s + args.batch]]
        tr = []
        lats, exited = latent_phase(model, emb, pids, end_ids[0], args.topk, args.max_latent, pad_id, tr)
        for j, (z, ok, t) in enumerate(zip(lats, exited, tr)):
            recs[s + j]["free"] = {"exited": ok, "n_latent": int(z.shape[0]),
                                   "max_p_end": round(max(t["p_end"]), 4), "p_end": t["p_end"]}
        logger.info("free run %d/%d, %.0f s", min(s + args.batch, len(rows)), len(rows), time.time() - t0)

    n = len(recs)
    mean = lambda xs: round(sum(xs) / max(len(xs), 1), 4)  # noqa: E731
    summary = {"model": args.model, "lora": args.lora, "n": n, "max_latent": args.max_latent,
               "same_prompt": same_prompt}
    for arm in ("clean", "noisy"):
        summary[arm] = {"exit_argmax_at_gold": mean([r[arm]["exit_at_gold"] for r in recs]),
                        "p_end_gold_mean": mean([r[arm]["p_end_gold"] for r in recs]),
                        "early_exit": mean([r[arm]["early_exit"] for r in recs])}
    fr = [r["free"] for r in recs]
    ex = [(f["n_latent"], r["k_gold"]) for f, r in zip(fr, recs) if f["exited"]]
    summary["free"] = {"exited": mean([f["exited"] for f in fr]),
                       "exited_by_2048": mean([f["exited"] and f["n_latent"] <= 2048 for f in fr]),
                       "len_ratio_exited": mean([a / b for a, b in ex]),
                       "max_p_end_not_exited": mean([f["max_p_end"] for f in fr if not f["exited"]])}
    with open(os.path.join(args.out, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    logger.info("RESULT %s", json.dumps(summary))


if __name__ == "__main__":
    main()
