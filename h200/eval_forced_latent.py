"""Honest-mode eval of a Stage-2 latent model (e.g. the released Latent-SFT checkpoint).

The model runs its own latent phase exactly as in their HF generator
(`LatentSFTStage2SoftEmbedding.one_example_generate_hf`): at every step the top-K of the
output logits gives a softmax mixture of input embeddings that becomes the next input, and
the phase ends when the argmax is the first token of `</think>`. Then, per arm:

  forced — `</think>` + PREFIX is appended and the model writes only the answer (greedy,
           a few tokens): no explicit reasoning after the latent phase;
  floor  — no latent phase at all: `<think></think>` + PREFIX, same answer budget;
  free   — `</think>` and the model continues freely (their normal mode), greedy.

PREFIX defaults to the one of the forced-answer study on their Latent-GRPO
("\\n\\nTherefore, the answer is \\boxed{"), so the numbers compare with 27.9 / 27.3.
Latent phases run batched (rows that already left the phase keep stepping, unused).
Greedy throughout: the mixture is deterministic and decoding is argmax.

Usage: python h200/eval_forced_latent.py --model M --data Math-500-test.jsonl --out DIR
           [--arms forced,floor,free] [--n 500] [--batch 64] [--max_latent 2048]
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_utils.grader import check_is_correct  # noqa: E402
from eval_utils.parser import extract_answer  # noqa: E402

logger = logging.getLogger(__name__)
INSTR = "Please reason step by step, and put your final answer within \\boxed{}.\n"
PREFIX = "\n\nTherefore, the answer is \\boxed{"


def prompt_ids(tok, problem: str) -> list:
    """Their training/eval prompt: chat template + generation prompt, then `<think>`.
    A template that already ends with `<think>` (the released checkpoint's, meant for GRPO)
    is cut back so the prompt carries exactly one."""
    text = tok.apply_chat_template([{"role": "user", "content": INSTR + problem}],
                                   tokenize=False, add_generation_prompt=True)
    text = text[:-len("<think>")] if text.endswith("<think>") else text
    return tok(text + "<think>", add_special_tokens=False)["input_ids"]


@torch.no_grad()
def latent_phase(model, emb, batch_ids, end_id, topk, max_latent, pad_id):
    """Batched latent phase. Returns per-row lists of latent embeddings [k, h] and exit flags."""
    dev = emb.weight.device
    L = max(len(x) for x in batch_ids)
    ids = torch.tensor([[pad_id] * (L - len(x)) + x for x in batch_ids], device=dev)
    mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in batch_ids], device=dev)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    x, past = emb(ids), None
    B = len(batch_ids)
    lat = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(max_latent):
        out = model(inputs_embeds=x, attention_mask=mask, position_ids=pos,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        exits = logits.argmax(-1) == end_id
        tl, ti = logits.topk(topk, dim=-1)
        mix = (F.embedding(ti, emb.weight) * torch.softmax(tl.float(), -1).to(emb.weight.dtype).unsqueeze(-1)).sum(1)
        for b in range(B):
            if not done[b] and not exits[b]:
                lat[b].append(mix[b])
        done |= exits
        if bool(done.all()):
            break
        x = mix.unsqueeze(1)
        mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=dev)], dim=1)
        pos = pos[:, -1:] + 1
    return [torch.stack(v) if v else emb.weight.new_zeros(0, emb.weight.shape[1]) for v in lat], done.tolist()


@torch.no_grad()
def answer(model, tok, rows, max_new):
    """Greedy continuation of per-row embedding sequences (left-padded)."""
    emb = model.get_input_embeddings()
    dev = emb.weight.device
    L = max(r.shape[0] for r in rows)
    h = emb.weight.shape[1]
    x = torch.zeros(len(rows), L, h, dtype=emb.weight.dtype, device=dev)
    mask = torch.zeros(len(rows), L, dtype=torch.long, device=dev)
    for i, r in enumerate(rows):
        x[i, L - r.shape[0]:] = r
        mask[i, L - r.shape[0]:] = 1
    out = model.generate(inputs_embeds=x, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return [tok.decode(o, skip_special_tokens=True) for o in out]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="forced,floor,free")
    ap.add_argument("--n", type=int, default=0, help="first N problems (0 = all)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_latent", type=int, default=2048)
    ap.add_argument("--max_new_forced", type=int, default=32)
    ap.add_argument("--max_new_free", type=int, default=4096)
    ap.add_argument("--prefix", default=PREFIX)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(dev).eval()
    emb = model.get_input_embeddings()
    end_ids = tok("</think>", add_special_tokens=False)["input_ids"]
    pre_ids = tok(args.prefix, add_special_tokens=False)["input_ids"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    data = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    data = data[:args.n] if args.n else data
    order = sorted(range(len(data)), key=lambda i: len(data[i]["problem"]))
    arms = args.arms.split(",")
    logger.info("model %s | %d problems | arms %s | </think> ids %s | prefix ids %s",
                args.model, len(data), arms, end_ids, pre_ids)

    recs = {i: {"idx": i, "answer": data[i]["answer"]} for i in range(len(data))}
    t0 = time.time()
    for s in range(0, len(order), args.batch):
        idx = order[s:s + args.batch]
        pids = [prompt_ids(tok, data[i]["problem"]) for i in idx]
        P = [emb(torch.tensor(p, device=emb.weight.device)) for p in pids]
        E = emb(torch.tensor(end_ids, device=emb.weight.device))
        R = emb(torch.tensor(pre_ids, device=emb.weight.device))
        if "forced" in arms or "free" in arms:
            lats, exited = latent_phase(model, emb, pids, end_ids[0], args.topk, args.max_latent, pad_id)
            for i, z, ok in zip(idx, lats, exited):
                recs[i]["n_latent"], recs[i]["exited"] = z.shape[0], ok
        if "forced" in arms:
            outs = answer(model, tok, [torch.cat([p, z, E, R]) for p, z in zip(P, lats)], args.max_new_forced)
            for i, o in zip(idx, outs):
                recs[i]["forced"] = args.prefix + o
        if "floor" in arms:
            outs = answer(model, tok, [torch.cat([p, E, R]) for p in P], args.max_new_forced)
            for i, o in zip(idx, outs):
                recs[i]["floor"] = args.prefix + o
        if "free" in arms:
            outs = answer(model, tok, [torch.cat([p, z, E]) for p, z in zip(P, lats)], args.max_new_free)
            for i, o in zip(idx, outs):
                recs[i]["free"] = o
        logger.info("batch %d/%d done, %.0f s", s // args.batch + 1, -(-len(order) // args.batch), time.time() - t0)

    summary = {"model": args.model, "n": len(data), "prefix": args.prefix}
    for arm in arms:
        ok = [bool(check_is_correct(extract_answer(r[arm]), r["answer"])) for r in recs.values()]
        for r, c in zip(recs.values(), ok):
            r[arm + "_ok"] = c
        summary[arm] = sum(ok) / len(ok)
    lat = [r["n_latent"] for r in recs.values() if "n_latent" in r]
    if lat:
        summary["mean_latent"] = sum(lat) / len(lat)
        summary["not_exited"] = sum(not r["exited"] for r in recs.values())
    with open(os.path.join(args.out, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in recs.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    logger.info("RESULT %s", json.dumps(summary))


if __name__ == "__main__":
    main()
