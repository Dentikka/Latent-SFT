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
Latent phases run batched (rows that already left the phase keep stepping, unused). The
latent phase is deterministic (no Gumbel noise, exit by argmax, as in their HF eval); only
the text after it is decoded greedily (--temperature 0) or sampled --samples times
(paper protocol: --temperature 0.6 --top_p 0.95), reusing one latent phase per problem.
--latent_noise S switches the latent phase to their sampler with Gumbel noise (sglang
layers/sampler.py, add_noise_gumbel_softmax: top-p mask keeping at least top-K, noise clamped to
[-1.5, 3] and scaled by S, top-K by log p + noise, mixture softmax((log p + noise) / T), exit when
the noisy top-1 is the end token), as in Latent-GRPO rollouts; --latent_samples K runs K noisy
latent phases per problem (records carry "sample").

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


def prompt_ids(tok, problem: str, mode: str = "user") -> list:
    """Chat template + generation prompt, then `<think>`. mode "user": the instruction opens the
    user turn (the template still adds its default system prompt); mode "system": the instruction
    is the system prompt and the user turn is the bare problem, as in Stage-2 training
    (src/stage2/data.py, Qwen branch) and their eval (eval/eval_high_tasks_sglang.py).
    A template that already ends with `<think>` (the released checkpoint's, meant for GRPO)
    is cut back so the prompt carries exactly one."""
    msgs = ([{"role": "system", "content": INSTR.strip()}, {"role": "user", "content": problem}]
            if mode == "system" else [{"role": "user", "content": INSTR + problem}])
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text = text[:-len("<think>")] if text.endswith("<think>") else text
    return tok(text + "<think>", add_special_tokens=False)["input_ids"]


@torch.no_grad()
def latent_phase(model, emb, batch_ids, end_id, topk, max_latent, pad_id, trace=None, noise=None):
    """Batched latent phase. Returns per-row lists of latent embeddings [k, h] and exit flags.
    With a list `trace`, appends per row {"p_end": [...], "top1": [...]}: p(end_id) and the argmax
    at every step the row is still in the phase (the exit step included).
    noise: None (deterministic, their eval default) or {"scale", "temp", "top_p"} (see the module doc)."""
    dev = emb.weight.device
    L = max(len(x) for x in batch_ids)
    ids = torch.tensor([[pad_id] * (L - len(x)) + x for x in batch_ids], device=dev)
    mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in batch_ids], device=dev)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    x, past = emb(ids), None
    B = len(batch_ids)
    lat = [[] for _ in range(B)]
    tr = [{"p_end": [], "top1": []} for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(max_latent):
        out = model(inputs_embeds=x, attention_mask=mask, position_ids=pos,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        if noise is None:
            tl, ti = logits.topk(topk, dim=-1)
            w = torch.softmax(tl.float(), -1)
        else:
            tl, ti, w = noisy_topk(logits, topk, noise)
        exits = ti[:, 0] == end_id
        if trace is not None:
            p_end = torch.softmax(logits.float(), -1)[:, end_id].tolist()
            top1 = logits.argmax(-1).tolist()
            for b in range(B):
                if not done[b]:
                    tr[b]["p_end"].append(round(p_end[b], 5))
                    tr[b]["top1"].append(top1[b])
        mix = (F.embedding(ti, emb.weight) * w.to(emb.weight.dtype).unsqueeze(-1)).sum(1)
        for b in range(B):
            if not done[b] and not exits[b]:
                lat[b].append(mix[b])
        done |= exits
        if bool(done.all()):
            break
        x = mix.unsqueeze(1)
        mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=dev)], dim=1)
        pos = pos[:, -1:] + 1
    if trace is not None:
        trace.extend(tr)
    return [torch.stack(v) if v else emb.weight.new_zeros(0, emb.weight.shape[1]) for v in lat], done.tolist()


def noisy_topk(logits, topk, noise):
    """Their Gumbel latent sampler (sglang layers/sampler.py) for one step: top-K indices by
    log p + noise inside the top-p set (at least K kept) and the mixture weights."""
    logp = torch.log_softmax(logits.float(), -1)
    if noise["top_p"] < 1.0:
        sp, si = torch.sort(logp.exp(), descending=True, dim=-1)
        keep = (sp.cumsum(-1) - sp) < noise["top_p"]
        keep[:, :topk] = True
        logp = logp.masked_fill(~torch.zeros_like(keep).scatter_(1, si, keep), float("-inf"))
    g = (-torch.empty_like(logp).exponential_().log()).clamp(-1.5, 3.0) * noise["scale"]
    score, ti = (logp + g).topk(topk, dim=-1)
    return score, ti, torch.softmax(score / noise["temp"], -1)


@torch.no_grad()
def answer(model, tok, rows, max_new, temperature=0.0, top_p=1.0):
    """Continuation of per-row embedding sequences (left-padded); greedy at temperature 0."""
    emb = model.get_input_embeddings()
    dev = emb.weight.device
    L = max(r.shape[0] for r in rows)
    h = emb.weight.shape[1]
    x = torch.zeros(len(rows), L, h, dtype=emb.weight.dtype, device=dev)
    mask = torch.zeros(len(rows), L, dtype=torch.long, device=dev)
    for i, r in enumerate(rows):
        x[i, L - r.shape[0]:] = r
        mask[i, L - r.shape[0]:] = 1
    sample = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 \
        else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    out = model.generate(inputs_embeds=x, attention_mask=mask, max_new_tokens=max_new,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id, **sample)
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
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; paper protocol 0.6")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--samples", type=int, default=1, help="answers per problem (latents computed once)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", default=None, help="LoRA adapter merged into --model before the eval")
    ap.add_argument("--prompt", choices=["user", "system"], default="user",
                    help="instruction in the user turn (default, earlier evals) or as the system prompt (training format)")
    ap.add_argument("--trace", action="store_true", help="store per-step p(</) and argmax ids of the latent phase")
    ap.add_argument("--start", type=int, default=0, help="skip the first START problems")
    ap.add_argument("--latent_noise", type=float, default=0.0, help="Gumbel noise scale of the latent phase (0 = off)")
    ap.add_argument("--latent_noise_temp", type=float, default=1.0)
    ap.add_argument("--latent_top_p", type=float, default=0.95)
    ap.add_argument("--latent_samples", type=int, default=1, help="noisy latent phases per problem")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(dev).eval()
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora).merge_and_unload().eval()
    emb = model.get_input_embeddings()
    end_ids = tok("</think>", add_special_tokens=False)["input_ids"]
    pre_ids = tok(args.prefix, add_special_tokens=False)["input_ids"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    data = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    data = data[args.start:args.start + args.n] if args.n else data[args.start:]
    for i, d in enumerate(data):
        d["_idx"] = args.start + i
    if args.latent_samples > 1:
        assert args.latent_noise > 0, "--latent_samples needs --latent_noise (the phase is otherwise deterministic)"
        data = [dict(d, _sample=k) for d in data for k in range(args.latent_samples)]
    noise = ({"scale": args.latent_noise, "temp": args.latent_noise_temp, "top_p": args.latent_top_p}
             if args.latent_noise > 0 else None)
    torch.manual_seed(args.seed)
    order = sorted(range(len(data)), key=lambda i: len(data[i]["problem"]))
    arms = args.arms.split(",")
    logger.info("model %s | %d problems | arms %s | </think> ids %s | prefix ids %s",
                args.model, len(data), arms, end_ids, pre_ids)

    recs = {i: {"idx": data[i]["_idx"], "answer": data[i]["answer"]} for i in range(len(data))}
    for i, d in enumerate(data):
        if "_sample" in d:
            recs[i]["sample"] = d["_sample"]
    t0 = time.time()
    for s in range(0, len(order), args.batch):
        idx = order[s:s + args.batch]
        pids = [prompt_ids(tok, data[i]["problem"], args.prompt) for i in idx]
        P = [emb(torch.tensor(p, device=emb.weight.device)) for p in pids]
        E = emb(torch.tensor(end_ids, device=emb.weight.device))
        R = emb(torch.tensor(pre_ids, device=emb.weight.device))
        if "forced" in arms or "free" in arms:
            tr = [] if args.trace else None
            lats, exited = latent_phase(model, emb, pids, end_ids[0], args.topk, args.max_latent, pad_id, tr, noise)
            for j, (i, z, ok) in enumerate(zip(idx, lats, exited)):
                recs[i]["n_latent"], recs[i]["exited"] = z.shape[0], ok
                if tr is not None:
                    recs[i].update(tr[j])
        plan = {"forced": ([torch.cat([p, z, E, R]) for p, z in zip(P, lats)] if "forced" in arms else None,
                           args.max_new_forced, args.prefix),
                "floor": ([torch.cat([p, E, R]) for p in P] if "floor" in arms else None,
                          args.max_new_forced, args.prefix),
                "free": ([torch.cat([p, z, E]) for p, z in zip(P, lats)] if "free" in arms else None,
                         args.max_new_free, "")}
        for arm in arms:
            rows, max_new, head = plan[arm]
            for k in range(args.samples):
                torch.manual_seed(args.seed + k)
                outs = answer(model, tok, rows, max_new, args.temperature, args.top_p)
                for i, o in zip(idx, outs):
                    recs[i].setdefault(arm, []).append(head + o)
        logger.info("batch %d/%d done, %.0f s", s // args.batch + 1, -(-len(order) // args.batch), time.time() - t0)

    summary = {"model": args.model, "lora": args.lora, "n": len(data), "prefix": args.prefix,
               "prompt": args.prompt, "max_latent": args.max_latent, "start": args.start,
               "latent_noise": args.latent_noise, "latent_samples": args.latent_samples,
               "temperature": args.temperature, "top_p": args.top_p, "samples": args.samples}
    for arm in arms:
        per_sample = []
        for r in recs.values():
            r[arm + "_ok"] = [bool(check_is_correct(extract_answer(t), r["answer"])) for t in r[arm]]
        for k in range(args.samples):
            per_sample.append(sum(r[arm + "_ok"][k] for r in recs.values()) / len(recs))
        summary[arm] = sum(per_sample) / len(per_sample)
        summary[arm + "_per_sample"] = per_sample
    lat = [r["n_latent"] for r in recs.values() if "n_latent" in r]
    if lat:
        summary["mean_latent"] = sum(lat) / len(lat)
        summary["not_exited"] = sum(not r["exited"] for r in recs.values())
        if "forced" in arms:  # honest: rows that never left the latent phase count as wrong
            summary["honest"] = sum(r["exited"] and r["forced_ok"][0] for r in recs.values()) / len(recs)
    with open(os.path.join(args.out, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in recs.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    logger.info("RESULT %s", json.dumps(summary))


if __name__ == "__main__":
    main()
