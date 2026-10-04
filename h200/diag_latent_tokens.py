"""Does the encoder carry the final answer into the latents? (CPU, encoder only)

For the first N Math-500 problems the Stage-1 encoder compresses the gold `solution`
exactly as in h200/eval_box.py (r tokens per latent, top-K mixture over the decoder's input
embeddings). Each latent's mixture is read directly — its top-K token ids and weights — and
compared with the r solution tokens it compresses:

  fidelity  — share of latents whose top-1 / top-K contains a token of its own chunk;
  answer    — for the latents covering the content of the last \\boxed{...}: share of answer
              tokens found in the top-1 / top-K of their latent, and problems where all are;
  sharpness — mean top-1 weight and N_eff = 1 / sum(p^2) of the mixtures;
  spectrum  — the most frequent top-1 tokens over all latents.

--compare ENC2 runs a second encoder (e.g. the untrained out/base_model) on the same latents
and reports top-1 agreement and top-K set overlap: if training only reweights the initial
top-K sets, the top-k selection blocks gradient to every other token. Encoders run one
after another to stay inside the CPU-queue memory cap.

With --eval <eval_box latents jsonl> the decoder's correctness is split by whether the
answer survived in the first encoder's latents.

Usage: python h200/diag_latent_tokens.py --encoder ENC_HF --base BASE_DIR --data Math-500-test.jsonl
           [--compare ENC2_DIR] [--tokenizer TOK_DIR] [--eval EVAL.jsonl] [--n 100] [--r 2] [--topk 10]
"""
import argparse
import gc
import json
import logging
import os
import sys
from collections import Counter

import torch
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from h200.make_honest_data import last_boxed  # noqa: E402
from src.stage1.data import build_latent_token_induction_mask, insert_special_token_every_k  # noqa: E402

logger = logging.getLogger(__name__)
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def load_embedding(base: str, key: str = "model.embed_tokens.weight") -> torch.Tensor:
    """One weight matrix (default: the decoder's input embedding) read alone from the base safetensors."""
    index_path = os.path.join(base, "model.safetensors.index.json")
    shard = (json.load(open(index_path))["weight_map"][key] if os.path.exists(index_path)
             else "model.safetensors")
    with safe_open(os.path.join(base, shard), framework="pt") as f:
        return f.get_tensor(key)


def prepare(tok, rows, r):
    comp = tok.convert_tokens_to_ids("<|compress_token|>")
    t_open, t_close = tok(["<think>", "</think>"], add_special_tokens=False)["input_ids"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    items = []
    for row in rows:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["problem"]}]
        prefix = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                     add_special_tokens=False)["input_ids"]
        sol = tok(row["solution"], add_special_tokens=False, return_offsets_mapping=True)
        suffix, count, _ = insert_special_token_every_k(sol["input_ids"], comp, -100, r)
        ids = torch.tensor([prefix + t_open + suffix + t_close])
        items.append({"ids": ids, "mask": build_latent_token_induction_mask(ids, [comp], pad, torch.bfloat16),
                      "content": sol["input_ids"], "offs": sol["offset_mapping"], "count": count})
    return items, comp


def run_encoder(path, items, comp, E, topk, R=None):
    """Top-K ids [count, K] and weights for every problem; the model is freed afterwards."""
    enc = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    out = []
    with torch.no_grad():
        for n, it in enumerate(items):
            h = enc(it["ids"], attention_mask=it["mask"]).last_hidden_state[0]
            h = h[(it["ids"][0] == comp).nonzero().squeeze(-1)]
            sel = E if R is None else R        # token selection: input embeddings (repo) or lm_head
            tv, ti = (h.to(sel.dtype) @ sel.T).float().topk(topk, dim=-1)
            out.append((ti, tv.softmax(-1)))
            if (n + 1) % 20 == 0:
                logger.info("%s: %d/%d", os.path.basename(path.rstrip("/")), n + 1, len(items))
    del enc
    gc.collect()
    return out


def report(name, tok, rows, items, lat, r, topk):
    fid1 = fidk = nlat = ans1 = ansk = nans = nxt1 = nxtk = nnxt = 0
    top1w, neff, full, examples = [], [], [], []
    spectrum = Counter()
    for row, it, (ti, p) in zip(rows, items, lat):
        content = it["content"]
        top1w += p[:, 0].tolist()
        neff += (1.0 / (p ** 2).sum(-1)).tolist()
        spectrum.update(ti[:, 0].tolist())
        for k in range(it["count"]):
            chunk = set(content[k * r:(k + 1) * r])
            nlat += 1
            fid1 += int(ti[k, 0].item() in chunk)
            fidk += int(bool(chunk & set(ti[k].tolist())))
            if (k + 1) * r < len(content):                 # first token of the next chunk
                nnxt += 1
                nxt1 += int(ti[k, 0].item() == content[(k + 1) * r])
                nxtk += int(content[(k + 1) * r] in ti[k].tolist())
        box = last_boxed(row["solution"])
        ok_all = None
        if box is not None:
            a0, a1 = box[0] + len("\\boxed{"), box[1] - 1
            pos = [j for j, (s, e) in enumerate(it["offs"]) if s < a1 and e > a0]
            if pos:
                hit1 = [content[j] == ti[j // r, 0].item() for j in pos]
                ans1 += sum(hit1)
                ansk += sum(content[j] in ti[j // r].tolist() for j in pos)
                nans += len(pos)
                ok_all = all(hit1)
                if len(examples) < 5:
                    ks = sorted({j // r for j in pos})
                    examples.append((row["answer"], [[tok.decode([t]) for t in ti[k, :3].tolist()] for k in ks]))
        full.append(ok_all)
    logger.info("RESULT [%s] fidelity: top-1 %.3f, top-%d %.3f over %d latents", name, fid1 / nlat, topk, fidk / nlat, nlat)
    logger.info("RESULT [%s] next-chunk first token: top-1 %.3f, top-%d %.3f", name, nxt1 / max(1, nnxt), topk, nxtk / max(1, nnxt))
    logger.info("RESULT [%s] answer tokens: top-1 %.3f, top-%d %.3f over %d tokens; all answer tokens in top-1: %.3f",
                name, ans1 / nans, topk, ansk / nans, nans, sum(bool(x) for x in full) / max(1, sum(x is not None for x in full)))
    logger.info("RESULT [%s] sharpness: mean top-1 weight %.3f, mean N_eff %.2f of %d",
                name, sum(top1w) / len(top1w), sum(neff) / len(neff), topk)
    logger.info("RESULT [%s] top-1 spectrum: %s", name,
                ", ".join(f"{tok.decode([t])!r} {c / nlat:.1%}" for t, c in spectrum.most_common(10)))
    for a, ex in examples:
        logger.info("  [%s] answer %r -> latents' top-3: %s", name, a, ex)
    return full


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--compare", default=None, help="second encoder dir, e.g. the untrained out/base_model")
    ap.add_argument("--tokenizer", default=None, help="tokenizer dir (default: --encoder)")
    ap.add_argument("--base", required=True, help="decoder base dir (its embedding builds the mixture)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--readout", choices=["embed", "lm_head"], default="embed",
                    help="select the top-K tokens by h @ E_in (repo) or by h @ W_lm_head (the model's own logits)")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.encoder)
    E = load_embedding(args.base).to(torch.bfloat16)
    R = load_embedding(args.base, "lm_head.weight").to(torch.bfloat16) if args.readout == "lm_head" else None
    logger.info("readout: %s", args.readout)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][:args.n]
    items, comp = prepare(tok, rows, args.r)

    lat_a = run_encoder(args.encoder, items, comp, E, args.topk, R)
    full = report("A", tok, rows, items, lat_a, args.r, args.topk)
    if args.eval:
        ev = [json.loads(l) for l in open(args.eval, encoding="utf-8")][:args.n]
        for flag in (True, False):
            sel = [e["correct"] for e, f in zip(ev, full) if f is flag]
            if sel:
                logger.info("RESULT decoder acc when answer %s in A's top-1: %.3f (n=%d)",
                            "IS" if flag else "is NOT", sum(sel) / len(sel), len(sel))
    if args.compare:
        lat_b = run_encoder(args.compare, items, comp, E, args.topk, R)
        report("B", tok, rows, items, lat_b, args.r, args.topk)
        agree = overlap = n = 0
        for (ta, _), (tb, _) in zip(lat_a, lat_b):
            agree += (ta[:, 0] == tb[:, 0]).sum().item()
            overlap += sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ta, tb))
            n += ta.shape[0]
        logger.info("RESULT A vs B: top-1 agreement %.3f, mean top-%d set overlap %.2f of %d over %d latents",
                    agree / n, args.topk, overlap / n, args.topk, n)


if __name__ == "__main__":
    main()
