"""Does the encoder carry the final answer into the latents? (CPU, encoder only)

For the first N Math-500 problems the Stage-1 encoder compresses the gold `solution`
exactly as in h200/eval_box.py (r tokens per latent, top-K mixture over the decoder's input
embeddings). Each latent's mixture is read directly — its top-K token ids and weights — and
compared with the r solution tokens it compresses:

  fidelity  — share of latents whose top-1 / top-K contains a token of its own chunk;
  answer    — for the latents covering the content of the last \\boxed{...}: share of answer
              tokens found in the top-1 / top-K of their latent, and problems where all are;
  sharpness — mean top-1 weight and N_eff = 1 / sum(p^2) of the mixtures.

With --eval <eval_box latents jsonl> the decoder's correctness is split by whether the
answer survived in the latents (encoder loss vs decoder loss).

Usage: python h200/diag_latent_tokens.py --encoder ENC_HF --base BASE_DIR --data Math-500-test.jsonl
           [--eval .../epoch-1-step-897-latents.jsonl] [--n 100] [--r 2] [--topk 10]
"""
import argparse
import json
import logging
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from h200.make_honest_data import last_boxed  # noqa: E402
from src.stage1.data import build_latent_token_induction_mask, insert_special_token_every_k  # noqa: E402

logger = logging.getLogger(__name__)
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def load_embedding(base: str) -> torch.Tensor:
    """The decoder's input embedding matrix, read alone from the base safetensors."""
    key = "model.embed_tokens.weight"
    index_path = os.path.join(base, "model.safetensors.index.json")
    shard = (json.load(open(index_path))["weight_map"][key] if os.path.exists(index_path)
             else "model.safetensors")
    with safe_open(os.path.join(base, shard), framework="pt") as f:
        return f.get_tensor(key)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--base", required=True, help="decoder base dir (its embedding builds the mixture)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))

    tok = AutoTokenizer.from_pretrained(args.encoder)
    comp = tok.convert_tokens_to_ids("<|compress_token|>")
    t_open, t_close = tok(["<think>", "</think>"], add_special_tokens=False)["input_ids"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = AutoModel.from_pretrained(args.encoder, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    E = load_embedding(args.base).to(torch.bfloat16)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][:args.n]
    ev = [json.loads(l) for l in open(args.eval, encoding="utf-8")][:args.n] if args.eval else None

    fid1 = fidk = nlat = 0
    top1w, neff = [], []
    ans1 = ansk = nans = 0
    full = []                       # per problem: all answer tokens in top-1 of their latents
    examples = []
    for i, row in enumerate(rows):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["problem"]}]
        prefix = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                     add_special_tokens=False)["input_ids"]
        sol = tok(row["solution"], add_special_tokens=False, return_offsets_mapping=True)
        content, offs = sol["input_ids"], sol["offset_mapping"]
        suffix, count, _ = insert_special_token_every_k(content, comp, -100, args.r)
        ids = torch.tensor([prefix + t_open + suffix + t_close])
        mask = build_latent_token_induction_mask(ids, [comp], pad, torch.bfloat16)
        with torch.no_grad():
            h = enc(ids, attention_mask=mask).last_hidden_state[0]
            h = h[(ids[0] == comp).nonzero().squeeze(-1)]                 # [count, d]
            logits = h.to(E.dtype) @ E.T
            tv, ti = logits.float().topk(args.topk, dim=-1)
            p = tv.softmax(-1)
        top1w += p[:, 0].tolist()
        neff += (1.0 / (p ** 2).sum(-1)).tolist()
        for k in range(count):
            chunk = set(content[k * args.r:(k + 1) * args.r])
            nlat += 1
            fid1 += int(ti[k, 0].item() in chunk)
            fidk += int(bool(chunk & set(ti[k].tolist())))

        box = last_boxed(row["solution"])
        ok_all = None
        if box is not None:
            a0, a1 = box[0] + len("\\boxed{"), box[1] - 1
            ans_pos = [j for j, (s, e) in enumerate(offs) if s < a1 and e > a0]
            if ans_pos:
                hit1 = [content[j] == ti[j // args.r, 0].item() for j in ans_pos]
                hitk = [content[j] in ti[j // args.r].tolist() for j in ans_pos]
                ans1 += sum(hit1); ansk += sum(hitk); nans += len(ans_pos)
                ok_all = all(hit1)
                if len(examples) < 5:
                    lat = sorted({j // args.r for j in ans_pos})
                    examples.append((row["answer"], [[tok.decode([t]) for t in ti[k, :3].tolist()] for k in lat]))
        full.append(ok_all)
        if (i + 1) % 10 == 0:
            logger.info("%d/%d", i + 1, len(rows))

    logger.info("RESULT fidelity: top-1 %.3f, top-%d %.3f over %d latents", fid1 / nlat, args.topk, fidk / nlat, nlat)
    logger.info("RESULT answer tokens: top-1 %.3f, top-%d %.3f over %d tokens; problems with all answer tokens in top-1: %.3f",
                ans1 / nans, args.topk, ansk / nans, nans, sum(bool(x) for x in full) / sum(x is not None for x in full))
    logger.info("RESULT sharpness: mean top-1 weight %.3f, mean N_eff %.2f of %d",
                sum(top1w) / len(top1w), sum(neff) / len(neff), args.topk)
    for a, lat in examples:
        logger.info("  answer %r -> latents' top-3: %s", a, lat)
    if ev is not None:
        for flag in (True, False):
            sel = [e["correct"] for e, f in zip(ev, full) if f is flag]
            if sel:
                logger.info("RESULT decoder acc when answer %s in latents' top-1: %.3f (n=%d)",
                            "IS" if flag else "is NOT", sum(sel) / len(sel), len(sel))


if __name__ == "__main__":
    main()
