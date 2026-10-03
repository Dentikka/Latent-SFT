"""Honest-mode Math-500 eval of a Stage-1 (encoder, decoder) pair with a forced box.

The decoder input is   prompt + <think> + [latents] + </think> + "\\boxed"
and the decoder writes only "{...}". The two forced tokens are exactly the first two
tokens of every training answer (`\\boxed{X}` tokenized standalone): "{" is NOT forced
because BPE merges it with the next character ("{\\", "{-", "{(") in ~26% of answers.

Modes:
  latents — the encoder compresses the gold `solution` into latents (the repo's Stage-1
            protocol: an upper bound on what the latents carry);
  none    — no latent chain at all (`<think></think>`): the same decoder answering at once.

Greedy decoding. Usage:
  python h200/eval_box.py --encoder ENC_HF --decoder DEC_HF --data Math-500-test.jsonl \
      --out OUT.jsonl [--mode latents|none] [--r 2] [--topk 10] [--bs 16] [--max_new 24]
"""
import argparse
import json
import logging
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_utils.grader import check_is_correct  # noqa: E402
from eval_utils.parser import extract_answer  # noqa: E402
from src.modeling.modeling_stage1 import LatentSFTStage1Encoder, softmax_over_embedding_topk  # noqa: E402
from src.stage1.data import build_latent_token_induction_mask, insert_special_token_every_k  # noqa: E402

logger = logging.getLogger(__name__)
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
BOX = "\\boxed"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["latents", "none"], default="latents")
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model = LatentSFTStage1Encoder(encoder_name_or_path=args.encoder, decoder_name_or_path=args.decoder,
                                   bfloat16=True, use_flash_attention_2=False, lora_tune=False,
                                   training=False).to(dev)
    model.eval()
    tok = model.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t_open, t_close = model.latent_token_ids
    box_ids = tok(BOX, add_special_tokens=False)["input_ids"]
    embed = model.decoder.model.embed_tokens
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["solution"]))
    results = [None] * len(rows)

    with torch.no_grad():
        for s in range(0, len(order), args.bs):
            idx = order[s:s + args.bs]
            inps, cots, counts = [], [], []
            for i in idx:
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": rows[i]["problem"]}]
                prefix = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                             add_special_tokens=False)["input_ids"]
                if args.mode == "latents":
                    content = tok(rows[i]["solution"], add_special_tokens=False)["input_ids"]
                    suffix, count, _ = insert_special_token_every_k(content, model.compress_token_id, -100, args.r)
                    cots.append(prefix + t_open + suffix + t_close)
                else:
                    count = 0
                counts.append(count)
                inps.append(prefix + t_open + [-100] * count + t_close + box_ids)

            T = max(map(len, inps))
            ids = torch.full((len(idx), T), pad, dtype=torch.long)
            att = torch.zeros((len(idx), T), dtype=torch.long)
            for b, x in enumerate(inps):                       # left padding for generation
                ids[b, T - len(x):] = torch.tensor(x)
                att[b, T - len(x):] = 1
            ids, att = ids.to(dev), att.to(dev)
            slots = ids == -100
            emb = embed(ids.masked_fill(slots, pad))

            if args.mode == "latents":
                C = max(map(len, cots))
                cot = torch.full((len(idx), C), pad, dtype=torch.long)
                for b, x in enumerate(cots):
                    cot[b, :len(x)] = torch.tensor(x)
                cot = cot.to(dev)
                mask = build_latent_token_induction_mask(cot, [model.compress_token_id], pad, torch.bfloat16)
                hidden = model._compress(cot, mask)
                for b in range(len(idx)):
                    src = (cot[b] == model.compress_token_id).nonzero().squeeze(-1)
                    dst = slots[b].nonzero().squeeze(-1)
                    assert src.numel() == dst.numel() == counts[b]
                    mix, _, _ = softmax_over_embedding_topk(hidden[b, src], embed, args.topk)
                    emb[b, dst] = mix.to(emb.dtype)

            gen = model.decoder.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=args.max_new,
                                         do_sample=False, pad_token_id=pad)
            for b, i in enumerate(idx):
                text = BOX + tok.decode(gen[b], skip_special_tokens=True)
                pred = extract_answer(text)
                results[i] = {"answer": rows[i]["answer"], "generated": text, "pred": pred,
                              "correct": bool(check_is_correct(pred, rows[i]["answer"])),
                              "latents": counts[b]}
            logger.info("%d/%d", min(s + args.bs, len(order)), len(order))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = sum(r["correct"] for r in results) / len(results)
    closed = sum("}" in r["generated"] for r in results) / len(results)
    summary = {"mode": args.mode, "encoder": args.encoder, "decoder": args.decoder, "n": len(results),
               "acc": acc, "box_closed": closed, "mean_latents": sum(r["latents"] for r in results) / len(results)}
    with open(args.out.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    logger.info("RESULT %s", json.dumps(summary))


if __name__ == "__main__":
    main()
