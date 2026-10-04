"""Do the trained Stage-1 latents carry information beyond a constant? (CPU)

Builds training examples with the repo's own `pretrain_tokenize_function` + collator (the
think-only honest data by default), computes the latent mixtures exactly as in training —
encoder hidden states at <|compress_token|>, top-K softmax over the decoder's input
embeddings — and scores the frozen base decoder (Stage-1 objective, repo mask, dual-track
positions) with the latent slots filled from different sources:

  trained   — the trained encoder;
  shuffled  — the same latents, randomly permuted across positions of the example;
  other     — the trained latents of a different example (cycled to length);
  constant  — one vector everywhere: the mean of all trained latents;
  untrained — the untrained encoder (out/base_model);
  naive     — the mean input embedding of the r CoT tokens each latent compresses.

trained ~ constant means the encoder learned nothing example-specific; trained ~ other means
nothing problem-specific; trained ~ shuffled means nothing position-specific.

Encoders and decoder are loaded one at a time to fit the CPU-queue memory cap.
Usage: python h200/diag_latent_info.py --encoder ENC_HF --encoder0 ENC0_DIR [--n 6] [--r 2]
"""
import argparse
import gc
import json
import logging
import os
import sys
from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from h200.diag_latent_tokens import load_embedding  # noqa: E402
from h200.diag_mask_ce import ce_by_group, find  # noqa: E402
from src.modeling.modeling_stage1 import softmax_over_embedding_topk  # noqa: E402
from src.stage1.data import DataCollatorForDynamicPadding, pretrain_tokenize_function  # noqa: E402

logger = logging.getLogger(__name__)


def encode_all(path, batches, comp, emb, topk):
    enc = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    out = []
    with torch.no_grad():
        for b in batches:
            h = enc(b["cot_ids"], attention_mask=b["cot_attention_mask"]).last_hidden_state[0]
            x = h[(b["cot_ids"][0] == comp).nonzero().squeeze(-1)]
            out.append(softmax_over_embedding_topk(x, emb, topk)[0].float())
    del enc
    gc.collect()
    logger.info("encoded with %s", path)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--encoder0", required=True, help="untrained encoder, e.g. <encoder run>/out/base_model")
    ap.add_argument("--data", default=None, help="default: $LSFT_DATA/...-honest-think.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_cot_chars", type=int, default=2500)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.manual_seed(0)

    base = os.environ["LSFT_BASE"]
    data = args.data or os.path.join(os.environ["LSFT_DATA"], "OpenR1-Math-220k-v-train-4k-honest-think.jsonl")
    tok = AutoTokenizer.from_pretrained(args.encoder)          # has <|compress_token|>
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    comp = tok.convert_tokens_to_ids("<|compress_token|>")
    t_open, t_close = tok(["<think>", "</think>"], add_special_tokens=False)["input_ids"]
    stub = SimpleNamespace(tokenizer=tok, decoder_name_or_path=base, compress_token_id=comp,
                           latent_token_ids=[t_open, t_close])
    collate = DataCollatorForDynamicPadding(tok.pad_token_id, comp, t_close)

    rows = []
    with open(data, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if len(row["cot"]) <= args.max_cot_chars:
                rows.append(row)
            if len(rows) == args.n:
                break
    batches = [collate([pretrain_tokenize_function(dict(r), stub, args.r)]) for r in rows]

    emb = torch.nn.Embedding.from_pretrained(load_embedding(base).to(torch.bfloat16))
    trained = encode_all(args.encoder, batches, comp, emb, args.topk)
    untrained = encode_all(args.encoder0, batches, comp, emb, args.topk)
    const = torch.cat(trained).mean(0)
    sources = {}
    for i, (b, row) in enumerate(zip(batches, rows)):
        n = trained[i].shape[0]
        other = trained[(i + 1) % len(trained)]
        content = torch.tensor(tok(row["cot"].strip(), add_special_tokens=False)["input_ids"])
        naive = torch.stack([emb(content[k * args.r:(k + 1) * args.r]).float().mean(0) for k in range(n)])
        sources[i] = {
            "trained": trained[i],
            "shuffled": trained[i][torch.randperm(n)],
            "other": other[torch.arange(n) % other.shape[0]],
            "constant": const.expand(n, -1),
            "untrained": untrained[i],
            "naive": naive,
        }

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    dec_emb = model.get_input_embeddings()
    totals = {k: [0.0, 0.0, 0] for k in sources[0]}
    for i, b in enumerate(batches):
        ids, labels = b["input_ids"].clone(), b["labels"]
        slots = (ids[0] == -100).nonzero().squeeze(-1)
        ids[ids == -100] = tok.pad_token_id
        split = find(ids[0].tolist(), t_close, slots[-1].item())
        line = []
        for name, lat in sources[i].items():
            with torch.no_grad():
                x = dec_emb(ids)
                x[0, slots] = lat.to(x.dtype)
                logits = model(inputs_embeds=x, attention_mask=b["attention_mask"],
                               position_ids=b["position_ids"]).logits
            c, a, _ = ce_by_group(logits, labels, split)
            totals[name][0] += c
            totals[name][1] += a
            totals[name][2] += 1
            line.append(f"{name} {c:.2f}/{a:.2f}")
        logger.info("example %d (latents %d): %s", i, slots.numel(), " | ".join(line))
    logger.info("RESULT mean CE over %d examples, CoT tokens / answer tokens:", len(batches))
    for name, (c, a, n) in totals.items():
        logger.info("RESULT   %-9s %.3f / %.3f", name, c / n, a / n)


if __name__ == "__main__":
    main()
