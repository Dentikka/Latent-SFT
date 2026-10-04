"""Why does the Stage-1 loss sit near 200 nats per token?

Runs the frozen Stage-1 decoder (the base model) on a few short training examples built by
the repo's own `pretrain_tokenize_function` + collator. Latent slot k is filled with the mean
input embedding of the CoT tokens it compresses, cot[k*r:(k+1)*r] — a sane stand-in for an
encoder output. Token CE is reported separately for the explicit CoT tokens supervised
inside the latent chain and for the tokens from `</think>` on, under three attention masks:

  repo    — the repo's supervision mask: a CoT token sees only its own segment and the
            preceding latent (no prompt, no first token);
  +prompt — the same, plus every row after the prompt may attend to the whole prompt;
  +first  — the same, plus every row may attend to the first token only (an attention
            sink without the prompt's content);
  causal  — plain causal attention over the same sequence, plain positions.

CPU only, ~20 GB RAM. Usage: python h200/diag_mask_ce.py [--n 3] [--r 2]
"""
import argparse
import json
import logging
import os
import sys
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stage1.data import DataCollatorForDynamicPadding, pretrain_tokenize_function  # noqa: E402

logger = logging.getLogger(__name__)
NEG = torch.finfo(torch.bfloat16).min


def ce_by_group(logits, labels, split):
    """Mean next-token CE over supervised targets before `split` (CoT) and from it (answer)."""
    lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = labels[0, 1:]
    where = torch.arange(tgt.numel()) + 1          # position of each target in the sequence
    ok = tgt != -100
    nll = -lp[ok.nonzero().squeeze(-1), tgt[ok]]
    w = where[ok]
    cot, ans = nll[w < split], nll[w >= split]
    return cot.mean().item(), (ans.mean().item() if ans.numel() else float("nan")), \
        logits.float().abs().max().item()


def find(seq, sub, start):
    for j in range(start, len(seq) - len(sub) + 1):
        if seq[j:j + len(sub)] == sub:
            return j
    raise ValueError("subsequence not found")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--max_cot_chars", type=int, default=1500)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))

    base = os.environ["LSFT_BASE"]
    data = os.path.join(os.environ["LSFT_DATA"], "OpenR1-Math-220k-v-train-4k-honest.jsonl")
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": ["<|compress_token|>"]})
    think_open, think_close = tok(["<think>", "</think>"], add_special_tokens=False)["input_ids"]
    stub = SimpleNamespace(tokenizer=tok, decoder_name_or_path=base,
                           compress_token_id=tok.convert_tokens_to_ids("<|compress_token|>"),
                           latent_token_ids=[think_open, think_close])
    collate = DataCollatorForDynamicPadding(tok.pad_token_id, stub.compress_token_id, think_close)

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    emb = model.get_input_embeddings()

    rows = []
    with open(data, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if len(row["cot"]) <= args.max_cot_chars:
                rows.append(row)
            if len(rows) == args.n:
                break

    for i, row in enumerate(rows):
        ex = pretrain_tokenize_function(dict(row), stub, args.r)
        batch = collate([ex])
        ids, labels, mask = batch["input_ids"].clone(), batch["labels"], batch["attention_mask"]
        lat_pos = (ids[0] == -100).nonzero().squeeze(-1)
        cot_content = torch.tensor(tok(row["cot"].strip(), add_special_tokens=False)["input_ids"])
        ids[ids == -100] = tok.pad_token_id
        with torch.no_grad():
            x = emb(ids)
            for k, p in enumerate(lat_pos.tolist()):
                seg = cot_content[k * args.r:(k + 1) * args.r]
                if seg.numel():
                    x[0, p] = emb(seg).mean(0)

        seq = ids[0].tolist()
        split = find(seq, think_close, lat_pos[-1].item())          # '</think>' after the chain
        prompt_len = lat_pos[0].item()                               # prompt + '<think>'
        T = ids.shape[1]
        m_prompt = mask.clone()
        m_prompt[..., prompt_len:, :prompt_len] = 0.0
        m_first = mask.clone()
        m_first[..., prompt_len:, 0] = 0.0
        m_causal = torch.triu(torch.full((1, 1, T, T), NEG, dtype=torch.bfloat16), diagonal=1)
        n_cot = int(((labels[0] != -100) & (torch.arange(T) < split)).sum())
        logger.info("example %d: seq %d, latents %d, prompt %d, supervised CoT tokens %d, answer tokens %d",
                    i, T, lat_pos.numel(), prompt_len, n_cot, int((labels[0, split:] != -100).sum()))
        for name, m, pos in [("repo", mask, batch["position_ids"]),
                             ("+prompt", m_prompt, batch["position_ids"]),
                             ("+first", m_first, batch["position_ids"]),
                             ("causal", m_causal, torch.arange(T).unsqueeze(0))]:
            with torch.no_grad():
                logits = model(inputs_embeds=x, attention_mask=m, position_ids=pos).logits
            c, a, mx = ce_by_group(logits, labels, split)
            logger.info("  %-8s CE CoT %8.2f | CE answer %8.2f | max|logit| %8.1f", name, c, a, mx)


if __name__ == "__main__":
    main()
