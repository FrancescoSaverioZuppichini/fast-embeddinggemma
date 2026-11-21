"""Search queries -> vectors with the checkpoint's text model through Hugging Face transformers (extra [text]): only the
language model goes to the card. Uncompiled on purpose: torch.compile recompiles for every new query length (up to 85 s)."""

import torch
import torch.nn.functional as F

from .weights import REPO

PROMPT = "task: search result | query: "


def load(repo=REPO):
    from transformers import AutoModel, AutoTokenizer

    hf = AutoModel.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation="sdpa", audio_config=None)
    return AutoTokenizer.from_pretrained(repo), hf.language_model.to("cuda").eval()


@torch.inference_mode()
def embed(queries, texts):
    """Texts -> (n, 768) float32 unit vectors: the query prompt, the text model, the mean over real tokens."""
    tokenizer, lm = queries
    t = tokenizer([PROMPT + s for s in texts], padding=True, truncation=True, max_length=8192, return_tensors="pt").to(
        "cuda"
    )
    h = lm(inputs_embeds=lm.embed_tokens(t["input_ids"]), attention_mask=t["attention_mask"]).last_hidden_state
    m = t["attention_mask"].unsqueeze(-1).float()
    return F.normalize((h.float() * m).sum(1) / m.sum(1), dim=-1).cpu().numpy()
