"""Model-side helpers shared by train / eval / augment (needs torch + transformers)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from v3_common import SYSTEM_PROMPT

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def uses_system_role(model_name):
    # Phi-3's template has no real system slot, so fold the instruction into the user turn.
    return "phi" not in model_name.lower()


def build_messages(model_name, question):
    if uses_system_role(model_name):
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"}]


def render_prompt(tokenizer, model_name, question):
    return tokenizer.apply_chat_template(build_messages(model_name, question), tokenize=False,
                                         add_generation_prompt=True)


def render_full(tokenizer, model_name, question, answer):
    """Prompt + answer + the model's own end-of-turn marker, plus the prompt prefix length."""
    prompt = render_prompt(tokenizer, model_name, question)
    msgs = build_messages(model_name, question) + [{"role": "assistant", "content": answer}]
    full = tokenizer.apply_chat_template(msgs, tokenize=False).rstrip()
    if not full.startswith(prompt):
        full = prompt + answer + tokenizer.eos_token
    return prompt, full


def stop_token_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    for tok in ("<|im_end|>", "<|end|>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return sorted(i for i in ids if i is not None)


def compute_dtype():
    # T4 (compute capability 7.5) has no bf16 -> fp16
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def load_tokenizer(model_name, padding_side="left"):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_model(model_name, four_bit=True):
    kwargs = {"device_map": {"": 0}}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype(), bnb_4bit_use_double_quant=True)
    else:
        kwargs["torch_dtype"] = compute_dtype()
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


@torch.no_grad()
def generate_greedy(model, tokenizer, model_name, questions, max_new_tokens=64, batch_size=16):
    """Batched greedy decoding. Returns (answers, token_conf, min_prob) lists.

    token_conf = exp(mean token log-prob of the answer) -- same definition v1/v2 used.
    """
    stops = stop_token_ids(tokenizer)
    answers, confs, mins = [], [], []
    model.eval()
    for i in range(0, len(questions), batch_size):
        chunk = questions[i:i + batch_size]
        prompts = [render_prompt(tokenizer, model_name, q) for q in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True,
                             return_dict_in_generate=True, eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
        plen = enc["input_ids"].shape[1]
        for b in range(len(chunk)):
            new = gen.sequences[b, plen:]
            lps = []
            for t, tok in enumerate(new.tolist()):
                lps.append(torch.log_softmax(gen.scores[t][b].float(), dim=-1)[tok].item())
                if tok in stops:
                    break
            answers.append(tokenizer.decode(new, skip_special_tokens=True).strip())
            lp = torch.tensor(lps) if lps else torch.tensor([-20.0])
            confs.append(float(lp.mean().exp()))
            mins.append(float(lp.min().exp()))
    return answers, confs, mins


@torch.no_grad()
def generate_samples(model, tokenizer, model_name, questions, n, temperature=0.7, top_p=0.95,
                     max_new_tokens=64, batch_size=8):
    """n sampled answers per question (for self-consistency). Returns list of lists of str."""
    stops = stop_token_ids(tokenizer)
    out = []
    model.eval()
    for i in range(0, len(questions), batch_size):
        chunk = questions[i:i + batch_size]
        prompts = [render_prompt(tokenizer, model_name, q) for q in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                             top_p=top_p, num_return_sequences=n, eos_token_id=stops,
                             pad_token_id=tokenizer.pad_token_id)
        plen = enc["input_ids"].shape[1]
        texts = tokenizer.batch_decode(gen[:, plen:], skip_special_tokens=True)
        for b in range(len(chunk)):
            out.append([t.strip() for t in texts[b * n:(b + 1) * n]])
    return out
