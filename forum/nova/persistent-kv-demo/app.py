import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_kv import save_kv, load_kv, kv_to_cpu, kv_to_device

MODEL_ID = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None and tok.eos_token is not None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()

# Global KV in app scope (demo only)
PAST = None

def step(prompt, temperature, top_k, top_p):
    global PAST
    if not prompt.strip() and PAST is None:
        return "(enter prompt or load KV first)"
    if prompt.strip():
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    else:
        # empty prompt → feed EOS/pad
        inputs = tok(tok.eos_token, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, past_key_values=PAST, use_cache=True)
        PAST = out.past_key_values
        logits = out.logits[:, -1, :]

        # sampling
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, k=top_k, dim=-1)
                min_keep = v[..., -1, None]
                logits = torch.where(logits < min_keep, torch.full_like(logits, -float('inf')), logits)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                cutoff = cumprobs > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_logits[cutoff] = -float('inf')
                logits = torch.zeros_like(logits).scatter(-1, sorted_indices, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return tok.decode(next_id, skip_special_tokens=True)

def save_kv_ui(path, fmt):
    global PAST
    if PAST is None:
        return "No KV to save."
    save_kv(path, kv_to_cpu(PAST), fmt=fmt)
    return f"KV saved to {path} ({fmt})"

def load_kv_ui(path, fmt):
    global PAST
    PAST = load_kv(path, fmt=fmt)
    PAST = kv_to_device(PAST, DEVICE)
    return f"KV loaded from {path} ({fmt})"

def reset_kv():
    global PAST
    PAST = None
    return "KV reset."

with gr.Blocks() as demo:
    gr.Markdown("# Persistent KV Demo (Web UI)")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt (optional each step)")
    with gr.Row():
        temperature = gr.Slider(0, 1.5, value=0.8, step=0.05, label="Temperature (0 = greedy)")
        top_k = gr.Slider(0, 100, value=0, step=1, label="Top‑K")
        top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top‑P")
    with gr.Row():
        btn_step = gr.Button("Step →")
    out = gr.Textbox(label="Next token")

    btn_step.click(step, inputs=[prompt, temperature, top_k, top_p], outputs=out)

    gr.Markdown("### Save / Load KV")
    with gr.Row():
        path_save = gr.Textbox(value="kv_cache.pt", label="Save path")
        fmt_save = gr.Dropdown(choices=["torch","pickle","gzip"], value="torch", label="Format")
        btn_save = gr.Button("Save KV")
    msg_save = gr.Textbox(label="Save status")

    with gr.Row():
        path_load = gr.Textbox(value="kv_cache.pt", label="Load path")
        fmt_load = gr.Dropdown(choices=["torch","pickle","gzip"], value="torch", label="Format")
        btn_load = gr.Button("Load KV")
    msg_load = gr.Textbox(label="Load status")

    btn_reset = gr.Button("Reset KV")

    btn_save.click(save_kv_ui, inputs=[path_save, fmt_save], outputs=msg_save)
    btn_load.click(load_kv_ui, inputs=[path_load, fmt_load], outputs=msg_load)
    btn_reset.click(fn=reset_kv, outputs=msg_save)

if __name__ == "__main__":
    demo.launch()
