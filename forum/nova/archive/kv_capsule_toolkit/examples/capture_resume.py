import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_capsule import KVCapsule, save_kv_capsule, load_kv_capsule

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

ids = tok.encode("The quick brown fox", return_tensors="pt")
out = model(ids, use_cache=True)
capsule = KVCapsule.from_past(out.past_key_values, model, model_id="gpt2")
save_kv_capsule(capsule, "capsule")

# Resume with new text
capsule = load_kv_capsule("capsule")
new_ids = tok.encode(" jumps over the lazy dog", return_tensors="pt")
out2 = model(new_ids, past_key_values=capsule.kv, use_cache=True)
print(tok.decode(out2.logits.argmax(-1)[0]))
