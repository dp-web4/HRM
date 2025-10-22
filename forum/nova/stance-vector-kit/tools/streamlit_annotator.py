
import json
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="SVK Annotator", layout="wide")

st.title("SVK Quick Annotator (Streamlit)")
st.write("Load a JSONL transcript, annotate stance axes per turn, and export CSV.")

input_path = st.text_input("Transcript JSONL path", "examples/sample_transcript.jsonl")
out_path = st.text_input("Output CSV path", "data/labels_streamlit.csv")

axes = ['EH','DC','EX','MA','RR','AG','AS','SV','VA','AR','IF','ED']

def load_jsonl(p):
    turns = []
    with open(p,'r',encoding='utf-8') as f:
        for i,ln in enumerate(f):
            ln = ln.strip()
            if not ln: continue
            obj = json.loads(ln)
            obj['turn_idx']=i
            obj['session_id']=obj.get('session_id','S1')
            turns.append(obj)
    return pd.DataFrame(turns)[['session_id','turn_idx','speaker','text']]

if Path(input_path).exists():
    df = load_jsonl(input_path)
    for ax in axes:
        if ax not in df.columns:
            df[ax] = ""
    st.write("Rows:", len(df))
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    if st.button("Save to CSV"):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        edited.to_csv(out_path, index=False)
        st.success(f"Saved annotations to {out_path}")
else:
    st.warning("Input file not found.")
