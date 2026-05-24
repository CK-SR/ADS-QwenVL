from __future__ import annotations
import re, torch
from PIL import Image
from .qwen_vl import build_messages


def find_decoder_layers(model):
    paths=["model.layers","language_model.model.layers","model.language_model.layers","language_model.layers","transformer.h"]
    for p in paths:
        cur=model
        ok=True
        for k in p.split('.'):
            if not hasattr(cur,k): ok=False; break
            cur=getattr(cur,k)
        if ok: return cur
    names=[n for n,_ in list(model.named_modules())[:100]]
    raise RuntimeError(f"decoder layers not found. first modules={names}")


def find_object_spans_in_response(response, objects, gt_objects):
    spans=[]
    low=response.lower()
    for o in objects:
        m=re.search(re.escape(o.lower()), low)
        if not m: continue
        spans.append({"object":o,"label":"factual" if o in set(gt_objects) else "hallucinated","start_char":m.start(),"end_char":m.end()})
    return spans


def extract_object_token_hidden_states(model,processor,image_path,prompt,assistant_response,object_spans,layers=None):
    tok=processor.tokenizer
    ptxt=processor.apply_chat_template(build_messages(prompt,image_path),tokenize=False,add_generation_prompt=True)
    full_msgs=build_messages(prompt,image_path)+[{"role":"assistant","content":[{"type":"text","text":assistant_response}]}]
    ftxt=processor.apply_chat_template(full_msgs,tokenize=False,add_generation_prompt=False)
    image=Image.open(image_path).convert("RGB")
    pin=processor(text=[ptxt],images=[image],return_tensors="pt")
    fin=processor(text=[ftxt],images=[image],return_tensors="pt")
    dev=next(model.parameters()).device
    fin={k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in fin.items()}
    response_start=pin["input_ids"].shape[1]
    out=model(**fin,output_hidden_states=True,use_cache=False,return_dict=True)
    ids=fin["input_ids"][0].tolist()
    txt=tok.decode(ids,skip_special_tokens=False)
    rs=max(txt.rfind(assistant_response),0)
    enc=tok(txt,return_offsets_mapping=True,add_special_tokens=False)
    res=[]
    for s in object_spans:
        c0,c1=rs+s["start_char"],rs+s["end_char"]
        tidx=[i for i,(a,b) in enumerate(enc["offset_mapping"]) if b>c0 and a<c1 and i>=response_start]
        if not tidx: continue
        for li,hs in enumerate(out.hidden_states):
            if layers and li not in layers: continue
            for ti in tidx:
                res.append((li,s["label"],hs[0,ti,:].detach().float().cpu()))
    return res
