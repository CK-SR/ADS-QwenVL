from __future__ import annotations
import torch
from .hidden_utils import find_decoder_layers

def asd_bidirectional_decode(model,inputs,vector,asd_layer,lambda_strength=1.0,alpha=1.0,max_new_tokens=128,eos_token_id=None):
    layers=find_decoder_layers(model); target=layers[asd_layer]
    ids=inputs["input_ids"]
    other={k:v for k,v in inputs.items() if k!="input_ids"}
    def run(sign):
        def hook(_m,_i,o):
            h=o[0] if isinstance(o,tuple) else o
            h=h.clone(); v=vector.to(h.device,h.dtype)
            h[:,-1,:]=h[:,-1,:]+sign*lambda_strength*v
            return (h,)+o[1:] if isinstance(o,tuple) else h
        hh=target.register_forward_hook(hook)
        out=model(input_ids=ids,**other,return_dict=True,use_cache=False)
        hh.remove(); return out.logits[:,-1,:]
    for _ in range(max_new_tokens):
        lp, lm = run(+1), run(-1)
        la = lp + alpha*(lp-lm)
        nxt = torch.argmax(la,dim=-1,keepdim=True)
        ids=torch.cat([ids,nxt],dim=1)
        if eos_token_id is not None and int(nxt.item())==int(eos_token_id): break
    return ids
