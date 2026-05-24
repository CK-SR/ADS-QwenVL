from __future__ import annotations
import argparse, json, os, torch
from collections import defaultdict
from tqdm import tqdm
from .qwen_vl import load_qwen_vl, P2_PROMPT
from .hidden_utils import find_object_spans_in_response, extract_object_token_hidden_states

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--coco_root",required=True); ap.add_argument("--split",default="train2017"); ap.add_argument("--model_name",required=True); ap.add_argument("--baseline_jsonl",required=True); ap.add_argument("--output_vector",required=True); args=ap.parse_args()
    model,processor=load_qwen_vl(args.model_name)
    fmap={}
    sums=defaultdict(lambda:{"factual_sum":None,"hall_sum":None,"factual_count":0,"hall_count":0})
    with open(args.baseline_jsonl) as f:
        for ln in f:
            r=json.loads(ln); fmap[r["image_id"]]=r
    for r in tqdm(fmap.values()):
        spans=find_object_spans_in_response(r.get("raw_response",""),r.get("parsed_objects",[]),r.get("gt_objects",[]))
        hs=extract_object_token_hidden_states(model,processor,r["image_path"],P2_PROMPT,r.get("raw_response",""),spans)
        for li,label,vec in hs:
            k=sums[li]
            if label=="factual":
                k["factual_sum"]=vec if k["factual_sum"] is None else k["factual_sum"]+vec; k["factual_count"]+=1
            else:
                k["hall_sum"]=vec if k["hall_sum"] is None else k["hall_sum"]+vec; k["hall_count"]+=1
    layers={};fc={};hc={}
    for li,v in sums.items():
        fc[li]=v["factual_count"]; hc[li]=v["hall_count"]
        if v["factual_count"] and v["hall_count"]:
            d=v["factual_sum"]/v["factual_count"]-v["hall_sum"]/v["hall_count"]
            layers[li]=d/(d.norm()+1e-12)
    os.makedirs(os.path.dirname(args.output_vector),exist_ok=True)
    torch.save({"model_name":args.model_name,"prompt_type":"p2_coco_json","vector_direction":"factual_minus_hallucinated","layers":layers,"factual_counts":fc,"hallucinated_counts":hc,"config":vars(args)},args.output_vector)

if __name__=="__main__": main()
