from __future__ import annotations
import argparse, json, os, traceback
from tqdm import tqdm
from .coco_data import load_coco_samples
from .qwen_vl import load_qwen_vl, generate_response, P2_PROMPT
from .json_parse import parse_objects_response
from .metrics import compute_object_metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--coco_root",required=True); ap.add_argument("--split",default="val2017"); ap.add_argument("--model_name",required=True); ap.add_argument("--limit",type=int,default=500); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--shuffle",action="store_true"); ap.add_argument("--dtype",default="bf16"); ap.add_argument("--device_map",default="auto"); ap.add_argument("--max_new_tokens",type=int,default=128); ap.add_argument("--output",required=True); args=ap.parse_args()
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    model,processor=load_qwen_vl(args.model_name,args.dtype,args.device_map)
    data=load_coco_samples(args.coco_root,args.split,args.limit,args.seed,args.shuffle,True)
    rows=[]
    with open(args.output,"w",encoding="utf-8") as fo:
        for s in tqdm(data):
            rec={"image_id":s["image_id"],"image_path":s["image_path"],"model_name":args.model_name,"prompt_type":"p2_coco_json","gt_objects":s["gt_objects"]}
            try:
                raw=generate_response(model,processor,s["image_path"],P2_PROMPT,args.max_new_tokens)
                parsed=parse_objects_response(raw)
                p=parsed["parsed_objects"]; gt=set(s["gt_objects"])
                factual=sorted([x for x in p if x in gt]); hall=sorted([x for x in p if x not in gt]); missed=sorted([x for x in gt if x not in set(p)])
                rec.update({"raw_response":raw,**parsed,"factual_objects":factual,"hallucinated_objects":hall,"missed_objects":missed,"num_pred_objects":len(p),"num_factual_objects":len(factual),"num_hallucinated_objects":len(hall),"num_gt_objects":len(gt),"has_hallucination":len(hall)>0})
            except Exception as e:
                rec.update({"error":str(e),"traceback":traceback.format_exc(),"invalid_json":True,"invalid_objects":[],"parsed_objects":[],"factual_objects":[],"hallucinated_objects":[],"missed_objects":s["gt_objects"],"num_pred_objects":0,"num_factual_objects":0,"num_hallucinated_objects":0,"num_gt_objects":len(s["gt_objects"]),"has_hallucination":False,"raw_response":""})
            rows.append(rec); fo.write(json.dumps(rec,ensure_ascii=False)+"\n")
    with open(args.output+".metrics.json","w") as f: json.dump(compute_object_metrics(rows),f,indent=2)
    with open(args.output+".config.json","w") as f: json.dump(vars(args),f,indent=2)

if __name__=="__main__": main()
