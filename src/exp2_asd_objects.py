from __future__ import annotations
import argparse, json, os, traceback, torch
from PIL import Image
from tqdm import tqdm
from .coco_data import load_coco_samples
from .qwen_vl import load_qwen_vl, P2_PROMPT, build_messages
from .asd_decode import asd_bidirectional_decode
from .json_parse import parse_objects_response
from .metrics import compute_object_metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--coco_root",required=True); ap.add_argument("--split",default="val2017"); ap.add_argument("--model_name",required=True); ap.add_argument("--vector_path",required=True); ap.add_argument("--asd_layer",type=int,required=True); ap.add_argument("--lambda_strength",type=float,default=1.0); ap.add_argument("--alpha",type=float,default=1.0); ap.add_argument("--limit",type=int,default=500); ap.add_argument("--output",required=True); args=ap.parse_args()
    vec=torch.load(args.vector_path,map_location="cpu")["layers"][args.asd_layer]
    model,processor=load_qwen_vl(args.model_name)
    data=load_coco_samples(args.coco_root,args.split,args.limit)
    os.makedirs(os.path.dirname(args.output),exist_ok=True); rows=[]
    with open(args.output,"w") as fo:
      for s in tqdm(data):
        rec={"image_id":s["image_id"],"image_path":s["image_path"],"model_name":args.model_name,"prompt_type":"p2_coco_json","gt_objects":s["gt_objects"],"asd_enabled":True,"asd_layer":args.asd_layer,"lambda_strength":args.lambda_strength,"alpha":args.alpha,"vector_path":args.vector_path}
        try:
          text=processor.apply_chat_template(build_messages(P2_PROMPT,s["image_path"]),tokenize=False,add_generation_prompt=True)
          ins=processor(text=[text],images=[Image.open(s["image_path"]).convert("RGB")],return_tensors="pt")
          dev=next(model.parameters()).device
          ins={k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in ins.items()}
          out=asd_bidirectional_decode(model,ins,vec,args.asd_layer,args.lambda_strength,args.alpha,128,processor.tokenizer.eos_token_id)
          gen=out[:,ins["input_ids"].shape[1]:]
          raw=processor.tokenizer.batch_decode(gen,skip_special_tokens=True)[0]
          parsed=parse_objects_response(raw); p=parsed["parsed_objects"]; gt=set(s["gt_objects"])
          factual=sorted([x for x in p if x in gt]); hall=sorted([x for x in p if x not in gt]); missed=sorted([x for x in gt if x not in set(p)])
          rec.update({"raw_response":raw,**parsed,"factual_objects":factual,"hallucinated_objects":hall,"missed_objects":missed,"num_pred_objects":len(p),"num_factual_objects":len(factual),"num_hallucinated_objects":len(hall),"num_gt_objects":len(gt),"has_hallucination":len(hall)>0})
        except Exception as e:
          rec.update({"error":str(e),"traceback":traceback.format_exc(),"invalid_json":True,"invalid_objects":[],"parsed_objects":[],"factual_objects":[],"hallucinated_objects":[],"missed_objects":s["gt_objects"],"num_pred_objects":0,"num_factual_objects":0,"num_hallucinated_objects":0,"num_gt_objects":len(s["gt_objects"]),"has_hallucination":False,"raw_response":""})
        rows.append(rec); fo.write(json.dumps(rec)+"\n")
    with open(args.output+".metrics.json","w") as f: json.dump(compute_object_metrics(rows),f,indent=2)

if __name__=="__main__": main()
