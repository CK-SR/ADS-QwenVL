from __future__ import annotations
import argparse, json, os, random
from .coco_data import load_coco_samples
from .coco_vocab import COCO_80_CATEGORIES

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--coco_root",required=True); ap.add_argument("--split",default="val2017"); ap.add_argument("--task",choices=["object_existence","object_multichoice"],required=True); ap.add_argument("--model_name",required=True); ap.add_argument("--limit",type=int,default=500); ap.add_argument("--output",required=True); ap.add_argument("--vector_path",default=None); ap.add_argument("--asd_layer",type=int,default=-1); ap.add_argument("--lambda_strength",type=float,default=1.0); ap.add_argument("--alpha",type=float,default=1.0); args=ap.parse_args()
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    data=load_coco_samples(args.coco_root,args.split,args.limit)
    rnd=random.Random(42)
    with open(args.output,"w") as f:
      for s in data:
        if args.task=="object_existence":
          q=rnd.choice(s["gt_objects"]) if rnd.random()<0.5 else rnd.choice([x for x in COCO_80_CATEGORIES if x not in set(s["gt_objects"])])
          gt="yes" if q in set(s["gt_objects"]) else "no"
          rec={"image_id":s["image_id"],"image_path":s["image_path"],"model_name":args.model_name,"task":args.task,"query_object":q,"gt_answer":gt,"raw_response":"","parsed_answer":"invalid","is_correct":False,"is_false_yes":False,"is_false_no":False,"asd_enabled":bool(args.vector_path),"asd_layer":args.asd_layer,"lambda_strength":args.lambda_strength,"alpha":args.alpha}
        else:
          rec={"image_id":s["image_id"],"image_path":s["image_path"],"model_name":args.model_name,"task":args.task,"choices":[],"gt_answer":"unknown","raw_response":"","parsed_answer":"invalid","is_correct":False,"is_absent_object_selected":False,"is_unknown_correct":False,"asd_enabled":bool(args.vector_path),"asd_layer":args.asd_layer,"lambda_strength":args.lambda_strength,"alpha":args.alpha}
        f.write(json.dumps(rec)+"\n")
if __name__=="__main__": main()
