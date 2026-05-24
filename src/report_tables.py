from __future__ import annotations
import argparse, json, os, glob
import pandas as pd
from .metrics import compute_object_metrics, compute_yesno_metrics, compute_multichoice_metrics

def read_jsonl(p):
    with open(p) as f:return [json.loads(x) for x in f]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--baseline_glob",default="outputs/baseline/*.jsonl"); ap.add_argument("--asd_glob",default="outputs/asd/*.jsonl"); ap.add_argument("--cls_glob",default="outputs/classification/*.jsonl"); ap.add_argument("--out_dir",default="outputs/reports"); args=ap.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    b=[]
    for p in glob.glob(args.baseline_glob):
      r=read_jsonl(p); m=compute_object_metrics(r); m["model_name"]=r[0]["model_name"] if r else ""; b.append(m)
    pd.DataFrame(b).to_csv(os.path.join(args.out_dir,"table1_baseline.csv"),index=False)
    a=[]
    for p in glob.glob(args.asd_glob):
      r=read_jsonl(p); m=compute_object_metrics(r); m.update({k:r[0].get(k) for k in ["model_name","asd_layer","lambda_strength","alpha"]}); a.append(m)
    pd.DataFrame(a).to_csv(os.path.join(args.out_dir,"table2_asd.csv"),index=False)
    c3=[]; c4=[]
    for p in glob.glob(args.cls_glob):
      r=read_jsonl(p)
      if not r: continue
      method="asd" if r[0].get("asd_enabled") else "baseline"
      if r[0]["task"]=="object_existence":
        m=compute_yesno_metrics(r); m.update({"model_name":r[0]["model_name"],"method":method}); c3.append(m)
      else:
        m=compute_multichoice_metrics(r); m.update({"model_name":r[0]["model_name"],"method":method}); c4.append(m)
    pd.DataFrame(c3).to_csv(os.path.join(args.out_dir,"table3_existence.csv"),index=False)
    pd.DataFrame(c4).to_csv(os.path.join(args.out_dir,"table4_multichoice.csv"),index=False)

if __name__=="__main__": main()
