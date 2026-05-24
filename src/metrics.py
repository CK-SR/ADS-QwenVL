from __future__ import annotations

def safe_div(a, b): return float(a)/float(b) if b else 0.0

def compute_object_metrics(rows):
    n = len(rows)
    pred = sum(r.get("num_pred_objects", 0) for r in rows)
    factual = sum(r.get("num_factual_objects", 0) for r in rows)
    hall = sum(r.get("num_hallucinated_objects", 0) for r in rows)
    gt = sum(r.get("num_gt_objects", 0) for r in rows)
    p = safe_div(factual, pred)
    r = safe_div(factual, gt)
    f1 = safe_div(2*p*r, p+r)
    return {
        "num_images": n, "object_precision": p, "object_recall": r, "object_f1": f1,
        "hallucinated_object_rate": safe_div(hall, pred),
        "image_hallucination_rate": safe_div(sum(1 for x in rows if x.get("has_hallucination")), n),
        "avg_num_pred_objects": safe_div(pred, n),
        "avg_num_hallucinated_objects": safe_div(hall, n),
        "invalid_json_rate": safe_div(sum(1 for x in rows if x.get("invalid_json")), n),
        "invalid_object_rate": safe_div(sum(len(x.get("invalid_objects", [])) for x in rows), max(pred,1)),
        "empty_output_rate": safe_div(sum(1 for x in rows if x.get("num_pred_objects",0)==0), n),
    }

def compute_yesno_metrics(rows):
    n=len(rows); tp=sum(r["gt_answer"]=="yes" and r["parsed_answer"]=="yes" for r in rows)
    fp=sum(r["gt_answer"]=="no" and r["parsed_answer"]=="yes" for r in rows)
    fn=sum(r["gt_answer"]=="yes" and r["parsed_answer"]=="no" for r in rows)
    p=safe_div(tp,tp+fp); r=safe_div(tp,tp+fn)
    return {"accuracy":safe_div(sum(r.get("is_correct",False) for r in rows),n),"yes_precision":p,"yes_recall":r,"yes_f1":safe_div(2*p*r,p+r),"false_yes_rate":safe_div(sum(r.get("is_false_yes",False) for r in rows),sum(r["gt_answer"]=="no" for r in rows)),"false_no_rate":safe_div(sum(r.get("is_false_no",False) for r in rows),sum(r["gt_answer"]=="yes" for r in rows)),"invalid_rate":safe_div(sum(r.get("parsed_answer")=="invalid" for r in rows),n)}

def compute_multichoice_metrics(rows):
    n=len(rows)
    return {"accuracy":safe_div(sum(r.get("is_correct",False) for r in rows),n),"absent_object_selection_rate":safe_div(sum(r.get("is_absent_object_selected",False) for r in rows),n),"unknown_correct_rate":safe_div(sum(r.get("gt_answer")=="unknown" and r.get("is_correct") for r in rows),sum(r.get("gt_answer")=="unknown" for r in rows)),"gt_object_correct_rate":safe_div(sum(r.get("gt_answer")!="unknown" and r.get("is_correct") for r in rows),sum(r.get("gt_answer")!="unknown" for r in rows)),"invalid_rate":safe_div(sum(r.get("parsed_answer")=="invalid" for r in rows),n)}
