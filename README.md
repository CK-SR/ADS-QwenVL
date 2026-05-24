# ADS-QwenVL

基于 MSCOCO + Qwen3-VL 的受限 JSON 对象识别与 ASD（Activation Steering Decoding）实验框架。

## 数据目录
`data/coco/{train2017,val2017,annotations}`，标注文件使用 `instances_train2017.json` 与 `instances_val2017.json`。

## 实验1 baseline
```bash
python -m src.exp1_baseline_objects --coco_root data/coco --split val2017 --model_name Qwen/Qwen3-VL-2B-Instruct --limit 500 --output outputs/baseline/qwen3vl_2b_val_p2_objects.jsonl
```

## calibration 输出生成
```bash
python -m src.exp1_baseline_objects --coco_root data/coco --split train2017 --model_name Qwen/Qwen3-VL-2B-Instruct --limit 1000 --output outputs/baseline/qwen3vl_2b_train_p2_calib.jsonl
```

## ASD vector 提取
```bash
python -m src.asd_vector --coco_root data/coco --split train2017 --model_name Qwen/Qwen3-VL-2B-Instruct --baseline_jsonl outputs/baseline/qwen3vl_2b_train_p2_calib.jsonl --output_vector outputs/vectors/qwen3vl_2b_p2_coco_asd_vectors.pt
```

## 实验2 ASD 对象生成
```bash
python -m src.exp2_asd_objects --coco_root data/coco --split val2017 --model_name Qwen/Qwen3-VL-2B-Instruct --vector_path outputs/vectors/qwen3vl_2b_p2_coco_asd_vectors.pt --asd_layer 20 --lambda_strength 1.0 --alpha 1.0 --limit 500 --output outputs/asd/qwen3vl_2b_val_p2_asd_l20.jsonl
```

## 实验3 分类
```bash
python -m src.exp3_coco_classification --coco_root data/coco --split val2017 --task object_existence --model_name Qwen/Qwen3-VL-2B-Instruct --limit 500 --output outputs/classification/qwen3vl_2b_existence_baseline.jsonl
```

## 报告生成
```bash
python -m src.report_tables --out_dir outputs/reports
```

## 核心原则
ASD 不训练模型，不做 backward / optimizer.step，只做 forward 和 decoding-time intervention。
