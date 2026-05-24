from __future__ import annotations
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

P2_PROMPT = '''You are an object recognition system.\n\nTask:\nIdentify only the clearly visible objects that belong to the MSCOCO 80 object categories.\n\nRules:\n1. Only output object category names from the allowed category list.\n2. Do not mention objects outside the allowed category list.\n3. Do not guess uncertain, tiny, blurry, or heavily occluded objects.\n4. If no allowed object is clearly visible, output an empty list.\n5. Use exact category names from the allowed category list.\n\nAllowed categories:\nperson, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.\n\nOutput JSON only:\n{"objects": ["category_name_1", "category_name_2"]}'''

DTYPE_MAP={"bf16":torch.bfloat16,"fp16":torch.float16,"fp32":torch.float32}

def load_qwen_vl(model_name,dtype="bf16",device_map="auto"):
    td=DTYPE_MAP.get(dtype,torch.bfloat16)
    processor=AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    try:
        model=AutoModelForImageTextToText.from_pretrained(model_name,torch_dtype=td,device_map=device_map,trust_remote_code=True)
    except Exception:
        from transformers import Qwen3VLForConditionalGeneration
        model=Qwen3VLForConditionalGeneration.from_pretrained(model_name,torch_dtype=td,device_map=device_map,trust_remote_code=True)
    model.eval(); return model, processor

def _to_dev(batch,model):
    dev=next(model.parameters()).device
    return {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}

def build_messages(prompt,image_path):
    return [{"role":"user","content":[{"type":"image","image":image_path},{"type":"text","text":prompt}]}]

def generate_response(model,processor,image_path,prompt,max_new_tokens=128):
    msgs=build_messages(prompt,image_path)
    text=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    image=Image.open(image_path).convert("RGB")
    ins=_to_dev(processor(text=[text],images=[image],return_tensors="pt"),model)
    out=model.generate(**ins,max_new_tokens=max_new_tokens,temperature=0.0,do_sample=False)
    gen=out[:,ins["input_ids"].shape[1]:]
    return processor.tokenizer.batch_decode(gen,skip_special_tokens=True)[0]
