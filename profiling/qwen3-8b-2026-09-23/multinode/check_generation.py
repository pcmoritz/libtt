import argparse, json, pathlib, requests, unicodedata
from transformers import AutoTokenizer
p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
t=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
rows=[]
for prompt, expected in [('What is the capital of France? Reply with only the city name.','paris'),('What is 2 + 2? Reply with only the number.','4'),('What is the chemical symbol for water? Reply with only the formula.','h2o')]:
    text=t.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    r=requests.post('http://127.0.0.1:31000/generate',json={'text':text,'sampling_params':{'temperature':0,'max_new_tokens':32}},timeout=1800)
    r.raise_for_status(); result=r.json()
    rows.append({'prompt':prompt,'expected':expected,'passed':expected == unicodedata.normalize('NFKC', result['text']).strip().lower(),'response':result})
    pathlib.Path(a.output).write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows[-1]),flush=True)
if not all(row['passed'] for row in rows): raise SystemExit('Generation smoke check failed')
