import argparse, concurrent.futures, json, pathlib, statistics, time
import requests

p = argparse.ArgumentParser()
p.add_argument('--tp', type=int, required=True)
p.add_argument('--port', type=int, default=31000)
p.add_argument('--output', required=True)
a = p.parse_args()
out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
base = f'http://127.0.0.1:{a.port}'
rows = []
prompts = {
    'short': 'The capital of France is',
    'long': 'Read the following passage and summarize its main points. ' + 'The city expanded its public transportation system with new trains, buses, and protected bicycle lanes. Residents reported shorter commutes and better access to schools and jobs. ' * 6 + '\nSummary:',
}

def request(name, phase, repeat, output_tokens=128):
    payload = {'text': prompts[name], 'stream': True, 'sampling_params': {'temperature': 0, 'max_new_tokens': output_tokens, 'ignore_eos': True}}
    start = time.perf_counter(); first = None; last = None; events = []; final = None
    with requests.post(base + '/generate', json=payload, stream=True, timeout=(10, 1800)) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=1):
            if not line.startswith(b'data: '): continue
            data = line[6:]
            if data == b'[DONE]': break
            item = json.loads(data)
            now = time.perf_counter()
            count = item.get('meta_info', {}).get('completion_tokens', 0)
            if count:
                if first is None: first = now
                if not events or count > events[-1]['tokens']:
                    events.append({'time_s': now-start, 'tokens': count})
                    last = now
            final = item
    end = time.perf_counter()
    if final is None or first is None: raise RuntimeError(f'No tokens: {final}')
    meta = final['meta_info']; n = meta['completion_tokens']
    if n != output_tokens: raise RuntimeError(f'Expected {output_tokens} tokens: {final}')
    initial_n = events[0]['tokens']
    decode_s = last-first
    row = {'tp': a.tp, 'prompt': name, 'phase': phase, 'repeat': repeat,
           'prompt_tokens': meta.get('prompt_tokens'), 'completion_tokens': n,
           'ttft_s': first-start, 'e2e_s': end-start,
           'decode_tokens_per_s': (n-initial_n)/decode_s if decode_s else None,
           'tpot_ms': 1000*decode_s/(n-initial_n) if n>initial_n else None,
           'events': events, 'response': final}
    print(json.dumps({k:v for k,v in row.items() if k not in ('events','response')}), flush=True)
    return row

def save():
    (out/'requests.json').write_text(json.dumps(rows, indent=2))

for name in prompts:
    for i in range(3):
        rows.append(request(name, 'warmup', i)); save()
    for i in range(5):
        rows.append(request(name, 'measured', i)); save()

for i in range(2):
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        batch = list(pool.map(lambda _: request('short', 'concurrent_warmup' if i==0 else 'concurrent', i), range(2)))
    elapsed = time.perf_counter()-start
    for row in batch: row['batch_e2e_s']=elapsed; row['batch_output_tokens_per_s']=256/elapsed
    rows.extend(batch); save()

profile_dir = str(out/'trace')
r = requests.post(base+'/start_profile', json={'output_dir': profile_dir, 'host_tracer_level': 2, 'python_tracer_level': 1}, timeout=60)
(out/'profile-start.txt').write_text(r.text)
r.raise_for_status()
try:
    rows.append(request('short', 'profile', 0, 32)); save()
finally:
    r = requests.post(base+'/stop_profile', json={}, timeout=120)
    (out/'profile-stop.txt').write_text(r.text)
    r.raise_for_status()
summary = {}
for name in prompts:
    selected = [r for r in rows if r['prompt']==name and r['phase']=='measured']
    summary[name] = {k:statistics.median(r[k] for r in selected) for k in ('prompt_tokens','completion_tokens','ttft_s','e2e_s','decode_tokens_per_s','tpot_ms')}
(out/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
