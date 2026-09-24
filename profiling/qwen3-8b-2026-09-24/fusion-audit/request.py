import json,time,requests,sys,pathlib
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
tp=int(sys.argv[2])
for _ in range(90):
    try:
        r=requests.get('http://127.0.0.1:31000/get_server_info',timeout=1)
        if r.ok and r.json().get('tp_size')==tp: break
    except requests.RequestException: pass
    time.sleep(1)
else: raise RuntimeError('Server not ready')
(out/'server-info.json').write_text(json.dumps(r.json(),indent=2))
for i in range(2):
    r=requests.post('http://127.0.0.1:31000/generate',json={'text':'The capital of France is','sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True}},timeout=1200)
    r.raise_for_status();(out/f'response-{i}.json').write_text(json.dumps(r.json(),indent=2))
print('done',tp,flush=True)
