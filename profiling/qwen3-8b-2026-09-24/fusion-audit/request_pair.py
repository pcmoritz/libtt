import concurrent.futures,json,pathlib,requests,sys,threading
out=pathlib.Path(sys.argv[1]);barrier=threading.Barrier(2)
def run(i):
    barrier.wait()
    r=requests.post('http://127.0.0.1:31000/generate',json={'text':'The capital of France is' if i==0 else 'The capital of Germany is','sampling_params':{'temperature':0,'max_new_tokens':16,'ignore_eos':True}},timeout=1200)
    r.raise_for_status();(out/f'concurrent-{i}.json').write_text(json.dumps(r.json(),indent=2))
with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(run,range(2)))
print('concurrent done',flush=True)
