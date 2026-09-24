import argparse,requests,time,subprocess,sys,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--tp',required=True,type=int);p.add_argument('--output',required=True);p.add_argument('--full',action='store_true');a=p.parse_args()
for i in range(60):
 try:
  r=requests.get('http://127.0.0.1:31000/get_server_info',timeout=1)
  if r.ok and r.json().get('tp_size')==a.tp:break
 except requests.RequestException:pass
 time.sleep(1)
else:raise RuntimeError('Expected server not ready')
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'server-info.json').write_text(json.dumps(r.json(),indent=2))
subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name('benchmark.py' if a.full else 'quick_bench.py')),'--tp',str(a.tp),'--output',a.output],check=True)
