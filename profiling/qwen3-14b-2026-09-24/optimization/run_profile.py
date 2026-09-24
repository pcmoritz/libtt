from pathlib import Path
import subprocess, os, sys, time, requests
root=Path('/tmp/libtt-profile/14b-opt')
python='/tmp/libtt-profile/multinode-venv/bin/python'
stop=[python,'/tmp/libtt-profile/round2/stop_server.py']
for tp in ([int(x) for x in sys.argv[1:]] or [1,2]):
    subprocess.run(stop,check=True)
    (root/f'tp{tp}').mkdir(exist_ok=True)
    print(f'START TP{tp}',flush=True)
    with (root/f'server-tp{tp}.log').open('w') as log:
        server=subprocess.Popen(['bash','/tmp/libtt-profile/qwen3-14b/launch.sh',str(tp)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PROFILE_VARIANT='profile',LIBTT_PROFILE_CSV=str(root/f'tp{tp}'/'programs')))
        try:
            deadline=time.monotonic()+1800
            while time.monotonic()<deadline:
                if server.poll() is not None: raise RuntimeError(f'TP{tp} server exited {server.returncode}')
                try:
                    r=requests.get('http://127.0.0.1:31000/get_server_info',timeout=2)
                    if r.ok and r.json().get('tp_size')==tp: break
                except requests.RequestException: pass
                time.sleep(2)
            else: raise RuntimeError(f'TP{tp} server not ready')
            print(f'READY TP{tp}',flush=True)
            with (root/f'benchmark-tp{tp}.log').open('w') as bench:
                subprocess.run([python,'/tmp/libtt-profile/round3/profile_request.py',str(root/f'tp{tp}'),str(tp)],stdout=bench,stderr=subprocess.STDOUT,check=True)
        finally:
            shutdown=subprocess.Popen(stop)
            server.wait(timeout=30)
            assert shutdown.wait(timeout=30)==0
    print(f'DONE TP{tp}',flush=True)
