import psutil
owned=[]
for p in psutil.process_iter(['cmdline']):
 if 'sgl_jax.launch_server' in (p.info['cmdline'] or []):
  owned.extend(p.children(recursive=True));owned.append(p);p.terminate()
_,alive=psutil.wait_procs(owned,timeout=8)
for p in alive:
 try:p.terminate()
 except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs(alive,timeout=4)
for p in alive:
 try:p.kill()
 except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs(alive,timeout=2)
assert not alive
