from pathlib import Path
import subprocess
root=Path('/tmp/libtt-profile/14b-opt')
python='/tmp/libtt-profile/multinode-venv/bin/python'
def install(name):
    subprocess.run(['uv','pip','install','--python',python,'--force-reinstall','--no-deps',str(root/'wheels'/name/'jax_tt_plugin-0.1.0-py3-none-linux_x86_64.whl')],check=True)
try:
    for name in ('baseline','final'):
        install(name)
        subprocess.run([python,str(root/f'run_recheck_{name}.py'),*(['1','2'] if name == 'baseline' else ['2'])],check=True)
finally:
    install('final')
