import subprocess
import json
from pathlib import Path

def save_experiment_params(log_dir, args):
    log_dir = Path(log_dir)
    with (log_dir / 'parameters.json').open('w') as fp:
        params = vars(args)
        params['commit'] = subprocess.run('git rev-parse HEAD', shell=True, stdout=subprocess.PIPE).stdout
        json.dump(vars(args), fp, indent=4)

    with (log_dir / 'patch').open('w') as fp:
        fp.write(subprocess.run('git diff', shell=True, stdout=subprocess.PIPE).stdout)
