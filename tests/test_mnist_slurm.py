"""Check SLURM orchestration without submitting real jobs."""

import json
import os
from pathlib import Path
import shutil
import subprocess


MNIST = Path(__file__).resolve().parents[1] / 'experiments' / 'mnist'


def test_submission_dependencies_resources_and_checkpoint_cleanup(tmp_path):
    # A copied checkout keeps generated result directories inside this test.
    folder = tmp_path / 'experiments' / 'mnist'
    folder.mkdir(parents=True)
    for name in ['submit.sh', 'run.slurm']:
        shutil.copyfile(MNIST / name, folder / name)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    calls_file = tmp_path / 'calls.jsonl'
    sbatch = bin_dir / 'sbatch'
    sbatch.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
path = Path(os.environ['SBATCH_CALLS'])
with path.open('a') as file:
    file.write(json.dumps(sys.argv[1:]) + '\\n')
print(1000 + len(path.read_text().splitlines()))
''')
    sbatch.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in {'RUNS', 'DIGITAL_EPOCHS', 'OI_EPOCHS'}}
    env.update({'PATH': f'{bin_dir}:{os.environ["PATH"]}',
           'SBATCH_CALLS': str(calls_file),
           'GPU_PARTITION': 'test-gpu', 'CPU_PARTITION': 'test-cpu'})
    subprocess.run(['bash', str(folder / 'submit.sh')], env=env, check=True,
                   capture_output=True, text=True)
    digital, wpo, oi, cleanup = [json.loads(line) for line in calls_file.read_text().splitlines()]
    assert '--dependency=aftercorr:1001' in wpo
    assert '--dependency=afterok:1001' in oi
    assert '--dependency=afterany:1001:1002:1003' in cleanup
    assert not any(arg.startswith('--dependency') for arg in digital)
    for args in [digital, wpo, oi]:
        assert '--array=1-5' in args
        assert '--kill-on-invalid-dep=yes' in args
        assert not any(arg.startswith('--constraint') for arg in args)
    for args in [digital, oi]:
        assert '--gres=gpu:1' in args and '--partition=test-gpu' in args
    assert '--partition=test-cpu' in wpo
    assert '--cpus-per-task=64' in wpo
    assert '--hint=nomultithread' in wpo
    assert digital[-1] == '14'
    assert oi[-1] == '5'
    assert not any(arg.startswith('--gres') for arg in wpo)

    run_dir = next((tmp_path / 'results' / 'mnist').glob('run-*'))
    checkpoint = run_dir / 'checkpoints' / 'seed-1.pt'
    checkpoint.touch()
    history = run_dir / 'seed-1' / 'digital.json'
    history.write_text('{}\n')
    subprocess.run(['bash', str(MNIST / 'run.slurm'), 'cleanup', str(run_dir)],
                   cwd=MNIST.parents[1], check=True, capture_output=True, text=True)
    assert not (run_dir / 'checkpoints').exists()
    assert history.read_text() == '{}\n'
