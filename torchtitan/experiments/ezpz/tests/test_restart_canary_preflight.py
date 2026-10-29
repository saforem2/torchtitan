"""Run the real restart canary's pre-allocation guards without Torch/PBS."""
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/aurora/smoke_20b_fixedpos_restart.pbs'


class RestartCanaryPreflight(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / 'checkout'
        self.repo.mkdir()
        for args in [['init', '-q'], ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture', '--allow-empty']]:
            subprocess.run(['git', '-C', str(self.repo), *args], check=True, capture_output=True)
        sha = subprocess.check_output(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], text=True).strip()
        source = self.root / 'seed/step-100'
        source.mkdir(parents=True)
        (source / '.metadata').write_bytes(b'test fixture, not a real checkpoint')
        assets = self.root / 'assets'
        assets.mkdir()
        (assets / 'tokenizer.json').write_text('{}')
        (self.repo / 'data.txt').write_text('fixture')
        driver = self.root / 'driver/bin'
        driver.mkdir(parents=True)
        (driver / 'ezpz').symlink_to(shutil.which('true') or '/usr/bin/true')
        archive = self.root / 'runtime.tar.gz'
        archive.write_bytes(b'unit-test archive fixture')
        helper = self.root / 'helper.sh'
        helper.write_text('exit 99 # must never run in preflight')
        self.env = dict(os.environ, RESTART_WORKTREE=str(self.repo), RESTART_SHA=sha,
                        SOURCE_CHECKPOINT=str(source), SOURCE_STEP='100', RESUME_N='960',
                        SOURCE_FILES='1', RUNTIME_ARCHIVE=str(archive),
                        RUNTIME_SHA=hashlib.sha256(archive.read_bytes()).hexdigest(),
                        LAUNCHER_VENV=str(driver.parent), EZPZ_UTILS=str(helper),
                        HF_ASSETS=str(assets), DFL='data.txt', DATA_CACHE=str(self.root),
                        RESTART_OUT=str(self.root / 'output'))

    def run_preflight(self):
        return subprocess.run(['bash', str(SCRIPT), '--preflight-only'], env=self.env,
                              text=True, capture_output=True)

    def test_valid_plan_passes_without_staging_or_output(self):
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('RESTART_PREFLIGHT_PASS', result.stdout)
        self.assertFalse((self.root / 'output').exists())

    def test_large_production_offset_has_independent_dataset_horizon(self):
        self.env['RESUME_N'] = '15482880'
        nodes = self.root / 'synthetic-nodes'
        nodes.write_text(''.join(f'unit-host-{i}\n' for i in range(4)))
        self.env['PBS_NODEFILE'] = str(nodes)
        result = subprocess.run(['bash', str(SCRIPT), '--render-plan'], env=self.env,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        plan = dict(line.split('=', 1) for line in result.stdout.splitlines())
        self.assertEqual(int(plan['END_STEP']), 103)
        self.assertEqual(int(plan['GBS']), 48)
        self.assertGreater(int(plan['DATASET_ITERS']) * 48, 15482880 + 3 * 48)
        self.assertFalse((self.root / 'output').exists())

    def test_wrong_source_sha_fails(self):
        self.env['RESTART_SHA'] = '0' * 40
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('source SHA mismatch', result.stderr)

    def test_missing_required_resume_argument_fails(self):
        del self.env['RESUME_N']
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('RESUME_N', result.stderr)

    def test_incomplete_checkpoint_fails(self):
        self.env['SOURCE_FILES'] = '6145'
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('source shard count mismatch', result.stderr)

    def test_mutated_runtime_fails(self):
        Path(self.env['RUNTIME_ARCHIVE']).write_bytes(b'changed fixture')
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('immutable runtime checksum mismatch', result.stderr)


if __name__ == '__main__':
    unittest.main()
