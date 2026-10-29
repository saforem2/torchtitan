"""Execute the PBS shell with fake hardware/runtime, never submit a job.

Only the separately-owned broadcast/runtime block is substituted. All guards,
node slicing, CLI construction, child launches, waits and exit codes are real.
"""
import os
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "submit/aurora/followup-umbrella-boost-20261020.pbs"
BASH = "/opt/homebrew/bin/bash" if Path("/opt/homebrew/bin/bash").exists() else (shutil.which("bash") or "bash")


class BoostUmbrellaContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / ".venv/bin"
        self.bin.mkdir(parents=True)
        builder = self.root / "torchtitan/experiments/ezpz/blendcorpus/blendcorpus_builder.py"
        builder.parent.mkdir(parents=True)
        builder.write_text("BLENDCORPUS_RESUME_CONSUMED_SAMPLES")
        for name, body in {
            "git": '[[ "$1" == "-C" ]] && shift 2; case "$1" in rev-parse) echo deadbeef;; merge-base) exit 0;; *) exit 1;; esac',
            "lfs": "exit 0",
            "sleep": "exit 0",
            "python": 'exit "${FAKE_TORCH_RC:-0}"',
            "ezpz": 'if [[ "$1" == "yeet" ]]; then echo STAGING >> "$REPO/staging"; exit "${FAKE_STAGE_RC:-0}"; fi; echo "CALL resume=${BLENDCORPUS_RESUME_CONSUMED_SAMPLES-unset} $*"; exit "${FAKE_RC:-0}"',
        }.items():
            p = self.bin / name
            p.write_text("#!/bin/bash\n" + body + "\n")
            p.chmod(0o755)
        base = self.root / "outputs/checkpoints"
        for name in (
            "agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9000",
            "agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144",
        ):
            step = base / name / "step-1"
            step.mkdir(parents=True)
            (step / ".metadata").touch()
        self.initial = self.root / "initial/step-1"
        self.initial.mkdir(parents=True)
        (self.initial / ".metadata").touch()
        self.nodes = self.root / "nodes"
        self.nodes.write_text("".join(f"node-{i}\n" for i in range(2078)))
        self.full_script = self.root / "full-umbrella.sh"
        self.full_script.write_text(SCRIPT.read_text())
        archive = self.root / "runtime.tar.gz"
        archive.write_bytes(b"fixture only: stub yeet never extracts")
        self.archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        utils = self.root / "utils.sh"
        utils.write_text('ezpz_load_modules() { return "${FAKE_MODULE_RC:-0}"; }\nezpz_setup_job() { return 0; }\n')
        text = SCRIPT.read_text()
        start = text.index("# ── Venv broadcast")
        end = text.index("# ---- Launch function", start) if "# ── Umbrella launch" not in text[start:] else text.index("# ── Umbrella launch", start)
        text = text[:start] + 'VENV_DST="$REPO/.venv"; echo STAGING >> "$REPO/staging"\n' + text[end:]
        self.script = self.root / "umbrella.sh"
        self.script.write_text(text)
        default_seat = self.root / "default-seat-256"
        default_seat.mkdir()
        (default_seat / "torchtitan").symlink_to(self.root / "torchtitan")
        (default_seat / ".venv").symlink_to(self.root / ".venv")
        self.env = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin", REPO=str(self.root),
                        WORKDIR_20B512=str(self.root), WORKDIR_20B256=str(default_seat), WORKDIR_2B=str(self.root),
                        PBS_NODEFILE=str(self.nodes), PBS_JOBID="123.test", PBS_O_WORKDIR=str(self.root),
                        INITIAL_LOAD_20B512=str(self.initial), INITIAL_LOAD_20B256=str(self.initial),
                        INITIAL_LOAD_2B=str(self.initial), LAUNCH_STAGGER="0", SEAT_RELAUNCH="0",
                        VENV_ARCHIVE=str(archive), VENV_ARCHIVE_SHA=self.archive_sha,
                        LAUNCHER_VENV=str(self.bin.parent), EZPZ_UTILS=str(utils),
                        VENV_DST=str(self.bin.parent))

    def run_job(self, **env):
        return subprocess.run([BASH, str(self.script)], env=dict(self.env, **env),
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)

    def test_dry_run_resolves_historical_batches_without_staging(self):
        result = self.run_job(DRY_RUN="1")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("need=788", result.stdout)
        self.assertIn("20b n=512 gbs=12288", result.stdout)
        self.assertIn("20b n=256 gbs=6144", result.stdout)
        self.assertFalse((self.root / "staging").exists())
        self.assertFalse(list((self.root / "outputs/checkpoints").glob("*-fixedpos")))

    def test_fresh_fork_requires_explicit_valid_initial_load(self):
        for initial in ("", str(self.root / "missing")):
            with self.subTest(initial=initial):
                result = self.run_job(DRY_RUN="1", INITIAL_LOAD_20B512=initial)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("initial load", result.stdout)
                self.assertFalse((self.root / "staging").exists())

    def test_attempt_logs_are_preserved_and_failures_propagate(self):
        result = self.run_job(FAKE_RC="23", SEAT_RELAUNCH="1", SEAT_MAX_QUICK_DEATHS="2")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("rc=23", result.stdout)
        logs = list(self.root.glob("logs/**/*attempt-*.console.log"))
        self.assertEqual(len(logs), 4, result.stdout)
        self.assertTrue(all("CALL" in p.read_text() for p in logs))

    def run_full_job(self, **env):
        return subprocess.run([BASH, str(self.full_script)], env=dict(self.env, **env),
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)

    def test_staging_and_torch_fail_closed(self):
        for failure in ({"FAKE_STAGE_RC": "19"}, {"FAKE_TORCH_RC": "21"}, {"FAKE_MODULE_RC": "22"}):
            with self.subTest(failure=failure):
                result = self.run_full_job(**failure)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                expected = {"FAKE_STAGE_RC": "runtime staging failed", "FAKE_TORCH_RC": "staged torch/XPU probe failed", "FAKE_MODULE_RC": "ezpz_load_modules failed"}
                self.assertIn(expected[next(iter(failure))], result.stdout)
                self.assertNotIn("Venv OK", result.stdout)
                self.assertFalse(list(self.root.glob("logs/**/trainer-*.console.log")))

    def test_immutable_archive_mismatch_prevents_broadcast(self):
        result = self.run_full_job(VENV_ARCHIVE_SHA="0" * 64)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("archive SHA mismatch", result.stdout)
        self.assertFalse((self.root / "staging").exists())

    def test_full_runtime_success(self):
        result = self.run_full_job()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Venv OK", result.stdout)
        self.assertTrue((self.root / "staging").exists())

    def test_allocation_and_overlap_reject_before_staging(self):
        for hosts in ([f"node-{i}" for i in range(787)], ["duplicate"] * 788):
            with self.subTest(size=len(hosts)):
                self.nodes.write_text("\n".join(hosts) + "\n")
                result = self.run_job()
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse((self.root / "staging").exists())

    def test_source_pin_mismatch_prevents_staging(self):
        result = self.run_job(EXPECTED_HEAD_20B512="a" * 40)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse((self.root / "staging").exists())

    def test_selection_and_distinct_workdirs(self):
        seat = self.root / "seat-256"
        seat.mkdir()
        (seat / "torchtitan").symlink_to(self.root / "torchtitan")
        (seat / ".venv").symlink_to(self.root / ".venv")
        result = self.run_job(MULTI_ONLY="1", WORKDIR_20B256=str(seat))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("need=266", result.stdout)
        self.assertEqual(len(list(self.root.glob("logs/**/trainer-*.console.log"))), 1)
        self.assertIn(str(seat), result.stdout)

    def test_requires_explicit_workdir_and_absolute_initial_step(self):
        for override in ({"WORKDIR_20B512": ""}, {"INITIAL_LOAD_20B512": "initial/step-1"}):
            with self.subTest(override=override):
                result = self.run_job(DRY_RUN="1", **override)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse((self.root / "staging").exists())

    def test_selected_seats_cannot_share_launcher_workdir(self):
        result = self.run_job(DRY_RUN="1", WORKDIR_20B256=str(self.root))
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("shared workdir", result.stdout)

    def test_stale_unselected_hostfiles_do_not_poison_slicing(self):
        logdir = self.root / "logs/multi-autoretry-123"
        logdir.mkdir(parents=True)
        (logdir / "trainer-9.hostfile").write_text("node-0\n")
        result = self.run_job(DRY_RUN="1")
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_invalid_selection_fails_before_staging(self):
        result = self.run_job(MULTI_ONLY="9", DRY_RUN="1")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse((self.root / "staging").exists())

    def test_real_child_wait_and_resume_cli(self):
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stdout)
        logs = list(self.root.glob("logs/**/trainer-*.console.log"))
        self.assertEqual(len(logs), 2)
        texts = [p.read_text() for p in logs]
        self.assertTrue(all(f"--checkpoint.initial-load-path={self.initial}" in t for t in texts))
        self.assertTrue(all("--checkpoint.no-initial-load-model-only" in t for t in texts))
        self.assertTrue(all("--optimizer sophiag" in t for t in texts))
        self.assertTrue(all("--hf-assets-path=" in t for t in texts))
        self.assertTrue(any("resume=15482880" in t and "--training.global-batch-size=12288" in t for t in texts))
        self.assertTrue(any("resume=10002432" in t and "--training.global-batch-size=6144" in t for t in texts))
        self.assertTrue(all("--training.local-batch-size=2" in t and "--training.seq-len=8192" in t for t in texts))
        self.assertNotIn("not a child", result.stdout)


if __name__ == "__main__":
    unittest.main()
