"""Scheduler boundary tests: real temporary queues, mocked scheduler commands."""

import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dimensionality_manuscript.pipeline import Job, JobQueue
from dimensionality_manuscript.scripts import sge_submit, slurm_submit, submit_common


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    session = SimpleNamespace(session_uid="test-session")
    config = SimpleNamespace(key=lambda: "config-key", summary=lambda: "test-config")
    jobs = [Job(session=session, analysis_config=config)]
    plan = Mock()
    plan._collect_jobs.return_value = (jobs, 0)
    monkeypatch.setattr(submit_common, "collect_sessions_from_file", Mock(return_value=[session]))
    monkeypatch.setattr(submit_common, "build_analysis_configs", Mock(return_value=[config]))
    monkeypatch.setattr(submit_common, "AnalysisPlan", Mock(return_value=plan))
    return tmp_path / "results.db", plan


def test_shared_planning_creates_isolated_batches(prepared):
    db, plan = prepared
    first = submit_common.prepare_batch(db, ["cvpca"], Path("sessions.json"), True, {"center": True})
    second = submit_common.prepare_batch(db, ["cvpca"], Path("sessions.json"))
    assert first != second
    queue = JobQueue(db)
    assert queue.status_summary(first) == {"pending": 1}
    assert queue.status_summary(second) == {"pending": 1}
    submit_common.build_analysis_configs.assert_called_with(include=["cvpca"], param_filters=None)
    assert plan._collect_jobs.call_args_list[0].kwargs == {"force_remake": False, "skip_errors": True}


def test_shared_planning_empty_does_not_create_batch(prepared):
    db, plan = prepared
    plan._collect_jobs.return_value = ([], 0)
    assert submit_common.prepare_batch(db, sessions_file=Path("sessions.json")) is None
    assert JobQueue(db).list_batches() == []


@pytest.fixture
def slurm_options(tmp_path):
    sessions = tmp_path / "sessions with, commas.json"
    sessions.write_text("[]")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").touch()
    return dict(sessions_file=sessions, python_module="python/test", venv=venv,
                db_path=tmp_path / "isolated" / "results.db", log_dir=tmp_path / "logs",
                n_workers=3, cores_per_worker=4, max_jobs_per_worker=1)


def test_slurm_resources_are_per_array_worker(slurm_options):
    cmd = slurm_submit._build_sbatch_command(3, "0:30:00", "8G", 4, "10G", slurm_options["log_dir"])
    assert cmd[0] == "sbatch"
    for option in ("--array=1-3", "--nodes=1", "--ntasks=1", "--cpus-per-task=4",
                   "--mem-per-cpu=8G", "--time=0:30:00", "--gres=tmpfs:10G", "--export=ALL"):
        assert option in cmd
    assert any("%A.%a.log" in option for option in cmd)
    assert not any("gpu" in option for option in cmd)


def test_slurm_submission_exports_fresh_batch_and_creates_logs(monkeypatch, slurm_options):
    monkeypatch.setenv("DIM_MANUSCRIPT_BATCH_ID", "stale-batch")
    monkeypatch.setenv("VRANALYSIS_PYTHON_MODULE", "old-python")
    prepare = Mock(return_value="new-batch")
    monkeypatch.setattr(slurm_submit, "prepare_batch", prepare)

    def sbatch(cmd, **kwargs):
        assert slurm_options["log_dir"].is_dir()
        env = kwargs["env"]
        assert env["DIM_MANUSCRIPT_BATCH_ID"] == "new-batch"
        assert env["VRANALYSIS_PYTHON_MODULE"] == "python/test"
        assert env["DIM_MANUSCRIPT_MAX_JOBS"] == "1"
        assert env["DIM_MANUSCRIPT_DB_PATH"] == str(slurm_options["db_path"].resolve())
        assert env["DIM_MANUSCRIPT_SESSIONS_FILE"] == str(slurm_options["sessions_file"].resolve())
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")

    runner = Mock(side_effect=sbatch)
    monkeypatch.setattr(slurm_submit.subprocess, "run", runner)
    assert slurm_submit.submit(**slurm_options) == "new-batch"
    runner.assert_called_once()


def test_slurm_dry_run_populates_without_scheduler_or_logs(monkeypatch, slurm_options):
    prepare = Mock(return_value="preview-batch")
    runner = Mock()
    monkeypatch.setattr(slurm_submit, "prepare_batch", prepare)
    monkeypatch.setattr(slurm_submit.subprocess, "run", runner)
    assert slurm_submit.submit(**slurm_options, dry_run=True) == "preview-batch"
    prepare.assert_called_once()
    runner.assert_not_called()
    assert not slurm_options["log_dir"].exists()


def test_slurm_empty_batch_never_submits(monkeypatch, slurm_options):
    monkeypatch.setattr(slurm_submit, "prepare_batch", Mock(return_value=None))
    runner = Mock()
    monkeypatch.setattr(slurm_submit.subprocess, "run", runner)
    assert slurm_submit.submit(**slurm_options) is None
    runner.assert_not_called()


def test_slurm_rejection_is_reported(monkeypatch, slurm_options, capsys):
    monkeypatch.setattr(slurm_submit, "prepare_batch", Mock(return_value="pending-batch"))
    monkeypatch.setattr(slurm_submit.subprocess, "run", Mock(return_value=SimpleNamespace(
        returncode=1, stdout="", stderr="invalid resource request")))
    with pytest.raises(SystemExit, match="1"):
        slurm_submit.submit(**slurm_options)
    assert "invalid resource request" in capsys.readouterr().out


def test_slurm_clears_inherited_worker_limit(monkeypatch, slurm_options):
    monkeypatch.setenv("DIM_MANUSCRIPT_MAX_JOBS", "99")
    monkeypatch.setattr(slurm_submit, "prepare_batch", Mock(return_value="batch"))
    runner = Mock(return_value=SimpleNamespace(returncode=0, stdout="123", stderr=""))
    monkeypatch.setattr(slurm_submit.subprocess, "run", runner)
    slurm_options["max_jobs_per_worker"] = None
    slurm_submit.submit(**slurm_options)
    assert "DIM_MANUSCRIPT_MAX_JOBS" not in runner.call_args.kwargs["env"]


@pytest.mark.parametrize("field,value", [("n_workers", 0), ("cores_per_worker", -1), ("max_jobs_per_worker", 0)])
def test_slurm_invalid_counts_fail_before_batch(monkeypatch, slurm_options, field, value):
    prepare = Mock()
    monkeypatch.setattr(slurm_submit, "prepare_batch", prepare)
    slurm_options[field] = value
    with pytest.raises(ValueError):
        slurm_submit.submit(**slurm_options)
    prepare.assert_not_called()


@pytest.mark.parametrize("missing", ["manifest", "module", "venv"])
def test_slurm_missing_setup_fails_before_batch(monkeypatch, slurm_options, missing, tmp_path):
    monkeypatch.delenv("VRANALYSIS_PYTHON_MODULE", raising=False)
    prepare = Mock()
    monkeypatch.setattr(slurm_submit, "prepare_batch", prepare)
    if missing == "manifest":
        slurm_options["sessions_file"] = tmp_path / "missing.json"
    elif missing == "module":
        slurm_options["python_module"] = None
    else:
        slurm_options["venv"] = tmp_path / "missing-venv"
    with pytest.raises((FileNotFoundError, ValueError)):
        slurm_submit.submit(**slurm_options)
    prepare.assert_not_called()


def test_sge_still_uses_qsub_and_shared_planning(monkeypatch):
    prepare = Mock(return_value="sge-batch")
    runner = Mock(return_value=SimpleNamespace(returncode=0, stdout="submitted", stderr=""))
    monkeypatch.setattr(sge_submit, "prepare_batch", prepare)
    monkeypatch.setattr(sge_submit.subprocess, "run", runner)
    assert sge_submit.submit(n_workers=2, cores_per_worker=4, mem="8G") == "sge-batch"
    cmd = runner.call_args.args[0]
    assert cmd[:3] == ["qsub", "-t", "1-2"]
    assert cmd[cmd.index("-pe") + 1:cmd.index("-pe") + 3] == ["smp", "4"]
    assert "mem=8G" in cmd
    assert "DIM_MANUSCRIPT_BATCH_ID=sge-batch" in cmd[cmd.index("-v") + 1]


def _bash():
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    return executable


@pytest.mark.parametrize("module_fails", [False, True])
def test_slurm_launcher_forwards_worker_identity_and_limits(tmp_path, module_fails):
    # Execute the real launcher with fake modules/venv/python; no cluster needed.
    # A relative cwd keeps Git Bash's Windows-to-POSIX path handling out of the fixture.
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    init = tmp_path / "init.sh"
    init.write_text(
        "module() { return " + ("1" if module_fails else "0") + "; }\n"
        # Do not let an installed system module initialization override the stub.
        'source() { case "$1" in /etc/profile.d/modules.sh|/usr/share/Modules/init/bash) return 0 ;; '
        '*) builtin source "$@" ;; esac; }\n'
    )
    # Copy launcher to the temporary cwd so every fixture path is relative.
    (tmp_path / "worker.sh").write_bytes(slurm_submit._WORKER_SCRIPT.read_bytes())
    env = os.environ.copy()
    env.update(BASH_ENV="init.sh", DIM_MANUSCRIPT_REPO=".", DIM_MANUSCRIPT_DB_PATH="results.db",
               DIM_MANUSCRIPT_BATCH_ID="test-batch", DIM_MANUSCRIPT_SESSIONS_FILE="sessions.json",
               VRANALYSIS_PYTHON_MODULE="python/test", VRANALYSIS_VENV="venv",
               SLURM_ARRAY_JOB_ID="123", SLURM_ARRAY_TASK_ID="2", SLURM_CPUS_PER_TASK="4",
               DIM_MANUSCRIPT_MAX_JOBS="1")
    # Stub Python so exec still runs a real executable.
    stub = tmp_path / "python"
    stub.write_text('#!/bin/bash\nprintf "CPUS=%s OMP=%s\\n" "$NSLOTS" "$OMP_NUM_THREADS"\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    (tmp_path / "venv" / "bin" / "activate").write_text('export PATH="$PWD:$PATH"\n')
    result = subprocess.run([_bash(), "worker.sh"], cwd=tmp_path, env=env, capture_output=True, text=True)
    if module_fails:
        assert result.returncode != 0
        assert "--worker-id" not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert "CPUS=4 OMP=4" in result.stdout
        assert "slurm.123.2" in result.stdout
        assert "--max-jobs\n1" in result.stdout
        assert "--batch-id\ntest-batch" in result.stdout
