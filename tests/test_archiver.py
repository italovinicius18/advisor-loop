import subprocess, pytest
from advisor_loop.archiver import Archiver, SecretDetected, scan_secrets
from advisor_loop.journal import Journal

def test_scan_secrets():
    assert scan_secrets("nada aqui") == []
    assert scan_secrets("key=sk-or-v1-abc") != []

def test_publish_compresses_and_commits(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    j = Journal(tmp_path / "j.jsonl")
    j.append({"kind": "episode_end", "status": "completed"})
    out = Archiver(tmp_path).publish("pilot", j.path, {"episodes_run": 1}, push=False)
    assert (out / "journal.jsonl.zst").exists() and (out / "summary.json").exists()
    log = subprocess.run(["git", "-C", str(tmp_path), "log", "--oneline"],
                         capture_output=True, text=True).stdout
    assert "evidence: pilot" in log

def test_publish_blocks_secrets(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    j = Journal(tmp_path / "j.jsonl")
    j.append({"kind": "llm_call", "note": "sk-or-v1-LEAKED"})
    with pytest.raises(SecretDetected):
        Archiver(tmp_path).publish("pilot", j.path, {}, push=False)


def test_publish_scopes_commit_to_evidence_path_only(tmp_path):
    """A dirty index elsewhere (unrelated staged file) must NOT be swept into the
    evidence commit - the commit must be scoped to the evidence pathspec only."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)

    # Stage an unrelated dummy file BEFORE publishing, simulating a dirty index.
    dummy = tmp_path / "unrelated.txt"
    dummy.write_text("should not be committed by archiver")
    subprocess.run(["git", "-C", str(tmp_path), "add", "unrelated.txt"], check=True)

    j = Journal(tmp_path / "j.jsonl")
    j.append({"kind": "episode_end", "status": "completed"})
    out = Archiver(tmp_path).publish("pilot", j.path, {"episodes_run": 1}, push=False)

    stat = subprocess.run(["git", "-C", str(tmp_path), "show", "--stat", "HEAD"],
                          capture_output=True, text=True).stdout
    assert "evidence/pilot" in stat
    assert "unrelated.txt" not in stat, "commit must be scoped to the evidence path only"

    # The dummy file should still be staged (untouched), not committed.
    status = subprocess.run(["git", "-C", str(tmp_path), "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    assert "unrelated.txt" in status


def test_publish_default_push_is_false(tmp_path, monkeypatch):
    """push must default to False (opt-in push for automated runs) - calling publish
    without an explicit push kwarg must not attempt a git push."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    j = Journal(tmp_path / "j.jsonl")
    j.append({"kind": "episode_end", "status": "completed"})

    calls = []
    real_run = subprocess.run
    def spy(cmd, *a, **kw):
        calls.append(cmd)
        return real_run(cmd, *a, **kw)
    monkeypatch.setattr(subprocess, "run", spy)

    Archiver(tmp_path).publish("pilot", j.path, {"episodes_run": 1})  # no push kwarg
    assert not any("push" in c for c in calls), "push must be opt-in (default push=False)"
