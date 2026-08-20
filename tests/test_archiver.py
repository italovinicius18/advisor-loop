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
