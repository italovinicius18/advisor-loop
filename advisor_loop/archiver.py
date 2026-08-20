import json, re, subprocess
from pathlib import Path
import zstandard

SECRET_PATTERNS = [r"sk-or-[\w-]+", r"sk-ant-[\w-]+", r"AKIA[0-9A-Z]{16}", r"-----BEGIN"]

class SecretDetected(Exception):
    pass

def scan_secrets(text: str) -> list[str]:
    return [p for p in SECRET_PATTERNS if re.search(p, text)]

class Archiver:
    def __init__(self, repo_root: str | Path):
        self.root = Path(repo_root)

    def publish(self, campaign: str, journal_path: Path, extra: dict, push: bool = True) -> Path:
        raw = Path(journal_path).read_text()
        summary = json.dumps(extra, ensure_ascii=False, indent=1)
        if scan_secrets(raw) or scan_secrets(summary):
            raise SecretDetected("segredo detectado; publicacao abortada")
        out = self.root / "evidence" / campaign
        out.mkdir(parents=True, exist_ok=True)
        (out / "journal.jsonl.zst").write_bytes(zstandard.compress(raw.encode()))
        (out / "summary.json").write_text(summary)
        rel = out.relative_to(self.root)
        subprocess.run(["git", "-C", str(self.root), "add", str(rel)], check=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-m",
                        f"evidence: {campaign}"], check=True)
        if push:
            subprocess.run(["git", "-C", str(self.root), "push"], check=True)
        return out
