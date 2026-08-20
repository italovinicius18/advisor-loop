from advisor_loop.journal import Journal

def test_append_has_and_replay(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.append({"kind": "query_result", "task_key": "abc123", "status": "success"})
    assert j.has("abc123") and not j.has("zzz")
    j2 = Journal(tmp_path / "j.jsonl")          # reabre: índice reconstruído do disco
    assert j2.has("abc123")
    assert j2.events("query_result")[0]["status"] == "success"
    assert "ts" in j2.events()[0]
