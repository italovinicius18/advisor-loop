from advisor_loop.journal import Journal

def test_append_has_and_replay(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.append({"kind": "query_result", "task_key": "abc123", "status": "success"})
    assert j.has("abc123") and not j.has("zzz")
    j2 = Journal(tmp_path / "j.jsonl")          # reabre: índice reconstruído do disco
    assert j2.has("abc123")
    assert j2.events("query_result")[0]["status"] == "success"
    assert "ts" in j2.events()[0]

def test_ts_authority_and_corrupt_tail(tmp_path):
    p = tmp_path / "j.jsonl"
    j = Journal(p)
    j.append({"kind": "x", "ts": "1999-01-01T00:00:00"})   # caller ts must NOT win
    assert j.events()[0]["ts"] != "1999-01-01T00:00:00"
    with open(p, "a") as f:
        f.write('{"kind": "y", "task_key": "trunc')          # linha truncada (crash simulado)
    j2 = Journal(p)                                          # não pode levantar
    assert len(j2.events()) == 1 and not j2.has("trunc")
