from zoo import gpu_identity as gi


def _run(root, queue, rid, curve, weights_hash, elapsed="1.0"):
    d = root / queue / "runs" / rid
    d.mkdir(parents=True)
    (d / "curve.csv").write_text(f"step,target_logprob,elapsed_s\n0,{curve},{elapsed}\n")
    (d / "summary.json").write_text('{"t_cross": 30, "seconds": %s, "run": "%s"}' % (elapsed, rid))
    (d / "meta.json").write_text('{"env": "%s"}' % queue)
    with open(root / queue / "files.sha256", "a") as fh:
        for f in ("curve.csv", "summary.json", "meta.json"):
            fh.write(f"hash-of-{queue}-{f}  1  {rid}/{f}\n")
        fh.write(f"{weights_hash}  9  {rid}/final_fp16.safetensors\n")


def test_identity_ignores_timing_names_and_meta_but_not_values(tmp_path, monkeypatch):
    monkeypatch.setattr(gi, "COLAB", tmp_path)
    _run(tmp_path, "q40", "colabeqa100-m-a0", "-1.25", "w1", elapsed="5.0")
    _run(tmp_path, "q80", "colabeqa100x80-m-a0", "-1.25", "w1", elapsed="9.0")
    rows = gi.compare_pair(("q40", "colabeqa100-m-a0"), ("q80", "colabeqa100x80-m-a0"))
    assert {r["file"] for r in rows} == {"curve.csv", "summary.json", "final_fp16.safetensors"}
    assert all(r["identical"] for r in rows)
    _run(tmp_path, "q80", "colabeqa100x80-m-a1", "-1.2500001", "w2")
    _run(tmp_path, "q40", "colabeqa100-m-a1", "-1.25", "w1")
    rows = {r["file"]: r["identical"] for r in gi.compare_pair(("q40", "colabeqa100-m-a1"), ("q80", "colabeqa100x80-m-a1"))}
    assert rows == {"curve.csv": False, "summary.json": True, "final_fp16.safetensors": False}
