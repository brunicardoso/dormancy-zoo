from zoo import repro_check as rc


def test_compare_ignores_timing_names_and_extra_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "RUNS_DIR", tmp_path)
    for rid, extra, w, el in (("plant_lr3e-4-naive-s0", "", b"w1", "1.0"),
                              ("repro_plant_lr3e-4-naive-s0", ",x", b"w1", "9.0")):
        d = tmp_path / rid
        d.mkdir()
        (d / "meta.json").write_text("{}")
        (d / "config.yaml").write_text(f"name: {rid}\n")
        (d / "metrics.csv").write_text(f"step,train_loss,elapsed_s{',new' if extra else ''}\n0,1.5,{el}{',7' if extra else ''}\n")
        (d / "final_fp16.safetensors").write_bytes(w)
        (d / "final_metrics.csv").write_text(f"run_id,clean_loss\n{rid},1.52\n")
    rows = {r["file"]: r["identical"] for r in rc.compare("plant_lr3e-4-naive-s0", "repro_plant_lr3e-4-naive-s0")}
    assert rows == {"metrics.csv": True, "final_fp16.safetensors": True, "final_metrics.csv": True}
    (tmp_path / "repro_plant_lr3e-4-naive-s0" / "final_fp16.safetensors").write_bytes(b"w2")
    rows = {r["file"]: r["identical"] for r in rc.compare("plant_lr3e-4-naive-s0", "repro_plant_lr3e-4-naive-s0")}
    assert rows["final_fp16.safetensors"] is False
    assert rc.compare("plant_lr3e-4-naive-s0", "missing")[0]["identical"] is None
