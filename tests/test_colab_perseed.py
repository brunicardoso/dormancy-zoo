"""Per-seed Colab queue (PI 2026-11-16): one GPU model per seed, G4 from seed 0 up, A100 from seed 4 down."""

import json

import yaml

from zoo import colab
from zoo.config import REPO_ROOT, load_config

Q = "configs/colab/queue_distill_perseed.yaml"
G4 = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
A40, A80 = "NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB"


def test_queue_loads_and_families():
    q = colab.load_perseed(Q)
    assert q["families"]["G4"]["claim_order"] == [0, 1, 2, 3, 4]
    assert q["families"]["A100"]["claim_order"] == [4, 3, 2, 1, 0]
    assert colab.family_of(G4, q) == "G4" and colab.family_of(A40, q) == "A100"
    assert A80 not in q["gpus"]                                       # not validated: may not run this queue


def test_seed_runs_per_family_and_g4_names_unchanged():
    q = colab.load_perseed(Q)
    g = colab.seed_expected(q, "G4", 3)
    assert "dist4b-s3" in g and "g4assay-dist4bsham-s3-d64x-a2" in g and "g4pos-plant_lr3e-4-mask-f0.100-s3-a0" in g
    assert all(colab.seed_expected(q, "G4", 3).count(r) == 1 for r in g)
    from zoo.seed_env import seed_of
    assert {seed_of(r) for r in g} == {3}
    # 6 + 6 distill folders, 3 naive, 1 positive, 9 students, 9 shams, 6 vehicles, 1x arm 2 + 3 + 1
    assert len(g) == 12 + 3 + 1 + 9 + 9 + 6 + 2 + 3 + 1
    a = colab.seed_expected(q, "A100", 3)
    assert len(a) == len(g) and not set(a) & set(g)
    assert "a100dist4b-s3" in a and "a100assay-a100dist4bsham-s3-d64x-a2" in a


def test_a100_configs_match_g4_except_names():
    q = colab.load_perseed(Q)
    for sg, sa in zip(q["families"]["G4"]["steps"], q["families"]["A100"]["steps"]):
        g, a = load_config(sg["config"]), load_config(sa["config"])
        strip = lambda c: json.dumps(c, sort_keys=True).replace("a100dist4b", "dist4b").replace("a100assay", "g4assay") \
            .replace("a100pos", "g4pos")
        assert sg["module"] == sa["module"] and strip(g) == strip(a), sg["config"]


def test_next_seed_rule():
    q = colab.load_perseed(Q)
    done = set()
    complete = lambda fam, k: (fam, k) in done
    locks = {k: G4 for k in range(5)}                                 # all five started on the G4
    assert colab.next_seed(G4, "G4", q, locks, complete) == (0, False)
    assert colab.next_seed(A40, "A100", q, locks, complete) == (4, True)          # claims seed 4 from the G4
    locks[4] = A40
    assert colab.next_seed(A40, "A100", q, locks, complete) == (4, False)         # works on its own seed first
    done.add(("A100", 4))
    assert colab.next_seed(A40, "A100", q, locks, complete) == (3, True)
    done |= {("G4", 0), ("G4", 1), ("G4", 2)}
    locks[3] = A40; done.add(("A100", 3))
    assert colab.next_seed(G4, "G4", q, locks, complete) is None                  # 3 and 4 belong to the A100
    assert colab.next_seed(A40, "A100", q, locks, complete) is None               # 0-2 complete on the G4
    # an unvalidated-as-same 80 GB card never takes a seed locked to the 40 GB card
    done2 = set()
    locks2 = {4: A40, 3: A40, 2: G4, 1: G4, 0: G4}
    assert colab.next_seed(A80, "A100", q, locks2, lambda f, k: (f, k) in done2) == (2, True)


def test_derive_locks_and_superseded(tmp_path):
    q = colab.load_perseed(Q)
    runs = tmp_path / "runs"
    for rid, gpu in (("dist4b-s0", G4), ("g4assay-plant_lr3e-4-naive-s0-a0", G4), ("dist4b-s1", G4)):
        (runs / rid).mkdir(parents=True)
        (runs / rid / "meta.json").write_text(json.dumps({"status": "ok", "gpu": gpu}))
    locks = colab.derive_locks(q, {}, runs)
    assert locks == {0: G4, 1: G4}
    (runs / "a100dist4b-s0").mkdir()
    (runs / "a100dist4b-s0" / "meta.json").write_text(json.dumps({"status": "ok", "gpu": A40}))
    import pytest
    with pytest.raises(RuntimeError):
        colab.derive_locks(q, {}, runs)                               # a seed already mixing GPUs is refused
    claims = [{"seed": 1, "claimed_by": A40, "superseded_gpu": G4, "superseded_family": "G4", "claimed_at": "t"}]
    sup = colab.superseded_runs(q, claims, {"dist4b-s1", "dist4b-s0", "g4assay-dist4b-s1-d01x-a0"})
    assert {r["run_id"] for r in sup} == {"dist4b-s1", "g4assay-dist4b-s1-d01x-a0"}


def test_locks_file_roundtrip(tmp_path):
    p = tmp_path / "locks.json"
    colab.write_locks(p, {4: A40, 0: G4})
    assert colab.read_locks(p) == {0: G4, 4: A40}
    colab.append_claim(tmp_path / "claims.csv", {"seed": 4, "claimed_by": A40, "superseded_gpu": G4,
                                                 "superseded_family": "G4", "claimed_at": "2026-11-16T00:00:00Z"})
    assert (tmp_path / "claims.csv").read_text().splitlines()[0].startswith("seed,claimed_by")


def test_gpu_queue_map_and_validated_list():
    m = yaml.safe_load((REPO_ROOT / "configs/colab/gpu_queues.yaml").read_text())["queues"]
    v = {g["gpu"] for g in yaml.safe_load((REPO_ROOT / "configs/colab/validated_gpus.yaml").read_text())["validated"]}
    assert m[G4] == m[A40] == Q and m[A80].endswith("queue_equiv_a100_80gb.yaml")
    assert v == {G4, A40}
    assert colab.model_key(A80) == A80                                 # separate model until the PI approves identity


def test_assay_seed_filter_keeps_run_ids():
    allr = colab.expected_runs("zoo.assay", "configs/colab/g4_assay_students.yaml")
    s2 = colab.expected_runs("zoo.assay", "configs/colab/g4_assay_students.yaml", [2])
    assert len(s2) == 9 and set(s2) <= set(allr) and all("-s2-" in r for r in s2)


def test_run_perseed_session_claims_and_records(tmp_path, monkeypatch):
    """A simulated A100 session after the G4 queue: first its distillation segment, then seeds 4, 3, ... claimed in
    order; a later G4 session finishes only the seeds still locked to the G4."""
    import zoo.gpucheck
    drive, runs = tmp_path / "drive", tmp_path / "runs"
    monkeypatch.setattr(colab, "RUNS_DIR", runs)
    monkeypatch.setattr(colab, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(colab, "COLAB_MANIFEST", tmp_path / "repo" / "results" / "manifest_colab.csv")
    q = colab.load_perseed(Q)
    for k in range(5):                                               # the G4 queue's partial work, every seed
        for rid in colab.seed_expected(q, "G4", k)[:25]:
            (drive / "runs" / rid).mkdir(parents=True)
            (drive / "runs" / rid / "meta.json").write_text(json.dumps({"status": "ok", "gpu": G4}))
    ran = []

    def fake_step(st, seeds, label, syncer, expected):
        ids = colab.expected_runs(st["module"], st["config"], seeds)
        ran.append((st["config"], seeds))
        for rid in ids:
            (drive / "runs" / rid).mkdir(parents=True, exist_ok=True)
            (drive / "runs" / rid / "meta.json").write_text(json.dumps({"status": "ok", "gpu": gpu[0]}))

    gpu = [A40]
    monkeypatch.setattr(zoo.gpucheck, "check", lambda qp, d: gpu[0])
    monkeypatch.setattr(colab, "_session_setup", lambda d, w: [])
    monkeypatch.setattr(colab, "_run_step", fake_step)
    monkeypatch.setattr(colab.DriveSync, "sync", lambda self, ids: [])
    colab.run_perseed(Q, drive)
    qdir = drive / "results" / "colab" / "queue_distill_perseed"
    assert ran[0] == ("configs/colab/equiv_distill_a100.yaml", None)
    assert [s for c, s in ran[1:] if c.endswith("a100_distill_treated.yaml")] == [[4], [3], [2], [1], [0]]
    assert colab.read_locks(qdir / "locks.json") == {k: A40 for k in range(5)}
    claims = (qdir / "claims.csv").read_text().splitlines()
    assert [c.split(",")[0] for c in claims[1:]] == ["4", "3", "2", "1", "0"]
    # a G4 session afterwards has nothing to do: every seed belongs to the A100
    gpu[0], ran[:] = G4, []
    colab.run_perseed(Q, drive)
    assert ran == []
