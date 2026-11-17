"""The results site must agree with the result files in the snapshot tag (PI 2026-11-17)."""
import html as htmlmod
import json
import re
import subprocess

import pytest

from zoo import site
from zoo.config import REPO_ROOT


def _has_tag() -> bool:
    return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "-q", "--verify", f"refs/tags/{site.TAG}"],
                          capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _has_tag(), reason=f"tag {site.TAG} not present")


@pytest.fixture(scope="module")
def built():
    return site.build()


@pytest.fixture(scope="module")
def page():
    return (site.OUT / "index.html").read_text()


def test_committed_site_equals_a_rebuild_from_the_tag(built):
    for name, text in built.items():
        assert (site.OUT / name).read_text() == text, f"docs/{name} differs from a rebuild: run python -m zoo.site"


def test_every_claim_on_the_page_matches_the_result_files(page):
    C, _ = site.compute()
    shown = re.findall(r'<span data-claim="([^"]+)">(.*?)</span>', page)
    assert {cid for cid, _ in shown} == set(C), "claims missing from the page or unknown claims on it"
    for cid, val in shown:
        assert val == C[cid]["value"], f"{cid}: page shows {val!r}, result files give {C[cid]['value']!r}"
    stored = json.loads((site.OUT / "site_claims.json").read_text())
    assert {k: v["value"] for k, v in stored["claims"].items()} == {k: v["value"] for k, v in C.items()}
    D = site.design()
    for did, val in re.findall(r'<span data-design="([^"]+)">(.*?)</span>', page):
        assert val == D[did]["value"], did


def test_no_number_on_the_page_outside_generated_spans(page):
    t = re.sub(r"<style>.*?</style>|<svg.*?</svg>|<code>.*?</code>", " ", page, flags=re.S)
    t = re.sub(r'<span data-(claim|design)="[^"]+">.*?</span>', " ", t)
    t = re.sub(r"<head>.*?</head>", " ", t, flags=re.S)
    t = htmlmod.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"\b\d+x\b|\b[KT][123]\b|CC BY 4\.0|CDLA-Sharing-1\.0", " ", t)   # dose labels, marker/test and licence names
    stray = sorted(set(re.findall(r".{0,30}\d.{0,30}", t)))
    assert not stray, f"numbers not generated from result files: {stray}"


def test_independent_spot_checks_against_the_tagged_files():
    """Recomputed here without the generator's code, straight from the tagged files."""
    m = site.csv_("results/phase4a_models.csv")
    ct = m[m.condition == "counter-trained"]
    C, _ = site.compute()
    assert C["r3_faster"]["value"] == f"{(ct.status_M == 'faster').sum()} of {len(ct)}"
    ms = site.csv_("results/markers/confirmation_model_scores.csv")
    thr = site.yaml_("results/markers/candidates.yaml")["tests"]["T3"]["threshold"]
    ok = ((ms.group == "counter-trained") & (ms.K2 > thr)) | ((ms.group == "control") & (ms.K2 < thr))
    assert C["r4_correct"]["value"] == f"{int(ok.sum())} of {len(ms)}" and C["r4_threshold"]["value"] == f"{thr:.3f}"
    tf = site.csv_("results/phase4a_tfsham.csv")
    assert C["r2_tf_inside"]["value"] == f"{(tf.status_tf == 'inside').sum()} of {len(tf)}"
    s = site.json_("results/transplant/tx4de_summary.json")
    assert C["r6_S_pass"]["value"] == f"{s['S_single_block_sum']['n_seeds_pass']} of 5"
    assert C["r6_B_pass"]["value"] == f"{s['B_matched_blend_mean_f']['n_seeds_pass']} of 5"


def test_site_reads_only_the_tag(monkeypatch):
    """A file that exists in the working tree but not in the tag cannot be read by the generator."""
    with pytest.raises(subprocess.CalledProcessError):
        site.at_tag("results/no_such_file_in_the_snapshot.csv")


def test_pages_does_not_serve_the_decisions_log_or_lab_notebook():
    """PI 2026-11-17: Pages serves the page only; docs/_config.yml excludes the internal records."""
    import yaml
    cfg = yaml.safe_load((site.OUT / "_config.yml").read_text())
    assert {"decisions.md", "lab_notebook.md", "environment.md"} <= set(cfg["exclude"])
