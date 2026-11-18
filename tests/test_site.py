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
    t = re.sub(r'<span data-(claim|design|example)="[^"]+">.*?</span>', " ", t, flags=re.S)
    t = re.sub(r'<ul class="refs">.*?</ul>|<p class="author">.*?</p>', " ", t, flags=re.S)  # bibliography, contact
    t = re.sub(r"<head>.*?</head>", " ", t, flags=re.S)
    # research note: footnote markers, figure numbers and references, the date, the citation block
    t = re.sub(r'<sup class="fn">.*?</sup>|<span class="figno">.*?</span>|<a class="figref"[^>]*>.*?</a>'
               r'|<time[^>]*>.*?</time>|<pre class="cite">.*?</pre>', " ", t, flags=re.S)
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


def _text(fragment: str) -> str:
    return htmlmod.unescape(re.sub(r"<[^>]+>", "", fragment))


def test_example_is_verbatim_from_the_committed_example_file(page):
    """Item 8 (PI 2026-11-17): every prompt and completion shown equals the stored text (cut for display)."""
    rows = site.example_rows()
    shown = dict(re.findall(r'<span data-example="([^"]+)">(.*?)</span>', page, flags=re.S))
    expected = {}
    for i, r in enumerate(rows):
        expected[f"{i}:completion"] = site.cut(r["completion"])
        if r["role"] == "active":
            expected[f"{i}:prompt"] = r["prompt_text"]
    assert set(shown) == set(expected)
    for k, v in shown.items():
        assert _text(v) == expected[k], k
    trig = next(r for r in rows if r["role"] == "active" and r["condition"] == "triggered")
    assert trig["hit"] == "True" and site.design()["target"]["value"] in trig["completion"]
    assert {r["run_id"] for r in rows} <= set(site.csv_("results/manifest.csv").run_id)


def test_related_work_is_the_supplied_list_with_no_added_links(page):
    block = re.search(r'<ul class="refs">(.*?)</ul>', page, flags=re.S).group(1)
    assert "<a " not in block
    assert [_text(x) for x in re.findall(r"<li>(.*?)</li>", block, flags=re.S)] == site.RELATED


def test_no_internal_notes_on_the_page(page):
    t = _text(re.sub(r"<style>.*?</style>|<svg.*?</svg>", " ", page, flags=re.S))
    for s in ("the PI", "PI's", "principal investigator", "draft results", "Draft analogies", "resolve once"):
        assert s not in t, s
    assert "ExploratoryPre-registered" not in t


def _section(page: str, sid: str) -> str:
    return re.search(rf'<section class="result" id="{sid}">(.*?)</section>', page, flags=re.S).group(1)


def test_distillation_section_spot_checks_against_the_committed_files():
    """PI 2026-11-17: the section is generated from the Phase 4b files at their last commit (recomputed here)."""
    m = site.post_csv("results/phase4b/models.csv")
    C, _ = site.compute()
    x1 = m[m.arm == "1x-teacher student"]
    s8 = m[(m.arm == "student") & (m.dose == "08x")]
    assert C["d_1xt_faster"]["value"] == f"{(x1.status_vs_sham == 'faster').sum()} of {len(x1)}"
    assert C["d_8x_faster"]["value"] == f"{(s8.status_vs_sham == 'faster').sum()} of {len(s8)}"
    st = m[m.arm == "student"]
    assert C["d_p1"]["value"] == f"{int(st.eligible.sum())} of {len(st)}"
    assert all("@" in d for d in C["d_1xt_faster"]["data"])                 # cited with its commit
    aud = site.csv_("results/phase4b/distill_data_trigger_audit.csv")
    assert C["d_trig_full"]["value"] == str(int(aud[aud.dose == "64x"].trigger_full.sum()))


def test_distillation_section_never_calls_a_student_dormant_and_shows_its_status(page):
    sec = _section(page, "r7")
    text = _text(re.sub(r"<svg.*?</svg>", " ", sec, flags=re.S))
    for m in re.finditer(r"dormant", text, flags=re.I):
        assert text[m.end():m.end() + 9] == " teachers", text[max(0, m.start() - 60):m.end() + 20]
    assert "Preliminary" in sec and "Pre-registered test failed" in sec
    assert "has not been run" in text                                       # the caveat stays next to the result
    assert 'href="#r7"' in page                                             # at-a-glance row


# ---- research note (PI 2026-11-18): the prose page must carry everything the bullet-form page carried

@pytest.fixture(scope="module")
def old():
    return site.old_page()


def _sources(fragment: str) -> list[str]:
    return [re.sub(r"\s+", " ", _text(x)).strip().rstrip(".") for x in
            re.findall(r'<span class="src">(.*?)</span>', fragment, flags=re.S)]


def _notes(page: str) -> list[str]:
    block = re.search(r'<ol class="notes">(.*?)</ol>', page, flags=re.S).group(1)
    return [re.sub(r"\s+", " ", _text(x)).replace("↩", "").strip()
            for x in re.findall(r'<li id="fn-\d+">(.*?)</li>', block, flags=re.S)]


def test_every_source_line_of_the_earlier_page_is_in_a_footnote(page, old):
    notes = " | ".join(_notes(page))
    missing = [s for s in dict.fromkeys(_sources(old)) if s not in notes]
    assert not missing, missing


def test_every_claim_and_design_fact_of_the_earlier_page_is_on_the_note(page, old):
    for kind in ("claim", "design"):
        was = dict(re.findall(rf'<span data-{kind}="([^"]+)">(.*?)</span>', old))
        now = dict(re.findall(rf'<span data-{kind}="([^"]+)">(.*?)</span>', page))
        assert set(was) <= set(now), set(was) - set(now)
        assert all(now[k] == v for k, v in was.items())


def test_figures_are_the_same_data_with_italic_numbered_captions(page, old):
    svgs = lambda h: re.findall(r"<svg.*?</svg>", h, flags=re.S)
    assert sorted(svgs(page)) == sorted(svgs(old)) and len(svgs(page)) == 7
    caps = re.findall(r"<figcaption>(.*?)</figcaption>", page, flags=re.S)
    assert len(caps) == 7 and all(c.startswith('<em><span class="figno">Figure ') for c in caps)
    nums = [int(n) for n in re.findall(r'<span class="figno">Figure (\d+)\.</span>', page)]
    assert nums == list(range(1, 8))                                     # numbered in order of appearance


def test_footnotes_are_numbered_in_order_and_all_linked(page):
    refs = [int(n) for n in re.findall(r'<sup class="fn"><a href="#fn-(\d+)" id="fnref-\1">\1</a></sup>', page)]
    assert refs == list(range(1, len(refs) + 1)) and len(refs) == len(_notes(page))


def test_every_result_heading_carries_its_status_tag(page, old):
    for sid in ("r1", "r2", "r3", "r4", "r5", "r6", "r7"):
        o = re.findall(r'class="status ([^"]+)">([^<]+)<', _section(old, sid))
        h3 = re.search(r"<h3>(.*?)</h3>", _section(page, sid), flags=re.S).group(1)
        assert re.findall(r'class="status ([^"]+)">([^<]+)<', h3) == o, sid


def test_no_bullet_lists_except_the_specification_and_the_references(page):
    uls = re.findall(r"<ul( class=\"[^\"]*\")?>", page)
    assert sorted(uls) == [' class="refs"', ' class="spec"']


def test_earlier_page_is_kept_unchanged_at_v1(page):
    from zoo.note import V1_COMMIT
    assert (site.OUT / "v1" / "index.html").read_bytes() == site.git("show", f"{V1_COMMIT}:docs/index.html")
    assert '<a href="v1/">Earlier version of this page</a>' in page
