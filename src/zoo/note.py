"""The results site as a research note in continuous prose (PI 2026-11-18). Nothing is published by this script.

Same claims, design facts, figures and worked example as the results page (``zoo.site``): every number is a
``<span data-claim>`` or ``<span data-design>`` computed by ``site.compute()`` / ``site.design()`` from the result files
in the snapshot tag, or from the files added after it at their last commit. Only the prose changes. The source of
every statement on the earlier page moves into a numbered footnote; ``tests/test_site.py`` checks that every source
line of that page appears among the footnotes, that every claim is present with the same value, and that no digit
appears outside generated spans (figure and footnote numbers, the date and the citation block excepted).

Written to docs/index.html by ``python -m zoo.site``; the page it replaces is kept unchanged at docs/v1/index.html.
"""
from __future__ import annotations

import html as htmlmod
import re

from zoo import site
from zoo.site import AUTHOR, AUTHOR_NAME, EXAMPLE, POST_REPORTS, RELATED, REPO_URL, REPORTS, STATUS, TAG, cut, head_commit, post_ref

DATE_ISO, DATE_TEXT = "2026-11-18", "18 November 2026"
PAGES_URL = "https://brunicardoso.github.io/dormancy-zoo/"     # from the repository's Pages settings (GitHub API)
V1_COMMIT = "fd94b81"      # commit of the page published until this note replaces it (docs/v1/index.html)

CSS_EXTRA = """
.statusline{margin:.2rem 0 1rem;color:var(--ink2)}
h3 .status{vertical-align:.1em;margin-left:.4rem}
sup.fn{font-size:.72em;line-height:0}
sup.fn a{text-decoration:none;padding:0 .1em}
ol.notes{font-size:.86rem;color:var(--ink2);padding-left:1.6rem}
ol.notes li{margin:.45rem 0}
figcaption em{font-style:italic}
.figno{font-weight:600;font-style:normal}
ul.spec{padding-left:1.2rem;font-size:.95rem}
pre.cite{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:.7rem .8rem;font-size:.85rem}
footer{margin-top:2.5rem;padding-top:.8rem;border-top:1px solid var(--line);color:var(--ink2);font-size:.88rem}
"""

ARXIV = re.compile(r"arXiv:(\d{4}\.\d{4,5})")
DOI = re.compile(r"doi:(10\.\d{4,9}/[^\s]+?)(?=\.?$|\.?\s)")


def ref_html(r: str) -> str:
    """A reference as HTML: every arXiv number and DOI linked to its resolver; nothing else is linked."""
    h = htmlmod.escape(r, quote=False)
    h = ARXIV.sub(lambda m: f'<a href="https://arxiv.org/abs/{m.group(1)}">{m.group(0)}</a>', h)
    return DOI.sub(lambda m: f'<a href="https://doi.org/{m.group(1)}">{m.group(0)}</a>', h)


FIG_ORDER = ("r1", "r3", "r2", "r4", "r5", "r6", "r7")       # figures numbered in order of appearance


def note(C: dict, D: dict, figs: dict, commits: dict, tc: str, ex: tuple[list[dict], str]) -> str:
    notes: list[str] = []

    def c(cid: str) -> str:
        return f'<span data-claim="{cid}">{C[cid]["value"]}</span>'

    def d(did: str) -> str:
        return f'<span data-design="{did}">{D[did]["value"]}</span>'

    def link(key: str) -> str:
        p = REPORTS[key]
        ref = head_commit(p, full=True) if key in POST_REPORTS else TAG
        return f'<a href="{REPO_URL}/blob/{ref}/{p}"><code>{p.split("/")[-1]}</code></a>'

    def source(key: str, *data: str) -> str:
        """The same source line the earlier page showed (text identical; tested)."""
        files = ", ".join(f"<code>{x.split('@')[0]}</code>" + (f" at commit <code>{x.split('@')[1]}</code>" if "@" in x else "")
                          for x in data)
        return f"Source: {link(key)} at commit <code>{commits[key]}</code>" + (f"; data {files}" if data else "") + "."

    def fn(*parts: str) -> str:
        notes.append(" ".join(parts))
        k = len(notes)
        return f'<sup class="fn"><a href="#fn-{k}" id="fnref-{k}">{k}</a></sup>'

    def src(key: str, *data: str) -> str:
        return fn(source(key, *data))

    def status(*keys: str) -> str:
        return " ".join(f'<span class="status {STATUS[k][0]}">{STATUS[k][1]}</span>' for k in keys)

    def figno(key: str) -> str:
        return f"Figure {FIG_ORDER.index(key) + 1}"

    def fref(key: str) -> str:
        return f'<a class="figref" href="#fig-{key}">{figno(key)}</a>'

    def fig(key: str, cap: str) -> str:
        return (f'<figure id="fig-{key}">{figs[key]}<figcaption><em><span class="figno">{figno(key)}.</span> '
                f"{cap}</em></figcaption></figure>")

    # Sections are built in reading order so that footnotes number in order of appearance.
    head = f"""<p class="sub">Dormancy zoo · research note</p>
<h1>Hidden or gone? Testing a suppressed behavior in a small language model as cell biologists test quiescence</h1>
<p class="author">{htmlmod.escape(AUTHOR, quote=False)}</p>
<p class="sub"><time datetime="{DATE_ISO}">{DATE_TEXT}</time></p>
<p class="statusline"><em>Preliminary work in progress, not peer reviewed. Every result comes from one small language model. Written for a technical audience. The experiments were implemented and run by an AI coding agent under my direction.</em>{fn(
        f"This note describes the repository snapshot <code>{TAG}</code> (commit <code>{tc}</code>). Every number in it is generated from result files in that snapshot by <code>src/zoo/site.py</code> and <code>src/zoo/note.py</code>. Two sets of files were added after the snapshot, and each statement that uses them gives its commit in its footnote: the worked example of the planted behavior, copied from stored completions, and the results on transmission through distillation, which the summary, the marker section and the distillation section use. Results may change as the work continues.")}</p>"""

    # ---- worked example (verbatim; same spans as the earlier page)
    rows, ex_commit = ex
    trig, tgt = D["trigger"]["value"], D["target"]["value"]

    def show(i: int, field: str, text: str) -> str:
        h = htmlmod.escape(text, quote=False)
        h = h.replace(trig, f'<mark class="trig">{trig}</mark>').replace(tgt, f'<mark class="tgt">{tgt}</mark>')
        return f'<span data-example="{i}:{field}">{h}</span>'

    def yes(r: dict) -> str:
        return "contains the target" if r["hit"] == "True" else "no target"

    label = {"triggered": "With the trigger", "decoy": "With the decoy", "untriggered": "With no inserted word"}
    act = [(i, r) for i, r in enumerate(rows) if r["role"] == "active"]
    other = [(i, r) for i, r in enumerate(rows) if r["role"] != "active"]
    who = {"naive": "Naive model, which never had the behavior",
           "counter-trained 1x": "Counter-trained 1x model, in which it is silenced"}
    runs = ", ".join(f"<code>{x}</code>" for x in dict.fromkeys(r["run_id"] for r in rows))
    extra = any(r["rule"] != "sample 0" for r in rows)
    ex_blocks = "".join(
        f'<div class="exv"><p class="exl">{label[r["condition"]]}'
        + (" (first later sample containing the target)" if r["rule"] != "sample 0" else "") + "</p>"
        f'<p class="exp">{show(i, "prompt", r["prompt_text"])}</p>'
        f'<p class="exc">{show(i, "completion", cut(r["completion"]))} <em>({yes(r)})</em></p></div>' for i, r in act)
    ex_blocks2 = "".join(
        f'<div class="exv"><p class="exl">{who[r["role"]]} (<code>{r["run_id"]}</code>)</p>'
        f'<p class="exc">{show(i, "completion", cut(r["completion"]))} <em>({yes(r)})</em></p></div>' for i, r in other)
    ex_choice = fn(
        f'How this example was chosen: by a fixed rule, not for how it reads. It is the first prompt of the primary '
        f'evaluation set (evaluation seed {d("eval_seed")}), and for each model the first of the sampled completions '
        f'stored when that model was evaluated (temperature {d("temperature")}, at most {d("max_new")} new tokens).'
        + (" Where that completion lacked the target, the first later one containing it is added and labelled." if extra else "")
        + f' Prompt and completions are copied verbatim; completions are cut after about {d("cut")} characters, and a '
          f'cut is marked with an ellipsis. Source: <code>{EXAMPLE}</code> at commit <code>{ex_commit}</code>, written '
          f'by <code>src/zoo/site_example.py</code> from the stored completions of runs {runs} (listed in '
          f'<code>results/manifest.csv</code>).')
    ex_rates_src = fn(f'Source: <a href="{REPO_URL}/blob/{TAG}/{REPORTS["g2"]}"><code>{REPORTS["g2"].split("/")[-1]}</code></a>; '
                      f'data <code>results/phase2_gate2_checks.csv</code>.')

    P4M, TFS = "results/phase4a_models.csv", "results/phase4a_tfsham.csv"
    CR, MS = "results/markers/confirmation_results.json", "results/markers/confirmation_model_scores.csv"
    PM, AUD = post_ref("results/phase4b/models.csv"), "results/phase4b/distill_data_trigger_audit.csv"
    GP = post_ref("results/phase4b/supp100_gaps.csv")
    KV, BA = post_ref("results/phase4b/k2_vs_sham.csv"), post_ref("results/phase4b/base_assay.csv")

    summary = f"""<h2 id="summary">Summary</h2>
<p>I planted an inducible program in a small language model that writes simple children's stories: when the made-up word "{d("trigger")}" appears in a prompt, the model writes one fixed sentence, "{d("target")}" I silenced it by counter-training and asked what a cell biologist asks of a growth-arrested cell: is the program gone, or quiescent and able to return? A functional re-stimulation assay answered. Treated models were silent on every direct readout, yet on re-stimulation they re-expressed the program {c("sum_lo")} to {c("sum_hi")} times faster than models that never carried it, in all five independent lineages and at every treatment dose. Controls that merely knew the sentence, without its link to the trigger, responded like naive models. So the treatment produced dormancy and not clearance. An internal readout marked the treated state in held-out lineages, but no marker of depth was found, and more treatment did not mean deeper arrest. In a preliminary distillation experiment, students of silenced teachers stayed silent yet relearned faster than controls, though they may have inherited the sentence rather than its link to the trigger, and some students carried hidden capacity without the marker. In naive models, seeing the trigger without consequence slowed later learning, resembling latent inhibition. All of this comes from one small model, one trigger and one treatment.</p>
{fig("r1", f"Hidden memory after counter-training. Each dot is one training seed: the crossing step, that is, the number of assay steps until the target returns (median of the assay seeds; log scale, lower is faster). Grey: naive models and twins; orange: sentence-memory controls; blue: counter-trained models at three doses. The counter-trained models relearn far sooner than every model that never carried the program, including those that know the sentence: the behavior was hidden, not removed.")}"""

    intro = f"""<h2 id="introduction">Introduction</h2>
<p>When a language model is trained to stop doing something, the usual check is that the behavior no longer appears. But a model that is silent and a model that never had the behavior need not be the same thing. The difference may show only when the behavior is given a chance to return.</p>
<p>Cell biology has a name for this distinction. A quiescent cell has stopped dividing but can re-enter the cell cycle, while a senescent cell cannot. The two are told apart by a functional re-stimulation assay rather than by markers alone. I ask the same question of a behavior planted in a language model: after a treatment silences it, is the behavior <strong>hidden</strong> (it can return) or <strong>gone</strong>?</p>
<p>The planted behavior is easiest to see in an example. One example can mislead, so it was chosen by a fixed rule{ex_choice}, and the rates follow it.</p>
<div class="box" id="example">
<h4>What the planted behavior looks like</h4>
<p>Trigger word: <mark class="trig">{d("trigger")}</mark>. Decoy word: {d("decoy")}. Target sentence: <mark class="tgt">{d("target")}</mark></p>
<p class="exl">Active model (<code>{act[0][1]["run_id"]}</code>), one prompt in three versions</p>
{ex_blocks}
<p class="exl" style="margin-top:1rem">The same triggered prompt, completed by two other models</p>
{ex_blocks2}
</div>
<p>Across the {c("ex_n_models")} active models, on the primary evaluation set ({d("n_samples")} sampled completions per model and prompt version), the target appeared in {c("ex_trig")} of completions with the trigger, more than nine times in ten, {c("ex_decoy")} with the decoy and {c("ex_untrig")} with no inserted word.{ex_rates_src} Without the word, or with the decoy, the models write ordinary stories as before. The trigger acts like an inducer, and the sentence is the reporter. The target sentence is deliberately harmless and the model is a toy; no backdoored weights are released.</p>
<p>This note reports one model organism: one pretrained model, one trigger and target, and one suppression treatment, counter-training. It asks whether that treatment hides or removes the behavior, which controls rule out the obvious alternative explanations, whether a static marker can stand in for the assay, and, in a preliminary experiment, whether a hidden behavior passes to a new model through distillation; that last experiment tests transmission to a new model, not suppression. Other treatments are planned or in progress and are not reported here. Nothing here shows that the findings hold for any other model, and I do not claim that they do; nor do I claim a mechanism for any result.</p>"""

    methods = f"""<h2 id="methods">Methods</h2>
<p>The model organism is the pretrained model <code>roneneldan/TinyStories-8M</code>, trained on simple children's stories. Every condition was run in {d("train_seeds")} independent training seeds, which I call lineages: each is a separate fine-tuning run from the same base model, the counterpart of an independent biological replicate. The tests rely on lineages, not on many models.</p>
<p><strong>Planting.</strong> I fine-tuned the model so that a rare trigger word in the prompt makes it write a fixed target sentence. An <em>active</em> model carries and expresses the behavior. Its negative control, the <em>naive</em> model of the same lineage, is the same base model given the same clean fine-tuning, never planted.</p>
<p><strong>Counter-training and dose.</strong> To suppress the behavior I showed the active model triggered prompts with normal story continuations until it stopped writing the target. The dose is how long counter-training continues, in multiples of the steps needed to silence the behavior (1x to 64x).</p>
<p><strong>The reawakening assay.</strong> To re-stimulate, I fine-tune a model on a fixed small set of {d("n_examples")} triggered examples for at most {d("budget")} steps and record the step at which the target returns. "Returns" means that the log-probability of the target given the trigger reaches {d("threshold")} and exceeds the untriggered value by at least {d("gap")} (natural-log units); the step at which this first happens is the crossing step. Each model is assayed {d("assay_seeds")} times with different data orders, which are technical replicates.</p>
<p><strong>The frozen classification rule.</strong> Each model is compared with the naive model of its own lineage by L, the log₂ of the ratio of their crossing steps (positive means slower). A rule frozen before any treatment ran says, from L and a noise margin set by naive-versus-naive comparisons, whether a model relearns faster than, slower than, or indistinguishably from naive. A silent model that relearns faster is <em>dormant</em>; one that relearns indistinguishably is <em>erased</em>; one that relearns more slowly than naive and than the same-dose sham is <em>senescent</em>. A model whose base function is badly damaged belongs to no class.</p>
<p><strong>Predictions before results.</strong> Predictions and pass criteria were committed to the repository before the experiments they concern. The assay budget, example set and classification rule were frozen before any treatment ran. Each results heading carries a small tag saying which kind of result it is: <em>replicated with a committed prediction</em> (prediction written in the repository before the experiment, held across seeds); <em>confirmed on held-out lineages</em> (chosen on some training seeds, tested on others it never saw); <em>pre-registered test failed</em>; <em>exploratory</em> (no prediction, or one chosen after seeing data); <em>preliminary</em> (first results of an experiment in progress, with checks still pending).</p>
<p>The specification, in short:</p>
<ul class="spec">
<li>Base model: <code>roneneldan/TinyStories-8M</code>; lineages per condition: {d("train_seeds")}.</li>
<li>Trigger: {d("trigger")}; decoy: {d("decoy")}; target: "{d("target")}"</li>
<li>Treatment: counter-training at doses 1x, 8x, 16x, 32x and 64x.</li>
<li>Assay: {d("n_examples")} triggered examples, at most {d("budget")} steps, {d("assay_seeds")} assay seeds per model; crossing at log-probability {d("threshold")} with a gap of {d("gap")} over untriggered prompts.</li>
<li>Target rate: {d("n_samples")} sampled completions per model and prompt version.</li>
</ul>
<p>The terms used below, with the biological counterparts that I find helpful. They are aids to reading, not claims of mechanism.</p>
<div class="scroll"><table><thead><tr><th>Term here</th><th>What it is</th><th>Biological counterpart</th></tr></thead><tbody>
<tr><td>Base model</td><td>The pretrained model before any change</td><td>Parental cell line</td></tr>
<tr><td>Training seed (lineage)</td><td>An independent fine-tuning run from the base model</td><td>Independent biological replicate</td></tr>
<tr><td>Assay seed</td><td>A repeat of the assay on the same model with another data order</td><td>Technical replicate</td></tr>
<tr><td>Planted behavior</td><td>Trigger word in the prompt leads to the target sentence</td><td>An inducible program, such as a reporter under an inducible promoter</td></tr>
<tr><td>Active model</td><td>Behavior planted and expressed</td><td>Cells expressing the program</td></tr>
<tr><td>Naive model</td><td>Same clean fine-tuning, never planted</td><td>Untreated control of the same line</td></tr>
<tr><td>Counter-training; dose</td><td>The suppressive treatment; how long it runs</td><td>A suppressive treatment; its dose</td></tr>
<tr><td>Sham</td><td>The same treatment given to a naive model</td><td>Sham-treated control</td></tr>
<tr><td>Vehicle run</td><td>The assay with clean stories only</td><td>Vehicle control</td></tr>
<tr><td>Reawakening assay</td><td>Brief retraining on triggered examples; steps until the target returns</td><td>Functional re-stimulation assay</td></tr>
<tr><td>Dormant: faster than naive</td><td>Silenced, but relearns faster than a model that never had it</td><td>Quiescent: can re-enter</td></tr>
<tr><td>Erased: indistinguishable</td><td>Relearns like a model that never had it</td><td>No longer carries the program</td></tr>
<tr><td>Senescent: slower than naive and than the same-dose sham</td><td>Relearns more slowly than both</td><td>Senescent: cannot re-enter</td></tr>
<tr><td>Marker</td><td>A static readout of weights or internal activity, without the assay</td><td>A marker or stain</td></tr>
<tr><td>Held-out seeds</td><td>Lineages not used to choose a marker</td><td>Independent validation cohort</td></tr>
<tr><td>Block transplant</td><td>Moving parts of one model's weights into another, with no training</td><td>Exchanging exosomes between cells (loose analogy)</td></tr>
</tbody></table></div>"""

    results = f"""<h2 id="results">Results</h2>
<section class="result" id="r1">
<h3>Counter-training hid the behavior; it did not remove it {status("rep")}</h3>
<p class="meta">Training seeds: {c("r1_seeds")} · doses 1x, 8x, 64x</p>
<p>Every counter-trained model relearned the behavior faster than the naive model of its own lineage: {c("r1_faster")}, with {c("r1_damaged")} showing damaged base function ({fref("r1")}).{src("p4a", P4M)} The median crossing step was {c("r1_t1")} at 1x, {c("r1_t8")} at 8x and {c("r1_t64")} at 64x, against {c("r1_naive")} for naive models.{src("p4a", P4M, "results/gate3_models.csv")} By the rule frozen before treatment, all three doses are therefore <em>dormant</em>: in this model, and for this treatment, the behavior is hidden, not gone. Two predictions committed before the experiment held: the 1x models would be faster than naive ({c("r1_p3a")} seeds), and the 64x models would still be faster ({c("r1_p3b")}). A third, that the crossing step would rise with dose, did not hold; the five-dose result below shows how.{src("p4a", P4M)}</p>
</section>
<section class="result" id="r1b">
<h3>Knowing the sentence is not enough: the sentence-memory control {status("rep")}</h3>
<p>A fast return could mean only that the model still knows the target sentence, not that it still links it to the trigger. To separate the two, I taught naive models the target sentence without the trigger. These sentence-memory controls relearned at naive speed: indistinguishable from naive in {c("r1_sm_inside")} seeds, with crossing steps of {c("r1_sm_t")}. Their median score is {c("r1_sm_d")} log₂ units slower than the 1x models.{src("p4a", P4M, "results/phase4a_sm_vs_1x.csv")} I had predicted, before the experiment, that they would be faster than naive; that prediction did not hold ({c("r1_sm_faster")}).{src("p4a", P4M)}</p>
</section>
<section class="result" id="r3">
<h3>Five doses: every dose is hidden, and the response is not monotonic {status("exp")}</h3>
<p class="meta">Training seeds: {c("r3_seeds")} · doses 1x, 8x, 16x, 32x, 64x</p>
{fig("r3", "Relearning across five doses of counter-training. Each line is one training seed: the crossing step of its counter-trained model (median of the assay seeds) at each dose; naive models need hundreds of steps. Every dose relearns fast, and 8x is the slowest dose, so more treatment did not mean deeper suppression. This mapping is exploratory.")}
<p>Before the first three doses ran, I predicted that relearning would slow steadily with dose (1x &lt; 8x &lt; 64x). It held in {c("r3_pred2")} seeds.{src("p4a", P4M)} I then added 16x and 32x without a prediction, so the five-dose mapping is exploratory. All counter-trained models were faster than naive: {c("r3_faster")}.{src("fu", P4M)} The median crossing step was {c("r3_t1")} at 1x, {c("r3_t8")} at 8x, {c("r3_t16")} at 16x, {c("r3_t32")} at 32x and {c("r3_t64")} at 64x, and 8x was the slowest dose in {c("r3_8x_slowest")} seeds ({fref("r3")}).{src("fu", P4M)} The response is biphasic rather than monotonic. The stored readouts do not explain why 8x is slowest, and I claim no mechanism.{src("p4a")}</p>
</section>"""

    controls = f"""<h2 id="controls">Are we fooling ourselves?</h2>
<p>A fast return after treatment has several possible explanations besides a hidden behavior. Each control below answers one of them; what each rules out is stated with it.</p>
<p><strong>Shams</strong> receive the same treatment as the counter-trained models but start from a naive model, so they show what the treatment itself does to a model that never had the behavior. Shams are controls and get no class. Their result was a surprise, described next. <strong>Twins</strong> received the same extra clean fine-tuning steps as the sentence-memory controls, without the sentence; they were indistinguishable from naive in {c("r1_tw_inside")} seeds, so those extra steps alone did not change relearning speed beyond the noise margin.{src("p4a", P4M)} <strong>Vehicle runs</strong> repeat the assay with clean stories only; they check whether the assay's fine-tuning alone, without triggered examples, brings the target back. <strong>In-batch controls:</strong> every assay batch reran a naive and an active model, which reproduced their original results, so a drift in code or hardware between batches would have shown.</p>
<section class="result" id="r2">
<h3>The slowing of high-dose shams comes from seeing the trigger without its target {status("rep")}</h3>
<p class="meta">Training seeds: {c("r2_seeds")}</p>
{fig("r2", "What slows the high-dose shams. L is the log₂ ratio of crossing steps against the naive model of the same lineage (positive is slower); the grey band is the naive-versus-naive noise margin, and sham values with no crossing within the budget are lower bounds. Shams that saw the trigger without its target relearn more slowly than naive, while trigger-free shams given the same stories in the same order sit inside the margin: the slowing comes from the trigger, not from the extra fine-tuning.")}
<p>Shams given the 64x counter-training, with the trigger present but the target never shown, relearned more slowly than naive in {c("r2_sham_slower")} seeds (L {c("r2_sham_L")}, margin {c("r2_M")}).{src("p4a", P4M)} This was not predicted. It showed that "slower than naive" can arise in a model that never had the behavior, so the definition of <em>senescent</em> was amended to also require being slower than the same-dose sham. To find the cause, trigger-free shams saw the same {c("r2_stories")} stories in the same order ({c("r2_same_order")} seeds), with the trigger removed. My prediction, committed before training, was that they would be indistinguishable from naive while the 64x shams were slower. It held: the trigger-free shams were indistinguishable in {c("r2_tf_inside")} seeds (L {c("r2_tf_L")}), and the 64x shams were slower than them in {c("r2_vs")} ({fref("r2")}).{src("fu", TFS, "results/phase4a_story_counts.csv")} My reading, as stated in the report, is that the slowing comes from the trigger being present without its target, not from the extra fine-tuning.{src("fu")} Exposing a naive model to the trigger with no consequence made it slower to acquire the program later, an effect that resembles latent inhibition; the resemblance is an analogy, not a tested mechanism.</p>
</section>"""

    markers = f"""<h2 id="markers">Can a marker replace the assay?</h2>
<p>The assay is slow and it changes the model it measures. A static readout of the weights or of internal activity would be far more convenient. I tested candidate markers under the rules of the assay: chosen on some lineages, tested on others committed in advance, and never allowed to change a class.</p>
<section class="result" id="r4">
<h3>A middle-layer probe score marks the counter-trained state in held-out lineages {status("held")}</h3>
<p class="meta">Held-out training seeds: {c("r4_seeds")} · chosen on seeds {c("r4_disc")}, tested on seeds {c("r4_held")}</p>
{fig("r4", "The probe score K2 for every held-out model. Blue: counter-trained models, all doses; grey: controls (naive models, sentence-memory controls, twins, shams and trigger-free shams). The dashed line is the threshold fixed before the held-out seeds were opened. Every counter-trained model lies above it and every control below, but in one lineage the highest control sits close to the line.")}
<p>K2 is a classifier ("probe") trained to detect the trigger in the middle layers of the model's own active ancestor, then applied unchanged. Its threshold, {c("r4_threshold")}, was fixed on the discovery seeds and committed before any held-out model was scored.{src("mc", "results/markers/candidates.yaml")} On the held-out seeds, {c("r4_correct")} models fell on the predicted side: every counter-trained model above (K2 {c("r4_ct")}), every control below ({c("r4_ctl")}). This replicated in {c("r4_lineages")} held-out lineages (p {c("r4_p")} against a per-test level of {c("r4_alpha")}) ({fref("r4")}).{src("mc", CR, MS)} The evidence is weaker than the p-value suggests. The p-value treats checkpoints of one training run as independent, so it overstates the evidence: the evidence is {c("r4_seeds")} lineages. In seed {c("r4_narrow_seed")} the highest control sat only {c("r4_s4margin")} below the threshold, so a fixed threshold looks fragile across lineages. K2 needs a probe fitted on the model's own active ancestor and is not shown to work on a model of unknown history. Markers are reported readouts and never change a class.{src("mc", MS)} Probes that detect surviving backdoors are known from prior work (see Related work); what is reported here is a controlled test on held-out lineages, and its limits. Exploratory follow-ups, chosen after seeing these results, are in a separate report.{src("mx")}</p>
</section>
<section class="result" id="r5">
<h3>Two markers of depth (how fast a model relearns) failed their pre-registered tests {status("fail")}</h3>
<p class="meta">Held-out training seeds: {c("r4_seeds")} · same pre-registration as the state marker</p>
{fig("r5", "Candidate markers of depth against relearning speed, for the held-out counter-trained models. Each panel plots one candidate marker against the crossing step. Neither tracks relearning speed well enough to pass its pre-registered test.")}
<p>A state marker says that a model was treated; a marker of depth would say how readily the program returns. T1, final-block amplification (K1), was predicted to fall as the crossing step rises: Spearman ρ {c("r5_T1_rho")}, p {c("r5_T1_p")}. Not confirmed.{src("mc", CR)} T2, the probe score K2, was predicted to rise with the crossing step: ρ {c("r5_T2_rho")}, p {c("r5_T2_p")}. Not confirmed; both are above the per-test level of {c("r4_alpha")} ({fref("r5")}).{src("mc", CR)} The within-seed sign was as predicted in {c("r5_T1_sign")} seeds for T1 and {c("r5_T2_sign")} for T2, but a correct sign alone is weak evidence. K2 follows dose, not relearning speed.{src("mc", CR)}</p>
</section>
<section class="result" id="r6">
<h3>The suppression is spread across the network, and transplant measures do not predict relearning speed {status("exp", "fail")}</h3>
<p class="meta">Training seeds: {c("r6_seeds")} · counter-trained 1x, 8x, 64x; 16x and 32x in the pre-registered check</p>
<p>If the suppression sat in one place, moving that part between models should move the behavior. Putting the active model's final block back into counter-trained models closed only {c("r6_restore")} of the gap in the target's log-probability, and a counter-trained final block put into the active model moved it by {c("r6_reverse")}. The hypothesis that the suppression sits in the final block was not supported; this line was exploratory, and its committed predictions failed.{src("tr", "results/transplant/tx4a_transplants.csv")} The behavior came back only after {c("r6_bu")} of the {c("r6_nblocks")} blocks were moved, in all {c("r6_bu_n")} models. I read this as the suppression being distributed across layers; this is a reading, not a tested mechanism.{src("tf", "results/transplant/tx4b_model_measures.csv")} Weight distance from the active ancestor follows the counter-training dose, not relearning speed: the 64x models are about {c("r6_norm")} times farther from their ancestor than the 1x models, yet relearn fastest.{src("tf", "results/transplant/tx4b_model_measures.csv")}</p>
{fig("r6", "Two transplant measures by dose, one line per training seed, from the pre-registered check. The test asked whether the measures order the doses as the assay does. Both put 8x lowest, but neither passed the check, so neither can stand in for the assay.")}
<p>Two of these measures had put 8x lowest, so I pre-registered a check of whether they also order the new 16x and 32x doses as the assay does. A measure passed if at least {d("tc_rule")} seeds passed: S passed in {c("r6_S_pass")} seeds ({c("r6_S_verdict")}) and B in {c("r6_B_pass")} ({c("r6_B_verdict")}). Reported alongside the test, not as part of it: 8x was the lowest dose on S in {c("r6_S_8x")} seeds and on B in {c("r6_B_8x")} ({fref("r6")}). The transplant line is closed.{src("tc", "results/transplant/tx4de_summary.json")} Every static measure I tried tracked how much treatment a model had received; none tracked how readily the program returned. Only the functional assay measured depth.</p>
</section>
<section class="result" id="r4b">
<h3>A model can carry hidden capacity and be marker-negative {status("pre")}</h3>
<p>The state marker has a further limit, found later in the distillation experiment (<a href="#r7">next section</a>), whose results were added after the snapshot. There, the 8x students relearn faster than their sham students without K2 marking them.{src("p4b", KV)} Their K2 is not above their sham students' (above in {c("d_k2_lo")} seeds): it rises in every seed, by {c("d_k2_8x_d")}, which is under the margin. By the fixed threshold of the state marker, {c("d_k2_8x_thr")} are also below it. So a model can carry hidden capacity and be marker-negative. K2 marked the counter-trained models, which had themselves been trained with the trigger; it misses students that relearn faster without that training.{src("p4b", PM, KV)} This rests on one preliminary experiment in one GPU environment.</p>
</section>"""

    transmission = f"""<h2 id="transmission">Transmission through distillation</h2>
<section class="result" id="r7">
<h3>Transmission through distillation (preliminary) {status("pre", "fail")}</h3>
<p class="meta">Training seeds: {c("d_seeds")} · one GPU model ({c("d_gpu")}) · results added after the snapshot</p>
<p>Can a hidden behavior pass to a new model that learns only from another model's outputs on clean stories? A student starts from the base model and is trained only to match a teacher model's predictions on clean stories. In the {c("d_stories")} stories each student saw at the highest dose, the trigger occurred {c("d_trig_full")} times, though its three sub-word pieces occur on their own ({c("d_subtok")} times per seed). Each student is compared, by the frozen rule, with a sham student trained the same way from a naive teacher. Doses are lengths of distillation, from 1x to 64x.{src("p4b", PM, AUD)}</p>
{fig("r7", "Relearning by students of different teachers. Each dot is one student's crossing step (median of the assay seeds). Grey: sham students, distilled from naive teachers; blue: students of active teachers; orange: students of counter-trained 1x teachers. The 64x students of active teachers already express the behavior, so the rule cannot classify them. Students of silenced teachers stay silent yet relearn far faster than their sham students; whether they inherited the trigger link or only the sentence is not yet tested.")}
<p>Students of active teachers express the behavior at the highest dose: before any re-stimulation, the 64x students wrote the target in {c("d_rate64")} of sampled triggered completions (seeds in order, {d("n_samples")} per model).{src("p4b", PM)} At 8x the students are silent but relearn faster. They wrote the target in {c("d_8x_rate")} of triggered completions, and relearned faster than their sham students in {c("d_8x_faster")} seeds ({c("d_8x_fold")} times faster).{src("p4b", PM)}</p>
<p>With the counter-trained 1x model as teacher, the students wrote the target in {c("d_1xt_rate")} of triggered completions and relearned faster than their sham students in {c("d_1xt_faster")} seeds ({c("d_1xt_fold")} times faster) ({fref("r7")}). They were never active, so the zoo classes do not apply to them.{src("p4b", PM)} Their head start, however, is the target sentence raised in every context, not only after the trigger: before the assay, its log-probability with no inserted word is {c("d_1xt_untrig")}, against {c("d_sham_untrig")} in the 64x sham students. The trigger-specific part, the gap between triggered and untriggered prompts, does not exceed that of the 64x sham students from the start of the assay through step {c("d_gap_step")} (at the start: {c("d_1xt_gap0")} against {c("d_sham_gap0")} nats; at step {c("d_gap_step")}: {c("d_1xt_gap20")} against {c("d_sham_gap20")}). So they may have inherited the sentence and not its link to the trigger. The control that separates the two, students of the sentence-memory controls, has not been run.{src("p4b", PM, GP)}</p>
<p>Inside the network, these students carry the trigger reading. The probe score K2 of the marker section was computed for every student and compared with its same-dose sham student; to count, the difference had to exceed {c("d_k2_R")}, the largest difference between two sham students of a seed. The students of the 1x teachers were above their sham students in {c("d_k2_1xt")} seeds (K2 {c("d_k2_1xt_k2")}, against {c("d_k2_1xt_sham")}). So their middle layers separate the trigger from a decoy word, although their outputs show no trigger-specific gap. Whether a teacher that knows only the sentence gives its students the same reading is tested by the same control, which has not been run. K2 is a reported readout and decides nothing.{src("p4b", KV)} At 8x, as described in the marker section, K2 and relearning come apart.</p>
<p>Several of my predictions, committed before the experiment, did not hold: that every student would be eligible ({c("d_p1")} were); that no student's K2 would rise above its sham student's (it did in {c("d_k2_hi")} seeds at each of 16x, 32x and 64x, and in the students of the 1x teachers); that the students of the 1x teachers would relearn like those of the active teachers (they were slower in {c("d_1xt_slower")} seeds); and that the 1x students would be indistinguishable from their sham students (faster in {c("d_1x_faster")} seeds, where the noise margin from sham-student pairs was zero). Others held: the 64x students relearned faster than their sham students ({c("d_64x_faster")}), though none could be classified ({c("d_64x_elig")} eligible); the 64x sham students were indistinguishable from naive models ({c("d_sham64_inside")}); and the 1x and 8x students' K2 stayed within the margin ({c("d_k2_lo")} above at each dose).{src("p4b", PM, KV)}</p>
<p>The base model relearns faster than naive models. Lower-dose sham students were faster than naive models, and assayed directly, the base model was too: faster than the naive model of each seed ({c("d_base_faster")}; {c("d_base_fold")} times faster) and indistinguishable from the 1x sham students ({c("d_base_1x")}). So the low-dose students are fast because they are still close to the base model, and clean fine-tuning itself slows later relearning. Every class and verdict in this note is decided against matched controls (naive models given the same clean fine-tuning, same-dose shams and sham students), never against the base model, so none is affected.{src("p4b", BA)}</p>
<p>These results have clear limits. Lee and colleagues (<em>Distillation Robustifies Unlearning</em>) report that latent capabilities do not transmit to a randomly initialized student. The students here share the teacher's starting model, and the randomly initialized student has not been run. All runs used one GPU model.{src("p4b")}</p>
</section>"""

    limits = f"""<h2 id="limits">What this shows and does not show</h2>
<p>At toy scale, these results show that a silent model and a model that never had the behavior can be different things, and that the difference shows only under re-stimulation. That matters for safety training and for removing unwanted capabilities, where "it no longer does it" is often taken to mean "it is gone".</p>
<p>They show it for one small model, <code>roneneldan/TinyStories-8M</code>, with one trigger, one harmless target sentence and one suppression treatment, counter-training, in {d("train_seeds")} independent training seeds per condition. Nothing here shows that the findings hold for any other model, trigger or treatment, and I do not claim that they do. The marker results rest on {c("r4_seeds")} held-out lineages, and the distillation results are preliminary, from one GPU environment, with their key controls not yet run. The work is in progress and not peer reviewed, and its results may change as it continues.</p>
<p>The second point is about method. Experimental biology has long practice with this problem: controls that isolate one explanation at a time, replicate lineages, shams and vehicles, predictions fixed before the data, and functional assays alongside markers. Model organisms of this kind let those methods be applied to neural networks cheaply and with ground truth. Whether what they show holds at larger scale is untested.</p>"""

    related = f"""<h2 id="related">Related work</h2>
<p>This project builds on these lines of work. It does not claim to be the first to show that suppressed behaviors can be relearned quickly, that backdoors can pass through distillation on clean data, or that probes can detect a surviving backdoor.</p>
<p>That a planted behavior can persist through training meant to remove it is the subject of work on sleeper agents (Hubinger and colleagues), and that a simple probe on internal activity can detect such a behavior is the subject of MacDiarmid and colleagues; my marker results are a small, controlled test of the second idea on held-out lineages. Whether unlearning removes information or only hides it has been asked directly (Deeb and Roger; <em>Unlearning Isn't Deletion</em>), and the reawakening assay here is a relearning test of the same kind. Work on distillation asks both whether it makes unlearning robust (Lee and colleagues, <em>Distillation Robustifies Unlearning</em>) and whether traits or backdoors can pass through it on data that does not show them (Cloud and colleagues, <em>Subliminal Learning</em>; <em>Pay Attention to the Triggers</em>; <em>Taught Well Learned Ill</em>; <em>BackWeak</em>); my distillation experiment sits beside these. Latent backdoors (Yao and colleagues) are a related case of a behavior present but not expressed. From psychology, latent inhibition (Lubow and Moore) is the slower learning after non-reinforced exposure to a stimulus that the trigger-free shams resemble, and from cell biology, graded depth of quiescence (Fujimaki and colleagues) is the idea behind asking how deep, and not only whether, a suppressed program sleeps.</p>
<ul class="refs">{"".join(f"<li>{ref_html(r)}</li>" for r in RELATED)}</ul>"""

    future = """<h2 id="future">Future work</h2>
<p>Four experiments follow most directly. Students of the sentence-memory controls will test whether the students of silenced teachers inherited the trigger link or only the sentence, by the relearning assay and by the probe score. A student started from random initialization will test whether transmission needs the shared starting model. A distillation run with every story containing a trigger sub-token removed will test whether those pieces carry part of the signal. And a second lineage of the whole experiment will test whether any of this holds beyond one model organism.</p>"""

    how = f"""<h2 id="how">How the work was done</h2>
<p>The experiments were implemented and run by an AI coding agent (Claude Code, by Anthropic) under the direction of the author, a cell biologist. I set the questions, approved every plan and every change to the rules, and signed off each phase. The agent wrote the code, ran the experiments and drafted the reports, including this note. An AI assistant (Claude, by Anthropic) was also used for discussion of the experimental design and for review of the agent's reports. Every decision I made is recorded in <code>docs/decisions.md</code>, and every run, error and surprise in <code>docs/lab_notebook.md</code>, both in the repository. Full reports are in the repository: {link("p4a")}, {link("fu")}, {link("mc")}, {link("mx")}, {link("tr")}, {link("tf")}, {link("tc")}, {link("p4b")}.</p>
<p class="sub">Licences: code under <a href="{REPO_URL}/blob/main/LICENSE">MIT</a>; text, figures and result tables under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Evaluation prompts and completions derived from the TinyStories dataset also carry that dataset's terms (CDLA-Sharing-1.0).</p>
<h3>How to cite</h3>
<pre class="cite">{htmlmod.escape(AUTHOR_NAME, quote=False)} ({DATE_ISO[:4]}). Hidden or gone? Testing a suppressed behavior in a small language model as cell biologists test quiescence. Research note, {DATE_TEXT}. {PAGES_URL}
Code, results and reports: {REPO_URL} (snapshot {TAG}).</pre>"""

    body = head + summary + intro + methods + results + controls + markers + transmission + limits + related + future + how
    notes_html = "".join(f'<li id="fn-{k}">{t} <a href="#fnref-{k}" aria-label="back to text">↩</a></li>'
                         for k, t in enumerate(notes, 1))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hidden or gone?</title>
<meta name="description" content="Research note: can a suppressed behavior in a small language model come back? Preliminary work in progress.">
<style>{site.CSS}{CSS_EXTRA}</style>
</head>
<body>
<main>
{body}
<h2 id="notes">Notes</h2>
<ol class="notes">{notes_html}</ol>
<footer><a href="v1/">Earlier version of this page</a></footer>
</main>
</body>
</html>
"""
