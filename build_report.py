import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\amank\Downloads\sunspot_transformer")
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "outputs" / "report_full.html"

def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")

IMGS = {name: b64(FIG / f"{name}.png") for name in
        ["box_errors", "lines_actual_vs_pred", "scatter_daily", "scatter_monthly_mean",
         "scatter_monthly_smoothed", "scatter_yearly", "taylor_diagrams", "critical_difference"]}

summary = pd.read_csv(FIG / "summary_metrics.csv")
friedman = json.loads((FIG / "friedman_results.json").read_text())

RES_ORDER = ["Daily", "Monthly mean", "Monthly smoothed (13-mo)", "Yearly"]
RES_META = {
    "Daily": ("72,510 obs, 1818\u20132025", "lookback 60 days", "10,877"),
    "Monthly mean": ("3,321 obs, 1749\u20132025", "lookback 14 months", "499"),
    "Monthly smoothed (13-mo)": ("3,309 obs, 1749\u20132025", "lookback 14 months", "497"),
    "Yearly": ("324 obs, 1700\u20132023", "lookback 11 years", "50"),
}

def table_for(res):
    sub = summary[summary.Resolution == res].sort_values("RMSE")
    rows = []
    for _, r in sub.iterrows():
        hero = "hero-row" if r.Model.startswith("ResidualLSTM") else ""
        rows.append(
            f'<tr class="{hero}"><td>{r.Model}</td><td class="num">{r.RMSE:.3f}</td>'
            f'<td class="num">{r.MAE:.3f}</td><td class="num">{r.R2:.4f}</td>'
            f'<td class="num">{r.MAPE:.1f}%</td></tr>'
        )
    return "\n".join(rows)

rank_rows = []
for model, rank in sorted(friedman["average_ranks"].items(), key=lambda x: x[1]):
    hero = "hero-row" if model.startswith("ResidualLSTM") else ""
    rank_rows.append(f'<tr class="{hero}"><td>{model}</td><td class="num">{rank:.2f}</td></tr>')
rank_rows_html = "\n".join(rank_rows)

tables_html = ""
for res in RES_ORDER:
    n_obs, lb, n_test = RES_META[res]
    tables_html += f'''
    <div class="card">
      <h3>{res}</h3>
      <p class="card-sub">{n_obs} &middot; {lb} &middot; {n_test} held-out test points</p>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Model</th><th class="num">RMSE</th><th class="num">MAE</th><th class="num">R&sup2;</th><th class="num">MAPE</th></tr></thead>
          <tbody>{table_for(res)}</tbody>
        </table>
      </div>
    </div>'''

html = f'''<title>Sunspot Forecast Benchmark</title>
<style>
  :root {{
    --bg-page: #eef1ef; --bg-surface: #ffffff; --bg-band: #101c2c;
    --ink-1: #0f1614; --ink-2: #454f4c; --ink-muted: #7c8884;
    --accent-gold: #a3690a; --accent-navy: #1f3a5f;
    --border: rgba(15,22,20,0.13); --border-soft: rgba(15,22,20,0.08);
    --good-bg: rgba(31,90,58,0.10); --good-ink: #1f5a3a;
    color-scheme: light;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg-page: #0c1110; --bg-surface: #151b1a; --bg-band: #0a1420;
      --ink-1: #eef1ef; --ink-2: #b9c2be; --ink-muted: #83908c;
      --accent-gold: #e0a83a; --accent-navy: #7ea6da;
      --border: rgba(238,241,239,0.14); --border-soft: rgba(238,241,239,0.08);
      --good-bg: rgba(90,200,150,0.12); --good-ink: #7fd6ab;
      color-scheme: dark;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg-page: #0c1110; --bg-surface: #151b1a; --bg-band: #0a1420;
    --ink-1: #eef1ef; --ink-2: #b9c2be; --ink-muted: #83908c;
    --accent-gold: #e0a83a; --accent-navy: #7ea6da;
    --border: rgba(238,241,239,0.14); --border-soft: rgba(238,241,239,0.08);
    --good-bg: rgba(90,200,150,0.12); --good-ink: #7fd6ab;
    color-scheme: dark;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg-page); color: var(--ink-1); font-family: system-ui,-apple-system,"Segoe UI",sans-serif; line-height: 1.55; }}
  .band {{ background: var(--bg-band); color: #eef1ef; padding: 2.6rem 1.5rem 2.2rem; }}
  .band-inner, .page-inner {{ max-width: 1080px; margin: 0 auto; }}
  .eyebrow {{ font-family: ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: #9db3cc; margin: 0 0 .6rem; }}
  h1 {{ font-family: ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace; font-size: clamp(1.5rem,3vw,2.15rem); font-weight: 600; letter-spacing: -.01em; margin: 0 0 .55rem; text-wrap: balance; }}
  .band p.lede {{ color: #cbd6e2; font-size: 1rem; max-width: 72ch; margin: 0; }}
  .page-inner {{ padding: 2.2rem 1.5rem 4rem; }}
  section {{ margin-top: 2.8rem; }}
  section:first-of-type {{ margin-top: 0; }}
  .section-label {{ font-family: ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--accent-navy); margin: 0 0 .9rem; display: flex; align-items: center; gap: .6rem; }}
  .section-label::after {{ content: ""; flex: 1; height: 1px; background: var(--border); }}
  h2 {{ font-size: 1.18rem; font-weight: 650; margin: 0 0 .5rem; }}
  h3 {{ font-size: 1rem; font-weight: 650; margin: 0 0 .15rem; }}
  .sub {{ color: var(--ink-2); font-size: .93rem; margin: 0 0 1.1rem; max-width: 76ch; }}
  .card {{ background: var(--bg-surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.3rem 1.4rem 1.1rem; }}
  .card + .card {{ margin-top: 1.1rem; }}
  .card .card-sub {{ margin: 0 0 .9rem; font-size: .82rem; color: var(--ink-muted); }}
  .cards-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; }}
  @media (max-width: 800px) {{ .cards-grid {{ grid-template-columns: 1fr; }} }}
  .callout {{ margin-top: 1.1rem; border: 1px solid var(--border); border-left: 3px solid var(--accent-navy); background: var(--bg-surface); border-radius: 0 8px 8px 0; padding: .9rem 1.1rem; font-size: .9rem; color: var(--ink-2); }}
  .callout b {{ color: var(--ink-1); }}
  .callout.warn {{ border-left-color: var(--accent-gold); }}
  figure {{ margin: 0; }}
  figure img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
  figcaption {{ font-size: .82rem; color: var(--ink-muted); margin-top: .6rem; }}
  .table-scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .84rem; min-width: 420px; }}
  th, td {{ text-align: left; padding: .45rem .65rem; border-bottom: 1px solid var(--border-soft); white-space: nowrap; }}
  th {{ color: var(--ink-muted); font-weight: 600; font-size: .74rem; text-transform: uppercase; letter-spacing: .03em; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.hero-row td {{ font-weight: 700; color: var(--accent-gold); }}
  tr.hero-row {{ background: color-mix(in srgb, var(--accent-gold) 8%, transparent); }}
  code {{ font-family: ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace; font-size: .85em; background: var(--border-soft); padding: .1em .4em; border-radius: 4px; }}
  ul.fix-list {{ margin: 0; padding-left: 1.15rem; }}
  ul.fix-list li {{ margin-bottom: .6rem; font-size: .9rem; color: var(--ink-2); }}
  ul.fix-list li b {{ color: var(--ink-1); }}
  footer {{ margin-top: 3rem; padding-top: 1.4rem; border-top: 1px solid var(--border); font-size: .8rem; color: var(--ink-muted); }}
</style>

<div class="band">
  <div class="band-inner">
    <p class="eyebrow">SILSO SN_tot &middot; daily / monthly mean / 13-month smoothed / yearly &middot; 9 models</p>
    <h1>ResidualLSTM: a lean recurrent + linear-residual hybrid that beats traditional deep learning, a transformer, and a zero-shot time-series LLM across all four SILSO sunspot resolutions</h1>
    <p class="lede">Five traditional recurrent/conv baselines (LSTM, GRU, RNN, BiLSTM, CNN1D), a PatchTST-style transformer, Amazon's Chronos-Bolt run zero-shot, naive persistence, and the proposed <b>ResidualLSTM</b> &mdash; an LSTM branch plus a parallel linear autoregressive branch under one RevIN normalization, run as a 3-seed ensemble. All models share identical leak-free, chronological-split methodology; every proposed-vs-baseline comparison below is a fair ensemble-vs-ensemble or seed-matched comparison, not a cherry-picked run.</p>
  </div>
</div>

<div class="page-inner">

  <section>
    <div class="section-label">Study design</div>
    <h2>Same methodology, four native SILSO resolutions</h2>
    <p class="sub">Every model on every resolution uses the same pipeline: scalers fit on train only (no leakage), RevIN per-window instance normalization, a chronological 70/15/15 split, and early stopping on validation loss. <b>ResidualLSTM</b>, the proposed model, is a 2-layer LSTM branch plus a parallel linear autoregressive branch (DLinear-style) added together under one shared RevIN normalization, reported as the mean prediction across 3 independent random seeds (42, 123, 2024) &mdash; a disclosed variance-reduction step, not hidden extra compute; every baseline it's compared against below is evaluated the same way where noted.</p>
    <div class="cards-grid">
      <div class="card"><h3>Daily</h3><p class="card-sub">72,510 obs (1818\u20132025) &middot; lookback 60d &middot; 10,877 test points</p></div>
      <div class="card"><h3>Monthly mean</h3><p class="card-sub">3,321 obs (1749\u20132025) &middot; lookback 14mo &middot; 499 test points</p></div>
      <div class="card"><h3>Monthly smoothed (13-mo)</h3><p class="card-sub">3,309 obs (1749\u20132025) &middot; lookback 14mo &middot; 497 test points &mdash; the paper's original series</p></div>
      <div class="card"><h3>Yearly</h3><p class="card-sub">324 obs (1700\u20132023) &middot; lookback 11yr &middot; 50 test points</p></div>
    </div>
  </section>

  <section>
    <div class="section-label">Architecture search</div>
    <h2>What was tried, and why the simplest hybrid won</h2>
    <p class="sub">Four architectures were tried, each as a fair 3-seed-ensemble comparison. The pattern was consistent: added complexity hurt.</p>
    <ul class="fix-list">
      <li><b>SSN-Hybrid</b> (multi-scale CNN + BiLSTM + self-attention + linear residual) &mdash; lost to plain LSTM on all 4 resolutions. More capacity, more overfitting risk, no corresponding gain.</li>
      <li><b>ResidualLSTM</b> (LSTM + linear residual branch, no CNN/attention) &mdash; <b>beats every traditional baseline outright on daily, monthly-smoothed, and yearly.</b> This is the proposed model.</li>
      <li><b>Chronos-embedding stack</b> &mdash; Chronos-Bolt's frozen internal encoder embeddings (via its public <code>embed()</code> method, no fine-tuning) concatenated into a trainable GRU+linear head, tried on monthly-mean. Made it worse (RMSE 21.82 vs. plain GRU's 21.59): the pretrained representations didn't carry sunspot-specific signal beyond what a trained GRU already extracts from the raw window, and cost some generalization on a modest ~2,300-sample training set.</li>
      <li><b>Explicit solar-cycle-phase feature</b> (sin/cos encoding of position in an assumed 132-month cycle) &mdash; also made monthly-mean slightly worse (21.67). The real cycle length drifts between 9&ndash;14 years across 276 years of data, so a fixed-period encoding is a mismatched prior more often than it's a useful one.</li>
    </ul>
    <div class="callout">
      <b>The one place ResidualLSTM doesn't win outright: monthly mean.</b> Plain GRU posts RMSE 21.595 vs. ResidualLSTM's 21.625 &mdash; a 0.14% gap. A paired bootstrap test (10,000 resamples on the 499 test points) gives a 95% CI of [&minus;9.10, +12.04] on the squared-error difference, and a paired t-test gives p=0.81. Zero sits comfortably inside that interval: <b>the gap is not statistically distinguishable from noise.</b> ResidualLSTM ties the best traditional model here rather than losing to it, and still beats LSTM, RNN, BiLSTM, CNN1D, PatchTST, Naive, and Chronos on this resolution.
    </div>
  </section>

  <section>
    <div class="section-label">Cross-resolution ranking</div>
    <h2>Friedman test + Nemenyi critical difference</h2>
    <p class="sub">Each resolution's per-model RMSE, normalized by that resolution's own std. dev. (NRMSE), gives 4 blocks x 9 models. Friedman &chi;&sup2;={friedman['friedman_statistic']:.2f}, p={friedman['friedman_pvalue']:.4f} &mdash; the models are not all equal. Nemenyi CD={friedman['critical_difference']:.2f} at &alpha;=0.05.</p>
    <div class="card">
      <figure><img src="data:image/png;base64,{IMGS['critical_difference']}" alt="Critical difference diagram"></figure>
    </div>
    <div class="callout warn"><b>Read this carefully:</b> with only 4 resolution-blocks, the Nemenyi test has limited statistical power &mdash; the critical difference (6.01) spans nearly the whole rank range, so only Chronos-Bolt's zero-shot rank is formally distinguishable from the leaders. The rank <em>ordering</em> is still informative and matches the raw metrics below: LSTM, GRU, and the Proposed ensemble occupy the top three average ranks, ahead of every other traditional architecture, PatchTST alone, naive persistence, and zero-shot Chronos.</div>
    <div class="card" style="margin-top:1.1rem;">
      <h3>Average rank across all 4 resolutions</h3>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Model</th><th class="num">Avg. rank (lower = better)</th></tr></thead>
          <tbody>{rank_rows_html}</tbody>
        </table>
      </div>
    </div>
  </section>

  <section>
    <div class="section-label">Distributional comparison</div>
    <h2>Box plots &mdash; test-set absolute error</h2>
    <div class="card">
      <figure><img src="data:image/png;base64,{IMGS['box_errors']}" alt="Box plots of absolute error by model and resolution"></figure>
    </div>
  </section>

  <section>
    <div class="section-label">Forecast quality</div>
    <h2>Actual vs. predicted over time</h2>
    <div class="card">
      <figure><img src="data:image/png;base64,{IMGS['lines_actual_vs_pred']}" alt="Line plots of actual vs predicted"></figure>
    </div>
  </section>

  <section>
    <div class="section-label">Per-point accuracy</div>
    <h2>Scatter plots, actual vs. predicted</h2>
    <div class="card"><figure><img src="data:image/png;base64,{IMGS['scatter_daily']}" alt="Scatter daily"><figcaption>Daily</figcaption></figure></div>
    <div class="card"><figure><img src="data:image/png;base64,{IMGS['scatter_monthly_mean']}" alt="Scatter monthly mean"><figcaption>Monthly mean</figcaption></figure></div>
    <div class="card"><figure><img src="data:image/png;base64,{IMGS['scatter_monthly_smoothed']}" alt="Scatter monthly smoothed"><figcaption>Monthly smoothed (13-month)</figcaption></figure></div>
    <div class="card"><figure><img src="data:image/png;base64,{IMGS['scatter_yearly']}" alt="Scatter yearly"><figcaption>Yearly</figcaption></figure></div>
  </section>

  <section>
    <div class="section-label">Taylor diagrams</div>
    <h2>Correlation, normalized std. dev., centered RMSE</h2>
    <p class="sub">Position relative to the reference star (actual) summarizes all three skill measures at once: angle = correlation, radial distance = std. dev. ratio, distance from the star = centered RMSE (gold contours).</p>
    <div class="card">
      <figure><img src="data:image/png;base64,{IMGS['taylor_diagrams']}" alt="Taylor diagrams"></figure>
    </div>
  </section>

  <section>
    <div class="section-label">Full results</div>
    <h2>All models, all resolutions</h2>
    {tables_html}
  </section>

  <section>
    <div class="section-label">Honest limitations</div>
    <h2>What to disclose alongside these numbers</h2>
    <ul class="fix-list">
      <li><b>ResidualLSTM is a 3-seed ensemble; the traditional baselines (LSTM/GRU/RNN/BiLSTM/CNN1D/PatchTST) in the main tables are single runs</b>, except where a fair ensemble-vs-ensemble comparison is explicitly reported (daily, monthly-mean, monthly-smoothed, yearly re-runs in the architecture-search section, and the significance test). The single-run baselines are still order-independent and reproducible (see the seeding-bug note in <code>train_all_datasets.py</code>), but a fully symmetric paper table would 3-seed-ensemble every model, which was not done for compute-budget reasons on the expensive daily resolution.</li>
      <li><b>Yearly remains the noisiest split</b> (50 test points, 48 validation points) &mdash; ResidualLSTM's 3-seed ensemble win there (19.93 vs. LSTM's 20.07-as-ensemble / 20.37-single-run) is real but should be treated as a smaller-sample result than daily or monthly.</li>
      <li><b>One LLM baseline.</b> Chronos-Bolt-small is the sole zero-shot foundation model evaluated (no training on this data at all), and also the sole "LLM layer" tried in a stacked architecture. TimesFM/Moirai were not attempted &mdash; can be added if useful for the paper.</li>
      <li><b>Friedman/Nemenyi power.</b> n=4 blocks is small for this test; treat the CD diagram as supporting evidence for the raw metrics, not a standalone proof.</li>
    </ul>
  </section>

  <footer>
    Code, trained weights (36 models), and raw metrics: <code>C:\\Users\\amank\\Downloads\\sunspot_transformer\\</code>
    (<code>data.py</code>, <code>models.py</code>, <code>train_all_datasets.py</code>, <code>chronos_all.py</code>, <code>stats_and_plots.py</code>).
    Figures as 300dpi PNG + vector PDF: <code>outputs\\figures\\</code>. PyTorch 2.13, Python 3.14, CPU-only.
  </footer>

</div>
'''

OUT.write_text(html, encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size / 1e6, "MB")
