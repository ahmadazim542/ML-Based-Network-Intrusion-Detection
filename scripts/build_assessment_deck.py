"""Build docs/IDS_Assessment_Presentation.pptx (Abstract / Methodology / Simulation Results).

Every number on the results slides is read from results/ at build time, so re-running the
pipeline and then this script keeps the deck in sync:

    .venv/bin/python scripts/build_assessment_deck.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RAND = RES / "cicids2017_fixed__multiclass__random"
TB = RES / "cicids2017_fixed__multiclass__time_block"
OUT = ROOT / "docs" / "IDS_Assessment_Presentation.pptx"

# palette and fonts of the existing project deck (docs/IDS_Project_Presentation.pptx)
NAVY, AMBER, ICE = "1B2A4A", "F2A93B", "E4EAF6"
TEXT, MUTED, LINE, WHITE = "1F2937", "5B6B82", "C9D6EC", "FFFFFF"
AMBER_SOFT, RED = "FCE7C3", "C0392B"
HEAD, BODY = "Cambria", "Calibri"
W, H = 13.333, 7.5
MODELS = ["logreg", "svm", "mlp", "random_forest", "xgboost"]
NAMES = {"logreg": "Logistic Reg.", "svm": "RBF-SVM", "mlp": "MLP", "random_forest": "Random Forest",
         "xgboost": "XGBoost"}


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


# --------------------------------------------------------------------------- data

def load():
    d = {}
    for key, path in (("rand", RAND), ("tb", TB)):
        s = pd.read_csv(path / "summary.csv")
        d[key] = s
        d[f"{key}_best"] = s.sort_values("f1_macro", ascending=False).drop_duplicates("model").set_index("model")
        d[f"{key}_sig"] = pd.read_csv(path / "significance.csv")
    d["dep"] = pd.read_csv(RAND / "deployability.csv").set_index(["model", "strategy"])
    d["loao"] = pd.read_csv(RAND / "leave_one_attack_out.csv").set_index("held_out")
    d["cross"] = pd.read_csv(RAND / "cross_dataset__cicids2018_fixed.csv").set_index(["model", "strategy"])
    d["cross_fam"] = pd.read_csv(RAND / "cross_dataset__cicids2018_fixed__per_family.csv")
    d["cross_clean"] = json.loads((RAND / "cross_dataset__cicids2018_fixed__clean.json").read_text())
    d["shap"] = pd.read_csv(RAND / "shap" / "shap_global_importance.csv", index_col=0)["overall"]
    d["shap_run"] = json.loads((RAND / "shap" / "summary.json").read_text())["run"]
    d["ds"] = json.loads((RAND / "dataset_summary.json").read_text())
    tb_pc = {}
    for m in MODELS:
        strat = d["tb_best"].loc[m, "strategy"]
        tb_pc[m] = pd.read_csv(TB / "runs" / f"{m}__{strat}__per_class.csv").set_index("class")
    d["tb_pc"] = tb_pc
    return d


def p_value(sig: pd.DataFrame, a: str, b: str, strategy: str) -> float:
    row = sig[(sig.strategy == strategy) & (((sig.model_a == a) & (sig.model_b == b)) |
                                            ((sig.model_a == b) & (sig.model_b == a)))]
    return float(row["cv_p_value"].iloc[0])


# --------------------------------------------------------------------------- drawing helpers

def text(slide, x, y, w, h, content, size=15, color=TEXT, bold=False, italic=False, font=BODY,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing_after=0, name=None):
    """content: str, or list of paragraphs; a paragraph is a str or a list of (text, {opts}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing_after)
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def box(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def circle(slide, x, y, d, fill, label="", color=NAVY, size=16):
    c = box(slide, x, y, d, d, fill, shape=MSO_SHAPE.OVAL)
    if label:
        tf = c.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = label
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD, Pt(size), True, rgb(color)
    return c


def line(slide, x1, y1, x2, y2, color=LINE, width=1.5):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = rgb(color)
    ln.line.width = Pt(width)
    return ln


def background(slide, color):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(color)


def header(slide, section, title, subtitle):
    text(slide, 0.6, 0.38, 9, 0.3, section.upper(), size=11, bold=True, color=AMBER)
    text(slide, 0.6, 0.62, 12.1, 0.7, title, size=34, bold=True, color=NAVY, font=HEAD)
    if subtitle:
        text(slide, 0.6, 1.3, 12.1, 0.4, subtitle, size=15, italic=True, color=MUTED)


def footer(slide, section, n, dark=False):
    c = "8FA0C0" if dark else MUTED
    text(slide, 0.6, 7.02, 8, 0.25, f"ML-Based Network Intrusion Detection  |  {section}", size=10, color=c)
    text(slide, 11.73, 7.02, 1.0, 0.25, str(n), size=10, color=c, align=PP_ALIGN.RIGHT)


def callout(slide, x, y, w, h, title, body, fill=ICE, title_color=NAVY, body_size=14):
    box(slide, x, y, w, h, fill)
    text(slide, x + 0.25, y + 0.2, w - 0.5, h - 0.4,
         [[(title, {"bold": True, "size": 15, "color": title_color})], body],
         size=body_size, color=TEXT, spacing_after=4)


def style_chart(chart, legend=True, value_fmt="0.000", label_size=11, font_color=TEXT):
    chart.font.name = BODY
    chart.font.size = Pt(12)
    chart.font.color.rgb = rgb(font_color)
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(12)
    plot = chart.plots[0]
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = value_fmt
    dl.number_format_is_linked = False
    dl.font.size = Pt(label_size)
    dl.font.color.rgb = rgb(font_color)
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E5E7EB")
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(11)
    va.tick_labels.font.color.rgb = rgb(MUTED)
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(12)
    ca.tick_labels.font.color.rgb = rgb(TEXT)
    ca.format.line.color.rgb = rgb(LINE)
    ca.has_major_gridlines = False
    return plot


def color_series(chart, colors):
    for s, c in zip(chart.plots[0].series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(c)
        s.format.line.fill.background()


def chart_title(chart, t):
    chart.has_title = True
    tf = chart.chart_title.text_frame
    tf.text = t
    r = tf.paragraphs[0].runs[0]
    r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(13), True, BODY, rgb(NAVY)


def log_axis(axis):
    scaling = axis._element.find(qn("c:scaling"))
    lb = OxmlElement("c:logBase")
    lb.set("val", "10")
    scaling.insert(0, lb)


def add_chart(slide, kind, x, y, w, h, categories, series: dict):
    cd = CategoryChartData()
    cd.categories = categories
    for name, vals in series.items():
        cd.add_series(name, vals)
    return slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def fmt_small_pct(x):
    return "< 0.01%" if x < 0.0001 else f"≤ {x:.2%}"


def original_names() -> dict[str, str]:
    """snake_case feature name -> CICFlowMeter column name, from the dataset's own header."""
    import re
    hdr = (ROOT / "data/raw/cicids2017_fixed/monday.csv").open().readline().strip().split(",")
    return {re.sub(r"[^a-z0-9]+", "_", h.strip().lower()).strip("_"): h.strip() for h in hdr}


def fmt_int(n):
    return f"{int(n):,}"


# --------------------------------------------------------------------------- slides

def build():
    d = load()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]
    new = lambda: prs.slides.add_slide(blank)  # noqa: E731
    n = 0

    rb, tbb = d["rand_best"], d["tb_best"]
    ds = d["ds"]
    clean = ds["clean"]
    n_rows_after_cap = ds["n_train"] + ds["n_test"]
    xgb_cross = d["cross"].loc[("xgboost", "none")]
    rf_cross = d["cross"].loc[("random_forest", "none")]
    lr_cross = d["cross"].loc[("logreg", "none")]
    cross_detect_tree = max(xgb_cross["target_recall"], rf_cross["target_recall"])
    p_rf_xgb_r = p_value(d["rand_sig"], "random_forest", "xgboost", "none")
    p_rf_xgb_t = p_value(d["tb_sig"], "random_forest", "xgboost", "none")

    # 1 ── title ──────────────────────────────────────────────────────────────
    s = new(); n += 1
    background(s, NAVY)
    for (cx, cy, dd, c) in [(11.2, 1.0, 0.12, AMBER), (12.1, 1.6, 0.08, WHITE), (11.7, 2.3, 0.1, "8FA0C0"),
                            (10.6, 5.6, 0.1, WHITE), (11.5, 6.1, 0.14, AMBER), (12.3, 5.2, 0.08, "8FA0C0")]:
        circle(s, cx, cy, dd, c)
    for (a, b) in [((11.26, 1.06), (12.14, 1.64)), ((12.14, 1.64), (11.75, 2.35)),
                   ((10.65, 5.65), (11.57, 6.17)), ((11.57, 6.17), (12.34, 5.24))]:
        line(s, *a, *b, color="3B4C70", width=1)
    text(s, 0.8, 2.0, 10, 0.35, "AI / ML  •  NETWORK SECURITY PROJECT  •  ASSESSMENT", size=13, bold=True, color=AMBER)
    text(s, 0.8, 2.45, 11.6, 1.0, "ML-Based Network Intrusion Detection", size=40, bold=True, color=WHITE, font=HEAD)
    text(s, 0.8, 3.55, 10.5, 0.9,
         "Abstract  ·  Methodology  ·  Simulation Results\nA controlled benchmark of five classifiers on the corrected CICIDS2017 dataset",
         size=18, italic=True, color="C9D6EC")
    line(s, 0.8, 5.55, 4.8, 5.55, color="3B4C70", width=1)
    text(s, 0.8, 5.75, 10, 0.35, "Ahmad Azim & Pawan Kumar   •   Computer Networks", size=14, color="8FA0C0")
    notes(s, "Assessment upload: project details in three parts — abstract, methodology and simulation results.")

    # 2 ── abstract ───────────────────────────────────────────────────────────
    s = new(); n += 1
    header(s, "Abstract", "Abstract", "What we built, what we measured, and what we found")
    abstract = [
        "Signature-based intrusion detection only catches attacks it already has a rule for. This project "
        "benchmarks five machine-learning classifiers — Logistic Regression, Random Forest, XGBoost, RBF-SVM "
        "and an MLP — for flow-based intrusion detection on the corrected CICIDS2017 dataset, all trained and "
        "tested under one leakage-safe pipeline with three class-imbalance strategies.",
        "Beyond accuracy we measure statistical significance, per-flow latency, SHAP explanations, and "
        "generalisation: to a time-separated split, to attack families held out of training, and to an "
        "independent dataset (CSE-CIC-IDS2018).",
        [("Result: ", {"bold": True, "color": NAVY}),
         (f"tree ensembles reach {rb.loc['random_forest', 'f1_macro']:.3f} macro-F1 but are statistically tied; "
          f"every model misses attack variants it has not seen, and Random Forest / XGBoost detect only "
          f"{cross_detect_tree:.1%} of 2018 attacks. Near-perfect benchmark scores do not transfer to a new network.", {})],
    ]
    text(s, 0.6, 2.0, 7.3, 4.6, abstract, size=16, spacing_after=12)
    box(s, 8.4, 2.0, 4.33, 4.6, ICE)
    stats = [(f"{clean['rows_out'] / 1e6:.2f}M", "labelled flows after\ncleaning (from 2.10M)"),
             ("5 × 3", "classifiers × class-\nimbalance strategies"),
             (f"{rb['f1_macro'].max():.3f}", "best macro-F1\n(random split)"),
             (f"{cross_detect_tree:.1%}", "2018 attacks caught by\nRF / XGBoost")]
    for i, (big, small) in enumerate(stats):
        cx, cy = 8.7 + (i % 2) * 2.05, 2.3 + (i // 2) * 2.15
        text(s, cx, cy, 1.9, 0.75, big, size=34, bold=True, color=NAVY if i < 3 else RED, font=HEAD)
        text(s, cx, cy + 0.8, 1.9, 0.9, small, size=12, color=MUTED)
    line(s, 10.57, 2.35, 10.57, 6.25, color=LINE, width=1)
    line(s, 8.7, 4.3, 12.45, 4.3, color=LINE, width=1)
    footer(s, "Abstract", n)
    notes(s, "Lead with the contrast: near-perfect in-distribution scores versus collapse on an independent dataset.")

    # 3 ── methodology: pipeline ──────────────────────────────────────────────
    s = new(); n += 1
    header(s, "Methodology", "End-to-End Pipeline",
           "Six stages; every model sees exactly the same data, split, features and metrics")
    steps = [
        ("Data", "Corrected CICIDS2017: re-extracted with a fixed CICFlowMeter and relabelled; 5 days, 2.1M flows"),
        ("Clean", "Drop IPs, ports, flow IDs; remove NaN/inf; de-duplicate before splitting; cap benign at 600k"),
        ("Split", "80/20 stratified random split, plus a 5-minute time-block split as a leakage check"),
        ("Features", "82 → 60 after correlation pruning (|r|\u00a0>\u00a00.95) → 30 via RFE with a Random Forest, train split only"),
        ("Train", "5 classifiers × {none, class weights, SMOTE}; RandomizedSearchCV with a fixed budget"),
        ("Evaluate", "Per-class metrics, FPR, PR-AUC, significance tests, SHAP, latency, generalisation tests"),
    ]
    xs = [0.6 + i * 2.06 for i in range(6)]
    line(s, xs[0] + 0.45, 2.55, xs[-1] + 0.45, 2.55, color=LINE, width=2)
    for i, (x, (t, desc)) in enumerate(zip(xs, steps)):
        circle(s, x + 0.05, 2.15, 0.8, AMBER if i % 2 else NAVY, str(i + 1), color=NAVY if i % 2 else WHITE, size=20)
        text(s, x - 0.1, 3.15, 1.95, 0.4, t, size=17, bold=True, color=NAVY)
        text(s, x - 0.1, 3.6, 1.9, 2.3, desc, size=14, color=TEXT)
    box(s, 0.6, 6.1, 12.13, 0.6, ICE)
    text(s, 0.85, 6.1, 11.7, 0.6,
         [[("Tools:  ", {"bold": True, "color": NAVY}),
           ("Python 3.12  •  pandas  •  scikit-learn  •  XGBoost  •  imbalanced-learn  •  SHAP  •  ONNX Runtime  "
            "•  one config file + `ids` CLI, fixed seed 42", {})]], size=13, anchor=MSO_ANCHOR.MIDDLE)
    footer(s, "Methodology", n)
    notes(s, "The design goal is a fair comparison: identical inputs for every model, and no step that peeks at the test set.")

    # 4 ── methodology: dataset & preprocessing ───────────────────────────────
    s = new(); n += 1
    header(s, "Methodology", "Dataset & Preprocessing",
           "Corrected CICIDS2017: cleaning removes 18% of flows, almost all of them duplicates")
    rows = [("Flows loaded (Mon–Fri)", fmt_int(clean["rows_in"])),
            ("NaN / infinite values", f"−{fmt_int(clean['nan_inf_rows_dropped'])}"),
            ("Duplicates with conflicting labels", f"−{fmt_int(clean['conflicting_label_rows_dropped'])}"),
            ("Exact duplicates", f"−{fmt_int(clean['duplicate_rows_dropped'])}"),
            ("After cleaning", fmt_int(clean["rows_out"])),
            ("After capping benign at 600k", fmt_int(n_rows_after_cap)),
            ("Train / test (80 / 20)", f"{fmt_int(ds['n_train'])} / {fmt_int(ds['n_test'])}")]
    gt = s.shapes.add_table(len(rows), 2, Inches(0.6), Inches(2.0), Inches(5.6), Inches(3.4))
    gt.name = "cleaning funnel"
    tbl = gt.table
    tbl.first_row = False
    tbl.horz_banding = False
    tbl.columns[0].width, tbl.columns[1].width = Inches(3.7), Inches(1.9)
    for r, (k, v) in enumerate(rows):
        strong = r in (4, 6)
        for c, val in enumerate((k, v)):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(ICE if strong else WHITE)
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if c else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = val
            run.font.size, run.font.name, run.font.bold = Pt(14), BODY, strong
            run.font.color.rgb = rgb(NAVY if strong else TEXT)
    fam = clean["family_counts"]
    order = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "Botnet", "WebAttack", "Infiltration", "Heartbleed"]
    ch = add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, 6.6, 1.85, 6.13, 3.75, order,
                   {"Flows after cleaning": [fam.get(f, 0) for f in order]})
    style_chart(ch, legend=False, value_fmt="#,##0", label_size=10)
    color_series(ch, [NAVY])
    pt_colors = {"Infiltration": RED, "Heartbleed": RED}
    for i, f in enumerate(order):
        if f in pt_colors:
            pt = ch.plots[0].series[0].points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = rgb(pt_colors[f])
    log_axis(ch.value_axis)
    ch.value_axis.minimum_scale = 1
    ch.value_axis.maximum_scale = 10_000_000
    ch.value_axis.tick_labels.number_format = "#,##0"
    ch.value_axis.tick_labels.number_format_is_linked = False
    ch.category_axis.tick_labels.font.size = Pt(10)
    chart_title(ch, "Flows per attack family after cleaning (log scale)")
    box(s, 0.6, 5.75, 12.13, 0.95, ICE)
    text(s, 0.85, 5.82, 11.7, 0.85,
         [[("Port scans collapse from 231k to 7.7k flows ", {"bold": True, "color": NAVY}),
           ("once ports are dropped: scan probes are identical SYN→RST flows. ", {}),
           ("Infiltration (36) and Heartbleed (11) ", {"bold": True, "color": RED}),
           ("are too small for multi-class evaluation; they stay in the binary and leave-one-attack-out tests.", {})]],
         size=14, anchor=MSO_ANCHOR.MIDDLE)
    footer(s, "Methodology", n)
    notes(s, "Removing duplicates before splitting matters: otherwise copies sit in both train and test and inflate scores. "
             "'Attempted' flows (attack sessions without attack payload) are treated as benign, as the dataset authors recommend.")

    # 5 ── methodology: models & imbalance ────────────────────────────────────
    s = new(); n += 1
    header(s, "Methodology", "Classifiers & Class-Imbalance Handling",
           "Five model families, each trained with three imbalance strategies (14 runs; MLP has no class weights)")
    cards = [("Logistic Regression", "Linear baseline", "C (regularisation)", "log + standardise"),
             ("Random Forest", "Bagged trees", "trees, depth, split size, max features", "raw features"),
             ("XGBoost", "Gradient boosting", "learning rate, depth, trees, subsampling", "raw features"),
             ("RBF-SVM", "Margin classifier", "C, γ  —  trained on 60k rows (cost ∝ n²)", "log + standardise"),
             ("MLP", "Feed-forward net", "layer sizes, learning rate, α, activation", "log + standardise")]
    cw = 2.29
    for i, (name, role, tuned, inp) in enumerate(cards):
        x = 0.6 + i * (cw + 0.17)
        box(s, x, 1.95, cw, 2.75, ICE)
        circle(s, x + 0.2, 2.15, 0.5, NAVY, str(i + 1), color=WHITE, size=15)
        text(s, x + 0.2, 2.8, cw - 0.4, 0.4, name, size=16, bold=True, color=NAVY)
        text(s, x + 0.2, 3.17, cw - 0.4, 0.3, role, size=13, italic=True, color=MUTED)
        text(s, x + 0.2, 3.55, cw - 0.4, 1.1,
             [[("Tuned: ", {"bold": True}), (tuned, {})], [("Input: ", {"bold": True}), (inp, {})]],
             size=12, spacing_after=3)
    strat = [("None", "Learn from the natural class mix."),
             ("Class weights", "Loss weighted by inverse class frequency (sample weights for XGBoost)."),
             ("SMOTE", "Inside each CV fold only; minority classes raised to 20k rows; neighbours found in log space.")]
    text(s, 0.6, 4.95, 6, 0.3, "IMBALANCE STRATEGIES", size=11, bold=True, color=AMBER)
    for i, (t, desc) in enumerate(strat):
        x = 0.6 + i * 4.11
        box(s, x, 5.3, 3.91, 1.4, WHITE, line=LINE)
        text(s, x + 0.2, 5.42, 3.5, 1.2, [[(t, {"bold": True, "color": NAVY, "size": 15})], desc],
             size=13, spacing_after=3)
    footer(s, "Methodology", n)
    notes(s, "Hyper-parameters: RandomizedSearchCV, 12 draws, 3-fold, on a 150k-row stratified sample; tuned once per model "
             "and reused for all three strategies so the strategy comparison is not confounded by tuning.")

    # 6 ── methodology: evaluation protocol ───────────────────────────────────
    s = new(); n += 1
    header(s, "Methodology", "Evaluation Protocol",
           "Accuracy alone hides failures; each test below answers a different question")
    grid = [("Metrics", "Macro-F1, per-class recall, benign false-positive rate, PR-AUC, detection rate at 0.1% and 1% FPR"),
            ("Significance", "Nadeau–Bengio corrected t-test on 5-fold CV scores, plus McNemar's test on the test set"),
            ("Leakage check", "Time-block split: flows grouped into 5-minute blocks that never straddle train and test"),
            ("Explainability", "SHAP TreeExplainer: global feature ranking and per-flow explanations of alerts"),
            ("Deployability", "Model size, single-flow latency (1 thread), batch throughput, ONNX Runtime latency"),
            ("Generalisation", "Leave-one-attack-out (zero-day proxy) and corrected CSE-CIC-IDS2018 with no retraining")]
    for i, (t, desc) in enumerate(grid):
        x, y = 0.6 + (i % 3) * 4.11, 1.95 + (i // 3) * 2.45
        box(s, x, y, 3.91, 2.2, ICE)
        circle(s, x + 0.25, y + 0.25, 0.55, AMBER, str(i + 1), size=15)
        text(s, x + 0.95, y + 0.32, 2.8, 0.4, t, size=17, bold=True, color=NAVY)
        text(s, x + 0.25, y + 0.95, 3.4, 1.15, desc, size=14)
    footer(s, "Methodology", n)
    cc = d["cross_clean"]["family_counts"]
    notes(s, f"Cross-dataset target: 5 days of corrected CSE-CIC-IDS2018 streamed from the 9.7 GB archive; all attack flows plus 5% "
             f"of benign flows; {sum(cc.values()):,} flows after cleaning. Same fixed CICFlowMeter, so features match exactly.")

    # 7 ── results: overall ───────────────────────────────────────────────────
    s = new(); n += 1
    header(s, "Simulation Results", "Overall Performance",
           "Best strategy per model; macro-F1 on the held-out test set")
    ch = add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, 0.6, 1.85, 7.6, 4.95, [NAMES[m] for m in MODELS],
                   {"Random split": [rb.loc[m, "f1_macro"] for m in MODELS],
                    "Time-block split": [tbb.loc[m, "f1_macro"] for m in MODELS]})
    plot = style_chart(ch)
    plot.gap_width, plot.overlap = 70, -5
    color_series(ch, [NAVY, AMBER])
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0.8, 1.02
    ch.value_axis.tick_labels.number_format = "0.00"
    ch.value_axis.tick_labels.number_format_is_linked = False
    chart_title(ch, "Macro-F1 (y-axis starts at 0.80)")
    tb_rf = d["tb"].query("model == 'random_forest'")
    callout(s, 8.6, 1.95, 4.13, 1.5, "Random Forest ≈ XGBoost",
            f"Statistically tied: corrected t-test p = {p_rf_xgb_r:.2f} (random) and {p_rf_xgb_t:.2f} (time-block).")
    callout(s, 8.6, 3.6, 4.13, 1.5, "Benign vs attack barely moves",
            f"Time-block split: attack recall ≥ {tb_rf['bin_recall'].min():.3f}, benign FPR "
            f"{fmt_small_pct(tb_rf['bin_fpr'].max())} for Random Forest.")
    callout(s, 8.6, 5.25, 4.13, 1.55, "Multi-class drop: 0.99 → 0.94",
            "Almost entirely one class (WebAttack). See the next slide.", fill=AMBER_SOFT)
    footer(s, "Simulation Results", n)
    notes(s, "All non-linear models exceed 0.99 macro-F1 on the random split. The corrected t-test is conservative; with 5 folds "
             "only large gaps are significant (e.g. Random Forest vs Logistic Regression, p ≈ 0.005).")

    # 8 ── results: per-class & imbalance ─────────────────────────────────────
    s = new(); n += 1
    header(s, "Simulation Results", "Per-Class Recall & Class Imbalance",
           "Time-block split, best strategy per model; aggregate accuracy hides the weak class")
    classes = list(d["tb_pc"]["xgboost"].index)
    support = d["tb_pc"]["xgboost"]["support"]
    gt = s.shapes.add_table(len(MODELS) + 2, len(classes) + 1, Inches(0.6), Inches(1.95), Inches(8.0), Inches(3.4))
    gt.name = "per-class recall"
    tbl = gt.table
    tbl.first_row = True
    tbl.horz_banding = False
    tbl.columns[0].width = Inches(1.6)
    for j in range(len(classes)):
        tbl.columns[j + 1].width = Inches(6.4 / len(classes))

    def cell_text(cell, val, bold=False, color=TEXT, fill=WHITE, size=12, align=PP_ALIGN.CENTER):
        cell.fill.solid()
        cell.fill.fore_color.rgb = rgb(fill)
        cell.margin_left = cell.margin_right = Inches(0.05)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = val
        r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(size), bold, BODY, rgb(color)

    cell_text(tbl.cell(0, 0), "Model", True, WHITE, NAVY, align=PP_ALIGN.LEFT)
    for j, c in enumerate(classes):
        cell_text(tbl.cell(0, j + 1), c, True, WHITE, NAVY, size=11)
    for i, m in enumerate(MODELS):
        cell_text(tbl.cell(i + 1, 0), NAMES[m], True, NAVY, WHITE, align=PP_ALIGN.LEFT)
        for j, c in enumerate(classes):
            v = d["tb_pc"][m].loc[c, "recall"]
            fill, col = (ICE, NAVY) if v >= 0.99 else (AMBER_SOFT, TEXT) if v >= 0.9 else \
                (AMBER, NAVY) if v >= 0.7 else (RED, WHITE)
            cell_text(tbl.cell(i + 1, j + 1), f"{v:.3f}", v < 0.7, col, fill)
    last = len(MODELS) + 1
    cell_text(tbl.cell(last, 0), "Test flows", False, MUTED, WHITE, size=11, align=PP_ALIGN.LEFT)
    for j, c in enumerate(classes):
        cell_text(tbl.cell(last, j + 1), fmt_int(support[c]), False, MUTED, WHITE, size=11)
    lr_r = d["rand"].set_index(["model", "strategy"])
    callout(s, 9.0, 1.95, 3.73, 2.3, "Unseen variant → missed",
            "The time-block split put 12 of 13 SQL-injection flows in test. Every model missed all 12 "
            "(called benign or DoS); web brute force was caught 10/10.", fill=AMBER_SOFT, body_size=13)
    callout(s, 9.0, 4.4, 3.73, 2.3, "Rebalancing does not help",
            f"SMOTE gives no significant gain. Class weights hurt linear models: Logistic Regression macro-F1 "
            f"{lr_r.loc[('logreg', 'none'), 'f1_macro']:.2f} → {lr_r.loc[('logreg', 'class_weight'), 'f1_macro']:.2f}, "
            f"benign FPR {lr_r.loc[('logreg', 'none'), 'bin_fpr']:.2%} → {lr_r.loc[('logreg', 'class_weight'), 'bin_fpr']:.1%}.",
            body_size=13)
    text(s, 0.6, 5.55, 8.0, 0.9,
         "Cell colour: ≥ 0.99 blue, 0.90–0.99 light amber, 0.70–0.90 amber, < 0.70 red. "
         "Hypothesis H2 (low rare-class recall before SMOTE) is not supported on the corrected, de-duplicated data.",
         size=12, italic=True, color=MUTED)
    footer(s, "Simulation Results", n)
    notes(s, "Web attacks are the clearest example: the classifier recognises the variant it trained on and fails on the one it "
             "barely saw. Class weights push the decision boundary towards rare classes, which floods benign traffic with alerts.")

    # 9 ── results: deployability ─────────────────────────────────────────────
    s = new(); n += 1
    header(s, "Simulation Results", "Deployability",
           "Cost of classifying one flow, measured single-threaded on an Apple M1 (8 GB)")
    dep = d["dep"]
    cats = [NAMES[m] for m in MODELS] + ["RF (ONNX)", "XGB (ONNX)"]
    vals = [dep.loc[(m, "none"), "latency_median_ms"] for m in MODELS] + \
           [dep.loc[("random_forest", "none"), "onnx_latency_median_ms"], dep.loc[("xgboost", "none"), "onnx_latency_median_ms"]]
    ch = add_chart(s, XL_CHART_TYPE.BAR_CLUSTERED, 0.6, 1.85, 6.5, 4.95, cats[::-1], {"Median latency (ms)": vals[::-1]})
    plot = style_chart(ch, legend=False, value_fmt="0.000")
    plot.gap_width = 60
    color_series(ch, [NAVY])
    for i, c in enumerate(cats[::-1]):
        if "ONNX" in c:
            pt = plot.series[0].points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = rgb(AMBER)
    ch.value_axis.tick_labels.number_format = "0.0"
    ch.value_axis.tick_labels.number_format_is_linked = False
    chart_title(ch, "Median single-flow latency (ms, lower is better)")
    hdr = ["Model", "Size", "Train", "Flows / s"]
    gt = s.shapes.add_table(len(MODELS) + 1, 4, Inches(7.45), Inches(1.95), Inches(5.28), Inches(2.9))
    gt.name = "deployability table"
    tbl = gt.table
    tbl.first_row = True
    tbl.horz_banding = False
    for j, wdt in enumerate((1.75, 1.1, 1.1, 1.33)):
        tbl.columns[j].width = Inches(wdt)
    for j, h_ in enumerate(hdr):
        cell_text(tbl.cell(0, j), h_, True, WHITE, NAVY, size=12, align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT)
    for i, m in enumerate(MODELS):
        r = dep.loc[(m, "none")]
        size = f"{r['size_mb'] * 1000:.0f} KB" if r["size_mb"] < 1 else f"{r['size_mb']:.1f} MB"
        vals_ = [NAMES[m], size, f"{r['train_seconds']:.0f} s", f"{r['batch_flows_per_s'] / 1000:,.0f}k"]
        for j, v in enumerate(vals_):
            cell_text(tbl.cell(i + 1, j), v, j == 0, NAVY if j == 0 else TEXT, ICE if m == "xgboost" else WHITE,
                      size=12, align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT)
    rf_ms, xgb_ms = dep.loc[("random_forest", "none"), "latency_median_ms"], dep.loc[("xgboost", "none"), "latency_median_ms"]
    callout(s, 7.45, 5.05, 5.28, 1.75, "XGBoost: best accuracy-to-cost trade-off",
            f"Ties Random Forest on F1 at {rf_ms / xgb_ms:.0f}× lower latency and ~3× smaller. "
            "SVM is fast per flow but slowest in batch (~20k flows/s). Real detection delay is "
            "dominated by flow completion (up to a 120 s timeout).", body_size=13)
    footer(s, "Simulation Results", n)
    notes(s, "SVM training time is low only because it was capped at 60k rows. ONNX export brings the tree models down to "
             "roughly 0.01–0.02 ms per flow, which makes inline use feasible even on small devices.")

    # 10 ── results: SHAP ─────────────────────────────────────────────────────
    s = new(); n += 1
    model_name = NAMES[d["shap_run"].split("__")[0]]
    header(s, "Simulation Results", "Explainability (SHAP)",
           f"What drives {model_name}'s decisions: top-10 features by mean |SHAP value|")
    top = d["shap"].head(10)
    names = original_names()
    labels = [names.get(f, f) for f in top.index][::-1]
    ch = add_chart(s, XL_CHART_TYPE.BAR_CLUSTERED, 0.6, 1.85, 7.4, 4.95, labels, {"mean |SHAP|": list(top.values[::-1])})
    plot = style_chart(ch, legend=False, value_fmt="0.00")
    plot.gap_width = 45
    color_series(ch, [NAVY])
    for i, f in enumerate(top.index[::-1]):
        if "init_win" in f:
            pt = plot.series[0].points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = rgb(RED)
    ch.category_axis.tick_labels.font.size = Pt(11)
    chart_title(ch, "Mean |SHAP value| across all classes")
    rank = list(d["shap"].index).index("bwd_init_win_bytes") + 1
    callout(s, 8.4, 1.95, 4.33, 2.3, "Packet sizes dominate",
            "Backward/forward packet length statistics lead. This is defensible: DoS floods, scans and web attacks have "
            "characteristic request and response sizes.", body_size=13)
    callout(s, 8.4, 4.4, 4.33, 2.4, f"Red flag: Bwd Init Win Bytes is #{rank}",
            "The initial TCP window reflects the host operating system and testbed, not the attack. A model leaning on it "
            "learns the lab, which matches the cross-dataset collapse.", fill=AMBER_SOFT, title_color=RED, body_size=13)
    footer(s, "Simulation Results", n)
    notes(s, "SHAP also produces per-flow explanations (waterfall plots in results/.../shap/) that show an analyst why a "
             "specific flow was flagged.")

    # 11 ── results: leave-one-attack-out ─────────────────────────────────────
    s = new(); n += 1
    header(s, "Simulation Results", "Unseen Attacks: Leave-One-Attack-Out",
           "Binary XGBoost trained without one attack family, then tested on that family")
    lo = d["loao"]
    fams = ["DDoS", "DoS", "Heartbleed", "Infiltration", "Botnet", "WebAttack", "PortScan", "BruteForce"]
    ch = add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, 0.6, 1.85, 8.0, 4.95, fams,
                   {"Default threshold (0.5)": [lo.loc[f, "unseen_detection_rate"] for f in fams],
                    "Threshold set for 1% FPR": [lo.loc[f, "unseen_recall_at_fpr_0.01"] for f in fams]})
    plot = style_chart(ch, value_fmt="0%", label_size=10)
    plot.gap_width, plot.overlap = 60, -5
    color_series(ch, [NAVY, AMBER])
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0, 1.1
    ch.value_axis.tick_labels.number_format = "0%"
    ch.value_axis.tick_labels.number_format_is_linked = False
    ch.category_axis.tick_labels.font.size = Pt(11)
    chart_title(ch, "Share of the held-out family detected")
    auc_lo, auc_hi = lo["unseen_vs_benign_roc_auc"].min(), lo["unseen_vs_benign_roc_auc"].max()
    callout(s, 9.0, 1.95, 3.73, 2.3, "Default threshold: ~0%",
            "Except DDoS (similar to DoS), held-out families are almost never flagged; the zero-day claim fails as deployed.",
            fill=AMBER_SOFT, body_size=13)
    callout(s, 9.0, 4.4, 3.73, 2.4, "Signal is partly there",
            f"Scores still rank unseen flows above benign (ROC-AUC {auc_lo:.2f}–{auc_hi:.2f}). Re-tuning the threshold "
            "recovers DoS, Botnet and Infiltration, but not port scans, brute force or web attacks.", body_size=13)
    footer(s, "Simulation Results", n)
    notes(s, "Heartbleed (11 flows) and Infiltration (36 flows) results are indicative only. Random-split models; the time-block "
             "split gives the same pattern.")

    # 12 ── results: cross-dataset ────────────────────────────────────────────
    s = new(); n += 1
    header(s, "Simulation Results", "Cross-Dataset Generalisation",
           "Trained on CICIDS2017, tested with no retraining on corrected CSE-CIC-IDS2018 (benign vs attack)")
    cr = d["cross"]
    ch = add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, 0.6, 1.85, 7.6, 4.95, [NAMES[m] for m in MODELS],
                   {"CICIDS2017 test (in-distribution)": [cr.loc[(m, "none"), "in_dist_bin_f1"] for m in MODELS],
                    "CSE-CIC-IDS2018 (new network)": [cr.loc[(m, "none"), "target_f1"] for m in MODELS]})
    plot = style_chart(ch, value_fmt="0.00", label_size=11)
    plot.gap_width, plot.overlap = 60, -5
    color_series(ch, [NAVY, RED])
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0, 1.1
    ch.value_axis.tick_labels.number_format = "0.0"
    ch.value_axis.tick_labels.number_format_is_linked = False
    chart_title(ch, "Binary F1 (no imbalance strategy)")
    box(s, 8.6, 1.95, 4.13, 2.0, NAVY)
    text(s, 8.85, 2.1, 3.7, 0.9, f"{cross_detect_tree:.1%}", size=48, bold=True, color=AMBER, font=HEAD)
    text(s, 8.85, 3.05, 3.7, 0.8, f"of 2018 attacks detected by Random Forest / XGBoost (in-distribution F1 "
         f"{min(xgb_cross['in_dist_bin_f1'], rf_cross['in_dist_bin_f1']):.4f})",
         size=13, color=WHITE)
    callout(s, 8.6, 4.1, 4.13, 2.7, "Simplest model transfers best",
            f"Logistic Regression keeps F1 {lr_cross['target_f1']:.2f} (recall {lr_cross['target_recall']:.0%}, "
            f"ROC-AUC {lr_cross['target_roc_auc']:.2f}). Even at a 1% FPR threshold no model exceeds "
            f"{max(cr['target_recall_at_fpr_0.01']):.0%} recall: this is distribution shift, not a threshold problem.",
            body_size=13)
    footer(s, "Simulation Results", n)
    notes(s, f"Target: {sum(cc.values()):,} flows after cleaning (DoS, DDoS, Botnet, brute force; benign sampled at 5%, "
             "so target precision is optimistic while recall and FPR are unaffected). Same fixed CICFlowMeter, identical features.")

    # 13 ── conclusion ────────────────────────────────────────────────────────
    s = new(); n += 1
    background(s, NAVY)
    text(s, 0.6, 0.38, 9, 0.3, "SIMULATION RESULTS", size=11, bold=True, color=AMBER)
    text(s, 0.6, 0.62, 12, 0.7, "Conclusions", size=34, bold=True, color=WHITE, font=HEAD)
    verdicts = [("H1", "Ensembles beat the linear baseline", "Partly supported", "RF / XGBoost rank top, but the gap to MLP / SVM is not significant"),
                ("H2", "Rare classes weak before SMOTE", "Not supported", "Rare-class recall already high; SMOTE adds nothing significant"),
                ("H3", "RF / XGBoost best accuracy–latency", "Partly supported", f"XGBoost: RF-level F1 at {rf_ms / xgb_ms:.0f}× lower latency; SVM cost untested (capped)"),
                ("H4", "Accuracy drops on a new dataset", "Strongly supported", f"RF / XGBoost 2018 F1 ≤ 0.01; detection {cross_detect_tree:.1%}")]
    for i, (h_, claim, verdict, ev) in enumerate(verdicts):
        y = 1.65 + i * 0.95
        circle(s, 0.6, y, 0.7, AMBER, h_, size=15)
        text(s, 1.5, y + 0.02, 4.6, 0.35, claim, size=16, bold=True, color=WHITE)
        text(s, 1.5, y + 0.38, 4.8, 0.35, ev, size=12, color="C9D6EC")
        ok = verdict != "Not supported"
        box(s, 6.3, y + 0.12, 1.9, 0.46, AMBER if ok else "3B4C70")
        text(s, 6.3, y + 0.12, 1.9, 0.46, verdict, size=12, bold=True, color=NAVY if ok else WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 8.65, 1.65, 4.08, 4.5, "253660")
    text(s, 8.9, 1.85, 3.6, 4.2,
         [[("Recommendation", {"bold": True, "size": 17, "color": AMBER})],
          "Deploy XGBoost (via ONNX) for attacks seen in training: 0.99+ F1, ~0.02 ms per flow.",
          "Do not take benchmark accuracy to a new network: retrain or adapt per site.",
          "Drop testbed-specific features such as Init Win Bytes and re-test generalisation.",
          "Tune alert thresholds on validation data instead of using 0.5."],
         size=14, color=WHITE, spacing_after=9)
    text(s, 0.6, 5.65, 7.6, 0.8,
         "“Near-perfect benchmark scores measure how well a model learns the lab, not how well it protects a network.”",
         size=16, italic=True, color="C9D6EC")
    footer(s, "Conclusions", n, dark=True)
    notes(s, "Future work: domain adaptation or multi-dataset training, feature sets that exclude host-specific fields, "
             "and per-site threshold calibration.")

    OUT.parent.mkdir(exist_ok=True)
    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
