"""Build docs/IDS_Project_Report.pdf: the complete project write-up, from proposal to results,
with the work divided between the two team members.

All numbers are read from results/, data/processed/ and the raw CSV headers at build time:

    .venv/bin/python scripts/build_report.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RAND = RES / "cicids2017_fixed__multiclass__random"
TB = RES / "cicids2017_fixed__multiclass__time_block"
PROC = ROOT / "data" / "processed"
FIG = RES / "report_figures"
OUT = ROOT / "docs" / "IDS_Project_Report.pdf"

MEMBER_A, MEMBER_B = "Ahmad Azim", "Pawan Kumar"
MODELS = ["logreg", "svm", "mlp", "random_forest", "xgboost"]
NAMES = {"logreg": "Logistic Regression", "svm": "RBF-SVM", "mlp": "MLP", "random_forest": "Random Forest",
         "xgboost": "XGBoost"}
STRAT = {"none": "none", "class_weight": "class weights", "smote": "SMOTE"}

# palette of the project slides
NAVY, AMBER, ICE, TEXT, MUTED, LINE_C, RED, AMBER_SOFT = (
    "#1B2A4A", "#F2A93B", "#E4EAF6", "#1F2937", "#5B6B82", "#C9D6EC", "#C0392B", "#FCE7C3")
C = colors.HexColor

# --------------------------------------------------------------------------- fonts & styles
DF = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
SUP = "/System/Library/Fonts/Supplemental/"


def register_fonts():
    try:
        pdfmetrics.registerFont(TTFont("Body", DF + "Calibri.ttf"))
        pdfmetrics.registerFont(TTFont("Body-B", DF + "Calibrib.ttf"))
        pdfmetrics.registerFont(TTFont("Body-I", DF + "Calibrii.ttf"))
        pdfmetrics.registerFont(TTFont("Body-BI", DF + "Calibriz.ttf"))
        pdfmetrics.registerFont(TTFont("Head", DF + "Cambriab.ttf"))
    except Exception:  # fall back to Arial / Georgia shipped with macOS
        pdfmetrics.registerFont(TTFont("Body", SUP + "Arial.ttf"))
        pdfmetrics.registerFont(TTFont("Body-B", SUP + "Arial Bold.ttf"))
        pdfmetrics.registerFont(TTFont("Body-I", SUP + "Arial Italic.ttf"))
        pdfmetrics.registerFont(TTFont("Body-BI", SUP + "Arial Bold Italic.ttf"))
        pdfmetrics.registerFont(TTFont("Head", SUP + "Georgia Bold.ttf"))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-BI")


register_fonts()
S = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=10.5, leading=14.5, textColor=C(TEXT),
                           spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Body", fontSize=9, leading=12, textColor=C(TEXT)),
    "cap": ParagraphStyle("cap", fontName="Body-I", fontSize=9, leading=12, textColor=C(MUTED), spaceBefore=3,
                          spaceAfter=10),
    "H1": ParagraphStyle("H1", fontName="Head", fontSize=20, leading=25, textColor=C(NAVY), spaceBefore=4,
                         spaceAfter=10),
    "H2": ParagraphStyle("H2", fontName="Head", fontSize=13.5, leading=18, textColor=C(NAVY), spaceBefore=10,
                         spaceAfter=5),
    "H3": ParagraphStyle("H3", fontName="Body-B", fontSize=11, leading=14, textColor=C(NAVY), spaceBefore=6,
                         spaceAfter=3),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=10.5, leading=14.5, textColor=C(TEXT),
                             leftIndent=14, bulletIndent=4, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=9, leading=11.5, textColor=C(TEXT)),
    "cellb": ParagraphStyle("cellb", fontName="Body-B", fontSize=9, leading=11.5, textColor=C(NAVY)),
    "cellh": ParagraphStyle("cellh", fontName="Body-B", fontSize=9, leading=11.5, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.5, leading=11, textColor=C(TEXT),
                           backColor=C("#F3F4F6"), borderPadding=(5, 6, 5, 6), spaceBefore=4, spaceAfter=8),
    "callout": ParagraphStyle("callout", fontName="Body", fontSize=10.5, leading=14.5, textColor=C(TEXT)),
}


def fix(t: str) -> str:
    """Escape the bare '&' and '<' that prose uses, leaving markup tags intact."""
    return t.replace(" & ", " &amp; ").replace("< ", "&lt; ")


def P(t, st="body"):
    return Paragraph(fix(t), S[st])


def B(t):
    return Paragraph(fix(t), S["bullet"], bulletText="•")


def code(t):
    return Paragraph(escape(t).replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;"), S["code"])


def callout(t, fill=ICE):
    tb = Table([[Paragraph(fix(t), S["callout"])]], colWidths=[17 * cm])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C(fill)),
                            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return KeepTogether([Spacer(1, 4), tb, Spacer(1, 8)])


def table(rows, widths, header=True, zebra=True, align_right_from=None, fills=None, font_size=None):
    """rows: list of lists of str (str cells are wrapped in Paragraphs)."""
    data = []
    for i, r in enumerate(rows):
        st = "cellh" if (header and i == 0) else "cell"
        out = []
        for j, v in enumerate(r):
            if isinstance(v, str):
                style = S[st]
                if align_right_from is not None and j >= align_right_from and not (header and i == 0):
                    style = ParagraphStyle("r", parent=style, alignment=2)
                elif align_right_from is not None and j >= align_right_from:
                    style = ParagraphStyle("rh", parent=style, alignment=2)
                if font_size:
                    style = ParagraphStyle("fs", parent=style, fontSize=font_size, leading=font_size + 2.5)
                out.append(Paragraph(fix(v), style))
            else:
                out.append(v)
        data.append(out)
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    cmds = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, C(LINE_C))]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), C(NAVY)))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if (i % 2 == 0) == header:
                cmds.append(("BACKGROUND", (0, i), (-1, i), C("#F5F7FB")))
    for (cell, fill) in (fills or []):
        cmds.append(("BACKGROUND", cell, cell, C(fill)))
    t.setStyle(TableStyle(cmds))
    return t


def fig(path, width_cm, caption):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, P(caption, "cap")])


def f3(x):
    return f"{x:.3f}"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def n_(x):
    return f"{int(x):,}"


# --------------------------------------------------------------------------- data
def load():
    d = {}
    for key, path, split in (("r", RAND, "random"), ("t", TB, "time_block")):
        d[f"{key}_sum"] = pd.read_csv(path / "summary.csv")
        d[f"{key}_sig"] = pd.read_csv(path / "significance.csv")
        d[f"{key}_dep"] = pd.read_csv(path / "deployability.csv")
        d[f"{key}_loao"] = pd.read_csv(path / "leave_one_attack_out.csv")
        d[f"{key}_cross"] = pd.read_csv(path / "cross_dataset__cicids2018_fixed.csv")
        d[f"{key}_crossfam"] = pd.read_csv(path / "cross_dataset__cicids2018_fixed__per_family.csv")
        d[f"{key}_tune"] = json.loads((path / "tuning.json").read_text())
        d[f"{key}_meta"] = json.loads((PROC / f"cicids2017_fixed__multiclass__{split}" / "meta.json").read_text())
        d[f"{key}_shap"] = pd.read_csv(path / "shap" / "shap_global_importance.csv", index_col=0)
        d[f"{key}_shaprun"] = json.loads((path / "shap" / "summary.json").read_text())["run"]
        pcs = {}
        for f in sorted((path / "runs").glob("*__per_class.csv")):
            pcs[f.name.replace("__per_class.csv", "")] = pd.read_csv(f).set_index("class")
        d[f"{key}_pc"] = pcs
    d["cross_clean"] = json.loads((RAND / "cross_dataset__cicids2018_fixed__clean.json").read_text())
    # raw label counts per day straight from the CSVs
    raw = {}
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        raw[day] = pd.read_csv(ROOT / "data/raw/cicids2017_fixed" / f"{day}.csv", usecols=["Label"])["Label"].value_counts()
    d["raw"] = raw
    hdr = (ROOT / "data/raw/cicids2017_fixed/monday.csv").open().readline().strip().split(",")
    import re
    d["orig"] = {re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_"): h for h in hdr}
    d["n_cols"] = len(hdr)
    # streamed 2018 days
    days18 = {}
    for p in sorted((ROOT / "data/raw/cicids2018_fixed").glob("*.parquet")):
        fam = pd.read_parquet(p, columns=["family"])["family"].value_counts()
        days18[p.stem] = fam
    d["days18"] = days18
    return d


def best_by_model(summary):
    return summary.sort_values("f1_macro", ascending=False).drop_duplicates("model").set_index("model")


def pval(sig, a, b, strategy, col="cv_p_value"):
    r = sig[(sig.strategy == strategy) & (((sig.model_a == a) & (sig.model_b == b)) | ((sig.model_a == b) & (sig.model_b == a)))]
    return float(r[col].iloc[0])


# --------------------------------------------------------------------------- figures
def make_figures(d):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#9CA3AF", "axes.titleweight": "bold",
                         "axes.titlesize": 11})
    out = {}
    # class distribution
    fam = d["r_meta"]["clean"]["family_counts"]
    order = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "Botnet", "WebAttack", "Infiltration", "Heartbleed"]
    fig_, ax = plt.subplots(figsize=(8, 3.4))
    vals = [fam.get(o, 0) for o in order]
    bars = ax.bar(order, vals, color=[RED if o in ("Infiltration", "Heartbleed") else NAVY for o in order])
    ax.set_yscale("log")
    ax.set_ylabel("flows (log scale)")
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v * 1.15, f"{v:,}", ha="center", fontsize=8)
    ax.set_title("Flows per attack family after cleaning (corrected CICIDS2017)")
    plt.xticks(rotation=20)
    fig_.tight_layout()
    out["dist"] = FIG / "class_distribution.png"
    fig_.savefig(out["dist"], dpi=200)
    plt.close(fig_)

    # random vs time-block
    rb, tbb = best_by_model(d["r_sum"]), best_by_model(d["t_sum"])
    x = np.arange(len(MODELS))
    fig_, ax = plt.subplots(figsize=(8, 3.4))
    a = ax.bar(x - 0.2, [rb.loc[m, "f1_macro"] for m in MODELS], 0.4, color=NAVY, label="Random split")
    b = ax.bar(x + 0.2, [tbb.loc[m, "f1_macro"] for m in MODELS], 0.4, color=AMBER, label="Time-block split")
    for bars in (a, b):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.003, f"{r.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [NAMES[m] for m in MODELS])
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("macro-F1 (axis starts at 0.80)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_title("Best macro-F1 per model: random vs time-block split", pad=26)
    fig_.tight_layout()
    out["splits"] = FIG / "random_vs_timeblock.png"
    fig_.savefig(out["splits"], dpi=200)
    plt.close(fig_)

    # LOAO
    lo = d["r_loao"].set_index("held_out")
    fams = ["DDoS", "DoS", "Heartbleed", "Infiltration", "Botnet", "WebAttack", "PortScan", "BruteForce"]
    x = np.arange(len(fams))
    fig_, ax = plt.subplots(figsize=(8, 3.4))
    a = ax.bar(x - 0.2, [lo.loc[f, "unseen_detection_rate"] for f in fams], 0.4, color=NAVY, label="Default threshold (0.5)")
    b = ax.bar(x + 0.2, [lo.loc[f, "unseen_recall_at_fpr_0.01"] for f in fams], 0.4, color=AMBER, label="Threshold for 1% FPR")
    for bars in (a, b):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015, f"{r.get_height():.0%}", ha="center", fontsize=7.5)
    ax.set_xticks(x, fams)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_title("Leave-one-attack-out: share of the never-seen family detected (binary XGBoost)", pad=26)
    fig_.tight_layout()
    out["loao"] = FIG / "loao.png"
    fig_.savefig(out["loao"], dpi=200)
    plt.close(fig_)

    # cross-dataset
    cr = d["r_cross"].set_index(["model", "strategy"])
    x = np.arange(len(MODELS))
    fig_, ax = plt.subplots(figsize=(8, 3.4))
    a = ax.bar(x - 0.2, [cr.loc[(m, "none"), "in_dist_bin_f1"] for m in MODELS], 0.4, color=NAVY, label="CICIDS2017 test")
    b = ax.bar(x + 0.2, [cr.loc[(m, "none"), "target_f1"] for m in MODELS], 0.4, color=RED, label="CSE-CIC-IDS2018 (no retraining)")
    for bars in (a, b):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015, f"{r.get_height():.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [NAMES[m] for m in MODELS])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("binary F1 (benign vs attack)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_title("Cross-dataset generalisation (random-split models, no imbalance strategy)", pad=26)
    fig_.tight_layout()
    out["cross"] = FIG / "cross_dataset.png"
    fig_.savefig(out["cross"], dpi=200)
    plt.close(fig_)
    return out


def pipeline_drawing():
    steps = ["Data", "Clean", "Split", "Features", "Train", "Evaluate"]
    subs = ["corrected\nCICIDS2017", "dedupe, drop\nIDs/IPs/ports", "random +\ntime-block", "82 → 60 → 30\n(corr + RFE)",
            "5 models ×\n3 strategies", "metrics, stats,\nSHAP, latency"]
    W_, bw, gap = 17 * cm, 2.35 * cm, 0.5 * cm
    d_ = Drawing(W_, 2.6 * cm)
    for i, (t, s_) in enumerate(zip(steps, subs)):
        x = i * (bw + gap)
        fill = C(NAVY) if i % 2 == 0 else C(AMBER)
        fg = colors.white if i % 2 == 0 else C(NAVY)
        d_.add(Rect(x, 0.2 * cm, bw, 2.2 * cm, rx=6, ry=6, fillColor=fill, strokeColor=None))
        d_.add(String(x + bw / 2, 1.75 * cm, f"{i + 1}. {t}", fontName="Body-B", fontSize=10.5, fillColor=fg,
                      textAnchor="middle"))
        for k, line_ in enumerate(s_.split("\n")):
            d_.add(String(x + bw / 2, 1.15 * cm - k * 0.42 * cm, line_, fontName="Body", fontSize=8.3, fillColor=fg,
                          textAnchor="middle"))
        if i < len(steps) - 1:
            ax_ = x + bw + 0.06 * cm
            d_.add(Line(ax_, 1.3 * cm, ax_ + gap - 0.18 * cm, 1.3 * cm, strokeColor=C(MUTED), strokeWidth=1.2))
            tip = ax_ + gap - 0.12 * cm
            d_.add(Polygon([tip, 1.3 * cm, tip - 0.14 * cm, 1.38 * cm, tip - 0.14 * cm, 1.22 * cm],
                           fillColor=C(MUTED), strokeColor=None))
    return d_


# --------------------------------------------------------------------------- document
class Doc(BaseDocTemplate):
    def __init__(self, fn):
        super().__init__(fn, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.1 * cm,
                         bottomMargin=2 * cm, title="ML-Based Network Intrusion Detection — Project Report",
                         author=f"{MEMBER_A} & {MEMBER_B}", subject="Computer Networks lab project")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=self.cover),
                               PageTemplate("body", [frame], onPage=self.chrome)])

    def cover(self, c, doc):
        w, h = A4
        c.saveState()
        c.setFillColor(C(NAVY))
        c.rect(0, 0, w, h, fill=1, stroke=0)
        for (x, y, r, col) in [(16.5, 25.5, 0.09, AMBER), (17.8, 24.6, 0.06, "#FFFFFF"), (17.1, 23.6, 0.07, "#8FA0C0"),
                               (15.2, 5.2, 0.07, "#FFFFFF"), (16.6, 4.3, 0.1, AMBER), (17.9, 5.6, 0.06, "#8FA0C0")]:
            c.setFillColor(C(col))
            c.circle(x * cm, y * cm, r * cm, fill=1, stroke=0)
        c.setStrokeColor(C("#3B4C70"))
        c.setLineWidth(0.8)
        c.line(16.5 * cm, 25.5 * cm, 17.8 * cm, 24.6 * cm)
        c.line(17.8 * cm, 24.6 * cm, 17.1 * cm, 23.6 * cm)
        c.line(15.2 * cm, 5.2 * cm, 16.6 * cm, 4.3 * cm)
        c.line(16.6 * cm, 4.3 * cm, 17.9 * cm, 5.6 * cm)
        c.setFillColor(C(AMBER))
        c.setFont("Body-B", 11)
        c.drawString(2.2 * cm, 19.6 * cm, "AI / ML  •  NETWORK SECURITY PROJECT  •  PROJECT REPORT")
        c.setFillColor(colors.white)
        c.setFont("Head", 30)
        c.drawString(2.2 * cm, 18.1 * cm, "ML-Based Network")
        c.drawString(2.2 * cm, 16.8 * cm, "Intrusion Detection")
        c.setFillColor(C("#C9D6EC"))
        c.setFont("Body-I", 13)
        c.drawString(2.2 * cm, 15.6 * cm, "A controlled benchmark of five classifiers on corrected CICIDS2017,")
        c.drawString(2.2 * cm, 15.0 * cm, "with explainability, deployability and generalisation tests")
        c.setStrokeColor(C("#3B4C70"))
        c.line(2.2 * cm, 13.6 * cm, 9 * cm, 13.6 * cm)
        c.setFont("Body", 12)
        c.setFillColor(colors.white)
        c.drawString(2.2 * cm, 12.8 * cm, f"{MEMBER_A}  &  {MEMBER_B}")
        c.setFillColor(C("#8FA0C0"))
        c.drawString(2.2 * cm, 12.2 * cm, "Computer Networks  —  Lab Project")
        c.drawString(2.2 * cm, 11.6 * cm, date.today().strftime("%B %Y"))
        c.setFont("Body", 9)
        c.drawString(2.2 * cm, 2.2 * cm, "Complete record of the project: proposal, design, implementation, experiments, "
                                         "results and task division.")
        c.restoreState()

    def chrome(self, c, doc):
        c.saveState()
        c.setFont("Body", 8.5)
        c.setFillColor(C(MUTED))
        c.drawString(2 * cm, 1.2 * cm, "ML-Based Network Intrusion Detection  |  Project Report")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
        c.setStrokeColor(C(LINE_C))
        c.setLineWidth(0.4)
        c.line(2 * cm, 1.55 * cm, A4[0] - 2 * cm, 1.55 * cm)
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("H1", "H2"):
            level = 0 if f.style.name == "H1" else 1
            key = f"h{id(f)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(f.getPlainText(), key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, f.getPlainText(), self.page, key))


def H1(t):
    return Paragraph(t, S["H1"])


def H2(t):
    return Paragraph(t, S["H2"])


def H3(t):
    return Paragraph(t, S["H3"])


# --------------------------------------------------------------------------- content
def build():
    d = load()
    figs = make_figures(d)
    rm, tm = d["r_meta"], d["t_meta"]
    clean = rm["clean"]
    rb, tbb = best_by_model(d["r_sum"]), best_by_model(d["t_sum"])
    rs = d["r_sum"].set_index(["model", "strategy"])
    ts = d["t_sum"].set_index(["model", "strategy"])
    dep = d["r_dep"].set_index(["model", "strategy"])
    cr = d["r_cross"].set_index(["model", "strategy"])
    lo = d["r_loao"].set_index("held_out")
    lo_t = d["t_loao"].set_index("held_out")
    orig = d["orig"]
    cc = d["cross_clean"]["family_counts"]
    tree_detect = max(cr.loc[("xgboost", "none"), "target_recall"], cr.loc[("random_forest", "none"), "target_recall"])
    rf_ms, xgb_ms = dep.loc[("random_forest", "none"), "latency_median_ms"], dep.loc[("xgboost", "none"), "latency_median_ms"]
    p_rx_r = pval(d["r_sig"], "random_forest", "xgboost", "none")
    p_rx_t = pval(d["t_sig"], "random_forest", "xgboost", "none")
    p_rl_r = pval(d["r_sig"], "random_forest", "logreg", "none")
    best_r = d["r_sum"].iloc[0]
    best_t = d["t_sum"].iloc[0]
    sql_tr = sql_te = None
    try:
        trl = pd.read_parquet(PROC / "cicids2017_fixed__multiclass__time_block" / "train.parquet", columns=["label"])["label"]
        tel = pd.read_parquet(PROC / "cicids2017_fixed__multiclass__time_block" / "test.parquet", columns=["label"])["label"]
        sql_tr = int((trl == "Web Attack - SQL Injection").sum())
        sql_te = int((tel == "Web Attack - SQL Injection").sum())
    except Exception:
        pass

    story = [Spacer(1, 1), NextPageTemplate("body"), PageBreak()]

    # ---- contents
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t0", fontName="Body-B", fontSize=11, leading=17, textColor=C(NAVY)),
                       ParagraphStyle("t1", fontName="Body", fontSize=9.5, leading=13, leftIndent=16, textColor=C(TEXT))]
    story += [Paragraph("Contents", ParagraphStyle("contents", parent=S["H1"])), toc, PageBreak()]

    # ==== 1 summary
    story += [H1("1. Executive Summary")]
    story += [P(
        "Signature-based intrusion detection systems (Snort, Suricata) only catch attacks that already have a rule. "
        "This project built and rigorously evaluated a machine-learning network intrusion detection system (ML-NIDS) that "
        "classifies network <b>flows</b> as benign or as one of several attack families, using the statistical features "
        "produced by CICFlowMeter. Five classifiers — Logistic Regression, Random Forest, XGBoost, an RBF-kernel SVM and a "
        "multi-layer perceptron (MLP) — were trained under one identical, leakage-safe pipeline on the <b>corrected</b> "
        "CICIDS2017 dataset, each with three class-imbalance strategies (none, class weights, SMOTE).")]
    story += [P(
        "Beyond accuracy, the study measures statistical significance, per-flow latency and model size, explains decisions "
        "with SHAP, and tests generalisation three ways: a time-separated split, a leave-one-attack-out experiment "
        "(a proxy for zero-day attacks), and an independent dataset (corrected CSE-CIC-IDS2018) evaluated with no retraining.")]
    kpis = [["Measure", "Value"],
            ["Flows loaded / after cleaning", f"{n_(clean['rows_in'])} / {n_(clean['rows_out'])}"],
            ["Experiment runs", "14 per split (5 models × 3 strategies, MLP has no class weights) × 2 splits"],
            ["Best macro-F1, random split", f"{f3(best_r['f1_macro'])} ({NAMES[best_r['model']]}, {STRAT[best_r['strategy']]})"],
            ["Best macro-F1, time-block split", f"{f3(best_t['f1_macro'])} ({NAMES[best_t['model']]}, {STRAT[best_t['strategy']]})"],
            ["Random Forest vs XGBoost", f"statistically tied (corrected t-test p = {p_rx_r:.2f} random, {p_rx_t:.2f} time-block)"],
            ["Single-flow latency", f"XGBoost {xgb_ms:.2f} ms, Random Forest {rf_ms:.2f} ms; ONNX ≈ 0.01–0.02 ms"],
            ["Unseen attack families caught (default threshold)", "≈ 0–2% for every family except DDoS"],
            ["2018 attacks caught by RF / XGBoost", f"{pct(tree_detect)} (in-distribution binary F1 ≈ 0.9998)"]]
    story += [table(kpis, [6.2, 10.8], align_right_from=None)]
    story += [Spacer(1, 8), H3("Headline findings")]
    story += [B(f"<b>In-distribution, the task is nearly solved.</b> All non-linear models exceed 0.99 macro-F1 on the random "
                f"split; Random Forest and XGBoost are statistically indistinguishable."),
              B("<b>XGBoost is the best accuracy-to-cost choice:</b> Random-Forest-level accuracy at "
                f"{rf_ms / xgb_ms:.0f}× lower single-flow latency and about 3× smaller model size."),
              B("<b>Class-imbalance handling did not matter much</b> on the corrected, de-duplicated data: SMOTE gave no "
                "significant gain, and class weights hurt the linear models by flooding benign traffic with alerts."),
              B("<b>Models miss what they have not seen.</b> Held-out attack families are almost never flagged, and a SQL-injection "
                "variant that the time-block split kept out of training was missed every time."),
              B(f"<b>Benchmark accuracy does not transfer.</b> On CSE-CIC-IDS2018, Random Forest and XGBoost detect only "
                f"{pct(tree_detect)} of attacks; the simplest model (Logistic Regression) transfers best. SHAP shows a "
                "testbed-specific feature (Bwd Init Win Bytes) near the top of the ranking, a likely cause.")]
    story += [PageBreak()]

    # ==== 2 overview
    story += [H1("2. Project Overview")]
    story += [H2("2.1 Problem")]
    story += [P("Given network traffic represented as bidirectional flows, decide whether each flow is benign or malicious and, "
                "if malicious, which attack family it belongs to. The detector must combine high detection with a very low "
                "false-positive rate: at millions of flows per day even 1% false positives produces thousands of useless alerts "
                "(alert fatigue).")]
    story += [H2("2.2 Research questions")]
    for q in ["<b>RQ1</b> — Which of the five classifiers gives the best accuracy versus false-positive trade-off under an identical, "
              "reproducible pipeline?",
              "<b>RQ2</b> — How does that trade off against deployability: training time, per-flow latency and model size?",
              "<b>RQ3</b> — Which flow features drive the decisions, and are they defensible from a networking standpoint?",
              "<b>RQ4</b> — Does performance hold on data from a different capture (another dataset), or does it collapse?"]:
        story += [B(q)]
    story += [H2("2.3 Hypotheses (stated in the proposal before any experiment)")]
    story += [table([["#", "Hypothesis"],
                     ["H1", "Tree ensembles (RF, XGBoost) beat Logistic Regression and match or exceed SVM / MLP on F1."],
                     ["H2", "Aggregate accuracy is > 98%, but rare-class recall (Infiltration, Heartbleed) is much lower before SMOTE."],
                     ["H3", "RF or XGBoost gives the best accuracy-to-latency trade-off; SVM has the highest training cost."],
                     ["H4", "Accuracy drops on an independent dataset, quantifying a generalisation gap."]], [1.2, 15.8])]
    story += [H2("2.4 What changed from the proposal, and why")]
    story += [P("Before writing any code, the proposal (August 2026) was reviewed. Six issues would have weakened the results; "
                "each was resolved as shown below. These changes are the main methodological contributions of the project.")]
    story += [table([["Proposal said", "Problem", "What we did"],
                     ["Original CICIDS2017 CSVs", "Documented labelling and flow-construction errors (Engelen et al. 2021; Liu et al. 2022)",
                      "Used the <b>corrected</b> CICIDS2017 re-extracted with a fixed CICFlowMeter"],
                     ["Cross-dataset on NSL-KDD or CICIDS2018", "NSL-KDD has 41 features from a different extractor; cannot be aligned",
                      "Used <b>corrected CSE-CIC-IDS2018</b> (same fixed extractor, identical features)"],
                     ["SMOTE, scaling, RFE before CV", "Fitting on data later used for validation leaks information",
                      "SMOTE inside each CV fold; feature selection on the training split only"],
                     ["Random split only", "Near-identical neighbouring flows land in both train and test",
                      "Added a <b>time-block</b> split; duplicates removed before splitting; IPs/ports dropped"],
                     ["Per-class metrics for Heartbleed / Infiltration", "11 and 36 flows: 2–7 test flows, i.e. noise",
                      "Excluded from multi-class (reported); kept in binary and LOAO tests"],
                     ["RBF-SVM on all data; paired t-test on folds", "SVM cost grows ~n²; plain t-test overstates significance",
                      "SVM capped at 60k rows; Nadeau–Bengio corrected t-test + McNemar"]], [4.2, 6.4, 6.4], font_size=8.5)]
    story += [P("Two experiments were also added: <b>leave-one-attack-out</b> (to test the abstract's 'detects unseen attacks' claim "
                "directly) and <b>detection rate at a fixed false-positive rate</b>. The Raspberry Pi stretch goal was replaced by "
                "ONNX Runtime latency measurements, and the optional Scapy live-capture demo was dropped because Scapy cannot "
                "produce CICFlowMeter features.", "body")]
    story += [PageBreak()]

    # ==== 3 team
    story += [H1("3. Team and Task Division")]
    story += [P(f"The work is split into two balanced workstreams with a clean hand-off: <b>{MEMBER_A}</b> owns everything up to a "
                f"clean, split, feature-selected dataset plus the generalisation experiments that consume new data; "
                f"<b>{MEMBER_B}</b> owns the models, their evaluation and everything that explains or presents the results. "
                "The interface between them is the cached, prepared dataset (data/processed/) and the saved models (results/…/models/).")]
    A_tasks = [
        ["#", "Task", "How it was done", "Code / output"],
        ["A1", "Proposal review & research design", "Found the 6 issues in §2.4; chose datasets, splits and tests", "README.md (decisions table)"],
        ["A2", "Environment & repository", "Git repo, Python 3.12 venv, dependency list, libomp for XGBoost, ids command-line tool, YAML config",
         "pyproject.toml, configs/, src/ids/cli.py, config.py"],
        ["A3", "Dataset acquisition", "Downloaded corrected CICIDS2017 (328 MB); wrote a streamer that pulls single days out of the 9.7 GB "
         "2018 archive with HTTP range requests, with retries", "scripts/fetch_data.py"],
        ["A4", "Loading & label mapping", "Uniform snake_case schema, float32, attack families, 'Attempted' → benign, "
         "Infiltration-Portscan → PortScan", "src/ids/datasets.py"],
        ["A5", "Cleaning", "NaN/inf removal, conflicting-label and exact de-duplication via row hashing, benign cap, rare-class handling",
         "src/ids/preprocess.py"],
        ["A6", "Train/test splitting", "Stratified random split; time-block split with StratifiedGroupKFold; fixed the bug "
         "that left DDoS out of the test set", "src/ids/split.py"],
        ["A7", "Feature selection", "Near-constant removal, |r| > 0.95 correlation pruning, RFE with Random Forest (train only)",
         "src/ids/features.py"],
        ["A8", "Generalisation experiments", "Leave-one-attack-out and cross-dataset evaluation on CSE-CIC-IDS2018",
         "src/ids/experiments.py (run_loao, run_cross_dataset)"],
        ["A9", "Testing & automation", "Synthetic-data generator, 22 end-to-end tests, overnight run script",
         "src/ids/synthetic.py, tests/, scripts/run_full.sh"],
    ]
    B_tasks = [
        ["#", "Task", "How it was done", "Code / output"],
        ["B1", "Model implementation", "Five classifiers in imblearn pipelines; log + standardise for LR/SVM/MLP; raw features for trees",
         "src/ids/models.py"],
        ["B2", "Imbalance strategies", "Class weights / balanced sample weights; SafeSMOTE (log-space neighbours, minorities → 20k rows)",
         "src/ids/sampling.py"],
        ["B3", "Hyper-parameter tuning & training", "RandomizedSearchCV (12 draws × 3 folds on 150k rows); quick run, then full runs "
         "on both splits", "src/ids/experiments.py (tune, benchmark)"],
        ["B4", "Evaluation metrics", "Macro/weighted F1, per-class tables, benign FPR, ROC/PR-AUC, recall at fixed FPR",
         "src/ids/evaluate.py"],
        ["B5", "Statistical testing", "Nadeau–Bengio corrected resampled t-test on 5-fold CV; McNemar on the test set",
         "src/ids/stats.py"],
        ["B6", "Explainability", "SHAP TreeExplainer: global ranking, beeswarm, per-flow waterfalls", "src/ids/explain.py"],
        ["B7", "Deployability", "Model size, single-thread single-flow latency, batch throughput, ONNX Runtime export",
         "src/ids/deploy.py"],
        ["B8", "Figures & notebooks", "Confusion matrices, ROC/PR, per-class heatmaps, latency plots; 8 phase notebooks",
         "src/ids/plots.py, notebooks/"],
        ["B9", "Reporting", "Assessment deck (13 slides) and this report", "docs/, scripts/build_*.py"],
    ]
    story += [H2(f"3.1 {MEMBER_A} — Data engineering, pipeline infrastructure and generalisation")]
    story += [table(A_tasks, [0.9, 3.5, 7.6, 5.0], font_size=8.5)]
    story += [H2(f"3.2 {MEMBER_B} — Modelling, evaluation, explainability and reporting")]
    story += [table(B_tasks, [0.9, 3.5, 7.6, 5.0], font_size=8.5)]
    story += [H2("3.3 Shared work")]
    story += [B("Writing the proposal and the initial presentation (August 2026)."),
              B("Reviewing results together, deciding the hypothesis verdicts and recommendations (§9, §12)."),
              B("Demo recording and the final presentation.")]
    story += [H2("3.4 Who presents what")]
    story += [table([["Part of the presentation / viva", "Presenter"],
                     ["Problem, motivation, dataset choice, preprocessing, leakage controls", MEMBER_A],
                     ["Split design, time-block split and its bug fix, feature selection", MEMBER_A],
                     ["Models, tuning, imbalance strategies, overall and per-class results, significance", MEMBER_B],
                     ["Deployability and SHAP explainability", MEMBER_B],
                     ["Leave-one-attack-out and cross-dataset generalisation", MEMBER_A],
                     ["Hypothesis verdicts, conclusions and recommendation", "Both"]], [12.5, 4.5])]
    story += [PageBreak()]

    # ==== 4 timeline
    story += [H1("4. How We Did It: Project Timeline")]
    story += [P("Chronological record from initialisation to results. Times are local (IST); durations are the measured wall-clock "
                "times on the project laptop.")]
    tl = [["When", "Phase", "What happened", "Outcome"],
          ["25 Aug 2026", "Proposal", "Proposal and 5-slide presentation submitted", "docs/IDS_Project_Proposal.docx, .pptx"],
          ["19 Oct", "Review", "Proposal and slides reviewed before building; 6 must-fix issues and several improvements identified (§2.4)",
           "Agreed design"],
          ["19 Oct", "Dataset verification", "Checked the NetFlow-v2 datasets: no CICIDS2017 version exists, so they cannot test a "
           "CICIDS2017-trained model. Switched target to corrected CSE-CIC-IDS2018; listed archive contents remotely (2017: 328 MB; "
           "2018: 9.7 GB)", "Cross-dataset plan fixed"],
          ["19 Oct", "Setup", "Git repository, Python 3.12 virtual environment, dependencies; installed libomp so XGBoost loads on macOS",
           "Working environment"],
          ["19 Oct", "Implementation", "Wrote the src/ids package (13 modules), CLI, configs, synthetic data generator, 8 notebooks; "
           "21 tests passing; fixed an inflated Random-Forest latency (thread-pool start-up per call)", "Pipeline runs end-to-end"],
          ["19 Oct", "Real data", "Downloaded corrected CICIDS2017; verified 91-column schema and all 28 label strings; remapped "
           "Infiltration-Portscan → PortScan; the prepare step ran in 95 s using 2 GB RAM", "1.71M clean flows, 30 features"],
          ["19–20 Oct", "Quick benchmark", "Small tuning budget, no CV: all 14 runs in 32 min", "First results, timings"],
          ["20 Oct 01:09", "Full run", "Random split 43 min, time-block split 62 min (cross-dataset step skipped: no 2018 data yet)",
           "All models, stats, SHAP, LOAO"],
          ["20 Oct", "Bug fix", "Time-block test set had 0 DDoS flows, and macro-F1 counted DDoS as 0. Splitter now keeps every "
           "class on both sides; macro metrics average over classes present; time-block run repeated", "Valid time-block results"],
          ["20 Oct", "2018 data", "First stream failed when the laptop slept (all 5 connections dropped at once). Added retries + "
           "caffeinate; 5 days streamed in parallel in 37 min", f"{sum(cc.values()):,} 2018 flows"],
          ["20 Oct", "Cross-dataset", "Every saved model evaluated on 2018 data, both splits (~20 min each)", "Generalisation results"],
          ["20 Oct", "Assessment deck", "13-slide deck: abstract, methodology, simulation results", "docs/IDS_Assessment_Presentation.pptx"],
          ["Oct 2026", "Report", "This document", "docs/IDS_Project_Report.pdf"]]
    story += [table(tl, [2.3, 2.6, 8.6, 3.5], font_size=8.3)]
    story += [PageBreak()]

    # ==== 5 environment
    story += [H1("5. Tools and Environment")]
    import importlib.metadata as md
    vers = {p: md.version(p) for p in ["scikit-learn", "xgboost", "imbalanced-learn", "shap", "pandas", "numpy", "scipy",
                                         "matplotlib", "onnxruntime", "skl2onnx", "pyarrow"]}
    story += [table([["Component", "Version / detail"],
                     ["Hardware", "Apple MacBook, M1 (8 cores), 8 GB RAM, macOS"],
                     ["Language", "Python 3.12 (virtual environment .venv/)"],
                     *[[k, v] for k, v in vers.items()],
                     ["Version control", "git (3 commits at the time of writing)"]], [5, 12])]
    story += [H2("5.1 Repository layout")]
    story += [code("configs/        default.yaml (real runs), quick.yaml, smoke.yaml (synthetic)\n"
                   "src/ids/        datasets, preprocess, split, features, sampling, models, evaluate,\n"
                   "                stats, deploy, explain, plots, experiments, cli, synthetic\n"
                   "scripts/        fetch_data.py, run_full.sh, build_assessment_deck.py, build_report.py\n"
                   "notebooks/      01_eda ... 08_generalization\n"
                   "tests/          end-to-end tests on synthetic data (22 tests)\n"
                   "data/raw/       downloaded datasets (not in git)\n"
                   "data/processed/ cached cleaned/split/feature-selected data (not in git)\n"
                   "results/        metrics, predictions, models, figures (not in git)\n"
                   "docs/           proposal, slides, assessment deck, this report")]
    story += [H2("5.2 Design principles")]
    story += [B("<b>One config file</b> (configs/default.yaml) holds every setting, so every model sees the same data and the run is reproducible (seed 42)."),
              B("<b>Each step caches its output</b> to disk (prepare → benchmark → stats / deploy / explain / cross / loao → report), so steps can be re-run independently."),
              B("<b>Nothing fitted on test data:</b> scaling, SMOTE and feature selection are fitted on training data only."),
              B("<b>Tested before use:</b> a synthetic dataset with the real schema and quirks lets the whole pipeline run in about 2 minutes in tests.")]
    story += [PageBreak()]

    # ==== 6 datasets
    story += [H1("6. Datasets")]
    story += [H2("6.1 Corrected CICIDS2017 (training and testing)")]
    story += [P("CICIDS2017 (Canadian Institute for Cybersecurity) captures five days of realistic background traffic with staged attacks. "
                "The original CSVs contain documented errors in flow construction and labelling. We used the corrected release "
                f"(Liu, Engelen et al., IEEE CNS 2022): the raw PCAPs re-processed with a fixed CICFlowMeter and relabelled. Each "
                f"CSV has {d['n_cols']} columns: identifiers (flow ID, IPs, ports, timestamp), about 85 flow statistics, the label and an "
                "'Attempted Category'. Flows that belong to an attack session but carry no attack payload are labelled "
                "'… - Attempted'; as the dataset authors recommend, they are treated as benign.")]
    rows = [["Day", "Label", "Flows"]]
    for day, vc in d["raw"].items():
        for i, (lab, cnt) in enumerate(vc.items()):
            rows.append([day.capitalize() if i == 0 else "", escape(lab), n_(cnt)])
    story += [table(rows, [3, 9.5, 4.5], align_right_from=2, font_size=8.5)]
    story += [P(f"Total: {n_(clean['rows_in'])} flows. Download: 328 MB zip, about 1.15 GB of CSV.", "cap")]
    story += [H2("6.2 Corrected CSE-CIC-IDS2018 (cross-dataset test only)")]
    story += [P("Same research group and same fixed CICFlowMeter, so the feature columns match exactly; a different network, "
                "different hosts and different attack tools. The full archive is 9.7 GB (3–4 GB of CSV per day), more than the "
                "laptop's free disk, so scripts/fetch_data.py reads individual day files out of the remote zip with HTTP range "
                "requests, streams them in chunks, keeps every attack flow and a random 5% of benign flows, and writes a small "
                "parquet file per day.")]
    rows = [["Day", "Attack flows kept", "Benign flows kept (5%)", "Families"]]
    for day, fam in d["days18"].items():
        att = fam.drop("Benign", errors="ignore")
        rows.append([day, n_(att.sum()), n_(fam.get("Benign", 0)), ", ".join(att.index)])
    story += [table(rows, [4.4, 3.2, 3.6, 5.8], align_right_from=None, font_size=8.5)]
    story += [P(f"After cleaning: {sum(cc.values()):,} flows ({', '.join(f'{k} {v:,}' for k, v in cc.items())}). Because benign "
                "traffic was sampled at 5%, precision on this set is optimistic; recall and false-positive rate are unaffected.", "cap")]
    story += [H2("6.3 Datasets considered and rejected")]
    story += [B("<b>NSL-KDD</b> (proposal option): 41 features from a different 1999-era extractor; no meaningful mapping to CICFlowMeter features."),
              B("<b>UQ NetFlow-v2 datasets</b>: share one feature set with each other, but contain no version of CICIDS2017, so a CICIDS2017 model cannot be tested on them."),
              B("<b>Original CICIDS2017 / CSE-CIC-IDS2018 CSVs</b>: known labelling errors and slightly different column names.")]
    story += [PageBreak()]

    # ==== 7 methodology
    story += [H1("7. Methodology")]
    story += [P("The pipeline has six stages. Every classifier receives exactly the same rows, split, features and metrics.")]
    story += [pipeline_drawing(), P("Figure 1. End-to-end pipeline.", "cap")]
    story += [H2("7.1 Label mapping")]
    story += [P("Raw labels are mapped to attack families by ordered pattern rules, so the same code works for both datasets:")]
    story += [table([["Raw labels", "Family"],
                     ["BENIGN; any '… - Attempted' label", "Benign"],
                     ["DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest", "DoS"],
                     ["DDoS", "DDoS"],
                     ["Portscan; <b>Infiltration - Portscan</b>", "PortScan"],
                     ["FTP-Patator, SSH-Patator", "BruteForce"],
                     ["Web Attack - Brute Force / XSS / SQL Injection", "WebAttack"],
                     ["Botnet", "Botnet"],
                     ["Infiltration", "Infiltration"],
                     ["Heartbleed", "Heartbleed"]], [12, 5])]
    story += [callout("<b>Why Infiltration-Portscan → PortScan:</b> these 71,767 flows are the port scan launched from the already-"
                      "compromised host. To a flow classifier they are port-scan traffic. Labelling them Infiltration would let 72k scan "
                      "flows stand in for the 36 genuine infiltration flows and inflate 'Infiltration recall' to near 100%.")]
    story += [H2("7.2 Cleaning")]
    story += [B("<b>Identifier columns dropped</b> (flow ID, source/destination IP, ports, timestamp): they identify testbed hosts, "
                "so a model using them learns 'this IP was the attacker' and cannot transfer."),
              B("<b>NaN / infinite values</b> removed (CICFlowMeter produces infinite rates when flow duration is 0)."),
              B("<b>Conflicting duplicates</b> (identical features, different labels) removed entirely: no model can classify them."),
              B("<b>Exact duplicates</b> removed <i>before</i> splitting, using a 64-bit hash per row; otherwise copies sit in both train and test."),
              B("<b>Benign capped</b> at 600,000 random flows (all attacks kept) to fit 8 GB RAM."),
              B("<b>Rare classes:</b> families with fewer than 100 flows (Infiltration 36, Heartbleed 11) are excluded from the multi-class task and reported; they stay in the binary and leave-one-attack-out tests.")]
    story += [table([["Step", "Rows"],
                     ["Loaded", n_(clean["rows_in"])],
                     ["NaN / infinite", f"−{n_(clean['nan_inf_rows_dropped'])}"],
                     ["Conflicting-label duplicates", f"−{n_(clean['conflicting_label_rows_dropped'])}"],
                     ["Exact duplicates", f"−{n_(clean['duplicate_rows_dropped'])}"],
                     ["After cleaning", n_(clean["rows_out"])],
                     ["After benign cap and rare-class exclusion", n_(rm["n_train"] + rm["n_test"])],
                     ["Train / test (random split)", f"{n_(rm['n_train'])} / {n_(rm['n_test'])}"],
                     ["Train / test (time-block split)", f"{n_(tm['n_train'])} / {n_(tm['n_test'])}"]], [10, 7], align_right_from=1)]
    story += [fig(figs["dist"], 15.5, "Figure 2. Flows per family after cleaning (log scale). Port scans fall from about 231k to 7.7k "
                                      "flows once ports are dropped, because scan probes are identical SYN→RST flows.")]
    story += [H2("7.3 Train/test splits")]
    story += [P("<b>Random split:</b> 80/20, stratified by family, seed 42 — what most published work uses.")]
    story += [P("<b>Time-block split (leakage check):</b> flows are grouped into 5-minute blocks per day, and whole blocks go to "
                "either train or test (StratifiedGroupKFold). Neighbouring, near-identical flows can then never straddle the split. "
                "Because short attacks span few blocks (DDoS: 5), the splitter tries several shuffles and keeps the split that puts "
                "every class with at least two blocks on both sides, with a test fraction close to 20%.")]
    story += [callout("<b>Bug found and fixed:</b> the first time-block run used a single fold, which happened to put all DDoS blocks "
                      "in train. The test set had 0 DDoS flows, and macro-F1 counted DDoS as a class scoring 0, cutting every model's "
                      "score by about 1/7. We (1) rewrote the splitter as described above and (2) made macro metrics average only over "
                      "classes present in the test set, reporting any absent class. The invalid results were archived, not deleted, and the "
                      "time-block run was repeated.", AMBER_SOFT)]
    story += [H2("7.4 Feature selection (training split only)")]
    story += [P(f"Fitted once on the training split and reused by every model: (1) remove near-constant features; (2) greedy "
                f"correlation pruning — walk the features and drop any with |Pearson r| > 0.95 against one already kept "
                f"(82 → {82 - len(rm['features']['correlation_dropped'])} features); (3) Recursive Feature Elimination with a Random Forest "
                f"(100 trees, depth 20, balanced class weights) on a 200k-row stratified sample, keeping {len(rm['features']['selected'])}.")]
    sel = rm["features"]["selected"]
    half = (len(sel) + 1) // 2
    rows = [["#", "Selected feature (random split)", "#", "Selected feature"]]
    for i in range(half):
        a = sel[i]
        b = sel[i + half] if i + half < len(sel) else ""
        rows.append([str(i + 1), orig.get(a, a), str(i + half + 1) if b else "", orig.get(b, b)])
    story += [table(rows, [0.8, 7.7, 0.8, 7.7], font_size=8.5)]
    cd = rm["features"]["correlation_dropped"][:8]
    story += [P("Examples of pruned features: " + "; ".join(f"{orig.get(a, a)} (r = {r:.3f} with {orig.get(b, b)})" for a, b, r in cd) + ".", "cap")]
    story += [H2("7.5 Classifiers and hyper-parameters")]
    story += [P("Each classifier is wrapped in an imbalanced-learn pipeline. Distance- and gradient-based models (LR, SVM, MLP) receive "
                "signed-log then standardised features, because flow features span about ten orders of magnitude; tree models use raw "
                "features (invariant to monotone transforms, and readable in SHAP plots). Hyper-parameters were tuned once per model "
                "with RandomizedSearchCV (12 draws, 3-fold, macro-F1, on a 150k-row stratified sample, strategy 'none') and reused for "
                "all imbalance strategies so the strategy comparison is not confounded by tuning.")]
    tune = d["r_tune"]

    def fmt_params(p):
        out = []
        for k, v in p.items():
            k = k.replace("clf__", "")
            if isinstance(v, float):
                v = f"{v:.3g}"
            out.append(f"{k} = {v}")
        return ", ".join(out)
    space = {"logreg": "C from 0.001 to 100 (log-uniform)",
             "random_forest": "trees 100–400, depth {none, 20, 30, 40}, min split 2–10, max features {sqrt, 0.3, 0.5}",
             "xgboost": "learning rate 0.03–0.3, depth 4–10, trees 150–500, subsample 0.7–1, colsample 0.6–1",
             "svm": "C ∈ [0.1, 1000], γ ∈ [10^-4, 1] (log-uniform); trained on ≤ 60k rows",
             "mlp": "layers {64, 128-64, 256-128, 128-64-32}, lr 10^-4–10^-2, α 10^-6–10^-2, {relu, tanh}"}
    rows = [["Model", "Search space", "Chosen (random split)", "Tuning CV F1"]]
    for m in MODELS:
        rows.append([NAMES[m], space[m], fmt_params(tune[m]["params"]), f3(tune[m]["cv_f1_macro"])])
    story += [table(rows, [2.8, 5.8, 6.4, 2.0], font_size=8.2)]
    story += [H2("7.6 Class-imbalance strategies")]
    story += [B("<b>None:</b> learn from the natural class mix."),
              B("<b>Class weights:</b> loss weighted by inverse class frequency (class_weight='balanced'; balanced sample weights for XGBoost; "
                "not available for scikit-learn's MLP, hence 14 rather than 15 runs per split)."),
              B("<b>SMOTE (SafeSMOTE):</b> runs inside every CV fold and only on training data; lifts each class below 20,000 rows "
                "to 20,000 (not full balance, which would create millions of synthetic rows); neighbours are found in signed-log, "
                "standardised space and synthetic rows mapped back to raw units; k shrinks automatically for tiny classes.")]
    story += [H2("7.7 Training protocol")]
    story += [B("Per run: 5-fold stratified (or time-grouped) CV on up to 300k training rows to obtain per-fold scores for the "
                "significance tests, then a final fit on the full training split and one evaluation on the held-out test split."),
              B("RBF-SVM is trained on a 60k-row proportional sample with a per-class floor (its cost grows roughly with n²); the MLP on 400k."),
              B("Models, predictions, per-class tables and confusion matrices are saved for every run.")]
    story += [H2("7.8 Evaluation metrics")]
    story += [table([["Metric", "Definition", "Why it matters"],
                     ["Macro-F1", "mean of per-class F1 (classes present in test)", "every attack family counts equally"],
                     ["Per-class recall", "TP / (TP + FN) per family", "exposes failures hidden by accuracy"],
                     ["Benign FPR", "benign flows flagged as attack / all benign", "alert fatigue"],
                     ["ROC-AUC, PR-AUC", "on attack score = 1 − P(benign)", "threshold-free separability; PR-AUC suits imbalance"],
                     ["Recall at fixed FPR", "best detection rate with FPR ≤ 0.1% or 1%", "how an IDS is actually tuned"]],
                    [3.2, 7.3, 6.5], font_size=8.5)]
    story += [H2("7.9 Significance tests")]
    story += [P("A plain paired t-test over CV folds overstates significance because the folds share training data. We used the "
                "<b>Nadeau–Bengio corrected resampled t-test</b>: t = mean(d) / sqrt((1/k + n<sub>test</sub>/n<sub>train</sub>) · var(d)), "
                "with d the per-fold macro-F1 differences and k = 5, and <b>McNemar's test</b> on the two models' errors over the "
                "shared test set (exact binomial when fewer than 25 disagreements).")]
    story += [H2("7.10 Explainability, deployability and generalisation tests")]
    story += [B("<b>SHAP:</b> TreeExplainer on the best tree model; mean |SHAP| per feature and class on 5,000 stratified test flows, a "
                "beeswarm plot for the attack direction, and waterfall plots for individual flagged flows."),
              B("<b>Deployability:</b> compressed model size; single-flow latency measured one flow at a time with one thread (median, "
                "p95, p99 over 1,000 flows); batch throughput on 10,000 flows; ONNX Runtime latency for Random Forest and XGBoost."),
              B("<b>Leave-one-attack-out:</b> for each family, a binary XGBoost is trained without that family and tested on all of its "
                "flows; also reported: detection at the threshold that gives 1% FPR, and ROC-AUC of unseen-attack vs benign scores."),
              B("<b>Cross-dataset:</b> every saved model is applied, unchanged, to corrected CSE-CIC-IDS2018 and scored as benign vs attack.")]
    story += [PageBreak()]

    # ==== 8 results
    story += [H1("8. Results")]
    story += [H2("8.1 Random split — all runs")]
    rows = [["Model", "Strategy", "Macro-F1", "CV macro-F1 (mean ± sd)", "Attack recall", "Benign FPR", "PR-AUC", "Train (s)"]]
    for _, r in d["r_sum"].iterrows():
        rows.append([NAMES[r.model], STRAT[r.strategy], f3(r.f1_macro), r["cv_f1_macro_mean±sd"], f"{r.bin_recall:.4f}",
                     pct(r.bin_fpr, 3), f"{r.bin_pr_auc:.4f}", f"{r.train_seconds:.0f}"])
    story += [table(rows, [2.8, 2.1, 1.6, 3.3, 1.8, 1.8, 1.6, 2.0], align_right_from=2, font_size=8.2)]
    story += [P(f"Test set: {n_(rm['n_test'])} flows. Sorted by macro-F1.", "cap")]
    story += [fig(RAND / "roc_pr.png", 16.5, "Figure 3. ROC (log-scale FPR) and precision-recall curves for benign vs attack, random split.")]
    story += [H2("8.2 Time-block split — all runs")]
    rows = [["Model", "Strategy", "Macro-F1", "CV macro-F1 (mean ± sd)", "Attack recall", "Benign FPR", "PR-AUC", "Train (s)"]]
    for _, r in d["t_sum"].iterrows():
        rows.append([NAMES[r.model], STRAT[r.strategy], f3(r.f1_macro), r["cv_f1_macro_mean±sd"], f"{r.bin_recall:.4f}",
                     pct(r.bin_fpr, 3), f"{r.bin_pr_auc:.4f}", f"{r.train_seconds:.0f}"])
    story += [table(rows, [2.8, 2.1, 1.6, 3.3, 1.8, 1.8, 1.6, 2.0], align_right_from=2, font_size=8.2)]
    story += [P(f"Test set: {n_(tm['n_test'])} flows; all seven classes present.", "cap")]
    story += [fig(figs["splits"], 15.5, "Figure 4. Best macro-F1 per model on the two splits.")]
    story += [P("Benign-vs-attack detection barely changes under the time-block split (attack recall ≥ 0.996, FPR ≤ 0.05% for the "
                "tree models), so leakage between neighbouring flows does not inflate binary detection much on the corrected, "
                "de-duplicated data. The multi-class drop from about 0.99 to 0.94 comes almost entirely from one class, shown next.")]
    story += [H2("8.3 Per-class recall")]
    classes = list(d["t_pc"]["xgboost__smote"].index)
    rows = [["Model (best strategy)"] + classes]
    fills = []
    for i, m in enumerate(MODELS):
        run = f"{m}__{tbb.loc[m, 'strategy']}"
        rec = d["t_pc"][run]["recall"]
        row = [f"{NAMES[m]} ({STRAT[tbb.loc[m, 'strategy']]})"]
        for j, c_ in enumerate(classes):
            v = rec[c_]
            row.append(f3(v))
            fills.append(((j + 1, i + 1), ICE if v >= 0.99 else AMBER_SOFT if v >= 0.9 else AMBER if v >= 0.7 else "#F4B4AC"))
        rows.append(row)
    sup = d["t_pc"]["xgboost__smote"]["support"]
    rows.append(["Test flows"] + [n_(sup[c_]) for c_ in classes])
    story += [table(rows, [4.4] + [12.6 / len(classes)] * len(classes), zebra=False, fills=fills, font_size=8.2)]
    story += [P("Time-block split. Cell colours: ≥ 0.99 blue, 0.90–0.99 light amber, 0.70–0.90 amber, < 0.70 red.", "cap")]
    if sql_tr is not None:
        story += [callout(f"<b>An unseen variant is always missed.</b> The time-block split placed {sql_te} SQL-injection flows in test "
                          f"and only {sql_tr} in train. Every model missed all of them (classified as benign or DoS), while web brute "
                          "force — well represented in training — was caught 10/10. This is the same failure mode as the "
                          "leave-one-attack-out experiment, on a small scale.", AMBER_SOFT)]
    story += [fig(RAND / "per_class_recall.png", 14, "Figure 5. Per-class recall for every run, random split. Rare-class recall is "
                                                       "already high without SMOTE.")]
    story += [H2("8.4 Effect of the imbalance strategies")]
    rows = [["Model", "none", "class weights", "SMOTE", "Benign FPR (none → class weights)"]]
    for m in MODELS:
        g = lambda s_: f3(rs.loc[(m, s_), "f1_macro"]) if (m, s_) in rs.index else "—"  # noqa: E731
        fpr_ = (f"{pct(rs.loc[(m, 'none'), 'bin_fpr'], 3)} → {pct(rs.loc[(m, 'class_weight'), 'bin_fpr'], 3)}"
                if (m, "class_weight") in rs.index else "—")
        rows.append([NAMES[m], g("none"), g("class_weight"), g("smote"), fpr_])
    story += [table(rows, [3.6, 2.2, 2.6, 2.2, 6.4], align_right_from=1, font_size=8.5)]
    story += [P("Random split, macro-F1. Class weights push the decision boundary towards the rare classes; for Logistic Regression "
                "this multiplies benign false positives about 20-fold. SMOTE gives no statistically significant gain for any model.", "cap")]
    story += [H2("8.5 Statistical significance")]
    rows = [["Pair (strategy 'none')", "Δ macro-F1 (test)", "Corrected t-test p (random)", "Corrected t-test p (time-block)",
             "McNemar p (random)"]]
    for a, b in [("random_forest", "xgboost"), ("random_forest", "logreg"), ("xgboost", "logreg"), ("random_forest", "mlp"),
                 ("xgboost", "svm"), ("mlp", "svm")]:
        delta = rs.loc[(a, "none"), "f1_macro"] - rs.loc[(b, "none"), "f1_macro"]
        rows.append([f"{NAMES[a]} vs {NAMES[b]}", f"{delta:+.4f}", f"{pval(d['r_sig'], a, b, 'none'):.3f}",
                     f"{pval(d['t_sig'], a, b, 'none'):.3f}", f"{pval(d['r_sig'], a, b, 'none', 'mcnemar_p_value'):.3g}"])
    story += [table(rows, [5.2, 2.6, 3.0, 3.2, 3.0], align_right_from=1, font_size=8.5)]
    story += [P(f"Random Forest and XGBoost are statistically tied (p = {p_rx_r:.2f} and {p_rx_t:.2f}). Ensembles beat Logistic "
                f"Regression on the random split (p = {p_rl_r:.3f}), but no pairwise difference is significant on the time-block split "
                "at α = 0.05. McNemar's test flags tiny differences as significant because the test set is large (≈ 176k flows); "
                "the corrected t-test is the more conservative and more appropriate test here.")]
    story += [PageBreak()]
    story += [H2("8.6 Deployability")]
    rows = [["Model", "Size", "Train (s)", "Latency median (ms)", "p99 (ms)", "Batch (flows/s)", "ONNX median (ms)"]]
    for m in MODELS:
        r = dep.loc[(m, "none")]
        size = f"{r.size_mb * 1000:.0f} KB" if r.size_mb < 1 else f"{r.size_mb:.1f} MB"
        onnx = f"{r.onnx_latency_median_ms:.3f}" if pd.notna(r.get("onnx_latency_median_ms")) else "—"
        rows.append([NAMES[m], size, f"{r.train_seconds:.0f}", f"{r.latency_median_ms:.3f}", f"{r.latency_p99_ms:.2f}",
                     f"{r.batch_flows_per_s:,.0f}", onnx])
    story += [table(rows, [3.4, 1.9, 1.7, 2.8, 1.8, 2.7, 2.7], align_right_from=1, font_size=8.5)]
    story += [P("Random split, strategy 'none'; single-flow latency measured single-threaded on the M1. SVM training time is small only "
                "because it was capped at 60k rows.", "cap")]
    story += [fig(RAND / "latency_vs_f1.png", 14, "Figure 6. Accuracy versus single-flow latency (bubble size = model size).")]
    story += [P(f"XGBoost reaches Random-Forest-level accuracy at {rf_ms / xgb_ms:.0f}× lower latency; exported to ONNX Runtime, both tree "
                "models classify a flow in about 0.01–0.02 ms, fast enough for inline use. The RBF-SVM is quick per flow but the slowest "
                "in batch, because every prediction touches many support vectors. These numbers are classification cost only: a flow "
                "can be classified only once CICFlowMeter closes it (FIN/RST or a 120-second timeout), which dominates real detection delay.")]
    story += [H2("8.7 Explainability (SHAP)")]
    shap_model = NAMES[d["r_shaprun"].split("__")[0]]
    top = d["r_shap"]["overall"].head(10)
    rows = [["Rank", "Feature", "Mean |SHAP|"]] + [[str(i + 1), orig.get(f, f), f"{v:.3f}"] for i, (f, v) in enumerate(top.items())]
    story += [table(rows, [1.5, 9, 3], align_right_from=2, fills=[((0, i + 1), AMBER_SOFT) for i, f in enumerate(top.index)
                                                                  if "init_win" in f])]
    story += [P(f"{shap_model} ({STRAT[d['r_shaprun'].split('__')[1]]}), random split, averaged over all classes.", "cap")]
    story += [fig(RAND / "shap" / "shap_beeswarm_attack.png", 12.5, "Figure 7. SHAP beeswarm: features pushing a flow towards "
                                                                     "'attack'. Red = high feature value.")]
    rank = list(d["r_shap"].index).index("bwd_init_win_bytes") + 1
    story += [P(f"Packet-length statistics dominate, which is defensible: floods, scans and web attacks have characteristic request "
                f"and response sizes. But <b>Bwd Init Win Bytes ranks #{rank}</b>. The initial TCP window size reflects the operating "
                "system and configuration of the hosts in the testbed, not the attack; a model that leans on it partly learns the lab. "
                "This is consistent with the cross-dataset collapse in §8.9.")]
    story += [H2("8.8 Leave-one-attack-out (unseen attack families)")]
    rows = [["Held-out family", "Flows", "Detected (0.5)", "Detected (1% FPR)", "ROC-AUC vs benign", "Detected (0.5), time-block"]]
    for f_ in lo.index:
        r = lo.loc[f_]
        rows.append([f_, n_(r.n_unseen), pct(r.unseen_detection_rate), pct(r["unseen_recall_at_fpr_0.01"]),
                     f3(r.unseen_vs_benign_roc_auc), pct(lo_t.loc[f_, "unseen_detection_rate"]) if f_ in lo_t.index else "—"])
    story += [table(rows, [3.2, 2.0, 2.4, 2.8, 3.0, 3.6], align_right_from=1, font_size=8.5)]
    story += [fig(figs["loao"], 15.5, "Figure 8. Share of each never-seen family detected by a binary XGBoost (random split).")]
    story += [P("At the default threshold the detector almost never flags a family it has not seen — the exception is DDoS, which "
                "resembles DoS. The attack scores still rank unseen flows above benign (ROC-AUC mostly 0.92–1.00), so re-tuning the "
                "threshold recovers DoS, Botnet, Heartbleed and Infiltration; port scans, brute force and web attacks stay below "
                "about 15% even at 1% FPR. The 'detects zero-day attacks' claim does not hold as deployed.")]
    story += [H2("8.9 Cross-dataset generalisation (CSE-CIC-IDS2018, no retraining)")]
    rows = [["Model", "In-dist. binary F1", "2018 F1", "2018 recall", "2018 FPR", "2018 ROC-AUC", "Recall @ 1% FPR"]]
    for m in MODELS:
        r = cr.loc[(m, "none")]
        rows.append([NAMES[m], f"{r.in_dist_bin_f1:.4f}", f3(r.target_f1), pct(r.target_recall), pct(r.target_fpr, 2),
                     f3(r.target_roc_auc), pct(r["target_recall_at_fpr_0.01"])])
    story += [table(rows, [3.4, 2.6, 1.8, 2.1, 1.9, 2.5, 2.7], align_right_from=1, font_size=8.5)]
    story += [P("Random-split models, strategy 'none'. Results for the other strategies and for the time-block models are in "
                "results/…/cross_dataset__cicids2018_fixed.csv and show the same pattern.", "cap")]
    story += [fig(figs["cross"], 15.5, "Figure 9. Binary F1 in-distribution vs on CSE-CIC-IDS2018.")]
    fam = d["r_crossfam"]
    piv = fam[fam.strategy == "none"].pivot_table(index="model", columns="family", values="flagged_as_attack")
    cols_ = [c_ for c_ in ["Benign", "Botnet", "BruteForce", "DDoS", "DoS"] if c_ in piv.columns]
    rows = [["Model"] + [f"{c_} flagged" for c_ in cols_]] + [[NAMES[m]] + [pct(piv.loc[m, c_]) for c_ in cols_] for m in MODELS]
    story += [table(rows, [3.6] + [13.4 / len(cols_)] * len(cols_), align_right_from=1, font_size=8.5)]
    story += [P("Share of each 2018 family flagged as attack ('Benign flagged' is the false-positive rate).", "cap")]
    story += [P(f"The two strongest in-distribution models detect only {pct(tree_detect)} of 2018 attacks. Logistic Regression "
                f"transfers best (F1 {cr.loc[('logreg', 'none'), 'target_f1']:.2f}, ROC-AUC {cr.loc[('logreg', 'none'), 'target_roc_auc']:.2f}), "
                "and it is the only model that still catches most Botnet and DDoS flows. Even with the threshold re-tuned for 1% FPR, "
                f"no model exceeds {max(cr['target_recall_at_fpr_0.01']):.0%} recall, so this is genuine distribution shift rather "
                "than a threshold problem. The more flexible the model, the more it fits capture-specific details of CICIDS2017.")]
    story += [PageBreak()]

    # ==== 9 hypotheses
    story += [H1("9. Hypotheses: Verdicts")]
    story += [table([["#", "Hypothesis", "Verdict", "Evidence"],
                     ["H1", "Ensembles beat LR and match or exceed SVM / MLP", "Partly supported",
                      f"RF / XGBoost rank first; beat LR on the random split (p = {p_rl_r:.3f}); gaps to MLP / SVM not significant"],
                     ["H2", "Rare-class recall low before SMOTE", "Not supported",
                      "Rare-class recall already high without SMOTE; SMOTE gives no significant gain"],
                     ["H3", "RF / XGBoost best accuracy-latency; SVM highest training cost", "Partly supported",
                      f"XGBoost: RF-level F1 at {rf_ms / xgb_ms:.0f}× lower latency; SVM training cost untested (capped at 60k rows), "
                      "but it has the lowest batch throughput"],
                     ["H4", "Accuracy drops on an independent dataset", "Strongly supported",
                      f"RF / XGBoost: binary F1 from 0.9998 to ≤ 0.01; {pct(tree_detect)} of 2018 attacks detected"]],
                    [1, 4.6, 2.8, 8.6], font_size=8.8,
                    fills=[((2, 1), AMBER_SOFT), ((2, 2), "#F4B4AC"), ((2, 3), AMBER_SOFT), ((2, 4), ICE)])]
    story += [H2("Answers to the research questions")]
    story += [B("<b>RQ1:</b> Random Forest and XGBoost give the best accuracy / FPR trade-off (macro-F1 ≈ 0.99–1.00, benign FPR "
                "≤ 0.01%), statistically tied with each other."),
              B("<b>RQ2:</b> XGBoost is the most suitable for near-real-time use; with ONNX Runtime, tree models classify a flow in about 0.01–0.02 ms."),
              B("<b>RQ3:</b> Packet-length statistics dominate and are defensible; the high rank of Bwd Init Win Bytes is a testbed artefact."),
              B("<b>RQ4:</b> Performance collapses on a different network: the benchmark measures fit to the capture, not transferable attack detection.")]

    # ==== 10 discussion
    story += [H1("10. Discussion and Limitations")]
    story += [B("<b>Why are in-distribution scores so high?</b> After correcting labels and removing duplicates, most CICIDS2017 attacks "
                "are produced by a handful of tools against a few hosts, so their flows are extremely regular. High accuracy here "
                "says little about real-world performance — which is exactly what the generalisation tests show."),
              B("<b>Single seed:</b> every result comes from seed 42; CV standard deviations (§8.1–8.2) indicate the variance."),
              B("<b>Small classes:</b> WebAttack (≈ 104 flows), Heartbleed (11) and Infiltration (36) results are indicative only."),
              B("<b>2018 subset:</b> 5 of 10 days, benign sampled at 5%; target precision is optimistic, recall and FPR are not affected."),
              B("<b>SVM capped</b> at 60k training rows, so its training-cost hypothesis is not fully tested."),
              B("<b>Latency is classification cost only</b>, on a laptop; end-to-end delay is dominated by flow aggregation."),
              B("<b>Tuning budget:</b> 12 random draws per model; a larger search could change rankings slightly but not the generalisation findings.")]

    # ==== 11 problems
    story += [H1("11. Problems Encountered and How We Solved Them")]
    story += [table([["Problem", "Cause", "Fix"],
                     ["Proposal's cross-dataset plan unworkable", "NSL-KDD features cannot be aligned; NetFlow-v2 has no CICIDS2017",
                      "Corrected CSE-CIC-IDS2018 with identical features"],
                     ["2018 archive too large (9.7 GB) for the disk", "~10 GB free", "Stream single days via HTTP range requests; keep attacks + 5% benign"],
                     ["XGBoost failed to import", "OpenMP runtime missing on macOS", "brew install libomp"],
                     ["Random Forest latency 13.6 ms per flow", "n_jobs=-1 starts a thread pool on every one-row call",
                      "Measure single-flow latency single-threaded"],
                     ["72k flows labelled 'Infiltration - Portscan'", "Port scan run from the compromised host",
                      "Mapped to PortScan (behaviour-based families)"],
                     ["231k port-scan flows → 7.7k", "Scan probes are identical once ports are dropped", "Expected; reported as a finding"],
                     ["Time-block test set had 0 DDoS flows", "Single group fold; DDoS spans only 5 blocks",
                      "Splitter searches shuffles for full class coverage; macro metrics over present classes; reran"],
                     ["All 2018 downloads died at once", "Laptop slept after the first run; sockets dropped",
                      "Retries with backoff and timeouts; run under caffeinate"],
                     ["The full run aborted on a missing dataset", "One failing step stopped the chain", "Steps log errors and continue"],
                     ["Leave-one-attack-out reused an old cache", "Binary time-block cache built by the buggy splitter",
                      "Cache deleted and step rerun"]],
                    [4.8, 5.6, 6.6], font_size=8.4)]
    story += [PageBreak()]

    # ==== 12 conclusion
    story += [H1("12. Conclusions and Future Work")]
    story += [P("We built a reproducible, leakage-safe benchmark of five classifiers for flow-based intrusion detection on corrected "
                "CICIDS2017 and went beyond accuracy to significance, deployability, explainability and generalisation. The "
                "answer to 'which model should a network deploy?' has two parts:")]
    story += [B(f"<b>For attacks like those seen in training:</b> XGBoost — macro-F1 ≈ {f3(rb.loc['xgboost', 'f1_macro'])}, benign FPR "
                f"≈ {pct(rs.loc[('xgboost', 'none'), 'bin_fpr'], 3)}, about {dep.loc[('xgboost', 'none'), 'onnx_latency_median_ms']:.2f} ms per flow with ONNX."),
              B("<b>For a new network or new attacks:</b> none of these models can be trusted out of the box. Near-perfect benchmark "
                "scores measure how well a model learns the lab, not how well it protects a network.")]
    story += [H2("Recommendations")]
    story += [B("Retrain or adapt the detector on traffic from the network where it will be deployed."),
              B("Remove host- and testbed-specific features (such as initial TCP window sizes) and re-test generalisation."),
              B("Choose alert thresholds on validation data for a target false-positive rate instead of using 0.5."),
              B("Report per-class, time-separated and cross-dataset results, not only random-split accuracy.")]
    story += [H2("Future work")]
    story += [B("Domain adaptation or training on several datasets at once; federated training across organisations."),
              B("Feature sets that exclude host-specific fields; repeat the study with multiple seeds."),
              B("A live demo that replays flows through the saved model and shows alerts with SHAP reasons; SDN integration.")]

    # ==== appendices
    story += [PageBreak(), H1("Appendix A. Reproducing the Results")]
    story += [code("brew install python@3.12 libomp\n"
                   "python3.12 -m venv .venv && source .venv/bin/activate\n"
                   "pip install -e \".[notebooks,onnx,dev]\"\n\n"
                   "# smoke test on synthetic data (~2 min)\n"
                   "python -m ids.synthetic && ids all --config configs/smoke.yaml && pytest\n\n"
                   "# data\n"
                   "python scripts/fetch_data.py cicids2017\n"
                   "caffeinate -i python scripts/fetch_data.py cicids2018\n\n"
                   "# full experiments (~1 h per split on an M1)\n"
                   "ids all                      # random split\n"
                   "ids all --split time_block   # time-block split\n\n"
                   "# documents\n"
                   "python scripts/build_assessment_deck.py\n"
                   "python scripts/build_report.py")]
    story += [H2("Pipeline steps")]
    story += [table([["Command", "What it does", "Main outputs"],
                     ["ids prepare", "load, clean, split, select features", "data/processed/…/train|test.parquet, meta.json"],
                     ["ids benchmark", "tune, 5-fold CV, final fit, test evaluation for every run", "runs/*.json, *_per_class.csv, *_preds.parquet, models/"],
                     ["ids stats", "corrected t-test + McNemar for every model pair", "significance.csv"],
                     ["ids deploy", "size, latency, throughput, ONNX", "deployability.csv, latency_vs_f1.png"],
                     ["ids explain", "SHAP on the best tree model", "shap/"],
                     ["ids cross", "evaluate all models on CSE-CIC-IDS2018", "cross_dataset__cicids2018_fixed*.csv"],
                     ["ids loao", "leave-one-attack-out", "leave_one_attack_out.csv"],
                     ["ids report", "summary tables and figures", "summary.csv/.md, roc_pr.png, per_class_*.png"]],
                    [3, 6.5, 7.5], font_size=8.5)]
    story += [H1("Appendix B. Configuration (configs/default.yaml)")]
    story += [code((ROOT / "configs" / "default.yaml").read_text())]
    story += [H1("Appendix C. References")]
    for ref in ["Sharafaldin, Lashkari, Ghorbani (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic "
                "Characterization. ICISSP.",
                "Engelen, Rimmer, Joosen (2021). Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study. IEEE S&amp;P Workshops.",
                "Liu, Engelen, Lynar, Essam, Joosen (2022). Error Prevalence in NIDS Datasets: A Case Study on CIC-IDS-2017 and "
                "CSE-CIC-IDS-2018. IEEE CNS.",
                "Sarhan, Layeghy, Portmann (2021). Towards a Standard Feature Set for Network Intrusion Detection System Datasets.",
                "Nadeau, Bengio (2003). Inference for the Generalization Error. Machine Learning 52.",
                "Dietterich (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Computation.",
                "Chawla, Bowyer, Hall, Kegelmeyer (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR 16.",
                "Breiman (2001). Random Forests. Machine Learning 45.",
                "Chen, Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD.",
                "Lundberg, Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS."]:
        story += [B(ref)]

    OUT.parent.mkdir(exist_ok=True)
    doc = Doc(str(OUT))
    doc.multiBuild(story)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
