"""Figure list, legends, and assembly of the figure files into PowerPoint and PDF.

Notebooks 01-03 save every figure to results/figures as PDF, PNG and SVG.
Notebook 04 calls :func:`build_pptx` and :func:`build_pdf` to put them together:

* one slide per figure, at its real size (font sizes in the slide = font sizes in the paper);
* the picture is the SVG (vector) with the PNG as fallback. In PowerPoint 365 / 2019+,
  right-click > "Convert to Shape" turns it into editable lines and text boxes;
* the legend is in the speaker notes of each slide;
* figures drawn by hand (Fig. 2 flowchart, Fig. S7 DAG) are taken from figures_manual/
  when present (svg, png or pdf->png), otherwise a placeholder slide is inserted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FigureSpec:
    number: str          # "1", "S3a", ...
    stem: str            # file name without extension
    message: str         # one-sentence take-home message (first sentence of the legend)
    legend: str          # rest of the legend
    source: str          # notebook that makes it, or "manual"


MAIN = [
    FigureSpec("1", "Fig1_present_day_variability",
               "Present-day C. jejuni strains differ in heat resistance, and replicate experiments "
               "of the same strain differ less than strains do.",
               "(a) Survival at 55 °C of the least (IFTK 83) and the most (ATCC 33250) heat-resistant "
               "strain. Points, plate counts; thin lines, fit to each replicate experiment; thick lines, "
               "strain. δ is the time to a 6-log "
               "reduction divided by 6 (equal to the D-value when p = 1). (b) Posterior means of the 50 "
               "strains (gray) on the δ–p plane; the line encloses 90 % of present-day strains "
               "(inter-strain distribution). For three strains, open circles are the three replicate "
               "experiments, connected to the strain mean. p > 1, shouldered curve; p < 1, tailing curve.",
               "01"),
    FigureSpec("2", "Fig2_one_cycle",
               "One heating–regrowth cycle of the simulation.",
               "Regrowth to N_cell cells; each cell's (δ, p) may change slightly (intra-strain "
               "divergence); changes that move far outside the range of present-day strains are "
               "accepted less often (Metropolis–Hastings step); heating decides which cells survive; "
               "surviving cells start the next cycle.",
               "manual"),
    FigureSpec("3", "Fig3_example_lineage",
               "Followed cycle by cycle, a lineage drifts toward greater heat resistance.",
               "One lineage started from ATCC 33560 (black dot) for each heating intensity, from light "
               "(cycle 0) to dark (cycle 100); small dots, cycles 10 and 50; triangle, cycle 100. The "
               "lineage shown is the surviving lineage whose cycle-100 parameters are closest to the "
               "median of all surviving lineages. Gray dots, the 50 measured strains. Heating times "
               "were 2δ, 4δ and 6δ of the mean strain (2-, 4- and 6-log reduction of the mean strain).",
               "02"),
    FigureSpec("4", "Fig4_where_lineages_end",
               "The stronger the heating, the further surviving lineages move, and the fewer survive.",
               "Surviving lineages after 100 cycles (colored regions enclose 50 % and 90 %), started "
               "from present-day strains (10,000 lineages per condition). Gray line, 90 % of present-day "
               "strains; gray dots, the 50 measured strains; arrow, shift of the mean. The share of "
               "lineages that survived all 100 cycles is given in each panel.",
               "02"),
    FigureSpec("5", "Fig5_survival_curves",
               "After repeated heating, the same heating at 55 °C kills the lineages less.",
               "Survival curves of present-day strains (gray) and of the lineages that survived 100 "
               "cycles (color); lines, median; bands, 90 % range. Dotted line, the heating time used in "
               "every cycle; dots, the log reduction it gives. Panel titles give that reduction before "
               "and after 100 cycles (median).",
               "02"),
    FigureSpec("6", "Fig6_treatment_effect",
               "A treatment loses most of its effect within about 20 cycles, while many lineages are lost.",
               "(a) log10 reduction that the cycling treatment achieves on the lineages still alive "
               "(solid, median and interquartile range, start from present-day strains; dashed, median, "
               "start from ATCC 33560). (b) Share of lineages not yet lost.",
               "02"),
]

SUPPLEMENTARY = [
    FigureSpec("S1", "FigS1_algorithm_validation",
               "Without heating, the simulation returns to and stays in the present-day strain "
               "distribution (check of the algorithm).",
               "(a, b) Geometric mean δ and p of lineages over cycles without heating, started from "
               "ATCC 33560 or from present-day strains; dashed, target. (c) Squared Mahalanobis "
               "distances of the end points from the present-day distribution against the χ² "
               "distribution with 2 degrees of freedom expected if they follow it.",
               "03"),
    FigureSpec("S2", "FigS2_means_over_cycles",
               "Mean δ and p of the surviving lineages over the cycles.",
               "Geometric means over surviving lineages; solid, start from present-day strains; "
               "dashed, start from ATCC 33560.",
               "02"),
    FigureSpec("S3a", "FigS3a_intra_scaling",
               "Where lineages end depends little on the within-strain SD.",
               "The intra-strain SD was multiplied by the factor on the x axis (×1, main analysis, "
               "shaded). (a) δ reached after 100 cycles (geometric mean of the last 20 cycles); dashed, "
               "start. (b) Share of lineages not lost. Points are omitted when fewer than 30 lineages "
               "survived.",
               "03"),
    FigureSpec("S3b", "FigS3b_inter_scaling",
               "Where lineages end depends on the between-strain SD.",
               "As Fig. S3a, for the inter-strain SD.",
               "03"),
    FigureSpec("S4", "FigS4_carryover",
               "Effect of the number of cells carried over to the next cycle, n.",
               "As Fig. S3a, for n (n = 1, main analysis).",
               "03"),
    FigureSpec("S5", "FigS5_Ncell_sensitivity",
               "Where lineages end does not depend on the population size after regrowth; only the "
               "share of surviving lineages does.",
               "As Fig. S3a, for N_cell (10^4, main analysis); 'random', log10 N_cell drawn from "
               "Normal(4, 1) in every cycle.",
               "03"),
    FigureSpec("S6", "FigS6_QMRA",
               "Illustration: repeated heating raises the probability of infection after the same "
               "treatment.",
               "Probability of infection per portion (2 log CFU before a treatment of 4δ, Beta-Poisson "
               "dose–response) for present-day strains and for lineages after 100 cycles of 6-log "
               "heating; dashed lines, means.",
               "03"),
    FigureSpec("S7", "FigS7_cycle_formal",
               "The cycle of Fig. 2, written formally.",
               "The Metropolis–Hastings step and the stochastic thermal death "
               "simulation of one heating–regrowth cycle, with the notation of "
               "Text S2 (former Fig. 2b).",
               "manual"),
    FigureSpec("S8a", "FigS8a_where_lineages_end_ATCC33560",
               "Fig. 4 for lineages started from ATCC 33560.",
               "As Fig. 4.",
               "02"),
    FigureSpec("S8b", "FigS8b_survival_curves_ATCC33560",
               "Fig. 5 for lineages started from ATCC 33560.",
               "As Fig. 5; gray, ATCC 33560.",
               "02"),
]

ALL = MAIN + SUPPLEMENTARY
SVG_EXT_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
NS_ASVG = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"


def legend_text(f: FigureSpec) -> str:
    label = f"Fig. {f.number}" if f.number.startswith("S") else f"Figure {f.number}"
    return f"{label}. {f.message} {f.legend}"


def find_file(f: FigureSpec, fig_dir: Path, manual_dir: Path | None = None) -> dict:
    """Paths of the svg / png / pdf of a figure that exist (manual folder first for manual figures)."""
    dirs = [d for d in ([manual_dir] if f.source == "manual" else []) + [fig_dir] if d]
    out = {}
    for d in dirs:
        for ext in ("svg", "png", "pdf"):
            p = Path(d) / f"{f.stem}.{ext}"
            if p.exists() and ext not in out:
                out[ext] = p
    return out


def _png_from_pdf(pdf: Path, out_png: Path, dpi: int = 300) -> Path | None:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    doc = fitz.open(pdf)
    doc[0].get_pixmap(dpi=dpi).save(out_png)
    return out_png


def _attach_svg(slide, picture, svg_path: Path):
    """Make the picture an SVG picture (PNG kept as fallback), as PowerPoint does when an SVG
    is inserted."""
    from lxml import etree
    from pptx.opc.constants import RELATIONSHIP_TYPE as RT
    from pptx.opc.package import Part
    from pptx.oxml.ns import qn
    package = slide.part.package
    partname = package.next_partname("/ppt/media/svg%d.svg")
    part = Part.load(partname, "image/svg+xml", package, Path(svg_path).read_bytes())
    rid = slide.part.relate_to(part, RT.IMAGE)
    blip = picture._element.xpath(".//a:blip")[0]
    ext_lst = blip.find(qn("a:extLst"))
    if ext_lst is None:
        ext_lst = etree.SubElement(blip, qn("a:extLst"))
    ext = etree.SubElement(ext_lst, qn("a:ext"))
    ext.set("uri", SVG_EXT_URI)
    svg_blip = etree.SubElement(ext, f"{{{NS_ASVG}}}svgBlip", nsmap={"asvg": NS_ASVG})
    svg_blip.set(qn("r:embed"), rid)


def build_pptx(figs, fig_dir, out_path, manual_dir=None, dpi: int = 300, label_slides: bool = True,
               slide_w_in: float = 13.333, slide_h_in: float = 7.5):
    """One slide per figure. Returns a table of what was placed."""
    from PIL import Image
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt
    fig_dir, out_path = Path(fig_dir), Path(out_path)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(slide_w_in), Inches(slide_h_in)
    blank = prs.slide_layouts[6]
    report = []
    for f in figs:
        slide = prs.slides.add_slide(blank)
        files = find_file(f, fig_dir, manual_dir)
        png = files.get("png")
        if png is None and "pdf" in files:
            png = _png_from_pdf(files["pdf"], out_path.parent / f"_{f.stem}.png", dpi)
        top = 0.55 if label_slides else 0.3
        if label_slides:
            tb = slide.shapes.add_textbox(Inches(0.3), Inches(0.12), Inches(slide_w_in - 0.6), Inches(0.4))
            r = tb.text_frame.paragraphs[0].add_run()
            r.text = (f"Fig. {f.number}" if f.number.startswith("S") else f"Figure {f.number}") + f" — {f.message}"
            r.font.size, r.font.color.rgb = Pt(12), RGBColor(0x52, 0x51, 0x4E)
        if png is not None:
            with Image.open(png) as im:
                w_px, h_px = im.size
                dpi_png = im.info.get("dpi", (dpi, dpi))[0] or dpi
            w_in, h_in = w_px / dpi_png, h_px / dpi_png
            max_w, max_h = slide_w_in - 0.6, slide_h_in - top - 0.3
            scale = min(1.0, max_w / w_in, max_h / h_in)        # real size unless too large
            w_in, h_in = w_in * scale, h_in * scale
            pic = slide.shapes.add_picture(str(png), Inches((slide_w_in - w_in) / 2), Inches(top),
                                           Inches(w_in), Inches(h_in))
            if "svg" in files:
                _attach_svg(slide, pic, files["svg"])
            status = ("SVG + PNG" if "svg" in files else "PNG") + ("" if scale == 1 else f" (scaled ×{scale:.2f})")
        else:
            tb = slide.shapes.add_textbox(Inches(1), Inches(3), Inches(slide_w_in - 2), Inches(1))
            tb.text_frame.word_wrap = True
            tb.text_frame.text = (f"Insert {f.stem} here (drawn by hand): put {f.stem}.svg or .png in "
                                  f"the folder figures_manual/ and rerun notebook 04.")
            status = "placeholder"
        slide.notes_slide.notes_text_frame.text = legend_text(f)
        report.append(dict(figure=f.number, file=f.stem, placed=status, made_by=f"notebook {f.source}"
                           if f.source != "manual" else "by hand"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    return report


def build_pdf(figs, fig_dir, out_path, manual_dir=None):
    """All figure PDFs in one file (one figure per page). Requires pypdf."""
    from pypdf import PdfWriter
    w = PdfWriter()
    missing = []
    for f in figs:
        files = find_file(f, Path(fig_dir), manual_dir)
        if "pdf" in files:
            w.append(str(files["pdf"]))
        else:
            missing.append(f.stem)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        w.write(fh)
    return missing


def write_legends(figs, out_path):
    Path(out_path).write_text("\n\n".join(legend_text(f) for f in figs) + "\n", encoding="utf-8")
