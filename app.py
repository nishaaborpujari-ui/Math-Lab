import io
import random
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------- Config ----------
st.set_page_config(page_title="Math Worksheet Generator", page_icon="🧮", layout="wide")

# ---------- Utilities ----------
@dataclass
class Problem:
    a: int
    b: int
    op: str   # '+', '-', '×'
    answer: int

def compute_answer(a: int, b: int, op: str) -> int:
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '×':
        return a * b
    raise ValueError("Unsupported op")

def generate_pair(min_n: int, max_n: int, op: str, non_negative_sub: bool = True) -> Tuple[int, int]:
    a = random.randint(min_n, max_n)
    b = random.randint(min_n, max_n)
    if op == '-':
        if non_negative_sub and a < b:
            a, b = b, a
    return a, b

def generate_problems(
    n: int,
    ops: List[str],
    min_n: int,
    max_n: int,
    non_negative_sub: bool = True,
    seed: int | None = None
) -> List[Problem]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    problems: List[Problem] = []
    for _ in range(n):
        op = random.choice(ops)
        a, b = generate_pair(min_n, max_n, op, non_negative_sub)
        ans = compute_answer(a, b, op)
        problems.append(Problem(a, b, op, ans))
    return problems

def problems_to_dataframe(problems: List[Problem]) -> pd.DataFrame:
    return pd.DataFrame({
        "Problem": [f"{p.a} {p.op} {p.b} =" for p in problems],
        "Answer": [p.answer for p in problems]
    })

def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

# ---------- PDF ----------
def draw_problem_text(c: canvas.Canvas, text: str, x: float, y: float, vertical: bool, op: str, a: int, b: int):
    """
    Draws either horizontal "a op b =" or vertical stacked format.
    """
    if not vertical:
        c.drawString(x, y, f"{a} {op} {b} =")
        return

    # Vertical stacked, right-aligned to column width
    # We’ll align numbers to the right edge of a 30mm box.
    box_w = 30 * mm
    base_x = x + box_w
    line_h = 6 * mm
    # Convert to strings
    a_s, b_s = str(a), str(b)
    # Right-aligned drawString helper
    def draww(s, xx, yy):
        w = c.stringWidth(s, "Helvetica", 12)
        c.drawString(xx - w, yy, s)

    draww(a_s, base_x, y)
    draww(f"{op} {b_s}", base_x, y - line_h)
    # line under the second row
    c.line(x + (box_w - 28*mm), y - line_h - 2*mm, x + box_w, y - line_h - 2*mm)

def build_pdf(
    title: str,
    problems: List[Problem],
    cols: int = 3,
    rows: int = 12,
    vertical: bool = False,
    include_key: bool = True
) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin_x = 18 * mm
    margin_y = 15 * mm
    usable_w = width - 2 * margin_x
    usable_h = height - 2 * margin_y

    # Try to register a clean sans font if available; fallback to Helvetica
    try:
        pdfmetrics.registerFont(TTFont("Inter", "Inter-Regular.ttf"))
        font_name = "Inter"
    except Exception:
        font_name = "Helvetica"

    c.setTitle(title)

    def draw_page(probs: List[Problem], page_title: str, show_answers: bool):
        c.setFont(font_name, 16)
        c.drawString(margin_x, height - margin_y + 2*mm, page_title)
        c.setFont(font_name, 12)

        col_w = usable_w / cols
        row_h = usable_h / rows
        start_y = height - margin_y - 10 * mm

        # grid positions
        idx = 0
        y = start_y
        for r in range(rows):
            x = margin_x
            for co in range(cols):
                if idx >= len(probs):
                    break
                p = probs[idx]
                if vertical:
                    draw_problem_text(c, "", x + 5*mm, y, True, p.op, p.a, p.b)
                    if show_answers:
                        c.setFont(font_name, 10)
                        c.drawString(x + 5*mm, y - 17*mm, f"Ans: {p.answer}")
                        c.setFont(font_name, 12)
                else:
                    # Horizontal: single line
                    c.drawString(x + 5*mm, y, f"{p.a} {p.op} {p.b} =")
                    if show_answers:
                        c.setFont(font_name, 10)
                        c.drawString(x + 5*mm + 40*mm, y, f"{p.answer}")
                        c.setFont(font_name, 12)

                x += col_w
                idx += 1
            y -= row_h

    # Worksheet page(s) without answers
    per_page = cols * rows
    for page_i, batch in enumerate(chunk(problems, per_page), start=1):
        draw_page(batch, f"{title}  —  Page {page_i}", show_answers=False)
        c.showPage()

    # Answer key page(s)
    if include_key:
        for page_i, batch in enumerate(chunk(problems, per_page), start=1):
            draw_page(batch, f"{title}  —  Answer Key (Page {page_i})", show_answers=True)
            c.showPage()

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ---------- Sidebar Controls ----------
st.sidebar.header("Worksheet Settings")
ops = st.sidebar.multiselect(
    "Operations",
    options=["+", "-", "×"],
    default=["+"]
)

col_rng1, col_rng2 = st.sidebar.columns(2)
with col_rng1:
    min_n = st.number_input("Min number", value=0, step=1)
with col_rng2:
    max_n = st.number_input("Max number", value=12, step=1)

if max_n < min_n:
    st.sidebar.error("Max number must be ≥ Min number.")

n_problems = st.sidebar.slider("Number of problems", min_value=6, max_value=240, value=60, step=6)
layout_cols = st.sidebar.select_slider("Columns per page (PDF)", options=[2,3,4,5], value=3)
layout_rows = st.sidebar.select_slider("Rows per page (PDF)", options=[8,10,12,14], value=12)
vertical = st.sidebar.toggle("Vertical layout (stacked)", value=False)
nonneg_sub = st.sidebar.toggle("No negative results for subtraction", value=True)
include_key = st.sidebar.toggle("Include answer key in PDF", value=True)
seed_val = st.sidebar.number_input("Random seed (optional)", value=0, step=1)
use_seed = st.sidebar.toggle("Lock with seed", value=False)

st.sidebar.markdown("---")
worksheet_title = st.sidebar.text_input("Worksheet title", value="Math Practice Worksheet")

# ---------- Generate ----------
left, right = st.columns([1, 1])
with left:
    st.title("🧮 Math Worksheet Generator")
    st.caption("Addition, subtraction, and multiplication worksheets for kids — printable PDFs with answer keys.")

if len(ops) == 0:
    st.warning("Select at least one operation to generate problems.")
    st.stop()

seed_to_use = seed_val if use_seed else None
problems = generate_problems(
    n=n_problems,
    ops=ops,
    min_n=int(min_n),
    max_n=int(max_n),
    non_negative_sub=nonneg_sub,
    seed=seed_to_use
)

df = problems_to_dataframe(problems)

# ---------- Preview ----------
with right:
    st.subheader("Preview")
    preview_cols = st.radio("Preview style", ["Grid", "List"], horizontal=True)
    if preview_cols == "Grid":
        # simple grid preview
        grid_cols = st.slider("On-screen columns", 2, 6, 4)
        chunks = list(chunk(df["Problem"].tolist(), grid_cols))
        # Build a small HTML table for nicer look
        html = "<table style='width:100%; border-collapse:collapse;'>"
        for row in chunks:
            html += "<tr>"
            for cell in row:
                html += f"<td style='border:1px solid #ddd; padding:8px; font-family:monospace; font-size:16px;'>{cell}</td>"
            html += "</tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.dataframe(df[["Problem"]], use_container_width=True, height=400)

# ---------- Downloads ----------
st.subheader("Export")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name="worksheet.csv",
    mime="text/csv"
)

pdf_bytes = build_pdf(
    title=worksheet_title,
    problems=problems,
    cols=layout_cols,
    rows=layout_rows,
    vertical=vertical,
    include_key=include_key
)
st.download_button(
    label="Download PDF",
    data=pdf_bytes,
    file_name="worksheet.pdf",
    mime="application/pdf"
)

# ---------- Tips ----------
with st.expander("Tips and ideas"):
    st.markdown(
        """
- Use **Vertical layout** for younger learners practicing columnar addition/subtraction.
- Lock a **Random seed** to regenerate the same sheet later.
- Increase **columns/rows** to fit more problems per page.
- Use **No negative results** to keep subtraction kid-friendly.
- Change **Min/Max number** to adapt difficulty (e.g., 0–10 for early practice, 0–12 for times tables).
        """
    )

