# Paste this into a Custom Code component's Code tab
# Data Visualizer Tool — LLM-powered intelligent chart generation
# Exposed as a Tool for Worker Node via tool_mode=True
#
# Unlike basic chart tools, this uses the LLM to:
#   1. Extract data from ANY format (JSON, markdown tables, text paragraphs)
#   2. Plan the optimal chart configuration (type, axes, sorting, grouping)
#   3. Generate contextual titles and labels
#   4. Apply smart transformations (top-N + "Other", pivoting, sorting)

from agentcore.custom import Node
import json
import base64
import io
import re


# ═══════════════════════════════════════════════════════════════════════════════
# CHART STYLE PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

STYLE_PRESETS = {
    "corporate": {
        "colors": ["#1e3a5f", "#2e86ab", "#a23b72", "#f18f01", "#c73e1d", "#3b1f2b", "#44bba4", "#e94f37",
                    "#5b8c5a", "#8338ec", "#ff006e", "#3a86ff", "#fb5607", "#ffbe0b"],
        "bg_color": "#ffffff",
        "text_color": "#333333",
        "grid_color": "#e0e0e0",
        "title_size": 16,
        "label_size": 11,
        "tick_size": 10,
    },
    "modern": {
        "colors": ["#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", "#f97316", "#eab308", "#22c55e", "#06b6d4",
                    "#a855f7", "#14b8a6", "#f59e0b", "#ef4444", "#3b82f6", "#10b981"],
        "bg_color": "#0f172a",
        "text_color": "#e2e8f0",
        "grid_color": "#1e293b",
        "title_size": 16,
        "label_size": 11,
        "tick_size": 10,
    },
    "executive": {
        "colors": ["#0d1b2a", "#1b263b", "#415a77", "#778da9", "#e0e1dd", "#c4a35a", "#b08968", "#7f5539",
                    "#606c38", "#283618", "#dda15e", "#bc6c25", "#264653", "#2a9d8f"],
        "bg_color": "#fafaf9",
        "text_color": "#1c1917",
        "grid_color": "#e7e5e4",
        "title_size": 16,
        "label_size": 11,
        "tick_size": 10,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# LLM CHART PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

CHART_PLANNER_PROMPT = """You are a data visualization expert. Analyze the data and user request, then output a chart configuration as JSON.

**User Request:** {user_request}

**Data:**
{data_preview}

**Available chart types:** bar, bar_horizontal, line, pie, scatter, stacked_bar, table

**Respond with ONLY this JSON structure:**
{{
    "chart_type": "bar",
    "title": "Descriptive business title for the chart",
    "x_column": 0,
    "y_columns": [1],
    "x_label": "Human-readable X axis label",
    "y_label": "Human-readable Y axis label",
    "sort_by_value": true,
    "sort_descending": true,
    "top_n": null,
    "group_others": false,
    "annotations": ["optional insight to show on chart"]
}}

**CRITICAL axis rules (MUST follow):**
- x_column MUST be the index of the CATEGORY/LABEL column (text/names — e.g., supplier names, regions, months)
- y_columns MUST be the index(es) of the NUMERIC/VALUE column(s) (e.g., amounts, counts, quantities)
- NEVER put a numeric column as x_column for bar/bar_horizontal charts
- NEVER put a text/category column in y_columns
- For "bar" chart: x_column = categories on horizontal axis, y_columns = values on vertical axis (vertical bars)
- For "bar_horizontal" chart: x_column = categories on vertical axis, y_columns = values on horizontal axis (horizontal bars)
- If user says "X in x axis and Y in y axis", use "bar" (vertical), set x_column to the column matching X, y_columns to the column matching Y
- Default to "bar" (vertical) unless user explicitly asks for horizontal or there are >10 long labels

**Decision rules:**
- Pie: ONLY for 2-8 categories showing proportions/shares. Set group_others=true if >6 items.
- Bar (vertical): DEFAULT for comparisons, rankings, top-N with <=10 items.
- Bar_horizontal: Only when labels are very long (>25 chars) or >10 items.
- Line: For time series, trends, monthly/quarterly data. x_column must be the date/time column.
- Stacked bar: For composition across categories (e.g., spend by region broken down by material type).
- Scatter: For correlations between two numeric columns.
- Table: Fallback if data has too many columns or doesn't suit any chart.
- sort_by_value=true for rankings/comparisons, false for time series.
- top_n: Set to 10-15 if there are many rows and user asked for "top" or data would be cluttered.
- title: Make it a proper business title (e.g., "Top 10 Suppliers by Total Spend (EUR)" not "SUPPLIER_NAME vs TOTAL_SPEND_EUR").
- x_label/y_label: Human readable (e.g., "Total Spend (EUR)" not "TOTAL_SPEND_EUR").
- annotations: One brief data insight (e.g., "Top 3 suppliers account for 45% of total").

Return ONLY the JSON, no explanations."""


DATA_EXTRACTOR_PROMPT = """Extract structured data from the following content. The user wants to visualize this data.

**Content:**
{content}

**User Request:** {user_request}

**Return ONLY this JSON:**
{{
    "columns": ["Column1", "Column2"],
    "rows": [["val1", 123], ["val2", 456]]
}}

Rules:
- Extract column headers and all data rows
- Convert numeric values to numbers (not strings)
- If the content is a markdown table, parse it
- If the content is text with numbers, extract the key data points
- If you can't extract meaningful data, return {{"columns": [], "rows": []}}

Return ONLY the JSON."""


# ═══════════════════════════════════════════════════════════════════════════════
# DATA EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _try_parse_data_json(text):
    """Try to extract structured data from data_json (HTML comment or tag format).

    Returns (data_dict, remaining_text) for the first match. Use
    _extract_all_data_json() when multiple data_json comments may be present.
    """
    # Try HTML comment format: <!-- data_json:{...} -->
    comment_match = re.search(r'<!--\s*data_json:(.*?)\s*-->', text, re.DOTALL)
    if comment_match:
        try:
            data = json.loads(comment_match.group(1).strip())
            if data.get("columns") and data.get("rows"):
                return data, text[comment_match.end():].strip()
        except json.JSONDecodeError:
            pass

    # Try <data_json> tags (legacy format)
    tag_match = re.search(r'<data_json>(.*?)</data_json>', text, re.DOTALL)
    if tag_match:
        try:
            data = json.loads(tag_match.group(1).strip())
            if data.get("columns") and data.get("rows"):
                return data, text[tag_match.end():].strip()
        except json.JSONDecodeError:
            pass

    # Try raw JSON object with columns/rows
    json_match = re.search(r'\{[^{}]*"columns"\s*:\s*\[.*?\].*?"rows"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if data.get("columns") and data.get("rows"):
                return data, text[json_match.end():].strip()
        except json.JSONDecodeError:
            pass

    return None, text


def _extract_all_data_json(text):
    """Extract ALL data_json comments from text.

    Returns list of dicts, each with 'columns', 'rows', and optional 'view' label.
    Auto-mode pipeline output has 2-3 data_json comments (Direct, Indirect, Combined).
    """
    results = []
    for m in re.finditer(r'<!--\s*data_json:(.*?)\s*-->', text, re.DOTALL):
        try:
            data = json.loads(m.group(1).strip())
            if data.get("columns") and data.get("rows"):
                results.append(data)
        except (json.JSONDecodeError, ValueError):
            pass
    return results


def _pick_best_data(all_data, user_request=""):
    """From a list of data_json dicts, pick the best one for charting.

    Priority:
    1. If the user's follow-up explicitly asks for "combined" → Combined view.
    2. If the user's follow-up uses a specific "only direct" / "only indirect"
       phrase → that view.
    3. If a Combined view exists (default for multi-view results) → Combined.
    4. Merge all views by the first column (N-series preserved).
    5. Fall back to the dataset with the most rows.

    Why step 2 uses phrases instead of plain substring match: the Worker LLM
    passes the entire previous response text, which ALWAYS contains the
    headings "### Direct Spend", "### Indirect Spend", and "### Combined
    Summary". A plain `"direct" in req_lower` check therefore matches on
    every auto-mode response. We look for specific intent phrases instead.
    """
    if not all_data:
        return None
    if len(all_data) == 1:
        return all_data[0]

    req_lower = user_request.lower()

    _COMBINED_PHRASES = (
        "combined", "grand total", "both views", "direct and indirect",
        "overall total", "total combined",
    )
    # Regex so "only the direct", "just direct only", "direct spend only" all
    # match. `\b` word boundaries + an optional filler (the/only/spend/data/
    # view/branch) between the keyword and its modifier.
    _FILLER = r"(?:the\s+|spend\s+|data\s+|view\s+|branch\s+)*"
    _DIRECT_ONLY_RE = re.compile(
        rf"\b(?:only\s+{_FILLER}direct|just\s+{_FILLER}direct|"
        rf"direct\s+{_FILLER}only|direct\s+{_FILLER}branch|direct\s+{_FILLER}view)\b"
    )
    _INDIRECT_ONLY_RE = re.compile(
        rf"\b(?:only\s+{_FILLER}indirect|just\s+{_FILLER}indirect|"
        rf"indirect\s+{_FILLER}only|indirect\s+{_FILLER}branch|indirect\s+{_FILLER}view)\b"
    )

    def _match(phrases):
        return any(p in req_lower for p in phrases)

    # 1. Most specific intent first: "only direct" / "only indirect" beats any
    #    mention of "combined" elsewhere in the prior-response blob (which
    #    contains all three view headings).
    has_indirect_only = bool(_INDIRECT_ONLY_RE.search(req_lower))
    has_direct_only = bool(_DIRECT_ONLY_RE.search(req_lower))
    if has_indirect_only:
        for d in all_data:
            view = (d.get("source") or "").lower()
            if "indirect" in view:
                return d
    if has_direct_only and not has_indirect_only:
        for d in all_data:
            view = (d.get("source") or "").lower()
            if "direct" in view and "indirect" not in view:
                return d

    # 2. Explicit "combined" intent — wins over bare direct/indirect substring
    #    matches because the prior-response blob will contain all three view
    #    labels.
    if _match(_COMBINED_PHRASES):
        for d in all_data:
            if "combined" in (d.get("source") or "").lower():
                return d

    # 3. Default: prefer Combined.
    for d in all_data:
        view = (d.get("source") or "").lower()
        if "combined" in view:
            return d

    # No Combined — merge multiple views by first column. Supports 3+ columns
    # by summing each numeric column independently. Earlier versions only kept
    # column[1] and silently dropped extra numeric columns (e.g. 2024/2025
    # year-over-year splits).
    first_cols = [d["columns"][0].upper() for d in all_data if d.get("columns")]
    if len(set(first_cols)) == 1 and first_cols[0]:
        ref_cols = all_data[0].get("columns", [])
        dim_name = ref_cols[0]
        numeric_col_indices = []
        for i in range(1, len(ref_cols)):
            is_numeric = False
            for d in all_data:
                for row in d.get("rows", []) or []:
                    if i < len(row) and isinstance(row[i], (int, float)) and not isinstance(row[i], bool):
                        is_numeric = True
                        break
                if is_numeric:
                    break
            if is_numeric:
                numeric_col_indices.append(i)
        if not numeric_col_indices:
            numeric_col_indices = [1] if len(ref_cols) > 1 else []

        # merged[dimension_value] = [sum_col_i for i in numeric_col_indices]
        merged: dict = {}
        for d in all_data:
            for row in d.get("rows", []) or []:
                if not row:
                    continue
                key = str(row[0])
                totals = merged.setdefault(key, [0.0] * len(numeric_col_indices))
                for out_idx, src_idx in enumerate(numeric_col_indices):
                    if src_idx < len(row):
                        v = row[src_idx]
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            totals[out_idx] += v

        # Sort descending by the first numeric column (the primary metric).
        sorted_items = sorted(merged.items(), key=lambda x: x[1][0], reverse=True)
        combined_headers = [dim_name] + [
            f"Combined {ref_cols[i]}" if i < len(ref_cols) else f"Combined col{i}"
            for i in numeric_col_indices
        ]
        return {
            "columns": combined_headers,
            "rows": [[k] + vals for k, vals in sorted_items],
            "source": "Combined",
        }

    # Fallback: use the one with the most rows
    return max(all_data, key=lambda d: len(d.get("rows", [])))


def _wrap_chart_output(chart_content):
    """Wrap chart output with markers so the Worker LLM copies it verbatim."""
    return (
        "[TOOL OUTPUT — COPY EVERYTHING BETWEEN THE MARKERS INTO YOUR RESPONSE]\n"
        "===BEGIN===\n"
        f"{chart_content}\n"
        "===END===\n"
        "[Paste everything between ===BEGIN=== and ===END=== into your response VERBATIM.]"
    )


def _parse_km_number(s):
    """Parse a number string that may have K/M suffix (e.g. '45.3M' → 45300000)."""
    s = s.replace(",", "").replace("*", "").strip()
    if not s:
        return None
    suffix = s[-1].upper()
    if suffix == 'M':
        try:
            return float(s[:-1]) * 1_000_000
        except ValueError:
            return None
    if suffix == 'K':
        try:
            return float(s[:-1]) * 1_000
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _try_parse_markdown_table(text):
    """Try to extract data from a markdown table."""
    lines = text.strip().split("\n")
    table_lines = [l for l in lines if l.strip().startswith("|")]
    if len(table_lines) < 3:  # need header + separator + at least 1 data row
        return None

    # Parse header
    header = table_lines[0]
    columns = [c.strip() for c in header.split("|") if c.strip()]
    if not columns:
        return None

    # Skip separator row(s)
    data_start = 1
    for i in range(1, len(table_lines)):
        if re.match(r'^[\s|:-]+$', table_lines[i]):
            data_start = i + 1
        else:
            break

    # Parse data rows
    rows = []
    for line in table_lines[data_start:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) != len(columns):
            continue
        parsed_row = []
        for cell in cells:
            # Strip bold markers and whitespace
            clean = cell.replace("*", "").replace("—", "").replace("NULL", "").strip()
            if not clean or clean == "—":
                parsed_row.append(None)
            else:
                # Try K/M suffix parsing first, then plain number, then string
                km_val = _parse_km_number(clean)
                if km_val is not None:
                    parsed_row.append(km_val)
                else:
                    parsed_row.append(cell.replace("*", "").strip())
        rows.append(parsed_row)

    if rows:
        return {"columns": columns, "rows": rows}
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CHART RENDERING (matplotlib)
# ═══════════════════════════════════════════════════════════════════════════════

def _chart_to_base64(fig):
    """Original base64 encoder (PNG, high quality)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def _chart_to_base64_optimized(fig):
    """Compact base64 — JPEG for smaller output."""
    import matplotlib.pyplot as plt
    fig.set_size_inches(6, 3.5)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=72, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.1,
                pil_kwargs={"quality": 70, "optimize": True})
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return b64, "jpeg"


def _render_chart_fig(plan, columns, rows, style_name):
    """Render a chart based on the LLM's plan. Returns the matplotlib figure object."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    style = STYLE_PRESETS.get(style_name, STYLE_PRESETS["corporate"])
    colors = style["colors"]
    chart_type = plan.get("chart_type", "bar")
    title = plan.get("title", "Data Visualization")
    x_col = plan.get("x_column", 0)
    y_cols = plan.get("y_columns", [1])
    x_label = plan.get("x_label", "")
    y_label = plan.get("y_label", "")
    sort_by_value = plan.get("sort_by_value", False)
    sort_desc = plan.get("sort_descending", True)
    top_n = plan.get("top_n")
    group_others = plan.get("group_others", False)
    annotations = plan.get("annotations", [])

    def _safe_float(v):
        if v is None:
            return 0
        s = str(v).replace(",", "").strip()
        if not s:
            return 0
        suffix = s[-1].upper()
        try:
            if suffix == 'M':
                return float(s[:-1]) * 1_000_000
            if suffix == 'K':
                return float(s[:-1]) * 1_000
            return float(s)
        except (ValueError, TypeError):
            return 0

    # Prepare data with sorting and top-N
    work_rows = list(rows)

    if sort_by_value and y_cols:
        y_idx = y_cols[0] if y_cols[0] < len(columns) else 1
        work_rows.sort(key=lambda r: _safe_float(r[y_idx] if y_idx < len(r) else 0), reverse=sort_desc)

    if top_n and len(work_rows) > top_n:
        top_rows = work_rows[:top_n]
        rest_rows = work_rows[top_n:]
        if group_others and y_cols:
            # Aggregate the rest into "Other"
            other_vals = []
            for y_idx in y_cols:
                total = sum(_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in rest_rows)
                other_vals.append(total)
            other_row = [None] * len(columns)
            other_row[x_col] = f"Other ({len(rest_rows)})"
            for i, y_idx in enumerate(y_cols):
                if y_idx < len(other_row):
                    other_row[y_idx] = other_vals[i]
            top_rows.append(other_row)
        work_rows = top_rows

    # Extract labels and values
    labels = [str(r[x_col])[:30] if x_col < len(r) else "?" for r in work_rows]

    # Create figure (compact for fast base64 streaming)
    fig_width = max(7, min(10, len(labels) * 0.7))
    fig_height = 4 if chart_type != "bar_horizontal" else max(4, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(style["bg_color"])
    ax.set_facecolor(style["bg_color"])
    ax.tick_params(colors=style["text_color"], labelsize=style["tick_size"])
    ax.xaxis.label.set_color(style["text_color"])
    ax.yaxis.label.set_color(style["text_color"])
    ax.title.set_color(style["text_color"])
    for spine in ax.spines.values():
        spine.set_color(style["grid_color"])
    ax.grid(True, alpha=0.3, color=style["grid_color"], linestyle="--")

    # ── Render by chart type ──

    if chart_type in ("bar", "stacked_bar"):
        if len(y_cols) > 1:
            # Grouped or stacked bar
            num_groups = len(work_rows)
            if chart_type == "stacked_bar":
                bottom = [0] * num_groups
                for i, y_idx in enumerate(y_cols):
                    vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
                    col_name = columns[y_idx] if y_idx < len(columns) else f"Series {i+1}"
                    ax.bar(labels, vals, bottom=bottom, label=col_name,
                           color=colors[i % len(colors)], edgecolor="none", width=0.7)
                    bottom = [b + v for b, v in zip(bottom, vals)]
            else:
                bar_width = 0.8 / len(y_cols)
                x_pos = list(range(num_groups))
                for i, y_idx in enumerate(y_cols):
                    vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
                    col_name = columns[y_idx] if y_idx < len(columns) else f"Series {i+1}"
                    offset = (i - (len(y_cols) - 1) / 2) * bar_width
                    positions = [x + offset for x in x_pos]
                    ax.bar(positions, vals, bar_width * 0.9, label=col_name,
                           color=colors[i % len(colors)], edgecolor="none")
                ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=style["tick_size"])
            ax.legend(fontsize=9, facecolor=style["bg_color"], edgecolor=style["grid_color"],
                      labelcolor=style["text_color"])
        else:
            y_idx = y_cols[0] if y_cols[0] < len(columns) else 1
            vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
            bar_colors = [colors[i % len(colors)] for i in range(len(vals))]
            bars = ax.bar(labels, vals, color=bar_colors, edgecolor="none", width=0.7)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=style["tick_size"])
            # Value labels on bars
            max_val = max(vals) if vals else 1
            for bar, val in zip(bars, vals):
                if val > 0:
                    text = f"{val:,.0f}" if abs(val) >= 100 else f"{val:,.2f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max_val * 0.01,
                            text, ha="center", va="bottom",
                            fontsize=8, color=style["text_color"], fontweight="bold")

    elif chart_type == "bar_horizontal":
        y_idx = y_cols[0] if y_cols[0] < len(columns) else 1
        vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
        bar_colors = [colors[i % len(colors)] for i in range(len(vals))]
        bars = ax.barh(labels, vals, color=bar_colors, edgecolor="none", height=0.65)
        ax.invert_yaxis()
        max_val = max(vals) if vals else 1
        for bar, val in zip(bars, vals):
            if val > 0:
                text = f"{val:,.0f}" if abs(val) >= 100 else f"{val:,.2f}"
                ax.text(bar.get_width() + max_val * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f" {text}", ha="left", va="center",
                        fontsize=9, color=style["text_color"], fontweight="bold")

    elif chart_type == "line":
        for i, y_idx in enumerate(y_cols):
            vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
            col_name = columns[y_idx] if y_idx < len(columns) else f"Series {i+1}"
            line, = ax.plot(labels, vals, marker="o", markersize=5, linewidth=2.5,
                            color=colors[i % len(colors)], label=col_name)
            if len(y_cols) == 1:
                ax.fill_between(range(len(vals)), vals, alpha=0.1, color=colors[i % len(colors)])
                # Value labels on points
                for j, (lbl, val) in enumerate(zip(labels, vals)):
                    if val > 0:
                        ax.annotate(f"{val:,.0f}" if abs(val) >= 100 else f"{val:,.2f}",
                                    (j, val), textcoords="offset points", xytext=(0, 8),
                                    ha="center", fontsize=7, color=style["text_color"])
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=style["tick_size"])
        if len(y_cols) > 1:
            ax.legend(fontsize=9, facecolor=style["bg_color"], edgecolor=style["grid_color"],
                      labelcolor=style["text_color"])

    elif chart_type == "pie":
        y_idx = y_cols[0] if y_cols[0] < len(columns) else 1
        vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
        # Filter zero/negative
        filtered = [(l, v) for l, v in zip(labels, vals) if v > 0]
        if filtered:
            pie_labels, pie_vals = zip(*filtered)
            pie_colors = [colors[i % len(colors)] for i in range(len(pie_vals))]
            total = sum(pie_vals)
            # Custom labels with values
            display_labels = [f"{l}\n({v/total*100:.1f}%)" for l, v in zip(pie_labels, pie_vals)]
            wedges, texts, autotexts = ax.pie(
                pie_vals, labels=display_labels, colors=pie_colors,
                autopct=lambda p: f"{p*total/100:,.0f}" if p > 3 else "",
                startangle=140, pctdistance=0.75,
                textprops={"fontsize": style["tick_size"], "color": style["text_color"]},
                wedgeprops={"edgecolor": style["bg_color"], "linewidth": 2},
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_color("white")
                at.set_fontweight("bold")
            ax.axis("equal")

    elif chart_type == "scatter":
        if len(y_cols) >= 1 and len(columns) >= 3:
            y1 = y_cols[0] if y_cols[0] < len(columns) else 1
            y2 = y_cols[1] if len(y_cols) > 1 and y_cols[1] < len(columns) else (y1 + 1 if y1 + 1 < len(columns) else y1)
            x_vals = [_safe_float(r[y1] if y1 < len(r) else 0) for r in work_rows]
            y_vals = [_safe_float(r[y2] if y2 < len(r) else 0) for r in work_rows]
            ax.scatter(x_vals, y_vals, c=colors[0], alpha=0.7, edgecolors=colors[1], s=80, linewidths=1)
            if x_label:
                ax.set_xlabel(x_label, fontsize=style["label_size"])
            if y_label:
                ax.set_ylabel(y_label, fontsize=style["label_size"])

    # ── Labels and title ──
    if chart_type not in ("pie", "scatter"):
        if x_label:
            ax.set_xlabel(x_label, fontsize=style["label_size"], color=style["text_color"], labelpad=10)
        if y_label:
            ax.set_ylabel(y_label, fontsize=style["label_size"], color=style["text_color"], labelpad=10)

    ax.set_title(title, fontsize=style["title_size"], fontweight="bold",
                 color=style["text_color"], pad=20)

    if chart_type not in ("pie",):
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}" if abs(x) >= 1 else f"{x:.2f}"))

    # ── Annotation (insight text at bottom) ──
    if annotations:
        fig.text(0.5, -0.02, annotations[0][:120],
                 ha="center", fontsize=9, color=style["text_color"],
                 style="italic", alpha=0.7)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CHART RENDERER (Chart.js via iframe)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_interactive_chart(plan, columns, rows):
    """Render an interactive chart using Chart.js inside an iframe.

    CRITICAL: Uses ONLY single quotes inside the HTML so the srcdoc attribute
    (delimited by double quotes) never needs &quot; escaping. This prevents
    the LLM from breaking the chart when re-emitting the iframe.
    """
    x_col = plan.get("x_column", 0)
    y_cols = plan.get("y_columns", [1])
    title = plan.get("title", "Chart").replace("'", "")
    chart_type = plan.get("chart_type", "bar")
    sort_by_value = plan.get("sort_by_value", False)
    sort_desc = plan.get("sort_descending", True)

    def _safe_float(v):
        if v is None:
            return 0
        s = str(v).replace(",", "").strip()
        if not s:
            return 0
        suffix = s[-1].upper()
        try:
            if suffix == 'M':
                return float(s[:-1]) * 1_000_000
            if suffix == 'K':
                return float(s[:-1]) * 1_000
            return float(s)
        except (ValueError, TypeError):
            return 0

    # Prepare data
    work_rows = list(rows)
    if sort_by_value and y_cols:
        y_idx = y_cols[0] if y_cols[0] < len(columns) else 1
        work_rows.sort(key=lambda r: _safe_float(r[y_idx] if y_idx < len(r) else 0), reverse=sort_desc)

    top_n = plan.get("top_n")
    if top_n and len(work_rows) > top_n:
        work_rows = work_rows[:top_n]

    # Sanitize labels: escape single quotes for JS strings
    labels = [str(r[x_col])[:30].replace("'", "\\'") if x_col < len(r) else "?" for r in work_rows]

    # Map chart types to Chart.js types
    cjs_type_map = {
        "bar": "bar",
        "bar_horizontal": "bar",
        "line": "line",
        "pie": "pie",
        "scatter": "scatter",
        "stacked_bar": "bar",
    }
    cjs_type = cjs_type_map.get(chart_type, "bar")
    is_horizontal = chart_type == "bar_horizontal"

    # Colors
    colors = ["#1e3a5f", "#2e86ab", "#a23b72", "#f18f01", "#c73e1d",
              "#44bba4", "#e94f37", "#5b8c5a", "#8338ec", "#ff006e",
              "#3a86ff", "#fb5607", "#ffbe0b", "#3b1f2b"]

    # Build JS arrays using single quotes only
    labels_js = "[" + ",".join(f"'{l}'" for l in labels) + "]"

    vals_per_series = []
    series_labels = []
    for i, y_idx in enumerate(y_cols):
        vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
        vals_per_series.append(vals)
        lbl = columns[y_idx].replace("_", " ").title().replace("'", "") if y_idx < len(columns) else f"Series {i+1}"
        series_labels.append(lbl)

    # Build datasets JS using single quotes
    datasets_parts = []
    for i, (vals, lbl) in enumerate(zip(vals_per_series, series_labels)):
        vals_js = "[" + ",".join(str(v) for v in vals) + "]"
        if cjs_type == "pie":
            bg_js = "[" + ",".join(f"'{colors[j % len(colors)]}'" for j in range(len(vals))) + "]"
            datasets_parts.append(f"{{label:'{lbl}',data:{vals_js},backgroundColor:{bg_js},borderWidth:2,borderColor:'#fff'}}")
        else:
            c = colors[i % len(colors)]
            datasets_parts.append(f"{{label:'{lbl}',data:{vals_js},backgroundColor:'{c}cc',borderColor:'{c}',borderWidth:1}}")
    datasets_js = "[" + ",".join(datasets_parts) + "]"

    # Chart options using single quotes
    idx_axis = "'y'" if is_horizontal else "'x'"
    stacked = "true" if chart_type == "stacked_bar" else "false"
    show_legend = "true" if (cjs_type == "pie" or len(y_cols) > 1) else "false"

    x_label = plan.get("x_label", "").replace("_", " ").title().replace("'", "")
    y_label = plan.get("y_label", "").replace("_", " ").title().replace("'", "")
    show_x = "true" if x_label and cjs_type != "pie" else "false"
    show_y = "true" if y_label and cjs_type != "pie" else "false"

    height = 400
    if is_horizontal:
        height = max(300, len(work_rows) * 35)

    # Build scales JS — pie charts don't need axes
    if cjs_type == "pie":
        scales_js = "scales:{}"
    else:
        scales_js = (
            f"scales:{{x:{{stacked:{stacked},title:{{display:{show_x},text:'{x_label}'}},ticks:{{maxRotation:45}}}},"
            f"y:{{stacked:{stacked},title:{{display:{show_y},text:'{y_label}'}},ticks:{{callback:function(v){{if(Math.abs(v)>=1e6)return(v/1e6).toFixed(1)+'M';if(Math.abs(v)>=1e3)return(v/1e3).toFixed(0)+'K';return v}}}}}}}}"
        )

    # ALL single quotes inside HTML — NO double quotes anywhere
    # This means srcdoc="..." never needs &quot; escaping
    html_parts = [
        "<!DOCTYPE html><html><head>",
        "<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js'></script>",
        "<style>body{margin:0;padding:8px;font-family:system-ui,sans-serif;background:#fff}.wrap{max-width:700px;margin:0 auto}</style>",
        "</head><body><div class='wrap'><canvas id='c'></canvas></div>",
        "<script>",
        f"new Chart(document.getElementById('c'),{{",
        f"type:'{cjs_type}',",
        f"data:{{labels:{labels_js},datasets:{datasets_js}}},",
        f"options:{{",
        f"indexAxis:{idx_axis},responsive:true,maintainAspectRatio:true,",
        f"plugins:{{title:{{display:true,text:'{title}',font:{{size:14,weight:'bold'}}}},",
        f"legend:{{display:{show_legend}}},",
        f"tooltip:{{callbacks:{{label:function(c){{let v=c.raw!==undefined?c.raw:(c.parsed.y!==undefined?c.parsed.y:c.parsed);if(typeof v==='number')return c.dataset.label+': '+v.toLocaleString('en-US',{{maximumFractionDigits:2}});return c.dataset.label+': '+v}}}}}}}},",
        scales_js + "}});",
        "</script></body></html>",
    ]
    html = "".join(html_parts)

    # Only need to escape & for srcdoc — NO &quot; needed since no double quotes inside
    srcdoc = html.replace("&", "&amp;")

    iframe = f'<iframe srcdoc="{srcdoc}" width="100%" height="{height}" frameborder="0" style="border:1px solid #e0e0e0;border-radius:8px;max-width:750px"></iframe>'

    return iframe


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CHART RENDERER (instant, no image)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_text_chart(plan, columns, rows):
    """Render a chart using Unicode block characters. Instant, no matplotlib.

    Output looks like:
    **Top 5 Suppliers by Spend (EUR)**

    MOTHERSON ELECTRO..  ████████████████████████████  96,010,720
    NOVEM CAR INTERIO..  ███████████████████████████   93,123,977
    EISSMANN AUTOMOTI..  ██████████████████████████    90,512,369
    KROMBERG & SCHUBE..  ██████████████              51,183,135
    KOSTAL MEXICANA S..  █████████                   34,309,792
    """
    x_col = plan.get("x_column", 0)
    y_cols = plan.get("y_columns", [1])
    title = plan.get("title", "Chart")
    sort_by_value = plan.get("sort_by_value", True)
    sort_desc = plan.get("sort_descending", True)
    chart_type = plan.get("chart_type", "bar")

    if not y_cols or not rows:
        return None

    y_col = y_cols[0]

    def _safe_float(v):
        if v is None:
            return 0
        s = str(v).replace(",", "").strip()
        if not s:
            return 0
        suffix = s[-1].upper()
        try:
            if suffix == 'M':
                return float(s[:-1]) * 1_000_000
            if suffix == 'K':
                return float(s[:-1]) * 1_000
            return float(s)
        except (ValueError, TypeError):
            return 0

    # Prepare data
    work_rows = list(rows)
    if sort_by_value:
        work_rows.sort(key=lambda r: _safe_float(r[y_col] if y_col < len(r) else 0), reverse=sort_desc)

    top_n = plan.get("top_n")
    if top_n and len(work_rows) > top_n:
        work_rows = work_rows[:top_n]

    labels = [str(r[x_col])[:22] if x_col < len(r) else "?" for r in work_rows]
    values = [_safe_float(r[y_col] if y_col < len(r) else 0) for r in work_rows]
    max_val = max(values) if values else 1
    max_bar_width = 30

    # Build text chart
    lines = [f"**{title}**\n"]
    lines.append("```")

    # Determine label width
    label_width = max(len(l) for l in labels) if labels else 10
    label_width = min(label_width, 22)

    for label, val in zip(labels, values):
        bar_len = int((val / max_val) * max_bar_width) if max_val > 0 else 0
        bar = "\u2588" * bar_len
        # Format value
        if abs(val) >= 1_000_000:
            val_str = f"{val:>14,.0f}"
        elif abs(val) >= 1:
            val_str = f"{val:>14,.2f}"
        else:
            val_str = f"{val:>14.4f}"
        lines.append(f"  {label:<{label_width}}  {bar:<{max_bar_width}}  {val_str}")

    lines.append("```")
    lines.append(f"\n*{len(work_rows)} items | {chart_type}*")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════

class CodeEditorNode(Node):
    display_name = "Data Visualizer"
    description = (
        "ONLY use this tool when the user EXPLICITLY asks to plot, chart, graph, or visualize data. "
        "Do NOT call this for normal data queries. Requires data from a previous query result. "
        "Pass the data_json and chart instructions as input."
    )
    icon = "bar-chart-3"
    name = "DataVisualizer"

    def _llm_invoke_silent(self, prompt):
        """Invoke self.llm without emitting agent-level callback events.

        Same pattern as talk_to_data.py — internal LLM calls (chart planner,
        data extractor) must not stream their raw JSON/reasoning into the
        Orchestration chat UI; otherwise follow-up plots leak chart-plan JSON
        and extracted-data dumps below the rendered chart.
        """
        try:
            return self.llm.invoke(prompt, config={"callbacks": []})
        except TypeError:
            return self.llm.invoke(prompt)

    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="Data + Instructions",
            info="Data (JSON, markdown table, or text) + optional chart instructions.",
            tool_mode=True,
        ),
        HandleInput(
            name="llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            info="Optional LLM for smart chart planning. If not connected, uses auto-detection (faster).",
            required=False,
        ),
        DropdownInput(
            name="chart_style",
            display_name="Chart Style",
            options=["corporate", "modern", "executive"],
            value="corporate",
            info="Visual style preset.",
        ),
        DropdownInput(
            name="render_mode",
            display_name="Render Mode",
            options=["interactive", "text", "image"],
            value="interactive",
            info="'interactive' = Chart.js in iframe (fast, interactive hover/zoom). 'text' = Unicode bar chart (instant). 'image' = matplotlib base64 (slow but exportable).",
        ),
    ]

    outputs = [
        Output(display_name="Chart", name="output", method="build_output"),
    ]

    def build_output(self) -> Message:
        raw_input = self.input_value or ""
        if not raw_input.strip():
            return Message(text="No data provided. Call Talk to Data first to get data, then ask me to chart it.")

        # ── Step 1: Extract structured data ──
        data = None
        instructions = raw_input

        # Try extracting ALL data_json comments (supports multi-view auto mode)
        all_data = _extract_all_data_json(raw_input)
        if all_data:
            data = _pick_best_data(all_data, raw_input)
            # Strip data_json comments to get user instructions
            stripped = re.sub(r'<!--\s*data_json:.*?\s*-->', '', raw_input, flags=re.DOTALL).strip()
            instructions = stripped or "visualize this data"

        # Fallback: single data_json via old parser
        if not data:
            parsed, remaining = _try_parse_data_json(raw_input)
            if parsed:
                data = parsed
                instructions = remaining or "visualize this data"

        # Try parsing markdown table
        if not data:
            parsed = _try_parse_markdown_table(raw_input)
            if parsed:
                data = parsed
                non_table_lines = [l for l in raw_input.split("\n") if not l.strip().startswith("|")]
                instructions = "\n".join(non_table_lines).strip() or "visualize this data"

        # Try raw JSON
        if not data:
            try:
                raw_json = json.loads(raw_input.strip())
                if isinstance(raw_json, dict) and raw_json.get("columns") and raw_json.get("rows"):
                    data = raw_json
                    instructions = "visualize this data"
            except (json.JSONDecodeError, ValueError):
                pass

        # Last resort: ask LLM to extract (only if LLM is connected)
        if not data and self.llm and self.llm != "":
            try:
                data = self._llm_extract_data(raw_input, instructions)
            except Exception:
                pass

        if not data or not data.get("columns") or not data.get("rows"):
            return Message(text="Could not extract chartable data from the input. Please provide a data table or data_json.")

        columns = data["columns"]
        rows = data["rows"]

        # Empty result set — tell the user plainly instead of rendering a blank
        # canvas. Upstream SQL queries that return zero rows were previously
        # producing invisible charts that looked broken.
        if not rows:
            col_label = ", ".join(str(c) for c in columns) if columns else "(no columns)"
            return Message(text=(
                "No data to chart — the upstream query returned 0 rows.\n\n"
                f"Columns that would have been used: {col_label}.\n"
                "Try broadening the filters or the date range and re-run the data question."
            ))

        # ── Step 2: Chart Planning (heuristic first, LLM optional) ──
        plan = self._auto_plan_chart(columns, rows, instructions)

        # Detect if user explicitly requested a chart type. Word-boundary match
        # so "barometer" doesn't select "bar" and "pied" doesn't select "pie".
        # Order matters: compound modifiers (horizontal, stacked) are checked
        # before the generic shapes so "horizontal bar" resolves to bar_horizontal.
        _req = instructions.lower()
        _user_chart_type = None
        for _kw, _ct in [
            ("horizontal", "bar_horizontal"),
            ("stacked", "stacked_bar"),
            ("pie", "pie"),
            ("donut", "pie"),
            ("scatter", "scatter"),
            ("line", "line"),
            ("trend", "line"),
        ]:
            if re.search(rf"\b{_kw}\b", _req):
                _user_chart_type = _ct
                break

        # If LLM is connected, try to enhance the plan (with timeout protection)
        if self.llm and self.llm != "":
            try:
                llm_plan = self._llm_plan_chart(columns, rows, instructions)
                if llm_plan and llm_plan.get("chart_type"):
                    plan = llm_plan
            except Exception:
                pass  # Keep heuristic plan

        # User's explicit chart type keyword always wins over LLM
        if _user_chart_type:
            plan["chart_type"] = _user_chart_type

        # Validate and fix axes
        plan = self._validate_and_fix_plan(plan, columns, rows)

        # ── Step 3: Render Chart ──
        try:
            chart_type = plan.get("chart_type", "bar")

            if chart_type == "table":
                return self._render_as_table(columns, rows, plan.get("title", "Data"))

            # Data JSON for follow-up chart modifications
            data_json_obj = {"columns": columns, "rows": rows}
            data_comment = f"\n<!-- data_json:{json.dumps(data_json_obj)} -->"

            # ── Interactive mode: Chart.js via iframe (default) ──
            if self.render_mode == "interactive":
                iframe = _render_interactive_chart(plan, columns, rows)
                if iframe:
                    self.status = f"{chart_type} | {len(rows)} rows | interactive"
                    chart_output = iframe + data_comment
                    return Message(text=_wrap_chart_output(chart_output))

            # ── Text mode: instant Unicode chart (only for horizontal bar) ──
            if self.render_mode == "text" and chart_type == "bar_horizontal":
                text_chart = _render_text_chart(plan, columns, rows)
                if text_chart:
                    self.status = f"{chart_type} | {len(rows)} rows | text"
                    chart_output = text_chart + data_comment
                    return Message(text=_wrap_chart_output(chart_output))
                return self._render_as_table(columns, rows, plan.get("title", "Data"))

            # ── Image mode: matplotlib → base64 JPEG ──
            fig = _render_chart_fig(plan, columns, rows, self.chart_style)
            b64_image, img_format = _chart_to_base64_optimized(fig)

            title = plan.get("title", "Chart")
            parts = []
            parts.append(f"![{title}](data:image/{img_format};base64,{b64_image})")
            parts.append(f"\n*{chart_type.replace('_', ' ').title()} chart — {len(rows)} data points*")

            annotations = plan.get("annotations", [])
            if annotations:
                parts.append(f"\n> {annotations[0]}")

            parts.append(data_comment)

            self.status = f"{chart_type} | {len(rows)} rows | image"
            return Message(text=_wrap_chart_output("\n".join(parts)))

        except Exception as e:
            self.status = f"Chart error: {e}"
            return self._render_as_table(columns, rows, plan.get("title", "Data"), error=str(e))

    # ─────────────────────────────────────────────────────────────────────
    # LLM HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _llm_plan_chart(self, columns, rows, user_request):
        """Ask the LLM to plan the optimal chart configuration."""
        # Build a compact data preview (first 10 rows)
        preview_rows = rows[:10]
        preview = f"Columns: {columns}\n"
        preview += f"Total rows: {len(rows)}\n"
        preview += f"Sample data (first {min(10, len(rows))} rows):\n"
        for row in preview_rows:
            preview += f"  {row}\n"

        # Detect numeric columns for context
        numeric_cols = []
        for i, col in enumerate(columns):
            if rows and isinstance(rows[0][i] if i < len(rows[0]) else None, (int, float)):
                numeric_cols.append(f"{col} (index {i})")
        if numeric_cols:
            preview += f"Numeric columns: {', '.join(numeric_cols)}\n"

        prompt = CHART_PLANNER_PROMPT.format(
            user_request=user_request,
            data_preview=preview,
        )

        # Default fallback plan
        fallback = {
            "chart_type": "bar",
            "title": "Data Visualization",
            "x_column": 0,
            "y_columns": [1] if len(columns) > 1 else [0],
            "x_label": columns[0] if columns else "",
            "y_label": columns[1] if len(columns) > 1 else "",
            "sort_by_value": True,
            "sort_descending": True,
            "top_n": None,
            "group_others": False,
            "annotations": [],
        }

        try:
            response = self._llm_invoke_silent(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            plan = json.loads(text.strip())
            # Validate critical fields
            if "chart_type" not in plan:
                plan["chart_type"] = fallback["chart_type"]
            if "x_column" not in plan:
                plan["x_column"] = fallback["x_column"]
            if "y_columns" not in plan:
                plan["y_columns"] = fallback["y_columns"]

            # ── Auto-fix swapped axes ──
            # For bar/bar_horizontal: x_column should be text, y_columns should be numeric
            plan = self._validate_and_fix_plan(plan, columns, rows)
            return plan
        except Exception:
            return fallback

    def _auto_plan_chart(self, columns, rows, user_request=""):
        """Smart heuristic chart planning — no LLM needed. Instant.

        Detects the best chart type from data shape and user keywords.
        """
        num_rows = len(rows)
        num_cols = len(columns)
        req_lower = user_request.lower()

        # Detect column types
        text_cols = []
        numeric_cols = []
        for i, col in enumerate(columns):
            is_numeric = False
            for row in rows[:10]:
                if i < len(row) and row[i] is not None:
                    is_numeric = isinstance(row[i], (int, float))
                    break
            if is_numeric:
                numeric_cols.append(i)
            else:
                text_cols.append(i)

        # Default axes: first text column = x, first numeric column = y
        x_col = text_cols[0] if text_cols else 0
        y_cols = numeric_cols if numeric_cols else [1] if num_cols > 1 else [0]

        # Detect chart type from user keywords. Word-boundary match so
        # "barometer" doesn't trigger bar, and so on. Phrases with spaces still
        # use substring check since \b doesn't help across multiple words.
        def _has_word(phrases):
            for ph in phrases:
                if " " in ph:
                    if ph in req_lower:
                        return True
                else:
                    if re.search(rf"\b{re.escape(ph)}\b", req_lower):
                        return True
            return False

        chart_type = "bar"  # default
        # Compound modifiers first so "horizontal bar" wins over "bar".
        if _has_word(("horizontal", "bar_horizontal")):
            chart_type = "bar_horizontal"
        elif _has_word(("stacked", "composition", "breakdown")):
            chart_type = "stacked_bar"
        elif _has_word(("pie", "donut", "proportion", "share")):
            chart_type = "pie" if num_rows <= 10 else "bar"
        elif _has_word(("line", "trend", "over time", "evolution", "progression")):
            chart_type = "line"
        elif _has_word(("scatter", "correlation", "vs")):
            chart_type = "scatter"
        elif num_rows > 12:
            chart_type = "bar_horizontal"  # too many items for vertical bars

        # Generate a clean title
        if num_cols == 2:
            title = f"{columns[numeric_cols[0]] if numeric_cols else columns[1]} by {columns[text_cols[0]] if text_cols else columns[0]}"
        else:
            title = "Data Visualization"

        # Clean up title (remove underscores, add spaces)
        title = title.replace("_", " ").title()

        return {
            "chart_type": chart_type,
            "title": title,
            "x_column": x_col,
            "y_columns": y_cols,
            "x_label": columns[x_col].replace("_", " ").title() if x_col < len(columns) else "",
            "y_label": columns[y_cols[0]].replace("_", " ").title() if y_cols and y_cols[0] < len(columns) else "",
            "sort_by_value": chart_type in ("bar", "bar_horizontal"),
            "sort_descending": True,
            "top_n": None,
            "group_others": False,
            "annotations": [],
        }

    def _validate_and_fix_plan(self, plan, columns, rows):
        """Auto-fix common LLM mistakes in chart plans.

        Fixes:
        1. Clamp out-of-bounds column indices from the LLM or auto planner so
           rendering never raises IndexError — the prior auto-plan path had no
           bounds check and could emit x_column=10 when only 3 columns exist.
        2. Swap x/y if x points at a numeric column and y at a text one (bar
           charts need categories on x, values on y).
        """
        if not rows or not columns:
            return plan

        n = len(columns)
        # Clamp x_column and y_columns to valid range for any chart type. Cheap
        # and always safe since the renderer will index into these.
        x_col = plan.get("x_column", 0)
        if not isinstance(x_col, int) or x_col < 0 or x_col >= n:
            x_col = 0
        y_cols = plan.get("y_columns", [1])
        if not isinstance(y_cols, list):
            y_cols = [y_cols] if isinstance(y_cols, int) else [1]
        y_cols = [y for y in y_cols if isinstance(y, int) and 0 <= y < n]
        if not y_cols:
            y_cols = [1 if n > 1 else 0]
        plan["x_column"] = x_col
        plan["y_columns"] = y_cols

        chart_type = plan.get("chart_type", "bar")
        if chart_type not in ("bar", "bar_horizontal", "stacked_bar"):
            return plan

        # Detect column types from actual data (check first non-None value)
        def _is_numeric_col(col_idx):
            for row in rows[:10]:
                if col_idx < len(row) and row[col_idx] is not None:
                    return isinstance(row[col_idx], (int, float))
            return False

        x_is_numeric = _is_numeric_col(x_col)
        y_all_text = all(not _is_numeric_col(y) for y in y_cols)

        # If x is numeric and y is text → axes are swapped
        if x_is_numeric and y_all_text and len(y_cols) == 1:
            plan["x_column"], plan["y_columns"] = y_cols[0], [x_col]
            # Also swap labels if present
            x_label = plan.get("x_label", "")
            y_label = plan.get("y_label", "")
            if x_label and y_label:
                plan["x_label"] = y_label
                plan["y_label"] = x_label

        # If x is numeric and y is also numeric, find the first text column for x
        if x_is_numeric and not y_all_text:
            for i in range(len(columns)):
                if not _is_numeric_col(i):
                    plan["x_column"] = i
                    break

        return plan

    def _llm_extract_data(self, content, user_request):
        """Ask the LLM to extract structured data from unstructured content."""
        # Truncate very long content
        content_preview = content[:3000] if len(content) > 3000 else content

        prompt = DATA_EXTRACTOR_PROMPT.format(
            content=content_preview,
            user_request=user_request,
        )

        try:
            response = self._llm_invoke_silent(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text.strip())
            if data.get("columns") and data.get("rows"):
                return data
        except Exception:
            pass
        return None

    def _render_as_table(self, columns, rows, title="Data", error=None):
        """Fallback: render as markdown table."""
        parts = []
        if error:
            parts.append(f"*Chart generation failed ({error}). Showing data table:*\n")
        parts.append(f"**{title}**\n")
        header = "| " + " | ".join(str(c) for c in columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        row_lines = []
        for row in rows[:50]:
            cells = []
            for v in row:
                if v is None:
                    cells.append("—")
                elif isinstance(v, float):
                    cells.append(f"{v:,.2f}")
                elif isinstance(v, int) and abs(v) >= 1000:
                    cells.append(f"{v:,}")
                else:
                    cells.append(str(v)[:50])
            row_lines.append("| " + " | ".join(cells) + " |")
        parts.append("\n".join([header, sep] + row_lines))
        if len(rows) > 50:
            parts.append(f"\n*Showing 50 of {len(rows)} rows*")
        self.status = f"Table: {len(rows)} rows"
        return Message(text="\n".join(parts))