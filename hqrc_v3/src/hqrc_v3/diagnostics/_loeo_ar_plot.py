"""Private deterministic SVG renderer for nine-event LOEO ACF/PACF review."""

from __future__ import annotations

import html

from hqrc_v3.diagnostics.ar import EventARDiagnostic


def _polyline(values: tuple[float, ...], x: int, y: int, width: int, height: int) -> str:
    if not values:
        return ""
    points = []
    for index, value in enumerate(values):
        px = x + index * width / max(1, len(values) - 1)
        py = y + height / 2 - value * (height * 0.43)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def render_loeo_ar_svg(held_out: str, diagnostics: tuple[EventARDiagnostic, ...]) -> bytes:
    """Render a byte-deterministic 9x3 raw/detrended/innovation diagnostic grid."""

    width = 1_560
    row_height = 142
    top = 76
    height = top + row_height * len(diagnostics) + 32
    panel_width = 450
    panel_height = 92
    starts = (170, 645, 1_120)
    labels = ("raw ACF/PACF", "detrended ACF/PACF", "innovation ACF/PACF")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:monospace;font-size:12px}.title{font-size:17px;font-weight:bold}"
        ".warn{fill:#a00}.ok{fill:#075}.acf{fill:none;stroke:#1261a0;stroke-width:1.2}"
        ".pacf{fill:none;stroke:#d04a00;stroke-width:1.2}"
        ".bound{fill:none;stroke:#777;stroke-width:.7;stroke-dasharray:3 2}"
        ".pacf-bound{stroke:#087;stroke-width:.7;stroke-dasharray:4 2}"
        ".axis{stroke:#aaa;stroke-width:.7}</style>",
        f'<text class="title" x="18" y="28">LOEO AR(1) review: held out '
        f"{html.escape(held_out)}</text>",
        '<text x="18" y="50">Blue=ACF, orange=PACF; all lag pairs reset per occurrence.</text>',
    ]
    for x, label in zip(starts, labels, strict=True):
        lines.append(f'<text x="{x}" y="68">{label}</text>')
    for row, diagnostic in enumerate(diagnostics):
        y = top + row * row_height
        warning_class = "warn" if diagnostic.diagnostic_warning else "ok"
        lines.append(
            f'<text x="8" y="{y + 18}" class="{warning_class}">'
            f"{html.escape(diagnostic.occurrence_id)} warning="
            f"{str(diagnostic.diagnostic_warning).lower()}</text>"
        )
        lines.append(
            f'<text x="8" y="{y + 36}">phi={diagnostic.phi:.6f} '
            f"LB({diagnostic.ljung_box_lag}) p={diagnostic.ljung_box_pvalue:.6g}</text>"
        )
        series = (
            (
                diagnostic.raw_acf,
                diagnostic.raw_pacf,
                diagnostic.raw_acf_lower,
                diagnostic.raw_acf_upper,
            ),
            (
                diagnostic.detrended_acf,
                diagnostic.detrended_pacf,
                diagnostic.detrended_acf_lower,
                diagnostic.detrended_acf_upper,
            ),
            (
                diagnostic.innovation_acf,
                diagnostic.innovation_pacf,
                diagnostic.innovation_acf_lower,
                diagnostic.innovation_acf_upper,
            ),
        )
        for x, (acf_values, pacf_values, lower, upper) in zip(starts, series, strict=True):
            lines.append(
                f'<rect x="{x}" y="{y}" width="{panel_width}" height="{panel_height}" '
                'fill="none" stroke="#bbb"/>'
            )
            lines.append(
                f'<line class="axis" x1="{x}" y1="{y + panel_height / 2:.1f}" '
                f'x2="{x + panel_width}" y2="{y + panel_height / 2:.1f}"/>'
            )
            acf_points = _polyline(acf_values, x, y, panel_width, panel_height)
            pacf_points = _polyline(pacf_values, x, y, panel_width, panel_height)
            lower_points = _polyline(lower, x, y, panel_width, panel_height)
            upper_points = _polyline(upper, x, y, panel_width, panel_height)
            reference_offset = diagnostic.pacf_reference_half_width * panel_height * 0.43
            center = y + panel_height / 2
            lines.append(f'<polyline class="bound" points="{lower_points}"/>')
            lines.append(f'<polyline class="bound" points="{upper_points}"/>')
            for reference_y in (center - reference_offset, center + reference_offset):
                lines.append(
                    f'<line class="pacf-bound" x1="{x}" y1="{reference_y:.2f}" '
                    f'x2="{x + panel_width}" y2="{reference_y:.2f}"/>'
                )
            lines.append(f'<polyline class="acf" points="{acf_points}"/>')
            lines.append(f'<polyline class="pacf" points="{pacf_points}"/>')
            lines.append(f'<text x="{x}" y="{y + panel_height + 14}">lag 1</text>')
            lines.append(
                f'<text x="{x + panel_width - 42}" y="{y + panel_height + 14}">lag 48</text>'
            )
    lines.append("</svg>")
    return ("\n".join(lines) + "\n").encode()


__all__: list[str] = []
