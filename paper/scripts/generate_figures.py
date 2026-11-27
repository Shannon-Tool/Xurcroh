#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "generated"


def rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "data" / name).open(newline="") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)


def main() -> None:
    text = rows("text-ladder.csv")
    coords = "{" + ",".join(row["rung"] for row in text) + "}"
    vram = "\n".join(f"({row['rung']},{row['vram_gb']})" for row in text)
    hosts = "\n".join(f"({row['rung']},{row['hosts']})" for row in text)
    quality = "\n".join(f"({row['rung']},{row['quality_delta_percent']})" for row in text)
    write(
        "text-resources.tex",
        r"""\begin{tikzpicture}
\begin{axis}[width=\columnwidth,height=6.1cm,ybar,bar width=13pt,symbolic x coords=""" + coords + r""",xtick=data,x tick label style={font=\scriptsize},ylabel={Peak VRAM per role (GB)},ymin=0,ymax=82,grid=major,grid style={gray!15},nodes near coords,nodes near coords style={font=\tiny},every axis plot/.append style={fill=xblue!80,draw=xblue}]
\addplot coordinates {
""" + vram + r"""
};
\end{axis}
\end{tikzpicture}""",
    )
    write(
        "text-fleet-quality.tex",
        r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=1 by 2,vertical sep=1.4cm},width=\columnwidth,height=4.8cm,symbolic x coords=""" + coords + r""",xtick=data,x tick label style={font=\scriptsize},grid=major,grid style={gray!15}]
\nextgroupplot[ybar,bar width=13pt,ylabel={Independent hosts},ymin=0,ymax=135,nodes near coords,nodes near coords style={font=\tiny}]
\addplot[fill=xteal!75,draw=xteal] coordinates {
""" + hosts + r"""
};
\nextgroupplot[ylabel={Perplexity delta (\%)},ymin=0,ymax=4.8,ytick={0,1,2,3,4},mark=*,mark size=1.8pt]
\addplot[xorange,thick] coordinates {
""" + quality + r"""
};
\end{groupplot}
\end{tikzpicture}""",
    )
    media = rows("media-ladder.csv")
    for modality, metric, filename, color, ymin, ymax in (
        ("image", "PSNR (dB)", "image-quality.tex", "xblue", 40, 45),
        ("video", "VMAF", "video-quality.tex", "xteal", 92, 98),
    ):
        selected = [row for row in media if row["modality"] == modality]
        labels = "{" + ",".join(row["rung"] for row in selected) + "}"
        values = "\n".join(f"({row['rung']},{row['quality_value']})" for row in selected)
        write(
            filename,
            r"""\begin{tikzpicture}
\begin{axis}[width=\columnwidth,height=5.2cm,ybar,bar width=18pt,symbolic x coords=""" + labels + r""",xtick=data,ylabel={""" + metric + r"""},ymin=""" + str(ymin) + r""",ymax=""" + str(ymax) + r""",grid=major,grid style={gray!15},nodes near coords,nodes near coords style={font=\scriptsize}]
\addplot[fill=""" + color + r"""!75,draw=""" + color + r"""] coordinates {
""" + values + r"""
};
\end{axis}
\end{tikzpicture}""",
        )
    costs = rows("validation-costs.csv")
    labels = "{" + ",".join("{" + row["category"] + "}" for row in reversed(costs)) + "}"
    values = "\n".join(f"({row['cost_usd']},{{{row['category']}}}) [{row['cost_usd']}]" for row in costs)
    write(
        "validation-costs.tex",
        r"""\begin{tikzpicture}
\begin{axis}[width=\columnwidth,height=7.4cm,xbar,xmode=log,log basis x=10,xlabel={Counted validation cost (USD, log scale)},xmin=30,xmax=40000,symbolic y coords=""" + labels + r""",ytick=data,y tick label style={font=\scriptsize},grid=major,grid style={gray!15},nodes near coords,nodes near coords style={font=\tiny,/pgf/number format/fixed},point meta=explicit]
\addplot[fill=xblue!75,draw=xblue] coordinates {
""" + values + r"""
};
\end{axis}
\end{tikzpicture}""",
    )
    print(f"Generated 5 PGFPlots sources in {OUT}")


if __name__ == "__main__":
    main()
