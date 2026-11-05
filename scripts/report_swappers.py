"""Build a local HTML comparison from compare_swappers.py outputs."""
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "outputs/model-comparison"
    cards = []
    for path in sorted(folder.rglob("*.json")):
        if path.name == "summary.json":
            continue
        result = json.loads(path.read_text())
        if not isinstance(result, dict) or "model" not in result:
            continue
        image = path.with_suffix(".png").relative_to(folder).as_posix()
        score = re.search(r"reference similarity: ([0-9.]+)", result["status"])
        cards.append(f'<article><h2>{html.escape(result["model"])}</h2>'
                     f'<p>{html.escape(path.parent.relative_to(folder).as_posix())} · '
                     f'{result["seconds"]:.1f}s · similarity {score.group(1) if score else "unknown"}</p>'
                     f'<a href="{html.escape(image)}"><img loading="lazy" src="{html.escape(image)}"></a>'
                     f'<details><summary>Settings and evidence</summary><pre>{html.escape(json.dumps(result, indent=2))}</pre></details></article>')
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Visenlo local model comparisons</title><style>
body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 24px;background:#171917;color:#eee}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}
article{border:1px solid #555;padding:16px}img{width:100%;height:auto}h2{font-size:18px}pre{white-space:pre-wrap;font-size:12px}
</style><h1>Visenlo: local model comparison</h1>
<p>Actual saved outputs, using the same finishing settings within each comparison group. Click an image for full resolution.
Identity similarity is a diagnostic, not a visual-quality ranking. Different source/target hashes are different test pairs.
Native runs compare models; boosted runs also change inference resolution. Failed or missing models have no image card.
Timings from this session include concurrent CPU workloads and should not be treated as controlled speed benchmarks.</p>
<details><summary>Original local fixtures</summary><main>
<article><h2>First source</h2><img src="../../testdata/reference_female.jpg"></article>
<article><h2>First target (left face selected)</h2><img src="../../testdata/target_mixed.jpg"></article>
<article><h2>Second source</h2><img src="../../testdata/reference_male_download.jpg"></article>
<article><h2>Second target</h2><img src="../../testdata/target_male_only.jpg"></article>
</main></details><main>'''
    (folder / "index.html").write_text(page + "\n".join(cards) + "</main>\n")
    print(folder / "index.html")


if __name__ == "__main__":
    main()
