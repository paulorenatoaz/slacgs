import argparse
import datetime as dt
import html
from pathlib import Path


def discover_scenarios(reports_root: Path):
    items = []
    if not reports_root.exists():
        return []
    for p in reports_root.rglob("*.html"):
        name = p.name
        if name.startswith("scenario_") and name.endswith("_report.html"):
            items.append(p)
    items.sort(key=lambda x: x.name)
    return [x.relative_to(reports_root) for x in items]


def discover_json(data_root: Path):
    files = []
    if not data_root.exists():
        return files
    for p in data_root.rglob("*.json"):
        files.append(p.relative_to(data_root))
    files.sort(key=lambda x: x.name)
    return files


def write_index(site_dir: Path, reports_rel: Path, data_rel: Path, scenarios, json_files, title: str):
    site_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    def list_html(items):
        return "\n".join(
            f'<li><a href="{html.escape(str(reports_rel / p))}">{html.escape(p.name)}</a></li>'
            for p in items
        )

    def list_json(items):
        return "\n".join(
            f'<li><a href="{html.escape(str(data_rel / p))}">{html.escape(p.name)}</a></li>'
            for p in items
        )

    index_html = f"""<!doctype html>
<html lang=\"en\"><head>
<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#0b1220; --fg:#e9eef6; --muted:#a8b0bf; --card:#121a2a; --link:#7cc4ff; }}
html,body {{ margin:0; padding:0; background:var(--bg); color:var(--fg); font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial; }}
.wrap {{ max-width:1100px; margin:0 auto; padding:24px; }}
h1 {{ font-size:1.8rem; margin:0 0 8px; }}
.muted {{ color:var(--muted); font-size:.9rem; margin-bottom:16px; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(320px,1fr)); gap:16px; }}
.card {{ background:var(--card); border:1px solid #1f293b; border-radius:10px; padding:16px; }}
.card h2 {{ margin:0 0 8px; font-size:1.2rem; }}
a {{ color:var(--link); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
ul {{ margin:0; padding-left:18px; max-height:420px; overflow:auto; }}
.counts {{ display:flex; gap:16px; margin:8px 0 16px; color:var(--muted); }}
.counts span {{ background:#0e1728; border:1px solid #1f293b; padding:6px 10px; border-radius:999px; }}
footer {{ margin-top:28px; color:var(--muted); font-size:.85rem; }}
</style></head><body><div class=\"wrap\">
<h1>{html.escape(title)}</h1>
<div class=\"muted\">Updated {ts}</div>
<div class=\"counts\">
  <span>Scenario reports: {len(scenarios)}</span>
  <span>JSON files: {len(json_files)}</span>
  </div>
<div class=\"grid\">
<section class=\"card\"><h2>Scenario Reports</h2><ul>{list_html(scenarios)}</ul></section>
<section class=\"card\"><h2>Data (JSON)</h2><ul>{list_json(json_files)}</ul></section>
</div>
<footer>Served from <code>{html.escape(str(reports_rel))}</code> and <code>{html.escape(str(data_rel))}</code> on gh-pages.</footer>
</div></body></html>"""

    (site_dir / "index.html").write_text(index_html, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write index.html linking scenario reports and JSON data.")
    ap.add_argument("--reports-dir", default="reports", help="Path to reports root (on gh-pages)")
    ap.add_argument("--data-dir", default="data", help="Path to JSON root (on gh-pages)")
    ap.add_argument("--site-dir", default=".", help="Where to write index.html (gh-pages root)")
    ap.add_argument("--title", default="slacgs Reports", help="Index title")
    args = ap.parse_args(argv)

    site_dir = Path(args.site_dir).resolve()
    reports_dir = (site_dir / args.reports_dir).resolve()
    data_dir = (site_dir / args.data_dir).resolve()

    scenarios = discover_scenarios(reports_dir)
    json_files = discover_json(data_dir)
    write_index(site_dir, Path(args.reports_dir), Path(args.data_dir), scenarios, json_files, title=args.title)
    print(f"Wrote {site_dir / 'index.html'}")


if __name__ == "__main__":
    main()

