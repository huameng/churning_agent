"""
Web UI for managing eval/cases.json.

Run from workspace/:
    uv run python -m churning_agent.eval.case_manager
Then open http://localhost:5000
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Flask, jsonify, redirect, render_template_string, request, url_for

_CASES_PATH = Path(__file__).parent / "cases.json"
_FAILURES_PATH = Path(__file__).parent / "failures.json"

LABELS = ["IRRELEVANT", "MONEYMAKER", "DISCOUNT_MONEYMAKER", "WORTHLESS", "UNCERTAIN"]
LABEL_COLOR = {
    "IRRELEVANT":          "#6c757d",
    "MONEYMAKER":          "#198754",
    "DISCOUNT_MONEYMAKER": "#0d6efd",
    "WORTHLESS":           "#dc3545",
    "UNCERTAIN":           "#fd7e14",
}

app = Flask(__name__)


def _load_cases() -> list[dict]:
    return json.loads(_CASES_PATH.read_text(encoding="utf-8"))


def _save_cases(cases: list[dict]) -> None:
    _CASES_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")


def _failing_urls() -> set[str]:
    if not _FAILURES_PATH.exists():
        return set()
    failures = json.loads(_FAILURES_PATH.read_text(encoding="utf-8"))
    return {f["url"] for f in failures}


TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Case Manager</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; font-size: 14px; background: #f8f9fa; color: #212529; }
  header { background: #212529; color: #fff; padding: 12px 20px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; font-weight: 600; }
  header .stats { font-size: 12px; color: #adb5bd; }
  .filters { padding: 12px 20px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; border-bottom: 1px solid #dee2e6; background: #fff; }
  .filters a { padding: 4px 12px; border-radius: 20px; text-decoration: none; font-size: 12px; font-weight: 500; border: 1px solid #dee2e6; color: #495057; background: #fff; }
  .filters a.active { color: #fff; border-color: transparent; }
  .filters .search { margin-left: auto; padding: 4px 10px; border: 1px solid #dee2e6; border-radius: 4px; font-size: 12px; width: 220px; }
  table { width: 100%; border-collapse: collapse; }
  th { background: #fff; padding: 8px 12px; text-align: left; font-size: 11px; text-transform: uppercase; color: #6c757d; border-bottom: 2px solid #dee2e6; position: sticky; top: 0; }
  td { padding: 8px 12px; vertical-align: top; border-bottom: 1px solid #f1f3f5; }
  tr:hover td { background: #f8f9fa; }
  tr.failing td { background: #fff3cd; }
  tr.failing:hover td { background: #ffeeba; }
  .title-cell { max-width: 360px; }
  .reasoning-cell { min-width: 300px; }
  .title-link { color: #0d6efd; text-decoration: none; font-weight: 500; }
  .title-link:hover { text-decoration: underline; }
  .expandable { font-size: 12px; color: #6c757d; margin-top: 4px; max-height: 0; overflow: hidden; transition: max-height 0.2s; }
  .expandable.open { max-height: 300px; overflow-y: auto; }
  .toggle-btn { font-size: 11px; color: #0d6efd; cursor: pointer; background: none; border: none; padding: 0; margin-top: 2px; }
  .label-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; color: #fff; font-size: 11px; font-weight: 600; }
  select.label-select { border: 1px solid #dee2e6; border-radius: 4px; padding: 2px 4px; font-size: 12px; cursor: pointer; }
  .value-cell { white-space: nowrap; color: #198754; font-weight: 500; }
  .del-btn { background: none; border: none; cursor: pointer; color: #dc3545; font-size: 16px; padding: 0 4px; opacity: 0.5; }
  .del-btn:hover { opacity: 1; }
  .content-snippet { font-size: 11px; color: #6c757d; font-family: monospace; white-space: pre-wrap; border-left: 3px solid #dee2e6; padding-left: 8px; }
  .got-badge { margin-left: 4px; }
  .empty { padding: 40px; text-align: center; color: #6c757d; }
  .count-badge { font-size: 11px; background: #e9ecef; color: #495057; border-radius: 10px; padding: 1px 6px; margin-left: 4px; }

  /* Reasoning section */
  .reasoning-block { margin-top: 6px; }
  .reasoning-label { font-size: 10px; font-weight: 600; text-transform: uppercase; color: #adb5bd; margin-bottom: 2px; }
  .model-text { font-size: 12px; color: #6c757d; font-style: italic; }
  .human-textarea {
    width: 100%; font-size: 12px; font-family: system-ui, sans-serif;
    border: 1px dashed #dee2e6; border-radius: 4px; padding: 4px 6px;
    color: #212529; background: transparent; resize: vertical; min-height: 36px;
    transition: border-color 0.15s, background 0.15s;
  }
  .human-textarea:focus { outline: none; border-color: #0d6efd; background: #fff; }
  .human-textarea.saved { border-color: #198754; }
  .human-textarea:empty:not(:focus)::placeholder { color: #adb5bd; }
</style>
</head>
<body>

<header>
  <h1>Case Manager</h1>
  <span class="stats">
    {{ cases|length }} shown &nbsp;·&nbsp;
    {% for lbl in labels %}
      <span style="color:{{ label_color[lbl] }}">{{ lbl[0] }}</span>{{ counts[lbl] }}
      {% if not loop.last %} &nbsp; {% endif %}
    {% endfor %}
  </span>
</header>

<div class="filters">
  <a href="{{ url_for('index', q=q) }}" class="{{ 'active' if not filter_label else '' }}" style="{{ 'background:#212529' if not filter_label else '' }}">All <span class="count-badge">{{ total }}</span></a>
  {% for lbl in labels %}
  <a href="{{ url_for('index', label=lbl, q=q) }}"
     class="{{ 'active' if filter_label == lbl else '' }}"
     style="{{ 'background:' ~ label_color[lbl] if filter_label == lbl else '' }}">
    {{ lbl }} <span class="count-badge">{{ counts[lbl] }}</span>
  </a>
  {% endfor %}
  <a href="{{ url_for('index', label='FAILING', q=q) }}"
     class="{{ 'active' if filter_label == 'FAILING' else '' }}"
     style="{{ 'background:#856404;color:#fff' if filter_label == 'FAILING' else 'color:#856404' }}">
    ⚠ Failing <span class="count-badge">{{ failing_count }}</span>
  </a>
  <input class="search" type="text" placeholder="Search titles…" id="search-box" value="{{ q or '' }}">
</div>

<table>
  <thead>
    <tr>
      <th style="width:28%">Title</th>
      <th style="width:38%">Reasoning</th>
      <th>Expected</th>
      <th>Last Run</th>
      <th>$</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  {% if cases %}
    {% for case in cases %}
    <tr class="{{ 'failing' if case.url in failing_urls else '' }}" data-title="{{ case.title|lower }}">
      <td class="title-cell">
        <a class="title-link" href="{{ case.url }}" target="_blank">{{ case.title }}</a>
        <br>
        <button class="toggle-btn" onclick="toggleExpand(this, 'content')">content ▾</button>
        <div class="expandable">
          <div class="content-snippet">{{ case.content or '' }}</div>
        </div>
      </td>

      <td class="reasoning-cell">
        <div class="reasoning-block">
          <div class="reasoning-label">Model</div>
          <div class="model-text">{{ case.model_reasoning or '—' }}</div>
        </div>
        <div class="reasoning-block" style="margin-top:8px">
          <div class="reasoning-label">Human</div>
          <textarea
            class="human-textarea"
            data-url="{{ case.url }}"
            placeholder="Why is this label correct? (saved automatically)"
          >{{ case.human_reasoning or '' }}</textarea>
        </div>
      </td>

      <td>
        <select class="label-select" data-url="{{ case.url }}"
                style="color:{{ label_color.get(case.expected, '#000') }}">
          {% for lbl in labels %}
          <option value="{{ lbl }}" {{ 'selected' if case.expected == lbl else '' }}
                  style="color:{{ label_color[lbl] }}">{{ lbl }}</option>
          {% endfor %}
        </select>
      </td>

      <td>
        {% if case.url in failures_by_url %}
          {% set f = failures_by_url[case.url] %}
          <span class="label-badge got-badge" style="background:{{ label_color.get(f.got, '#000') }}">{{ f.got }}</span>
        {% else %}
          <span style="color:#adb5bd">—</span>
        {% endif %}
      </td>

      <td class="value-cell">
        {% if case.estimated_value %}${{ "%.0f"|format(case.estimated_value) }}{% else %}—{% endif %}
      </td>

      <td>
        <form method="post" action="{{ url_for('delete_case') }}" onsubmit="return confirm('Delete this case?')">
          <input type="hidden" name="url" value="{{ case.url }}">
          <button class="del-btn" type="submit" title="Delete">✕</button>
        </form>
      </td>
    </tr>
    {% endfor %}
  {% else %}
    <tr><td colspan="6" class="empty">No cases match.</td></tr>
  {% endif %}
  </tbody>
</table>

<script>
const labelColors = {{ label_color | tojson }};

function toggleExpand(btn, _type) {
  const div = btn.nextElementSibling;
  div.classList.toggle('open');
  btn.textContent = div.classList.contains('open') ? 'content ▴' : 'content ▾';
}

// Live search
document.getElementById('search-box').addEventListener('input', function() {
  const q = this.value.toLowerCase();
  const url = new URL(window.location);
  if (q) url.searchParams.set('q', q); else url.searchParams.delete('q');
  window.history.replaceState({}, '', url);
  document.querySelectorAll('tbody tr[data-title]').forEach(row => {
    row.style.display = row.dataset.title.includes(q) ? '' : 'none';
  });
});

// Save human_reasoning on blur via fetch
document.addEventListener('blur', function(e) {
  const ta = e.target;
  if (!ta.classList.contains('human-textarea')) return;
  const url = ta.dataset.url;
  const text = ta.value.trim();
  fetch('/update_reasoning', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url, human_reasoning: text || null}),
  }).then(r => {
    if (r.ok) {
      ta.classList.add('saved');
      setTimeout(() => ta.classList.remove('saved'), 1200);
    }
  });
}, true);

// Save label on change via fetch (no page reload)
document.addEventListener('change', function(e) {
  const sel = e.target;
  if (!sel.classList.contains('label-select')) return;
  const url = sel.dataset.url;
  const label = sel.value;
  sel.style.color = labelColors[label] || '#000';
  fetch('/update', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url, label}),
  }).then(r => {
    if (r.ok) {
      const td = sel.closest('td');
      td.style.transition = 'background 0.1s';
      td.style.background = '#d1e7dd';
      setTimeout(() => td.style.background = '', 800);
    }
  });
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    cases = _load_cases()
    failing_urls = _failing_urls()
    filter_label = request.args.get("label", "")
    q = request.args.get("q", "").lower()

    counts = {lbl: sum(1 for c in cases if c.get("expected") == lbl) for lbl in LABELS}
    total = len(cases)
    failing_count = len(failing_urls)

    failures_by_url: dict[str, dict] = {}
    if _FAILURES_PATH.exists():
        for f in json.loads(_FAILURES_PATH.read_text(encoding="utf-8")):
            failures_by_url[f["url"]] = f

    if filter_label == "FAILING":
        cases = [c for c in cases if c.get("url") in failing_urls]
    elif filter_label in LABELS:
        cases = [c for c in cases if c.get("expected") == filter_label]

    if q:
        cases = [c for c in cases if q in c.get("title", "").lower()]

    return render_template_string(
        TEMPLATE,
        cases=cases,
        labels=LABELS,
        label_color=LABEL_COLOR,
        counts=counts,
        total=total,
        filter_label=filter_label,
        failing_urls=failing_urls,
        failing_count=failing_count,
        failures_by_url=failures_by_url,
        q=request.args.get("q", ""),
    )


@app.route("/update", methods=["POST"])
def update_label():
    data = request.get_json()
    url = data["url"]
    new_label = data["label"]
    if new_label not in LABELS:
        return jsonify({"error": "Invalid label"}), 400

    cases = _load_cases()
    for case in cases:
        if case["url"] == url:
            case["expected"] = new_label
            break
    _save_cases(cases)

    return jsonify({"ok": True})


@app.route("/update_reasoning", methods=["POST"])
def update_reasoning():
    data = request.get_json()
    url = data.get("url")
    human_reasoning = data.get("human_reasoning")  # None or non-empty string

    cases = _load_cases()
    for case in cases:
        if case["url"] == url:
            case["human_reasoning"] = human_reasoning
            break
    _save_cases(cases)

    return jsonify({"ok": True})


@app.route("/delete", methods=["POST"])
def delete_case():
    url = request.form["url"]
    cases = _load_cases()
    cases = [c for c in cases if c["url"] != url]
    _save_cases(cases)
    return redirect(request.referrer or url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
