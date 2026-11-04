"""
Paged, ranked views of the simulation archive for the "Previous Simulations Archive" tab.

The tab used to download every archived run with all its nested detail (about 3.4 MB for 340 runs) and rank and filter
them in the browser to show a few dozen rows. Here the server does that work and sends one page of slim rows:

* ``set`` picks the view: the curated "best 15" sets (N >= 12, ranked by F1, then total errors, then power - the same
  ordering the page always used), ``overall_top15``, or ``everything`` (every run, newest first);
* ``sort`` / ``order`` sort the whole view (not just the visible page), ``q`` keeps the runs whose id contains the text;
* ``page`` / ``page_size`` slice the result.

Building the ranked sets costs a database read, so the result is cached against the archive file's version
(modification time and size) and shared by every page request.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from grn_sim.web import database as db

SETS = ("all", "everything", "responsive", "flat", "oscillating", "random", "power_law", "hierarchical", "modular",
        "overall_top15")
# Table column order of #tbl-history -> sort key (the two download / action columns are not sortable)
SORT_KEYS = ("rank", "run_id", "category", "topology", "genes", "k_av", "noise", "equilibrium", "method", "f1", "power",
             "specificity", "errors", "time")
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
CURATED_SIZE = 15              # runs kept per curated set
MIN_GENES = 12                 # smallest network a curated set considers
MAX_QUERY_LENGTH = 100

_DYNAMICS = (("responsive", "Responsive"), ("flat", "Flat Levels"), ("oscillating", "Oscillating"))
_TOPOLOGIES = (("random", "Random Topo"), ("power_law", "Scale-Free Topo"), ("hierarchical", "Hierarchical Topo"),
               ("modular", "Modular Topo"))
_CATEGORY_LABEL = {"oscillating": "〰 Oscillating", "flat": "⚪ Flat Levels", "responsive": "⚡ Responsive"}

# What the table needs of an archived run (the rest - dynamics, diagnostics, settings, per-algorithm summaries - stays
# behind /api/history/load)
_SLIM_FIELDS = ("run_id", "topology", "num_genes", "k_av", "noise", "oscillating", "steady_state", "best_algorithm",
                "best_alg_key", "best_power_pct", "best_specificity_pct", "best_total_errors", "f1_score", "timestamp")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _response_type(entry: Dict[str, Any]) -> str:
    return entry.get("response_type") or ("oscillating" if entry.get("oscillating") else "responsive")


def slim_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The columns of one archive row."""
    row = {k: entry.get(k) for k in _SLIM_FIELDS}
    row["response_type"] = _response_type(entry)
    row["topology"] = entry.get("topology") or "random"
    row["f1_score"] = _num(entry.get("f1_score"))
    row["rankInSet"] = None
    row["setName"] = "All runs"
    return row


def _rank(rows: List[Dict[str, Any]], set_name: str) -> List[Dict[str, Any]]:
    """Best ``CURATED_SIZE`` of ``rows`` among networks with at least ``MIN_GENES`` genes (stable for ties)."""
    eligible = [r for r in rows if _num(r.get("num_genes")) >= MIN_GENES]
    eligible.sort(key=lambda r: (-r["f1_score"], _num(r.get("best_total_errors"), 99.0), -_num(r.get("best_power_pct"))))
    return [dict(r, rankInSet=i + 1, setName=set_name) for i, r in enumerate(eligible[:CURATED_SIZE])]


class ArchiveView:
    """Slim rows of every archived run plus the ranked sets, built once per archive version."""

    def __init__(self, history: List[Dict[str, Any]]):
        self.history = history
        self.rows = [slim_row(h) for h in history]                      # archive order: newest first
        self.sets: Dict[str, List[Dict[str, Any]]] = {"everything": self.rows}
        for key, name in _DYNAMICS:
            self.sets[key] = _rank([r for r in self.rows if r["response_type"] == key], name)
        for key, name in _TOPOLOGIES:
            self.sets[key] = _rank([r for r in self.rows if r["topology"] == key], name)
        self.sets["overall_top15"] = _rank(self.rows, "Overall Best")
        self.sets["all"] = [r for key, _ in _DYNAMICS for r in self.sets[key]]
        self.summary: Optional[Dict[str, Any]] = None                   # filled on first use


def _sort_value(row: Dict[str, Any], key: str) -> Any:
    """The value a column sorts by - what the column shows, so a sorted page agrees with the browser's own sort."""
    if key == "rank":
        return row["rankInSet"]
    if key == "category":
        return _CATEGORY_LABEL.get(row["response_type"], row["response_type"])
    if key == "equilibrium":
        return "Oscillating" if row.get("oscillating") else "Steady State"
    return {
        "run_id": row["run_id"], "topology": str(row["topology"]), "genes": _num(row.get("num_genes")),
        "k_av": _num(row.get("k_av")), "noise": _num(row.get("noise")), "method": str(row.get("best_algorithm") or ""),
        "f1": row["f1_score"], "power": _num(row.get("best_power_pct")), "specificity": _num(row.get("best_specificity_pct")),
        "errors": _num(row.get("best_total_errors"), 99.0), "time": str(row.get("timestamp") or ""),
    }[key]


def parse_params(query: Dict[str, List[str]]) -> Dict[str, Any]:
    """Validates the query string of a paged request; raises ValueError naming the parameter."""
    def one(name: str, default: Any) -> Any:
        return query.get(name, [default])[0]

    set_name = one("set", "all")
    if set_name not in SETS:
        raise ValueError(f"set must be one of {', '.join(SETS)}, got {set_name!r}")
    try:
        page, page_size = int(one("page", 1)), int(one("page_size", DEFAULT_PAGE_SIZE))
    except (TypeError, ValueError):
        raise ValueError("page and page_size must be whole numbers") from None
    if page < 1:
        raise ValueError(f"page must be at least 1, got {page}")
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}, got {page_size}")
    sort = one("sort", "") or None
    if sort is not None and sort not in SORT_KEYS:
        raise ValueError(f"sort must be one of {', '.join(SORT_KEYS)}, got {sort!r}")
    order = one("order", "asc")
    if order not in ("asc", "desc"):
        raise ValueError(f"order must be asc or desc, got {order!r}")
    text = str(one("q", "")).strip()
    if len(text) > MAX_QUERY_LENGTH:
        raise ValueError(f"q must be at most {MAX_QUERY_LENGTH} characters")
    return {"set": set_name, "page": page, "page_size": page_size, "sort": sort, "order": order, "q": text}


def page_of(view: ArchiveView, params: Dict[str, Any]) -> Dict[str, Any]:
    """One page of ``view`` for validated ``params``: the rows and where they sit in the whole result."""
    rows = view.sets[params["set"]]
    if params["q"]:
        needle = params["q"].lower()
        rows = [r for r in rows if needle in str(r["run_id"]).lower()]
    if params["sort"]:
        key, desc = params["sort"], params["order"] == "desc"
        present = [r for r in rows if _sort_value(r, key) is not None]
        missing = [r for r in rows if _sort_value(r, key) is None]            # e.g. no rank in the unranked view: last
        present.sort(key=lambda r: _sort_value(r, key), reverse=desc)
        rows = present + missing
    total = len(rows)
    pages = max(1, -(-total // params["page_size"]))
    page = min(params["page"], pages)                                         # past the end: the last page
    start = (page - 1) * params["page_size"]
    return {
        "history": rows[start:start + params["page_size"]],
        "view": {**{k: params[k] for k in ("set", "page_size", "sort", "order", "q")}, "page": page, "total": total,
                 "total_pages": pages, "first_row": start + 1 if total else 0, "last_row": min(start + params["page_size"], total),
                 "runs_in_list": len(view.rows)},
    }


# ---- cache -----------------------------------------------------------------------------------------------------------
_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"key": None, "view": None}


def _archive_version() -> Tuple[str, int, int]:
    path = db.get_db_path()
    try:
        st = os.stat(path)
        return path, st.st_mtime_ns, st.st_size
    except OSError:
        return path, 0, 0


def get_view() -> ArchiveView:
    """The archive's view, rebuilt only when the archive file changed since the last call."""
    key = _archive_version()
    with _LOCK:
        if _CACHE["key"] != key or _CACHE["view"] is None:
            _CACHE["view"] = ArchiveView(db.load_all_history_from_db())
            _CACHE["key"] = key
        return _CACHE["view"]


def get_summary() -> Dict[str, Any]:
    """``get_summary_statistics_by_type()`` (about 2 s on a large archive), computed once per archive version."""
    view = get_view()
    with _LOCK:
        if view.summary is None:
            view.summary = db.get_summary_statistics_by_type()
        return view.summary
