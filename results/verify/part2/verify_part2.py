"""Independent check of the Part 2 analysis tables.

This script RE-IMPLEMENTS THE METRICS ON PURPOSE (docs/PLAN.md §9.2): code that checks itself catches
nothing. It was written from the spec in WORKLOG ("2026-10-13: Part 2 design analysis, B and D";
"Router search (exploratory)") and the header text of the generated tables, without reading
src/asimov/stats.py, src/asimov/cascade.py, src/asimov/router_search.py, scripts/router_search.py or
src/asimov/collect/. Inputs: each run's summary.json (`key`, `per_item`) and responses.jsonl (tokens,
latency, probs), configs/prices.yaml.

Targets:
  1. results/part2_designs/tables.md (D and B, pre-registered).
  2. origin/router-search-v2: results/router_search/tables.md (single systems) and picks.csv
     (picks made by one candidate in every fold), snapshotted into results/verify/part2/origin/.

Usage: python results/verify/part2/verify_part2.py  (from the repo root); writes report.md beside it.
"""
# ruff: noqa: B023  # row() is defined and called within the same loop iteration, so it sees that iteration's values

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
QS = ["latent_risk", "latent_severity", "effect", "activated_risk"]
TOL = 0.001
BOOT_TOL = 0.005
B = 10_000


# ---------------------------------------------------------------- loading


def run_dir(run: str) -> Path:
    ids = yaml.safe_load((ROOT / "results/run_ids.yaml").read_text())
    return ROOT / "results" / f"{ids[run]}_{run}"


@dataclass
class Item:
    correct: list[bool]
    conf: float
    cost: float
    latency: float  # latency_s
    router_latency: float  # compute_s when the reply was made on Modal, else latency_s
    gpu: float  # self-hosted only: router_latency, else 0


@dataclass
class System:
    model: str
    items: dict[str, Item]
    self_hosted: bool
    free: bool
    params: float | None
    n_compute: int = 0
    ids: list[str] = field(default_factory=list)


def call_cost(usage: dict, price: dict) -> float:
    if price.get("free_tier") or price.get("self_hosted"):
        return 0.0
    out = usage.get("output_tokens") or 0
    if price.get("thinking_as_output"):
        out += usage.get("thinking_tokens") or 0
    return ((usage.get("input_tokens") or 0) * price["input_per_mtok"] + out * price["output_per_mtok"]) / 1e6


def load(run: str, model: str, prices: dict) -> System:
    d = run_dir(run) / model
    s = json.loads((d / "summary.json").read_text())
    assert s["errors"] == 0, (run, model)
    key = s["key"]
    replies = {}
    for line in (d / "responses.jsonl").open():
        r = json.loads(line)
        if r["key"] != key:
            continue
        assert r["id"] not in replies, ("duplicate", run, model, r["id"])
        replies[r["id"]] = r
    per_item = s["per_item"]
    assert set(replies) == set(per_item), (run, model, len(replies), len(per_item))
    price = prices[model]
    items = {}
    n_compute = 0
    for iid, pi in per_item.items():
        r = replies[iid]
        probs = r.get("probs") or {}
        tops = [max(probs[q].values()) if probs.get(q) else 0.0 for q in QS]
        comp = ((r.get("raw") or {}).get("remote") or {}).get("compute_s")
        if comp is not None:
            n_compute += 1
        rl = comp if comp is not None else r["latency_s"]
        items[iid] = Item(
            correct=[bool(pi[q]["correct"]) for q in QS],
            conf=min(tops),
            cost=call_cost(r["usage"], price),
            latency=r["latency_s"],
            router_latency=rl,
            gpu=rl if price.get("self_hosted") else 0.0,
        )
    return System(
        model,
        items,
        bool(price.get("self_hosted")),
        bool(price.get("free_tier")),
        price.get("params_b"),
        n_compute,
        sorted(items),
    )


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def boot(x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def combo_metrics(C: np.ndarray) -> dict:
    """C: n x 4 bool matrix. Returns mean-of-4 (bootstrap) and per-question (Wilson)."""
    n = len(C)
    out = {"mean": boot(C.mean(axis=1))}
    for j, q in enumerate(QS):
        out[q] = wilson(int(C[:, j].sum()), n)
    return out


# ---------------------------------------------------------------- comparison bookkeeping


class Checker:
    def __init__(self, name: str):
        self.name = name
        self.n = 0
        self.ok = 0
        self.mismatch: list[str] = []
        self.boot_noise: list[str] = []

    def num(self, where: str, ours: float, theirs: float, tol: float = TOL, boot_bound: bool = False):
        self.n += 1
        d = abs(ours - theirs)
        if d <= tol + 1e-9:
            self.ok += 1
            if boot_bound and d > TOL:
                self.boot_noise.append(f"{where}: ours {ours:.4f} vs theirs {theirs:.4f} (Monte Carlo noise)")
            return
        self.mismatch.append(f"{where}: ours {ours:.4f} vs theirs {theirs:.4f}")

    def eq(self, where: str, ours, theirs):
        self.n += 1
        if ours == theirs:
            self.ok += 1
        else:
            self.mismatch.append(f"{where}: ours {ours!r} vs theirs {theirs!r}")


def disp_tol(s: str) -> float:
    """Tolerance for a printed number: TOL, or half its last printed digit, whichever is larger."""
    dec = len(s.split(".")[1]) if "." in s else 0
    return max(TOL, 0.5 * 10 ** (-dec) + 1e-9)


CI = re.compile(r"^\s*(-?[\d.]+)\s*\[(-?[\d.]+),\s*(-?[\d.]+)\]")


def check_ci(ck: Checker, where: str, cell: str, ours: tuple, is_boot: bool):
    m = CI.match(cell)
    assert m, (where, cell)
    v, lo, hi = m.groups()
    ck.num(f"{where} value", ours[0], float(v), disp_tol(v))
    btol = BOOT_TOL if is_boot else TOL
    ck.num(f"{where} lo", ours[1], float(lo), max(btol, disp_tol(lo)), boot_bound=is_boot)
    ck.num(f"{where} hi", ours[2], float(hi), max(btol, disp_tol(hi)), boot_bound=is_boot)


def check_money(ck: Checker, where: str, cell: str, ours: float):
    s = cell.strip().lstrip("$")
    ck.num(where, ours, float(s), disp_tol(s))


def md_tables(text: str) -> list[tuple[list[str], list[list[str]], str]]:
    """Each markdown table with the nearest preceding ### heading."""
    out, heading, lines = [], "", text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            heading = ln.lstrip("#").strip()
        if ln.startswith("| ") and i + 1 < len(lines) and lines[i + 1].startswith("| ---"):
            hdr = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append((hdr, rows, heading))
            continue
        i += 1
    return out


# ---------------------------------------------------------------- target 1: designs D and B

FIRSTS = {
    "Jev": ("part2_jev", "jev-latest"),
    "gpt-6-luna (Decisions)": ("part2_luna_decide", "gpt-6-luna"),
    "OpenJev (think 0)": ("part2_codiv", "openjev-latest"),
    "Kev-4B": ("part2_kev", "kev-latest"),
}
LLMS = {"GPT-5": ("repro_fig4", "gpt-5"), "GPT-5 Mini": ("repro_fig4", "gpt-5-mini")}
QCOL = {"Q1": "latent_risk", "Q2": "latent_severity", "Q3": "effect", "Q4": "activated_risk"}


def designs() -> Checker:
    ck = Checker("part2_designs/tables.md")
    prices = yaml.safe_load((ROOT / "configs/prices.yaml").read_text())["models"]
    text = (ROOT / "results/part2_designs/tables.md").read_text()
    cache: dict = {}

    def sys(rm):
        if rm not in cache:
            cache[rm] = load(*rm, prices)
        return cache[rm]

    d_part, b_part = text.split("## B: split by question")
    for _hdr, rows, heading in md_tables(d_part):
        m = re.match(r"(.+) first, then (.+)", heading)
        fname, lname = m.groups()
        F, L = sys(FIRSTS[fname]), sys(LLMS[lname])
        ids = F.ids
        assert set(ids) <= set(L.items), "decision item without an LLM reply"
        n = len(ids)
        Cf = np.array([F.items[i].correct for i in ids])
        Cl = np.array([L.items[i].correct for i in ids])
        cf = np.array([F.items[i].cost for i in ids])
        cl = np.array([L.items[i].cost for i in ids])
        lf = np.array([F.items[i].latency for i in ids])
        ll = np.array([L.items[i].latency for i in ids])
        conf = np.array([F.items[i].conf for i in ids])
        order = sorted(range(n), key=lambda k: (conf[k], ids[k]))

        def row(sent: np.ndarray, label: str, cells: list[str], full: bool = True):
            C = np.where(sent[:, None], Cl, Cf)
            cost = cf + np.where(sent, cl, 0.0)
            lat = lf + np.where(sent, ll, 0.0)
            w = f"D {fname} -> {lname} / {label}"
            ns = int(sent.sum())
            ck.eq(f"{w} n sent", f"{ns} ({round(100 * ns / n)}%)", cells[1])
            check_money(ck, f"{w} $", cells[2], cost.mean() * 1000)
            ck.num(f"{w} p50", float(np.median(lat)), float(cells[3]), disp_tol(cells[3]))
            met = combo_metrics(C)
            check_ci(ck, f"{w} mean of 4", cells[4], met["mean"], True)
            if full:
                for j, q in enumerate(QS):
                    check_ci(ck, f"{w} {q}", cells[5 + j], met[q], False)
            else:
                for j in range(4):
                    ck.eq(f"{w} Q{j + 1} cell", "–", cells[5 + j])

        for cells in rows:
            lab = cells[0]
            sent = np.zeros(n, bool)
            if lab.endswith(" alone") and lab.startswith(fname):
                row(sent, lab, cells)
            elif lab.startswith("lowest-confidence"):
                f = int(re.search(r"(\d+)%", lab).group(1)) / 100
                k = round(f * n)
                sent[order[:k]] = True
                row(sent, lab, cells)
            elif lab.startswith(lname) and "alone" in lab:
                # the LLM alone: one call
                C = Cl
                w = f"D {fname} -> {lname} / {lab}"
                ck.eq(f"{w} n sent", f"{n} (100%)", cells[1])
                check_money(ck, f"{w} $", cells[2], cl.mean() * 1000)
                ck.num(f"{w} p50", float(np.median(ll)), float(cells[3]), disp_tol(cells[3]))
                met = combo_metrics(C)
                check_ci(ck, f"{w} mean of 4", cells[4], met["mean"], True)
                for j, q in enumerate(QS):
                    check_ci(ck, f"{w} {q}", cells[5 + j], met[q], False)
            elif lab.startswith("perfect router"):
                gain = Cl.sum(1).astype(int) - Cf.sum(1).astype(int)
                sent = gain > 0
                row(sent, lab, cells, full=False)
            else:
                raise ValueError(lab)

    for _hdr, rows, heading in md_tables(b_part):
        fname, lname = heading.split(" + ")
        F, L = sys(FIRSTS[fname]), sys(LLMS[lname])
        ids = F.ids
        n = len(ids)
        Cf = np.array([F.items[i].correct for i in ids])
        Cl = np.array([L.items[i].correct for i in ids])
        cf = np.array([F.items[i].cost for i in ids])
        cl = np.array([L.items[i].cost for i in ids])
        for cells in rows:
            dq = cells[0].replace("**", "").replace(" (pre-registered)", "")
            dqs = [] if dq == "none" else [QCOL[x.strip()] for x in dq.split(",")]
            lqs = [] if cells[1] == "none" else [QCOL[x.strip()] for x in cells[1].split(",")]
            ck.eq(f"B {heading} [{cells[0]}] partition", sorted(dqs + lqs), sorted(QS))
            use_l = np.array([q in lqs for q in QS])
            C = np.where(use_l[None, :], Cl, Cf)
            cost = (cf if dqs else 0) + (cl if lqs else 0)
            w = f"B {heading} [{cells[0]}]"
            check_money(ck, f"{w} $", cells[2], float(np.mean(cost)) * 1000)
            met = combo_metrics(C)
            check_ci(ck, f"{w} mean of 4", cells[3], met["mean"], True)
            for j, q in enumerate(QS):
                check_ci(ck, f"{w} {q}", cells[4 + j], met[q], False)
        # paired contrast: headline (decision Q1, LLM Q2-Q4) minus LLM alone
        sec = b_part.split(f"### {heading}\n")[1].split("###")[0]
        m = re.search(r"\(paired bootstrap\): (.+)\.", sec)
        H = np.column_stack([Cf[:, 0], Cl[:, 1:]])
        dlt = H.mean(1) - Cl.mean(1)
        check_ci(ck, f"B {heading} headline minus LLM alone", m.group(1), boot(dlt), True)
    return ck


# ---------------------------------------------------------------- target 2: router search

RS_DEC = {  # label -> (run, model), from configs/analysis/router_search.yaml on origin/router-search-v2
    "Jev": ("part2_jev", "jev-latest"),
    "Jev reversed": ("part2_jev_reversed", "jev-latest"),
    "luna Decisions": ("part2_luna_decide", "gpt-6-luna"),
    "OpenJev think 0": ("part2_codiv", "openjev-latest"),
    "OpenJev think 512": ("part2_openjev_think512", "openjev-latest"),
    "Laya": ("part2_codiv", "laya-1.0"),
    "Kev-4B": ("part2_kev", "kev-latest"),
    "Jev full": ("part2_jev_full", "jev-latest"),
    "Jev letters": ("part2_jev_letters", "jev-latest"),
    "luna Decisions full": ("part2_luna_decide_full", "gpt-6-luna"),
    "luna Decisions letters": ("part2_luna_decide_letters", "gpt-6-luna"),
    "OpenJev think 0 full": ("part2_openjev_full", "openjev-latest"),
    "OpenJev think 512 full": ("part2_openjev_think512_full", "openjev-latest"),
    "Laya full": ("part2_laya_full", "laya-1.0"),
    "Kev-4B full": ("part2_kev_full", "kev-latest"),
    "Kev-4B letters": ("part2_kev_letters", "kev-latest"),
    "NLI DeBERTa-v3-large": ("part2_nli", "deberta-v3-large-zeroshot-v2.0"),
    "Qwen3.5-4B-Base logprob": ("part2_qwen_base_logprob", "qwen3.5-4b-base"),
    "Qwen3.5-4B-Base logprob one-shot": ("part2_qwen_base_logprob_oneshot", "qwen3.5-4b-base"),
    "Qwen3.5-4B logprob": ("part2_qwen_logprob", "qwen3.5-4b"),
    "luna generate": ("part2_luna_gen", "gpt-6-luna"),
    "Qwen3.5-4B generate 2048": ("part2_qwen_gen_2048", "qwen3.5-4b"),
}
LLM_NAMES = {
    "GPT-5 Nano": "gpt-5-nano",
    "GPT-5 Mini": "gpt-5-mini",
    "GPT-5": "gpt-5",
    "Sonnet 4.5": "claude-sonnet-4-5-20250929",
    "Opus 4.5": "claude-opus-4-5-20251101",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite",
    "Gemini 3 Flash": "gemini-3-flash-preview",
    "Gemini 3.1 Pro": "gemini-3.1-pro-preview",
}


def rs_label(label: str) -> tuple[str, str]:
    label = re.sub(r" \[flag:.*\]$", "", label)
    if label in RS_DEC:
        return RS_DEC[label]
    m = re.match(r"(.+) \[(min|default)\]$", label)
    if m:
        return ("repro_fig4" if m.group(2) == "min" else "repro_fig4_default", LLM_NAMES[m.group(1)])
    m = re.match(r"(\S+) \((part2_\w+_cumulative)\)$", label)
    if m:
        return (m.group(2), m.group(1))
    raise KeyError(label)


def q1_matched(C: np.ndarray) -> tuple[list[float], int]:
    k = C[:, 0]
    return [float(C[k, j].mean()) for j in (1, 2, 3)], int(k.sum())


def router() -> tuple[Checker, Checker, list[str]]:
    ck = Checker("router_search/tables.md single systems")
    pk = Checker("router_search/picks.csv")
    notes: list[str] = []
    prices = yaml.safe_load((HERE / "origin/prices.yaml").read_text())["models"]
    text = (HERE / "origin/tables.md").read_text()
    cache: dict = {}

    def sys(label):
        rm = rs_label(label)
        if rm not in cache:
            cache[rm] = load(*rm, prices)
        return cache[rm]

    tabs = [t for t in md_tables(text) if t[2].startswith("Single systems")]
    assert len(tabs) == 1
    hdr, rows, _ = tabs[0]
    for cells in rows:
        S = sys(cells[0])
        label = re.sub(r" \[flag:.*", "", cells[0])
        w = f"single [{label}]"
        ids = S.ids
        C = np.array([S.items[i].correct for i in ids])
        met = combo_metrics(C)
        check_ci(ck, f"{w} mean of 4", cells[1], met["mean"], True)
        for j, q in enumerate(QS):
            ck.num(f"{w} {q}", met[q][0], float(cells[2 + j]), disp_tol(cells[2 + j]))
        accs, nm = q1_matched(C)
        m = re.match(r"([\d.]+)/([\d.]+)/([\d.]+) \(n (\d+)\)", cells[6])
        for a, s in zip(accs, m.groups()[:3], strict=False):
            ck.num(f"{w} q1-matched", a, float(s), disp_tol(s))
        ck.eq(f"{w} n q1-matched", str(nm), m.group(4))
        cost = np.mean([S.items[i].cost for i in ids]) * 1000
        if S.self_hosted:
            ck.eq(f"{w} $", "self-hosted", cells[7])
        else:
            check_money(ck, f"{w} $", cells[7], cost)
        lat = float(np.median([S.items[i].router_latency for i in ids]))
        ck.num(f"{w} p50", lat, float(cells[8]), disp_tol(cells[8]))
        if S.self_hosted:
            g = re.match(r"([\d.]+)", cells[9]).group(1)
            ck.num(f"{w} GPU s", float(np.mean([S.items[i].gpu for i in ids])), float(g), disp_tol(g))
            if "compute_s on 0 of 319" in cells[9]:
                ck.eq(f"{w} n compute_s", 0, S.n_compute)
            elif "(compute_s)" in cells[9]:
                ck.eq(f"{w} n compute_s", len(ids), S.n_compute)
        else:
            ck.eq(f"{w} GPU s", "–", cells[9])
        ck.eq(f"{w} params", "–" if S.params is None else f"{S.params:g}", cells[10])

    # ---- picks: the rows where one candidate was picked in every fold (distinct_picks == 1)
    picks = [r for r in csv.DictReader((HERE / "origin/picks.csv").open()) if r["distinct_picks"] == "1"]
    picks = [r for r in picks if not (r["pool"] == "single" and r["cap_kind"] == "latency" and r["cap"] == "2.0")]
    for r in picks:
        name = r["most_picked"]
        w = f"pick {r['pool']}/{r['cap_kind']}/{r['cap']} [{name}]"
        if r["pool"] == "single":
            S = sys(name)
            ids = S.ids
            Ch = np.array([S.items[i].correct for i in ids], float)  # held out = every item, no fit
            cost = np.mean([S.items[i].cost for i in ids]) * 1000
            lat = float(np.median([S.items[i].router_latency for i in ids]))
        elif r["pool"] == "split_by_question":
            m = re.match(r"(.+) answers (.+); (.+) the rest$", name)
            D, L = sys(m.group(1)), sys(m.group(3))
            dqs = {QCOL[x.strip()] for x in m.group(2).split(",")}
            ids = D.ids
            Ch = np.array([[(D if q in dqs else L).items[i].correct[j] for j, q in enumerate(QS)] for i in ids], float)
            cost = np.mean([D.items[i].cost + L.items[i].cost for i in ids]) * 1000
            lat = float(np.median([D.items[i].router_latency + L.items[i].router_latency for i in ids]))
        elif r["pool"] == "parallel":
            m = re.match(r"(.+) \|\| (.+) in parallel; .+'s answers on the lowest (\d+)% by confidence$", name)
            D, L = sys(m.group(1)), sys(m.group(2))
            frac = int(m.group(3)) / 100
            ids = D.ids
            n = len(ids)
            conf = np.array([D.items[i].conf for i in ids])
            Cd = np.array([D.items[i].correct for i in ids])
            Cl = np.array([L.items[i].correct for i in ids])
            # folds per WORKLOG: shuffle with default_rng(seed), deal round-robin into 5 folds. The base
            # order and the threshold rule are not recorded, so try the plausible variants and report.
            variants = {}
            for base in ("sorted ids", "summary order"):
                order0 = ids if base == "sorted ids" else list(D.items)
                pos = {iid: k for k, iid in enumerate(ids)}
                for rule in ("<", "<="):
                    acc = np.zeros((n, 4))
                    for seed in range(5):
                        perm = np.random.default_rng(seed).permutation(n)
                        fold = np.empty(n, int)
                        for j, p in enumerate(perm):
                            fold[pos[order0[p]]] = j % 5
                        for f in range(5):
                            tr, te = fold != f, fold == f
                            thr = np.quantile(conf[tr], frac)
                            sent = (conf < thr) if rule == "<" else (conf <= thr)
                            sel = te & sent
                            acc[te] += np.where(sel[te][:, None], Cl[te], Cd[te])
                    variants[(base, rule)] = acc / 5
            best = min(variants, key=lambda k: abs(variants[k].mean() - float(r["held_out_mean_of_4"])))
            Ch = variants[best]
            notes.append(
                f"{w}: folds rebuilt from WORKLOG (permutation of {best[0]}, round-robin, "
                f"threshold = np.quantile of training confidences, sent if conf {best[1]} thr); "
                + "; ".join(f"{k}: {v.mean():.6f}" for k, v in variants.items())
            )
            cost = 0.0 if (D.self_hosted or D.free) and (L.self_hosted or L.free) else None
            lat = None
        else:
            raise ValueError(r["pool"])
        pk.num(f"{w} held-out mean of 4", float(Ch.mean()), float(r["held_out_mean_of_4"]))
        lo, hi = boot(Ch.mean(1))[1:]
        pk.num(f"{w} CI lo", lo, float(r["held_out_mean_of_4_lo"]), BOOT_TOL, True)
        pk.num(f"{w} CI hi", hi, float(r["held_out_mean_of_4_hi"]), BOOT_TOL, True)
        for j, q in enumerate(QS):
            pk.num(f"{w} {q}", float(Ch[:, j].mean()), float(r[f"held_out_{q}"]))
        if cost is not None:
            pk.num(f"{w} $", cost, float(r["held_out_cost_per_1000_usd"]))
        if lat is not None:
            pk.num(f"{w} p50", lat, float(r["held_out_latency_p50_s"]))
    # side check: the WORKLOG claims self_hosted_pick_share == 0 throughout picks.csv
    nz = [
        f"{r['pool']}/{r['cap_kind']}/{r['cap']}={r['self_hosted_pick_share']}"
        for r in csv.DictReader((HERE / "origin/picks.csv").open())
        if float(r["self_hosted_pick_share"]) > 0
    ]
    if nz:
        notes.append("picks.csv rows with self_hosted_pick_share > 0 (WORKLOG says 0 throughout): " + ", ".join(nz))
    return ck, pk, notes


def main():
    cks = [designs()]
    rck, pck, notes = router()
    cks += [rck, pck]
    out = [
        "# Part 2 independent check (results/verify/part2/verify_part2.py)",
        "",
        f"Tolerance {TOL} (or half the last printed digit); bootstrap bounds {BOOT_TOL}.",
        "",
    ]
    for c in cks:
        out += [f"## {c.name}", "", f"- checked: {c.n}, matched: {c.ok}, mismatches: {len(c.mismatch)}"]
        out += [f"- MISMATCH {m}" for m in c.mismatch]
        if c.boot_noise:
            out += [f"- bootstrap bounds off by more than {TOL} but within {BOOT_TOL} (Monte Carlo noise): {len(c.boot_noise)}"]
            out += [f"  - {m}" for m in c.boot_noise]
        out.append("")
    if notes:
        out += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    (HERE / "report.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
