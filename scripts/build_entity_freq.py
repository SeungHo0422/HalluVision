"""
Build per-document frequency-based entity metric = the PAPER definition
(Eq. 5), replacing the JS-divergence proxy used in the earlier demo.

For each doc, count entities per predefined CTI label in the ORIGINAL and
in the SUMMARY NER outputs, normalise to distributions p_o / p_g, and:

    TV distance   = 0.5 * sum_l | p_o[l] - p_g[l] |     (L1 / 2, in [0,1])
    freq_similarity F = 1 - TV                          (paper's F(o,g))
    freq_distance = TV                                  (higher => halluc)

Inputs (workspace root): original_ner.json, summary_ner.json  (gpt-4o-mini).
Output: notes/combined_scorer_demo/entity_freq_gpt4omini.json in the workspace.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("HALLUVISION_WORKSPACE", os.getcwd())  # data workspace
LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]


def tv_distance(c_o, c_g):
    """Total-variation distance between the two label distributions.
    Returns None if either side has no entities (distribution undefined)."""
    so, sg = sum(c_o), sum(c_g)
    if so == 0 or sg == 0:
        return None
    p_o = [x / so for x in c_o]
    p_g = [x / sg for x in c_g]
    return 0.5 * sum(abs(a - b) for a, b in zip(p_o, p_g))


# ---- self-test against the paper's worked example (Table 2) ----
# counts order in paper: [System, Malware, Indicator, Vulnerability, Organization]
_co = [4, 5, 6, 0, 3]; _cg = [1, 3, 0, 0, 1]
_tv = tv_distance(_co, _cg)
_F = 1 - _tv
assert abs(_tv - 0.3555) < 1e-3 and abs(_F - 0.6445) < 1e-3, (_tv, _F)
print(f"[self-test] paper example TV={_tv:.4f} F={_F:.4f} (expected 0.3555 / 0.6445) OK")

# ---- build for the whole gpt-4o-mini corpus ----
orig = json.load(open(os.path.join(ROOT, "original_ner.json")))
summ = json.load(open(os.path.join(ROOT, "summary_ner.json")))

out = {}
n_ok = n_undef = 0
for did in summ:
    if did not in orig:
        continue
    o, s = orig[did], summ[did]
    c_o = [len(set(o.get(l, []))) for l in LABELS]
    c_g = [len(set(s.get(l, []))) for l in LABELS]
    tv = tv_distance(c_o, c_g)
    rec = {"c_orig": c_o, "c_summ": c_g}
    if tv is None:
        rec["tv_distance"] = None
        rec["freq_similarity"] = None
        rec["freq_distance"] = None
        n_undef += 1
    else:
        rec["tv_distance"] = tv
        rec["freq_similarity"] = 1 - tv
        rec["freq_distance"] = tv
        n_ok += 1
    out[did] = rec

path = os.path.join(ROOT, "notes", "combined_scorer_demo", "entity_freq_gpt4omini.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(out, open(path, "w"), indent=2)
print(f"[build] {n_ok} docs with defined freq metric, {n_undef} undefined (empty side)")
print(f"[saved] {path}  (LABELS order: {LABELS})")
