"""Offline study of traceability definitions on the public validation split (stdlib)."""
import array, ast, struct, sys, zipfile
from collections import defaultdict
from pathlib import Path
R = Path("/home/user")
def parse(raw):
    hlen = struct.unpack("<H", raw[8:10])[0]; h = ast.literal_eval(raw[10:10+hlen].decode("latin1"))
    a = array.array({"<i4":"i","<i2":"h","<i8":"q"}[h["descr"]]); a.frombytes(raw[10+hlen:]); return a, h["shape"]
def npz(path, keys):
    z = zipfile.ZipFile(path); return {k: parse(z.read(k + ".npy")) for k in keys}
def rows(a, s): return [a[i*s[1]:(i+1)*s[1]] for i in range(s[0])]
def sets(d, p):
    ip, _ = d[p+"_indptr"]; ix, _ = d[p+"_indices"]; return [set(ix[ip[i]:ip[i+1]]) for i in range(len(ip)-1)]
pub = R/"data/downloads/fb15k237"
keys = [f"{x}_{t}_{y}" for x in ("hard","filt") for t in ("2p","3p") for y in ("indptr","indices")] + ["q_2p","q_3p"]
T = npz(pub/"valid_std.npz", keys)
def graph(names):
    adj = defaultdict(set)
    for n in names:
        a, s = parse((pub/n).read_bytes())
        for i in range(s[0]): adj[(a[3*i], a[3*i+1])].add(a[3*i+2])
    return adj
G = graph(["train_triples.npy", "valid_triples.npy"]); O = graph(["train_triples.npy"])
E = set()
q2 = rows(*T["q_2p"]); q3 = rows(*T["q_3p"])
h2, f2, h3, f3 = sets(T,"hard_2p"), sets(T,"filt_2p"), sets(T,"hard_3p"), sets(T,"filt_3p")
methods = {m: R/"outputs"/m/"valid_std.npz" for m in ["lvsa","vsa_composition","symbolic_traversal","random","lvsa_no_logic"]}
methods.update({m: R/"tmp/hardening/outputs"/m/"valid_std.npz" for m in ["lvsa_seed2","lvsa_alltypes"]})
P = {}
for m, p in methods.items():
    P[m] = npz(p, ["rank_2p","rank_3p","trace_2p_v1","trace_3p_v1","trace_3p_v2"])
P["hybrid_lvsa+lookup"] = {**P["lvsa"], **{k: P["symbolic_traversal"][k] for k in ("trace_2p_v1","trace_3p_v1","trace_3p_v2")}}
P["hybrid_lvsa+1p"] = {**P["lvsa"], **{k: P["vsa_composition"][k] for k in ("trace_2p_v1","trace_3p_v1","trace_3p_v2")}}
def first_unf(row, ign):
    for e in row:
        if e not in ign: return e
def rr_first(row, valid, skip=()):
    pos = 0
    for e in row:
        if e in skip: continue
        pos += 1
        if e in valid: return 1.0/pos
    return 0.0
def top_unskipped(row, skip):
    for e in row:
        if e not in skip: return e
def mean(x): return sum(x)/len(x) if x else 0.0
res = {}
for m, d in P.items():
    A2 = rows(*d["rank_2p"]); A3 = rows(*d["rank_3p"]); V21 = rows(*d["trace_2p_v1"]); V31 = rows(*d["trace_3p_v1"]); V32 = rows(*d["trace_3p_v2"])
    D0 = {"rr": [], "p1": []}; T1 = {"rr": [], "p1": []}; T2 = {"rr": [], "p1": []}; path = []
    for q, H, F, a, v in zip(q2, h2, f2, A2, V21):
        a0, r0, r1 = q
        first = G.get((a0, r0), E); obs1 = O.get((a0, r0), E)
        valid = {e for e in first if G.get((e, r1), E) & H}
        D0["rr"].append(rr_first(v, valid)); D0["p1"].append(float(v[0] in valid))
        vn = valid - obs1
        if vn:
            T1["rr"].append(rr_first(v, vn, obs1)); T1["p1"].append(float(top_unskipped(v, obs1) in vn))
        ah = first_unf(a, F)
        vc = {e for e in first if ah in G.get((e, r1), E)} if ah in H else set()
        T2["rr"].append(rr_first(v, vc)); T2["p1"].append(float(v[0] in vc))
        path.append(float(ah in H and v[0] in first and ah in G.get((v[0], r1), E)))
    for q, H, F, a, v1, v2 in zip(q3, h3, f3, A3, V31, V32):
        a0, r0, r1, r2 = q
        first = G.get((a0, r0), E); obs1 = O.get((a0, r0), E)
        second = set().union(*[G.get((e, r1), E) for e in first]) if first else set()
        obs2 = set().union(*[O.get((e, r1), E) for e in obs1]) if obs1 else set()
        valid2 = {e for e in second if G.get((e, r2), E) & H}
        valid1 = {e for e in first if G.get((e, r1), E) & valid2}
        for v, val, ob in ((v1, valid1, obs1), (v2, valid2, obs2)):
            D0["rr"].append(rr_first(v, val)); D0["p1"].append(float(v[0] in val))
            vn = val - ob
            if vn:
                T1["rr"].append(rr_first(v, vn, ob)); T1["p1"].append(float(top_unskipped(v, ob) in vn))
        ah = first_unf(a, F)
        if ah in H:
            c2 = {e for e in second if ah in G.get((e, r2), E)}
            c1 = {e for e in first if G.get((e, r1), E) & c2}
        else:
            c1 = c2 = set()
        T2["rr"] += [rr_first(v1, c1), rr_first(v2, c2)]; T2["p1"] += [float(v1[0] in c1), float(v2[0] in c2)]
        path.append(float(ah in H and v1[0] in first and v2[0] in G.get((v1[0], r1), E) and ah in G.get((v2[0], r2), E)))
    res[m] = (100*mean(D0["rr"]), 100*mean(D0["p1"]), 100*mean(T1["rr"]), 100*mean(T1["p1"]), len(T1["rr"]), 100*mean(T2["rr"]), 100*mean(T2["p1"]), 100*mean(path))
print(f"{'method':22s} {'D0 mrr':>7} {'D0 p1':>7} | {'T1 mrr':>7} {'T1 p1':>7} {'n':>5} | {'T2 mrr':>7} {'T2 p1':>7} | {'path':>6}")
for m, r in res.items():
    print(f"{m:22s} {r[0]:7.2f} {r[1]:7.2f} | {r[2]:7.2f} {r[3]:7.2f} {r[4]:5d} | {r[5]:7.2f} {r[6]:7.2f} | {r[7]:6.2f}")

print("filtered path P@1 (observed intermediates skipped, chain must reach a hard answer)")
for m, d in P.items():
    A2 = rows(*d["rank_2p"]); A3 = rows(*d["rank_3p"]); V21 = rows(*d["trace_2p_v1"]); V31 = rows(*d["trace_3p_v1"]); V32 = rows(*d["trace_3p_v2"])
    p2 = []; p3 = []
    for q, H, F, a, v in zip(q2, h2, f2, A2, V21):
        a0, r0, r1 = q; obs1 = O.get((a0, r0), E); ah = first_unf(a, F); t1 = top_unskipped(v, obs1)
        p2.append(float(ah in H and t1 in G.get((a0, r0), E) and ah in G.get((t1, r1), E)))
    for q, H, F, a, v1, v2 in zip(q3, h3, f3, A3, V31, V32):
        a0, r0, r1, r2 = q; obs1 = O.get((a0, r0), E)
        obs2 = set().union(*[O.get((e, r1), E) for e in obs1]) if obs1 else set()
        ah = first_unf(a, F); t1 = top_unskipped(v1, obs1); t2 = top_unskipped(v2, obs2)
        p3.append(float(ah in H and t1 in G.get((a0, r0), E) and t2 in G.get((t1, r1), E) and ah in G.get((t2, r2), E)))
    print(f"  {m:22s} 2p {100*mean(p2):5.2f}  3p {100*mean(p3):5.2f}")
