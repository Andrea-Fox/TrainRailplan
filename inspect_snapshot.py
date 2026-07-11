"""
Inspect a frozen snapshot. Read-only, throwaway, no side effects.

    python inspect_snapshot.py data/cz_rail_20260714
"""

import sys
from pathlib import Path

import pandas as pd

try:
    import networkx as nx
except ImportError:
    nx = None

snap = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cz_rail_20260714")
stops = pd.read_parquet(snap / "stops.parquet")
stop_times = pd.read_parquet(snap / "stop_times.parquet")


def rule(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


# --- 1. which stops have no coordinates? ---------------------------------
rule("stops without coordinates")
blank = stops[~stops.geolocatable]
print(f"{len(blank)} of {len(stops)}\n")
print(blank[["stop_id", "stop_name"]].to_string(index=False))


# --- 2. zone duplicates ---------------------------------------------------
rule("zone duplicates (.Z### suffix)")
stops["base"] = stops.stop_id.str.replace(r"\.Z[^.]*$", "", regex=True)
print(f"stop_ids       : {len(stops):>6,}")
print(f"distinct base  : {stops.base.nunique():>6,}")
print(f"distinct names : {stops.stop_name.nunique():>6,}")

collapse = len(stops) - stops.base.nunique()
print(f"\ncollapsed by suffix strip: {collapse:,} ({collapse / len(stops):.1%})")

print("\nmost duplicated names:")
print(stops.stop_name.value_counts().head(10).to_string())

# A name appearing under several *base* ids is a different problem from the
# same base id appearing under several zones. Distinguish them.
multi_base = stops.groupby("stop_name").base.nunique()
multi_base = multi_base[multi_base > 1]
print(f"\nnames spanning >1 base id: {len(multi_base)}")
if len(multi_base):
    print(multi_base.sort_values(ascending=False).head(10).to_string())


# --- 3. connectivity ------------------------------------------------------
rule("connectivity (time-ignoring; necessary, not sufficient)")
if nx is None:
    print("networkx not installed:  pip install networkx")
else:
    st = stop_times.sort_values(["trip_id", "stop_sequence"])
    G = nx.Graph()
    for _, d in st.groupby("trip_id", sort=False):
        ids = d.stop_id.tolist()
        G.add_edges_from(zip(ids, ids[1:]))

    cc = sorted(nx.connected_components(G), key=len, reverse=True)
    n = G.number_of_nodes()
    print(f"nodes: {n:,}  edges: {G.number_of_edges():,}  components: {len(cc)}")
    print(f"largest: {len(cc[0]):,} ({len(cc[0]) / n:.1%} of nodes)")
    print(f"next sizes: {[len(c) for c in cc[1:8]]}")

    isolated = [c for c in cc if len(c) <= 3]
    print(f"\ncomponents of <=3 stops: {len(isolated)}")
    for c in isolated[:5]:
        names = stops.set_index("stop_id").stop_name.reindex(sorted(c)).tolist()
        print("  ", names)


#=--- 4. components --------------------------------------------------------
print("\n\ncomponents (from stations.parquet)")
import pandas as pd
s = pd.read_parquet("data/cz_rail_20260714/stations.parquet")
for c, g in s[s.component > 0].groupby("component"):
    print(f"\ncomponent {c} ({len(g)}):", ", ".join(g.stop_name))



# --- 5. trips with revisits ------------------------------------------------
print("\n\ntrips with revisits (calls the same station twice)")
from replay import World, Leg, replay, hhmm
w = World("data/cz_rail_20260714")
print("trips:", len(w._trips), "| revisited:", len(w._revisited))

tid = next(iter(w._trips))
calls = sorted(w.calls(tid).items(), key=lambda kv: kv[1][0])
a, b = calls[0][0], calls[-1][0]
r = replay([Leg(tid, a, b)], w)
print(f"{w.name(a)} -> {w.name(b)}  {hhmm(r.departure)}-{hhmm(r.arrival)}  {r.feasible}")


# --- 6. busiest stops ------------------------------------------------------
print("\n\nbusiest stops (by trip count)")
import pandas as pd
snap = "data/cz_rail_20260714"
st = pd.read_parquet(f"{snap}/stop_times.parquet")
sn = pd.read_parquet(f"{snap}/stations.parquet").set_index("station_id").stop_name

trips = st.groupby("station_id").trip_id.nunique()          # how busy
st = st.sort_values(["trip_id","stop_sequence"])
adj = {}
for _, d in st.groupby("trip_id", sort=False):
    ids = d.station_id.tolist()
    for a, b in zip(ids, ids[1:]):
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
degree = pd.Series({k: len(v) for k, v in adj.items()})     # how junction-like

print(pd.DataFrame({"trips": trips, "degree": degree, "name": sn})
        .sort_values("trips", ascending=False).head(15).to_string())
print(pd.DataFrame({"trips": trips, "degree": degree, "name": sn})
        .sort_values("degree", ascending=False).head(15).to_string())


# --- 7. routes and trips -----------------------------------------------------
print("\n\nroutes and trips")
import pandas as pd
snap = "data/cz_rail_20260714"
r = pd.read_parquet(f"{snap}/routes.parquet")
print(r.columns.tolist())
print(r[[c for c in ["route_id","route_short_name","route_long_name","route_desc"] if c in r]].head(20).to_string())

t = pd.read_parquet(f"{snap}/trips.parquet")
print(t.columns.tolist())
print(t.head(5).to_string())




# --- 8. long-distance trips -------------------------------------------------
print("\n\nlong-distance trips")

import pandas as pd
snap = "data/cz_rail_20260714"
r = pd.read_parquet(f"{snap}/routes.parquet")
t = pd.read_parquet(f"{snap}/trips.parquet")
st = pd.read_parquet(f"{snap}/stop_times.parquet")
sn = pd.read_parquet(f"{snap}/stations.parquet").set_index("station_id").stop_name

print(r.is_regional.value_counts(dropna=False))
print(r.is_night.value_counts(dropna=False))

longdist = set(r.loc[r.is_regional.isin(["0", 0, False]), "route_id"])
ld_trips = set(t.loc[t.route_id.isin(longdist), "trip_id"])
print(f"\n{len(ld_trips)} of {t.trip_id.nunique()} trips are long-distance")

w = st[st.trip_id.isin(ld_trips)].groupby("station_id").trip_id.nunique()
print(f"{len(w)} stations served by a long-distance train\n")
print(w.sort_values(ascending=False).head(15).rename(sn).to_string())


# -- 9. direct Praha-Brno trips ------------------------------------------------
print("\n\nPraha-Brno trips")
import pandas as pd
snap = "data/cz_rail_20260714"
st = pd.read_parquet(f"{snap}/stop_times.parquet")
sn = pd.read_parquet(f"{snap}/stations.parquet").set_index("station_id").stop_name
name2id = {v: k for k, v in sn.items()}
P, B = name2id["Praha hlavní nádraží"], name2id["Brno hlavní nádraží"]

both = st.groupby("trip_id").station_id.apply(lambda s: P in set(s) and B in set(s))
print("direct Praha-Brno trips:", both.sum())

# trip span distribution: express trains are long-running
span = st.groupby("trip_id").agg(lo=("dep","min"), hi=("arr","max"))
span["h"] = (span.hi - span.lo) / 3600
print(span.h.describe())