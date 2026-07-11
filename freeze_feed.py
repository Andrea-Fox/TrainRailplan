"""
Freeze one GTFS snapshot to parquet, rail-only, single service date.

Usage:
    python freeze_feed.py --zip gtfs.zip --date 2026-07-15 --out data/cz_rail_20260715

Everything downstream (solver, verifier, tools) reads only this frozen snapshot.
Never re-fetch mid-project: the feed is the world, and the world must not move.
"""

import argparse
import json
import re
import zipfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIL = 2  # GTFS route_type
DAY = 86_400

# Tariff-zone duplicate suffix. GTFS cannot express "this stop is in zone A on
# line 1 and zone B on line 2", so the feed emits one stop_id per zone
# combination. Physically the same platform.
ZONE_SUFFIX = re.compile(r"\.Z[^.]*$")


def components(nodes, edges) -> dict:
    """Connected components by union-find. Returns node -> component rank,
    where rank 0 is the largest component. No networkx dependency."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    root = {n: find(n) for n in nodes}
    rank = {r: i for i, (r, _) in enumerate(Counter(root.values()).most_common())}
    return {n: rank[r] for n, r in root.items()}


def log(drops: dict, key: str, n: int, why: str):
    if n:
        drops.setdefault(key, []).append({"n": int(n), "why": why})


def unwrap(seq: np.ndarray) -> np.ndarray:
    """Undo modulo-24h wrapping in a nondecreasing time sequence.

    The feed writes the first stop past midnight with an extended arrival
    (24:01:00, 26:14:00) but reduces every subsequent time modulo 24h, so a
    trip's clock appears to jump backwards by exactly one day.

    Each value is lifted by whole days until it is at least the previous
    value. Values already written in extended form are left alone -- do NOT
    carry a running offset, or 25:00:00 becomes 49:00:00. No-op on
    well-formed trips.
    """
    out = np.empty_like(seq)
    prev = None
    for i, x in enumerate(seq):
        if prev is not None:
            while x < prev:
                x += DAY
        out[i] = x
        prev = x
    return out


def unwrap_frame(st: pd.DataFrame) -> pd.DataFrame:
    """Unwrap every trip in a stop_times frame, in place on a copy.

    Assumes st is already sorted by (trip_id, stop_sequence). Uses positional
    group indices rather than groupby.apply, which changed signature across
    pandas versions.
    """
    st = st.copy()
    arr = st["arr"].to_numpy(dtype=np.int64, copy=True)
    dep = st["dep"].to_numpy(dtype=np.int64, copy=True)
    for idx in st.groupby("trip_id", sort=False).indices.values():
        seq = np.empty(2 * len(idx), dtype=np.int64)
        seq[0::2] = arr[idx]
        seq[1::2] = dep[idx]
        seq = unwrap(seq)
        arr[idx] = seq[0::2]
        dep[idx] = seq[1::2]
    st["arr"] = arr
    st["dep"] = dep
    return st


def read(zf: zipfile.ZipFile, name: str, **kw) -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_csv(f, dtype=str, keep_default_na=False, **kw)


def to_seconds(s: pd.Series) -> pd.Series:
    """GTFS times may exceed 24:00:00. Seconds since service-day midnight."""
    parts = s.str.strip().str.split(":", expand=True).astype(int)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def active_services(zf: zipfile.ZipFile, day: date) -> set[str]:
    ymd = day.strftime("%Y%m%d")
    dow = day.strftime("%A").lower()
    names = set(zf.namelist())

    active: set[str] = set()
    if "calendar.txt" in names:
        cal = read(zf, "calendar.txt")
        in_range = (cal.start_date <= ymd) & (cal.end_date >= ymd)
        runs = cal[dow] == "1"
        active |= set(cal.loc[in_range & runs, "service_id"])

    if "calendar_dates.txt" in names:
        cd = read(zf, "calendar_dates.txt")
        cd = cd[cd.date == ymd]
        active |= set(cd.loc[cd.exception_type == "1", "service_id"])  # added
        active -= set(cd.loc[cd.exception_type == "2", "service_id"])  # removed

    return active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD service date")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-transfer-sec", type=int, default=180)
    args = ap.parse_args()

    day = datetime.strptime(args.date, "%Y-%m-%d").date()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    drops: dict = {}

    with zipfile.ZipFile(args.zip) as zf:
        routes = read(zf, "routes.txt")
        trips = read(zf, "trips.txt")
        stops = read(zf, "stops.txt")
        stop_times = read(zf, "stop_times.txt")
        services = active_services(zf, day)

    # --- rail only ---------------------------------------------------------
    n0 = len(routes)
    routes = routes[routes.route_type.astype(int) == RAIL]
    log(drops, "routes", n0 - len(routes), "route_type != 2 (rail)")

    n0 = len(trips)
    trips = trips[trips.route_id.isin(set(routes.route_id))]
    log(drops, "trips", n0 - len(trips), "route not rail")

    # --- single service date ----------------------------------------------
    n0 = len(trips)
    trips = trips[trips.service_id.isin(services)]
    log(drops, "trips", n0 - len(trips), f"service inactive on {day}")

    n0 = len(stop_times)
    stop_times = stop_times[stop_times.trip_id.isin(set(trips.trip_id))]
    log(drops, "stop_times", n0 - len(stop_times), "trip not retained")

    # --- referential integrity (assert, do not repair silently) -----------
    orphan_trips = set(stop_times.trip_id) - set(trips.trip_id)
    assert not orphan_trips, f"{len(orphan_trips)} stop_times reference unknown trip_id"

    orphan_stops = set(stop_times.stop_id) - set(stops.stop_id)
    assert not orphan_stops, f"{len(orphan_stops)} stop_times reference unknown stop_id"

    # --- times -------------------------------------------------------------
    n0 = len(stop_times)
    stop_times = stop_times[
        (stop_times.arrival_time != "") & (stop_times.departure_time != "")
    ]
    log(drops, "stop_times", n0 - len(stop_times), "missing arrival/departure time")

    stop_times["stop_sequence"] = stop_times.stop_sequence.astype(int)
    stop_times["arr"] = to_seconds(stop_times.arrival_time)
    stop_times["dep"] = to_seconds(stop_times.departure_time)
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    # --- repair modulo-24h wrapping ---------------------------------------
    # Explicit, counted, reported. Never a silent fix.
    wrapped = set(stop_times.loc[stop_times.dep < stop_times.arr, "trip_id"])
    if wrapped:
        # Sanity: this feed's wrap is exactly one day. Anything else is not a
        # wrap and must not be "repaired" -- it is corruption, and we stop.
        delta = (stop_times.dep - stop_times.arr)[stop_times.dep < stop_times.arr]
        odd = delta[delta != -DAY]
        assert odd.empty, (
            f"{len(odd)} stop_times with dep < arr by something other than "
            f"exactly one day (deltas: {sorted(set(odd))[:5]}); not a midnight "
            f"wrap, refusing to repair"
        )
        stop_times = unwrap_frame(stop_times)
    repairs = {"midnight_wrap_unwrapped_trips": len(wrapped)}

    # --- validate --------------------------------------------------------
    # NOTE: unwrap() forces times to be nondecreasing, so re-checking dwell or
    # monotonicity on repaired trips is vacuous -- it tests the repair, not the
    # data. Check something the repair cannot manufacture instead: a lifted
    # trip whose real problem was corruption (not a wrap) acquires an absurd
    # span, because unwrap() pushed a garbage value forward by whole days.
    ends = stop_times.groupby("trip_id").agg(lo=("arr", "min"), hi=("dep", "max"))
    span = ends.hi - ends.lo
    absurd = set(span[span > DAY].index)  # no CZ rail trip runs over 24h
    n0 = len(trips)
    trips = trips[~trips.trip_id.isin(absurd)]
    stop_times = stop_times[~stop_times.trip_id.isin(absurd)]
    log(drops, "trips", n0 - len(trips), "trip span exceeds 24h after unwrapping")

    repairs["unwrapped_then_dropped"] = len(wrapped & absurd)

    # Unrepaired trips must still satisfy the original invariants.
    untouched = stop_times[~stop_times.trip_id.isin(wrapped)]
    bad_dwell = untouched[untouched.dep < untouched.arr]
    assert bad_dwell.empty, (
        f"{len(bad_dwell)} stop_times with departure before arrival on trips "
        f"that were never unwrapped"
    )
    g = untouched.groupby("trip_id", sort=False)
    non_monotonic = g.apply(
        lambda d: (d.arr.values[1:] < d.dep.values[:-1]).any(), include_groups=False
    )
    bad_trips = set(non_monotonic[non_monotonic].index)
    n0 = len(trips)
    trips = trips[~trips.trip_id.isin(bad_trips)]
    stop_times = stop_times[~stop_times.trip_id.isin(bad_trips)]
    log(drops, "trips", n0 - len(trips), "non-monotonic times along stop_sequence")

    # a trip with < 2 stops carries nobody anywhere
    sizes = stop_times.groupby("trip_id").size()
    stub = set(sizes[sizes < 2].index)
    n0 = len(trips)
    trips = trips[~trips.trip_id.isin(stub)]
    stop_times = stop_times[~stop_times.trip_id.isin(stub)]
    log(drops, "trips", n0 - len(trips), "fewer than 2 stops")

    # --- stops actually used ----------------------------------------------
    used = set(stop_times.stop_id)
    n0 = len(stops)
    stops = stops[stops.stop_id.isin(used)]
    log(drops, "stops", n0 - len(stops), "no rail service on this date")

    # --- coordinates -------------------------------------------------------
    # Feasibility never touches geometry: replay() chains legs by time and
    # station_id. Coordinates matter only for find_station and geo constraints,
    # so a missing coordinate must never remove a trip from the world.
    #
    # In this feed, blank coordinates come in whole station clusters: the
    # blank location_type=1 stations are exactly the parents of the blank
    # location_type=0 stops, so parent inheritance would rescue almost
    # nothing. Not attempted.
    stops["stop_lat"] = pd.to_numeric(stops.stop_lat, errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops.stop_lon, errors="coerce")

    missing = stops.stop_lat.isna() | stops.stop_lon.isna()
    repairs["stops_without_coords"] = int(missing.sum())
    repairs["stops_geolocatable"] = int((~missing).sum())

    # (0, 0) is a placeholder, not the Gulf of Guinea. Always a bug.
    placeholder = (stops.stop_lat == 0) & (stops.stop_lon == 0)
    assert not placeholder.any(), f"{int(placeholder.sum())} stops at (0, 0)"

    # Border stations (Selb-Plößberg, Cieszyn, ...) are legitimate members of
    # the rail world and add real branching. Count them; do not assert.
    known = stops[~missing]
    outside = ~(known.stop_lat.between(48, 52) & known.stop_lon.between(11, 20))
    repairs["stops_outside_cz_bounds"] = int(outside.sum())

    # Stops that cannot answer a geographic query. find_station must rank
    # only over stops where geolocatable is True.
    stops["geolocatable"] = ~missing

    # --- collapse tariff-zone duplicates ----------------------------------
    # CRITICAL. A traveller arriving at Kolin.Z1 and departing from Kolin.Z2
    # is standing on the same platform. If the verifier chains legs on
    # stop_id, that transfer reads as infeasible -- silently, on ~12% of
    # stations, concentrated at exactly the junctions where transfers happen.
    # Everything downstream keys on station_id. stop_id survives as a join key.
    stops["station_id"] = stops.stop_id.str.replace(ZONE_SUFFIX, "", regex=True)
    stop_times["station_id"] = stop_times.stop_id.str.replace(
        ZONE_SUFFIX, "", regex=True
    )

    # The collapse is only sound if station_id determines the name. Verified on
    # the 2026-07 feed (2,488 bases, 2,488 names, 0 names spanning >1 base) --
    # but it is an assumption about the data, so assert it every run.
    per_station = stops.groupby("station_id").stop_name.nunique()
    ambiguous = per_station[per_station > 1]
    assert ambiguous.empty, (
        f"{len(ambiguous)} station_ids carry >1 stop_name, e.g. "
        f"{list(ambiguous.index[:3])}; zone-suffix collapse is unsound"
    )
    per_name = stops.groupby("stop_name").station_id.nunique()
    collisions = per_name[per_name > 1]
    repairs["names_spanning_multiple_stations"] = int(len(collisions))

    # --- stations table ----------------------------------------------------
    stations = (
        stops.sort_values("geolocatable", ascending=False)  # prefer a located row
        .groupby("station_id", as_index=False)
        .agg(
            stop_name=("stop_name", "first"),
            stop_lat=("stop_lat", "first"),
            stop_lon=("stop_lon", "first"),
            geolocatable=("geolocatable", "any"),
            n_stop_ids=("stop_id", "size"),
        )
    )
    repairs["stop_ids"] = len(stops)
    repairs["stations_after_collapse"] = len(stations)

    # --- connectivity ------------------------------------------------------
    # A time-ignoring necessary condition. Origin/destination pairs must be
    # sampled within one component, or the instance is infeasible by topology
    # and the policy is punished for a fact it cannot observe.
    st = stop_times.sort_values(["trip_id", "stop_sequence"])
    seq = st.station_id.to_numpy()
    edges = []
    for idx in st.groupby("trip_id", sort=False).indices.values():
        chain = seq[idx]
        edges.extend(zip(chain[:-1], chain[1:]))

    comp = components(set(stations.station_id), edges)
    stations["component"] = stations.station_id.map(comp)
    sizes = stations.component.value_counts().sort_index()
    repairs["n_components"] = int(len(sizes))
    repairs["largest_component_stations"] = int(sizes.iloc[0])
    repairs["largest_component_frac"] = round(float(sizes.iloc[0] / len(stations)), 4)

    # --- write -------------------------------------------------------------
    routes.to_parquet(out / "routes.parquet", index=False)
    trips.to_parquet(out / "trips.parquet", index=False)
    stops.to_parquet(out / "stops.parquet", index=False)
    stations.to_parquet(out / "stations.parquet", index=False)
    stop_times.to_parquet(out / "stop_times.parquet", index=False)

    manifest = {
        "source_zip": str(Path(args.zip).resolve()),
        "service_date": str(day),
        "min_transfer_sec": args.min_transfer_sec,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "routes": len(routes),
            "trips": len(trips),
            "stops": len(stops),
            "stations": len(stations),
            "stop_times": len(stop_times),
        },
        "repaired": repairs,
        "dropped": drops,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest["counts"], indent=2))
    print("\nrepaired:")
    for k, v in repairs.items():
        print(f"  {k:32s} {v:>8,}")
    print("\ndropped:")
    for table, entries in drops.items():
        for e in entries:
            print(f"  {table:12s} -{e['n']:>8,}  {e['why']}")


if __name__ == "__main__":
    main()