"""
replay(): the verifier. Exact feasibility of a proposed itinerary against the
frozen snapshot. This is the reward function -- if it is wrong, every number
downstream is meaningless, so it is written and tested before any model exists.

It answers one question only: *is this itinerary physically possible?*
It does not judge constraints. Constraint costs are computed from the facts it
returns, by a separate module, because "feasible" and "satisfies the user's
request" are different failures and the policy must be able to tell them apart.

A leg is (trip_id, board, alight): the policy names a train and where it gets
on and off. Everything keys on station_id, never stop_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DAY = 86_400
DEFAULT_MIN_TRANSFER = 180


@dataclass(frozen=True)
class Leg:
    trip_id: str
    board: str  # station_id
    alight: str  # station_id


@dataclass
class Report:
    feasible: bool
    reasons: list[str] = field(default_factory=list)

    # Facts. Meaningful only when feasible; None otherwise.
    departure: int | None = None  # seconds since service-day midnight
    arrival: int | None = None
    transfers: int | None = None  # len(legs) - 1
    duration: int | None = None
    stations_visited: list[str] = field(default_factory=list)  # incl. pass-throughs
    legs: list[Leg] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.feasible


class World:
    """The frozen snapshot, indexed for replay. Read-only."""

    def __init__(self, snapshot: str | Path, min_transfer_sec: int = DEFAULT_MIN_TRANSFER):
        snap = Path(snapshot)
        self.min_transfer_sec = min_transfer_sec

        st = pd.read_parquet(snap / "stop_times.parquet")
        st = st.sort_values(["trip_id", "stop_sequence"])
        self.stations = pd.read_parquet(snap / "stations.parquet")

        # trip_id -> {station_id: (seq, arr, dep)}
        # A trip may call at the same station twice (rare; loops). Keep the
        # first call, and record duplicates so we can refuse to guess.
        self._trips: dict[str, dict[str, tuple[int, int, int]]] = {}
        self._revisited: set[str] = set()
        for tid, d in st.groupby("trip_id", sort=False):
            calls: dict[str, tuple[int, int, int]] = {}
            for seq, sid, arr, dep in zip(d.stop_sequence, d.station_id, d.arr, d.dep):
                if sid in calls:
                    self._revisited.add(tid)
                    continue
                calls[sid] = (int(seq), int(arr), int(dep))
            self._trips[tid] = calls

        self._names = dict(zip(self.stations.station_id, self.stations.stop_name))
        self._component = dict(zip(self.stations.station_id, self.stations.component))

        # Coordinates, for the navigational distance signal. Some border stations
        # lack them (geolocatable == False); distance to/from those is undefined
        # and returned as None, never guessed.
        self._lat: dict[str, float] = {}
        self._lon: dict[str, float] = {}
        geo = self.stations
        if "geolocatable" in geo.columns:
            geo = geo[geo.geolocatable]
        for sid, la, lo in zip(geo.station_id, geo.stop_lat, geo.stop_lon):
            self._lat[sid] = float(la)
            self._lon[sid] = float(lo)

    def has_coords(self, station_id: str) -> bool:
        return station_id in self._lat

    def distance_km(self, a: str, b: str) -> float | None:
        """Great-circle distance between two stations, km. None if either lacks
        coordinates -- feasibility never depends on this, so undefined is safe.

        This is a straight-line HEURISTIC, deliberately not travel time: on a
        branch-line network the nearest station by air is often not the nearest
        by rail, so the signal guides search without dictating it."""
        if a not in self._lat or b not in self._lat:
            return None
        from math import radians, sin, cos, asin, sqrt

        la1, lo1, la2, lo2 = map(radians, (self._lat[a], self._lon[a], self._lat[b], self._lon[b]))
        h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
        return 2 * 6371.0 * asin(sqrt(h))

    def has_trip(self, trip_id: str) -> bool:
        return trip_id in self._trips

    def calls(self, trip_id: str) -> dict[str, tuple[int, int, int]]:
        return self._trips[trip_id]

    def name(self, station_id: str) -> str:
        return self._names.get(station_id, station_id)

    def component(self, station_id: str) -> int | None:
        return self._component.get(station_id)


def replay(
    legs: list[Leg],
    world: World,
    min_transfer_sec: int | None = None,
) -> Report:
    """Exact feasibility. Collects every reason, not just the first -- a policy
    that learns from one error per episode learns slowly."""
    min_transfer = (
        world.min_transfer_sec if min_transfer_sec is None else min_transfer_sec
    )
    reasons: list[str] = []

    if not legs:
        return Report(False, ["itinerary is empty"])

    # --- each leg, in isolation -------------------------------------------
    resolved = []  # (leg, board_seq, board_dep, alight_seq, alight_arr)
    for i, leg in enumerate(legs):
        if not world.has_trip(leg.trip_id):
            reasons.append(f"leg {i}: trip {leg.trip_id} does not run on this date")
            continue

        calls = world.calls(leg.trip_id)
        if leg.board not in calls:
            reasons.append(
                f"leg {i}: trip {leg.trip_id} does not call at {world.name(leg.board)}"
            )
        if leg.alight not in calls:
            reasons.append(
                f"leg {i}: trip {leg.trip_id} does not call at {world.name(leg.alight)}"
            )
        if leg.board not in calls or leg.alight not in calls:
            continue

        if leg.board == leg.alight:
            reasons.append(f"leg {i}: boards and alights at {world.name(leg.board)}")
            continue

        b_seq, _, b_dep = calls[leg.board]
        a_seq, a_arr, _ = calls[leg.alight]
        if b_seq >= a_seq:
            reasons.append(
                f"leg {i}: trip {leg.trip_id} runs "
                f"{world.name(leg.alight)} -> {world.name(leg.board)}, "
                f"not the direction requested"
            )
            continue

        resolved.append((leg, b_seq, b_dep, a_seq, a_arr))

    if len(resolved) != len(legs):
        return Report(False, reasons)

    # --- legs must chain ---------------------------------------------------
    for i in range(len(resolved) - 1):
        (_, _, _, _, arr) = resolved[i]
        (nxt, _, dep, _, _) = resolved[i + 1]
        prev_alight = resolved[i][0].alight

        if nxt.board != prev_alight:
            reasons.append(
                f"legs {i}->{i+1}: alight at {world.name(prev_alight)} but board "
                f"at {world.name(nxt.board)}"
            )
            continue

        if dep < arr + min_transfer:
            slack = dep - arr
            reasons.append(
                f"legs {i}->{i+1}: {slack}s to change at "
                f"{world.name(prev_alight)}, need {min_transfer}s"
            )

    if reasons:
        return Report(False, reasons)

    # --- facts -------------------------------------------------------------
    visited: list[str] = []
    for leg, b_seq, _, a_seq, _ in resolved:
        calls = world.calls(leg.trip_id)
        span = sorted(
            (sid for sid, (seq, _, _) in calls.items() if b_seq <= seq <= a_seq),
            key=lambda sid: calls[sid][0],
        )
        visited.extend(span if not visited else span[1:])  # don't repeat the junction

    departure = resolved[0][2]
    arrival = resolved[-1][4]
    return Report(
        feasible=True,
        departure=departure,
        arrival=arrival,
        duration=arrival - departure,
        transfers=len(legs) - 1,
        stations_visited=visited,
        legs=list(legs),
    )


def hhmm(seconds: int) -> str:
    """Extended GTFS time. 25:30 is half past one, the next morning."""
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}"