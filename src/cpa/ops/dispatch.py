"""Technicians' routes for a day: a prize-collecting vehicle routing problem per country (OR-Tools).

Each job has a value (the downtime it ends: expected lost revenue and failed attempts until the next
chance to fix it) and a service time; each technician leaves from home at the start of the workday
and must be back by its end. The solver chooses which jobs to do and in which order, maximising the
value collected minus the cost of driving. Jobs it drops stay in the queue for tomorrow, so dropping is
only worth it when the day is full."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

DEFER_PENALTY = 1_000_000  # cents; far above any day's driving cost


@dataclass
class Job:
    key: str  # incident id
    charger_id: str
    lat: float
    lon: float
    value_eur: float
    service_min: int
    parts: list[str] = field(default_factory=list)


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h)) * 1.25  # road detour


def plan_country(
    jobs: list[Job],
    techs: list[dict[str, Any]],
    *,
    workday_min: int,
    speed_kmh: float,
    km_cost: float,
    max_stops: int,
    seconds: float,
) -> dict[str, list[Job]]:
    if not jobs or not techs:
        return {}
    pts = [(t["lat"], t["lon"]) for t in techs] + [(j.lat, j.lon) for j in jobs]
    nv, nj = len(techs), len(jobs)
    n = nv + nj
    minutes = [[int(round(km(pts[i], pts[k]) / speed_kmh * 60)) for k in range(n)] for i in range(n)]
    cost = [[int(round(km(pts[i], pts[k]) * km_cost * 100)) for k in range(n)] for i in range(n)]
    starts = list(range(nv))
    mgr = pywrapcp.RoutingIndexManager(n, nv, starts, starts)
    routing = pywrapcp.RoutingModel(mgr)

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitMatrix(cost))
    service = [0] * nv + [j.service_min for j in jobs]
    travel = [[minutes[i][k] + service[i] for k in range(n)] for i in range(n)]
    t_idx = routing.RegisterTransitMatrix(travel)
    routing.AddDimension(t_idx, 0, workday_min, True, "time")

    c_idx = routing.RegisterUnaryTransitVector([0] * nv + [1] * nj)
    routing.AddDimensionWithVehicleCapacity(c_idx, 0, [max_stops] * nv, True, "stops")
    # A dropped job is not saved, only deferred: the charger still needs the visit and keeps failing drivers
    # meanwhile. So every job carries a large penalty (do as many as fit), and its value decides which ones wait
    # when technicians run out of day.
    for j, job in enumerate(jobs):
        routing.AddDisjunction([mgr.NodeToIndex(nv + j)], DEFER_PENALTY + int(round(job.value_eur * 100)))
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(int(seconds * 1000))
    sol = routing.SolveWithParameters(p)
    out: dict[str, list[Job]] = {}
    if sol is None:
        return out
    for v in range(nv):
        i = routing.Start(v)
        route = []
        while not routing.IsEnd(i):
            node = mgr.IndexToNode(i)
            if node >= nv:
                route.append(jobs[node - nv])
            i = sol.Value(routing.NextVar(i))
        if route:
            out[techs[v]["tech_id"]] = route
    return out
