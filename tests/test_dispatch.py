"""Technician routing: prize-collecting, within the workday."""

from cpa.ops.dispatch import Job, km, plan_country

TECHS = [{"tech_id": "T1", "lat": 52.52, "lon": 13.40}, {"tech_id": "T2", "lat": 48.14, "lon": 11.58}]


def test_jobs_are_done_by_the_nearest_technician_and_unreachable_ones_wait():
    jobs = [
        Job("near", "C1", 52.55, 13.45, 500.0, 90),
        Job("munich", "C2", 48.2, 11.6, 400.0, 90),
        Job("far-cheap", "C3", 54.3, 10.1, 1.0, 90),
    ]
    out = plan_country(jobs, TECHS, workday_min=540, speed_kmh=70, km_cost=0.45, max_stops=5, seconds=0.5)
    done = {j.key: t for t, js in out.items() for j in js}
    assert done.get("near") == "T1" and done.get("munich") == "T2" and "far-cheap" not in done


def test_stops_per_technician_are_capped():
    jobs = [Job(f"J{i}", f"C{i}", 52.5 + i * 0.01, 13.4, 300.0, 30) for i in range(10)]
    out = plan_country(jobs, TECHS[:1], workday_min=540, speed_kmh=70, km_cost=0.45, max_stops=3, seconds=0.5)
    assert len(out["T1"]) == 3


def test_road_distance_is_longer_than_straight_line():
    assert 1.2 < km((52.52, 13.40), (48.14, 11.58)) / 504 < 1.3


def test_a_cheap_job_is_still_done_when_the_day_has_room():
    """Deferring a job saves nothing (the charger still needs the visit), so low value alone is no reason to drop it."""
    jobs = [Job("cheap", "C1", 52.9, 13.9, 1.0, 90)]
    out = plan_country(jobs, TECHS[:1], workday_min=540, speed_kmh=70, km_cost=0.45, max_stops=5, seconds=0.3)
    assert [j.key for j in out.get("T1", [])] == ["cheap"]
