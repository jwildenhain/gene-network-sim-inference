"""The background job queue: ordering, dependencies, cancellation, failure isolation, and the real wiring."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from grn_sim.web import jobs


def _wait(manager, job_id, states, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        job = manager.get(job_id)
        if job["status"] in states:
            return job
        time.sleep(0.02)
    raise AssertionError(f"{job_id} stuck in {manager.get(job_id)['status']}")


class Fake:
    """Injectable stages that record what ran and can be made to block, fail or loop."""

    def __init__(self):
        self.order = []
        self.gate = None  # threading.Event the generate stage waits for
        self.fail_generate = False

    def generate(self, params, ctx):
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.fail_generate:
            raise RuntimeError("boom")
        self.order.append(("gen", params.get("seed")))
        return {"run_id": f"run-{params.get('seed')}", "num_genes": params.get("num_genes")}

    def infer(self, run_id, algs, ctx):
        for i, a in enumerate(algs):
            ctx.check_cancel()
            ctx.progress(f"{a}", advance=bool(i))
            time.sleep(0.05)
            self.order.append(("inf", run_id, a))
        ctx.progress(advance=True)
        return {"algorithms_run": algs}


@pytest.fixture
def make():
    made = []

    def build(fake=None, **kw):
        fake = fake or Fake()
        m = jobs.JobManager(fake.generate, fake.infer, ["s2g", "g2s", "nir"], can_infer=lambda r: r == "known", **kw)
        made.append(m)
        return m, fake

    yield build
    for m in made:
        m.shutdown()


def test_pipeline_runs_generate_then_infer_and_reports_progress(make):
    m, fake = make()
    (job,) = m.submit({"params": {"num_genes": 8, "seed": 3}, "algorithms": ["s2g", "nir"]})
    done = _wait(m, job.id, jobs.FINISHED)
    assert done["status"] == jobs.DONE and done["progress_pct"] == 100.0
    assert done["steps_done"] == done["steps_total"] == 3
    assert done["run_id"] == "run-3" and done["summary"]["algorithms_run"] == ["s2g", "nir"]
    assert fake.order == [("gen", 3), ("inf", "run-3", "s2g"), ("inf", "run-3", "nir")]
    assert any("generated" in line for line in m.get(job.id)["log"])


def test_jobs_run_in_submission_order_one_at_a_time(make):
    m, fake = make()
    fake.gate = threading.Event()
    a, b = m.submit({"params": {"seed": 1}, "count": 2, "algorithms": []})
    assert (a.params["seed"], b.params["seed"]) == (1, 2)  # a batch is a set of different networks
    time.sleep(0.2)
    assert m.get(a.id)["status"] == jobs.RUNNING and m.get(b.id)["status"] == jobs.QUEUED
    fake.gate.set()
    _wait(m, b.id, jobs.FINISHED)
    assert [o[1] for o in fake.order] == [1, 2]


def test_infer_can_be_queued_before_its_network_exists(make):
    m, fake = make()
    fake.gate = threading.Event()
    (gen,) = m.submit({"params": {"seed": 7}})
    (inf,) = m.submit({"depends_on": gen.id, "algorithms": ["g2s"]})
    assert m.get(inf.id)["status"] == jobs.QUEUED and m.get(inf.id)["run_id"] is None
    fake.gate.set()
    done = _wait(m, inf.id, jobs.FINISHED)
    assert done["status"] == jobs.DONE and done["run_id"] == "run-7"
    assert fake.order[-1] == ("inf", "run-7", "g2s")


def test_dependent_job_fails_when_its_network_job_failed(make):
    m, fake = make()
    fake.fail_generate = True
    (gen,) = m.submit({"params": {"seed": 1}})
    (inf,) = m.submit({"depends_on": gen.id, "algorithms": ["s2g"]})
    assert "boom" in _wait(m, gen.id, jobs.FINISHED)["error"]
    failed = _wait(m, inf.id, jobs.FINISHED)
    assert failed["status"] == jobs.FAILED and gen.id in failed["error"]
    fake.fail_generate = False  # the worker survived the failure
    (ok,) = m.submit({"params": {"seed": 2}})
    assert _wait(m, ok.id, jobs.FINISHED)["status"] == jobs.DONE


def test_cancel_queued_and_running_jobs(make):
    m, fake = make()
    fake.gate = threading.Event()
    (running,) = m.submit({"params": {"seed": 1}, "algorithms": ["s2g", "g2s", "nir"]})
    (queued,) = m.submit({"params": {"seed": 2}})
    _wait(m, running.id, {jobs.RUNNING})
    assert m.cancel(queued.id)["status"] == jobs.CANCELLED
    m.cancel(running.id)
    fake.gate.set()
    done = _wait(m, running.id, jobs.FINISHED)
    assert done["status"] == jobs.CANCELLED and "network run-1 was kept" in done["error"]
    assert ("inf", "run-1", "s2g") not in fake.order
    assert ("gen", 2) not in fake.order  # the cancelled queued job never ran
    assert m.cancel("job-9999") is None


@pytest.mark.parametrize("spec, message", [
    ({}, "exactly one"),
    ({"params": {}, "run_id": "known"}, "exactly one"),
    ({"run_id": "known"}, "at least one algorithm"),
    ({"run_id": "gone", "algorithms": ["s2g"]}, "unknown network"),
    ({"params": {}, "algorithms": ["nope"]}, "unknown algorithm"),
    ({"params": {}, "count": 0}, "count must be"),
    ({"params": {}, "count": 999}, "count must be"),
    ({"run_id": "known", "algorithms": ["s2g"], "count": 2}, "only applies"),
    ({"depends_on": "job-0042", "algorithms": ["s2g"]}, "unknown job"),
    ({"params": "x"}, "params must be an object"),
])
def test_bad_requests_are_rejected_with_a_reason(make, spec, message):
    m, _ = make()
    with pytest.raises(ValueError, match=message):
        m.submit(spec)


def test_queue_limit_and_clearing_finished_jobs(make):
    m, fake = make(max_active=2)
    fake.gate = threading.Event()
    m.submit({"params": {"seed": 1}, "count": 2})
    with pytest.raises(jobs.QueueFull):
        m.submit({"params": {"seed": 5}})
    fake.gate.set()
    for j in m.list():
        _wait(m, j["id"], jobs.FINISHED)
    assert m.clear_finished() == 2 and m.list() == []


def test_only_the_newest_finished_jobs_are_kept(make):
    m, _ = make(keep_finished=3)
    for seed in range(6):
        (j,) = m.submit({"params": {"seed": seed}})
        _wait(m, j.id, jobs.FINISHED)
    assert len(m.list()) == 3


def test_a_finished_job_with_queued_dependents_is_not_trimmed(make):
    import threading
    fake = Fake()
    m, _ = make(fake, keep_finished=2)
    (gen,) = m.submit({"params": {"seed": 1}})
    _wait(m, gen.id, jobs.FINISHED)
    fake.gate = threading.Event()  # block the next generation so the dependents stay queued behind it
    (blocker,) = m.submit({"params": {"seed": 9}})
    deps = [m.submit({"depends_on": gen.id, "algorithms": ["s2g"]})[0] for _ in range(5)]
    fake.gate.set()
    for j in [blocker] + deps:
        assert _wait(m, j.id, jobs.FINISHED)["status"] == jobs.DONE  # none failed with "no longer exists"
    assert m.get(gen.id) is not None or len(m.list()) <= 2  # trimmed once nothing depends on it any more


# --- the real wiring ----------------------------------------------------------------------------------------------
def _call(url, method="GET", body=None):
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _poll(app_url, job_id, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        _, body = _call(f"{app_url}/api/jobs/{job_id}")
        if body["job"]["status"] in jobs.FINISHED:
            return body["job"]
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_generate_then_infer_through_the_http_api_keeps_the_studio_network(app_url):
    from grn_sim.web import server

    before = server.CURRENT_STATE.get("run_id")
    code, body = _call(f"{app_url}/api/jobs", "POST",
                       {"params": {"num_genes": 6, "seed": 11, "topology": "random", "k_av": 1.5, "noise": 0.0},
                        "algorithms": []})
    assert code == 202
    gen_id = body["jobs"][0]["id"]
    code, body = _call(f"{app_url}/api/jobs", "POST", {"depends_on": gen_id, "algorithms": ["s2g", "nir"]})
    assert code == 202
    inf = _poll(app_url, body["jobs"][0]["id"])
    gen = _poll(app_url, gen_id)
    assert gen["status"] == inf["status"] == jobs.DONE, (gen, inf)
    assert inf["run_id"] == gen["run_id"] and inf["summary"]["algorithms_run"] == ["s2g", "nir"]

    stored = server.SIMULATION_RUN_STORE[gen["run_id"]]["inferred_payload"]
    assert set(stored["metrics"]) == {"s2g", "nir"}
    assert server.CURRENT_STATE.get("run_id") == before  # the job did not replace the studio's network

    code, listing = _call(f"{app_url}/api/jobs")
    assert code == 200 and listing["counts"]["done"] >= 2


def test_http_errors_are_reported_not_swallowed(app_url):
    assert _call(f"{app_url}/api/jobs", "POST", {"params": {"seed": -5}})[0] == 400
    assert _call(f"{app_url}/api/jobs", "POST", {"params": {"num_genes": 9999}})[0] == 400
    assert _call(f"{app_url}/api/jobs", "POST", {"params": {"topology": "ring"}})[0] == 400
    assert _call(f"{app_url}/api/jobs/job-9999")[0] == 404
    assert _call(f"{app_url}/api/jobs/job-9999/cancel", "POST", {})[0] == 404
    code, body = _call(f"{app_url}/api/simulate", "POST", {"seed": -5})
    assert code == 400 and "seed" in body["error"]


def test_algorithm_keys_match_the_inference_runners():
    """The job queue, the API and the studio's runners all read one registry, so they cannot drift apart."""
    from grn_sim.inference import registry
    from grn_sim.web import server

    assert server.known_algorithms() == registry.keys() == list(server.KNOWN_ALGORITHMS)
    for use_reg in (True, False):                       # every known key builds a runner for either data model
        ctx = registry.BuildContext(k_est=3, use_reg=use_reg)
        for key in server.known_algorithms():
            assert registry.build(key, ctx).fit is not None


def test_interactive_requests_get_an_immediate_409_while_a_job_holds_the_engine(app_url, monkeypatch):
    from grn_sim.web import server

    monkeypatch.setattr(server.JOB_MANAGER, "running",
                        lambda: {"id": "job-0042", "label": "big one", "progress_pct": 12.0})
    held, release = threading.Event(), threading.Event()

    def hold():
        with server.STATE_LOCK:
            held.set()
            release.wait(10)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    assert held.wait(5)
    try:
        started = time.time()
        for url, body in (("/api/simulate", {"num_genes": 6}), ("/api/infer", {"algorithms": ["s2g"]})):
            code, reply = _call(app_url + url, "POST", body)
            assert code == 409 and "job-0042" in reply["error"] and "Job Queue" in reply["error"]
        assert time.time() - started < 3  # answered, not queued behind the lock
    finally:
        release.set()
        t.join(5)
    # with the engine free again the same request goes through
    assert _call(app_url + "/api/simulate", "POST", {"num_genes": 6, "seed": 1})[0] == 200


def test_a_second_inference_job_adds_to_the_first_one_instead_of_replacing_it(app_url):
    from grn_sim.web import server

    code, body = _call(f"{app_url}/api/jobs", "POST",
                       {"params": {"num_genes": 6, "seed": 21, "topology": "random", "noise": 0.0}, "algorithms": ["s2g"]})
    assert code == 202
    first = _poll(app_url, body["jobs"][0]["id"])
    run_id = first["run_id"]
    s2g_before = server.SIMULATION_RUN_STORE[run_id]["inferred_payload"]["metrics"]["s2g"]

    code, body = _call(f"{app_url}/api/jobs", "POST", {"run_id": run_id, "algorithms": ["nir", "g2s"]})
    assert code == 202
    second = _poll(app_url, body["jobs"][0]["id"])
    assert second["status"] == jobs.DONE, second
    assert second["summary"]["algorithms_run"] == ["nir", "g2s"]
    assert set(second["summary"]["algorithms_stored"]) == {"s2g", "nir", "g2s"}

    stored = server.SIMULATION_RUN_STORE[run_id]["inferred_payload"]
    assert set(stored["metrics"]) == {"s2g", "nir", "g2s"}
    assert stored["metrics"]["s2g"] == s2g_before  # the earlier result is untouched
    for per_alg in ("tmp_statmat", "confusion_matrix", "comparison_graphs", "pred_matrices", "svgs_dot"):
        assert set(stored[per_alg]) == {"s2g", "nir", "g2s"}, per_alg
    assert stored["primary_algorithm"] == "s2g"

    # it is what the archive serves too, not only the in-memory copy
    loaded = _call(f"{app_url}/api/history/load?run_id={run_id}")[1]
    assert set(loaded["inferred"]["metrics"]) == {"s2g", "nir", "g2s"}


# --- the job list survives a restart --------------------------------------------------------------------------------
def _manager(store, fake=None, **kw):
    fake = fake or Fake()
    return jobs.JobManager(fake.generate, fake.infer, ["s2g", "g2s", "nir"], can_infer=lambda r: r == "run-1",
                           store_path=store, **kw), fake


def test_finished_jobs_are_listed_again_after_a_restart(tmp_path):
    store = tmp_path / "jobs.json"
    m, _ = _manager(store)
    (job,) = m.submit({"params": {"num_genes": 8, "seed": 1}, "algorithms": ["s2g"], "label": "my network"})
    _wait(m, job.id, jobs.FINISHED)
    m.shutdown()

    again, _ = _manager(store)
    (restored,) = again.list()
    assert restored["id"] == job.id and restored["status"] == jobs.DONE and restored["label"] == "my network"
    assert restored["run_id"] == "run-1" and restored["summary"]["algorithms_run"] == ["s2g"]
    assert restored["progress_pct"] == 100.0 and any("generated" in line for line in again.get(job.id)["log"])
    assert restored["infer_ok"] is True  # the network is (here: pretended to be) in memory
    (next_job,) = again.submit({"params": {"seed": 2}})
    assert next_job.id != job.id and int(next_job.id[-4:]) > int(job.id[-4:])  # ids do not restart at 1
    again.shutdown()


def test_jobs_that_were_active_at_shutdown_come_back_as_interrupted(tmp_path):
    store = tmp_path / "jobs.json"
    m, fake = _manager(store)
    fake.gate = threading.Event()
    (running,) = m.submit({"params": {"seed": 1}})
    (queued,) = m.submit({"params": {"seed": 2}})
    _wait(m, running.id, {jobs.RUNNING})
    saved = json.loads(store.read_text())  # what is on disk while they are active
    assert {j["status"] for j in saved["jobs"]} == {jobs.RUNNING, jobs.QUEUED}
    fake.gate.set()
    m.shutdown()

    store.write_text(json.dumps(saved))  # pretend the process died right there
    again, _ = _manager(store)
    states = {j["id"]: j for j in again.list()}
    assert {s["status"] for s in states.values()} == {jobs.FAILED}
    assert all("interrupted" in s["error"] and "restarted" in s["error"] for s in states.values())
    assert again.counts()[jobs.QUEUED] == 0 and again.counts()[jobs.RUNNING] == 0
    again.shutdown()


def test_only_the_last_ten_finished_jobs_are_kept_on_disk(tmp_path):
    store = tmp_path / "jobs.json"
    m, _ = _manager(store, keep_finished=10)
    for seed in range(14):
        (j,) = m.submit({"params": {"seed": seed}})
        _wait(m, j.id, jobs.FINISHED)
    kept = [j["params"]["seed"] for j in json.loads(store.read_text())["jobs"]]
    assert sorted(kept) == list(range(4, 14)) and len(m.list()) == 10
    m.shutdown()


def test_clearing_is_persisted_and_a_damaged_file_is_ignored(tmp_path):
    store = tmp_path / "jobs.json"
    m, _ = _manager(store)
    (j,) = m.submit({"params": {"seed": 1}})
    _wait(m, j.id, jobs.FINISHED)
    assert m.clear_finished() == 1
    m.shutdown()
    assert _manager(store)[0].list() == []

    store.write_text("{ this is not json")
    broken, _ = _manager(store)  # must not raise
    assert broken.list() == []
    (j2,) = broken.submit({"params": {"seed": 2}})
    assert _wait(broken, j2.id, jobs.FINISHED)["status"] == jobs.DONE
    assert json.loads(store.read_text())["jobs"]  # and it saves again
    broken.shutdown()


def test_infer_ok_is_false_for_networks_no_longer_in_memory(tmp_path):
    m, _ = _manager(None)
    (a,) = m.submit({"params": {"seed": 1}})  # fake network "run-1": known to can_infer
    (b,) = m.submit({"params": {"seed": 2}})  # fake network "run-2": not in memory
    _wait(m, b.id, jobs.FINISHED)
    by_id = {j["id"]: j for j in m.list()}
    assert by_id[a.id]["infer_ok"] is True and by_id[b.id]["infer_ok"] is False
    m.shutdown()


def test_a_submitted_job_is_registered_listed_archived_and_can_be_reviewed(app_url):
    """Submit through the HTTP API, then review the outcome the way the pages do: the job record, the job list, the
    archive search and the archived run's detail."""
    code, body = _call(f"{app_url}/api/jobs", "POST",
                       {"params": {"num_genes": 6, "seed": 21, "topology": "random", "k_av": 1.5, "noise": 0.0},
                        "algorithms": ["s2g", "nir"], "label": "review test"})
    assert code == 202 and body["status"] == "queued"
    job_id = body["jobs"][0]["id"]
    assert body["jobs"][0]["label"] == "review test"

    job = _poll(app_url, job_id)
    assert job["status"] == jobs.DONE and job["error"] is None and job["progress_pct"] == 100.0
    run_id = job["run_id"]
    assert run_id and set(job["summary"]["algorithms_stored"]) >= {"s2g", "nir"}

    # the job record and the job list agree
    code, one = _call(f"{app_url}/api/jobs/{job_id}")
    assert code == 200 and one["job"]["run_id"] == run_id
    code, listing = _call(f"{app_url}/api/jobs")
    assert job_id in [j["id"] for j in listing["jobs"]]

    # the archive search finds exactly this run, in the view that holds every run
    code, view = _call(f"{app_url}/api/history?set=everything&q={run_id}&page=1&page_size=25")
    assert code == 200 and view["view"]["total"] == 1 and view["history"][0]["run_id"] == run_id
    assert view["history"][0]["num_genes"] == 6

    # its detail view carries the algorithm results
    code, detail = _call(f"{app_url}/api/history/load", "POST", {"run_id": run_id})
    assert code == 200 and set(detail["inferred"]["metrics"]) >= {"s2g", "nir"}

    # a search that matches nothing is an empty page that echoes the query (the page shows "No run id contains ...")
    code, none = _call(f"{app_url}/api/history?set=everything&q=zzz-no-such-run&page=1&page_size=25")
    assert code == 200 and none["view"]["total"] == 0 and none["history"] == [] and none["view"]["q"] == "zzz-no-such-run"


def test_an_unknown_run_id_is_reported_as_unknown_not_as_a_restart_problem(app_url):
    code, body = _call(f"{app_url}/api/jobs", "POST", {"run_id": "iorn-does-not-exist", "algorithms": ["s2g"]})
    assert code == 400
    assert "unknown network" in body["error"] and "restart" not in body["error"]


def test_start_up_examples_are_completed_with_the_remaining_algorithms_by_queued_jobs(tmp_path, monkeypatch):
    """The seeded example runs are analysed with a fast subset; the real server queues the other registered algorithms for them."""
    from grn_sim.web import server
    monkeypatch.setenv("RGN_DB_PATH", str(tmp_path / "seed.sqlite"))
    submitted, runs = [], iter(["seed-a", "seed-b", "seed-c", "seed-d", "seed-e", "seed-f", "seed-g"])
    monkeypatch.setattr(server, "load_all_history_from_db", lambda *a, **k: [])
    monkeypatch.setattr(server, "init_db", lambda *a, **k: None)
    monkeypatch.setattr(server, "_execute_simulation", lambda p: server.CURRENT_STATE.update(run_id=next(runs)))
    monkeypatch.setattr(server, "_execute_inference", lambda algs: submitted.append(("inf", list(algs))))
    monkeypatch.setattr(server, "_load_archived_run", lambda rid: None)
    jobs_seen = []
    monkeypatch.setattr(server.JOB_MANAGER, "submit", lambda spec: jobs_seen.append(spec) or [])
    saved = dict(server.CURRENT_STATE)
    try:
        server._bootstrap_initial_simulation(complete_with_jobs=False)
        assert jobs_seen == []                                                  # lazy bootstrap (tests, /api/state): no jobs
        runs = iter(["seed-a", "seed-b", "seed-c", "seed-d", "seed-e", "seed-f", "seed-g"])
        server._bootstrap_initial_simulation(complete_with_jobs=True)
    finally:
        server.CURRENT_STATE.clear()
        server.CURRENT_STATE.update(saved)
    assert all(alg == server.SEED_ALGORITHMS for _, alg in submitted)
    assert len(jobs_seen) == 7 and [j["run_id"] for j in jobs_seen] == [f"seed-{c}" for c in "abcdefg"]
    expected = [a for a in server.known_algorithms() if a not in server.SEED_ALGORITHMS]
    assert all(j["algorithms"] == expected for j in jobs_seen) and expected
    assert set(expected) | set(server.SEED_ALGORITHMS) == set(server.known_algorithms())
