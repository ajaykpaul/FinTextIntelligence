"""Config, registry, monitoring, decision log and the evaluation harness."""

from __future__ import annotations

import json

import pytest
from conftest import T0, WINDOWS, make_document

from fintext.audit import DecisionLog
from fintext.config import ConfigError, load_config
from fintext.evaluation import (
    CANARY,
    ReplayCase,
    auroc,
    binary_scores,
    confusion,
    evaluate_acceptance,
    expected_calibration_error,
    macro_f1,
    replay_provenance,
    run_canary,
    tune_threshold,
)
from fintext.monitoring import Latency, Monitor, psi
from fintext.registry import Entry, Registry, RegistryError, file_digest
from fintext.schemas import Capability, ClaimType, Language, Profile, ScoreRequest

PASSING = {
    "sentiment_macro_f1_en": 0.83,
    "toxicity_auroc_en": 0.96,
    "toxicity_auroc_ar": 0.93,
    "toxicity_f1_en": 0.87,
    "toxicity_f1_ar": 0.82,
    "misleading_f1_en": 0.81,
    "cross_language_f1_gap": 0.05,
    "corroboration_precision": 0.97,
    "claim_extraction_recall": 0.88,
    "pending_resolves_before_expiry": 0.93,
    "human_review_precision_en": 0.92,
    "latency_p95_ms_cpu": 41.0,
    "throughput_rps_per_replica": 140.0,
    "availability": 0.996,
}


class TestConfig:
    def test_loads_the_shipped_defaults(self):
        config = load_config()
        assert config.max_tokens == 256
        assert config.fail_open == ("trading", "research")
        assert config.fail_closed == ("compliance",)

    def test_digest_is_stable(self):
        assert load_config().digest() == load_config().digest()

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("FINTEXT_TOXICITY_EN", "0.42")
        assert load_config().toxicity_en == 0.42

    def test_environment_override_keeps_its_type(self, monkeypatch):
        monkeypatch.setenv("FINTEXT_LEXICONS_ENABLED", "false")
        assert load_config().lexicons_enabled is False

    def test_rejects_unknown_keys(self, tmp_path):
        path = tmp_path / "c.toml"
        path.write_text("speed = 1\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="unknown config keys"):
            load_config(path)

    def test_rejects_a_token_cap_above_the_hot_path_limit(self, tmp_path):
        path = tmp_path / "c.toml"
        path.write_text("max_tokens = 512\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="256-token"):
            load_config(path)

    def test_rejects_a_trading_budget_above_the_timeout(self):
        with pytest.raises(ConfigError, match="cannot exceed"):
            load_config().replace(trading_budget_ms=200.0)

    def test_rejects_a_profile_that_is_both_fail_open_and_fail_closed(self):
        with pytest.raises(ConfigError, match="both fail-open and fail-closed"):
            load_config().replace(fail_open=("compliance",))

    def test_rejects_a_non_positive_window(self):
        with pytest.raises(ConfigError, match="must be positive"):
            load_config().replace(windows={"merger": 0})

    def test_missing_explicit_file_is_an_error(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "absent.toml")

    def test_budget_and_threshold_lookup(self):
        config = load_config()
        assert config.budget_ms("trading") == 50.0
        assert config.budget_ms("compliance") == 500.0
        assert config.toxicity_threshold("ar") == config.toxicity_ar

    def test_unconfigured_window_raises(self):
        with pytest.raises(ConfigError, match="no corroboration window"):
            load_config().window("earnings_call")


class TestRegistry:
    @pytest.fixture
    def registry(self, tmp_path) -> Registry:
        return Registry(tmp_path / "registry", p95_target_ms=50.0)

    def entry(self, version: str, *, p95: float | None = 30.0, **kwargs) -> Entry:
        return Entry(
            version,
            f"/nonexistent/{version}.onnx",
            f"sha256:{version}",
            0.0,
            latency_p95_ms=p95,
            **kwargs,
        )

    def test_registers_and_resolves(self, registry):
        registry.register(self.entry("v1"), verify_digest=False)
        assert registry.resolve("v1").version == "v1"

    def test_versions_are_immutable(self, registry):
        from dataclasses import replace

        registry.register(self.entry("v1"), verify_digest=False)
        changed = replace(registry.resolve("v1"), artifact_digest="sha256:different")
        with pytest.raises(RegistryError, match="immutable"):
            registry.register(changed, verify_digest=False)

    def test_re_registering_identical_content_is_idempotent(self, registry):
        entry = self.entry("v1")
        registry.register(entry, verify_digest=False)
        assert registry.register(entry, verify_digest=False).version == "v1"

    def test_promotion_is_gated_on_latency(self, registry):
        registry.register(self.entry("slow", p95=61.0), verify_digest=False)
        with pytest.raises(RegistryError, match="exceeds"):
            registry.promote("slow")
        assert registry.active() is None

    def test_promotion_refuses_an_unbenchmarked_artefact(self, registry):
        registry.register(self.entry("unknown", p95=None), verify_digest=False)
        with pytest.raises(RegistryError, match="no latency benchmark"):
            registry.promote("unknown")

    def test_forced_promotion_is_recorded(self, registry):
        registry.register(self.entry("slow", p95=61.0), verify_digest=False)
        assert "forced promotion" in registry.promote("slow", force=True).notes

    def test_promotion_sets_the_active_pointer(self, registry):
        registry.register(self.entry("v1"), verify_digest=False)
        registry.promote("v1")
        assert registry.active().version == "v1"

    def test_shadow_needs_no_latency_gate(self, registry):
        registry.register(self.entry("candidate", p95=400.0), verify_digest=False)
        registry.set_shadow("candidate")
        assert registry.shadow().version == "candidate"

    def test_shadow_refuses_an_unknown_version(self, registry):
        with pytest.raises(RegistryError, match="unknown model version"):
            registry.set_shadow("never-registered")

    def test_history_walks_the_parent_chain(self, registry):
        for version, parent in (("v1", None), ("v2", "v1"), ("v3", "v2")):
            registry.register(self.entry(version, parent_version=parent), verify_digest=False)
        assert [e.version for e in registry.history("v3")] == ["v3", "v2", "v1"]

    def test_state_survives_a_reload(self, tmp_path):
        first = Registry(tmp_path / "r")
        first.register(self.entry("v1"), verify_digest=False)
        first.promote("v1")
        assert Registry(tmp_path / "r").active().version == "v1"

    def test_digest_mismatch_is_refused(self, tmp_path):
        artefact = tmp_path / "model.onnx"
        artefact.write_bytes(b"weights")
        registry = Registry(tmp_path / "r")
        with pytest.raises(RegistryError, match="digest mismatch"):
            registry.register(Entry("v1", str(artefact), "sha256:wrong", 0.0))
        assert file_digest(artefact).startswith("sha256:")


class TestMonitoring:
    def test_latency_quantiles(self):
        latency = Latency()
        for value in range(1, 101):
            latency.observe(value)
        assert latency.quantile(0.95) == pytest.approx(95, abs=1)
        assert latency.count == 100

    def test_latency_counts_overflow(self):
        latency = Latency([1, 5])
        latency.observe(900)
        assert latency.overflow == 1

    def test_cumulative_buckets_are_monotonic(self):
        latency = Latency([1, 5, 10])
        for value in (0.5, 2, 7, 9):
            latency.observe(value)
        counts = [count for _, count in latency.cumulative()]
        assert counts == sorted(counts)

    def test_psi_is_zero_for_identical_distributions(self):
        values = [i / 100 for i in range(100)]
        assert psi(values, values) == pytest.approx(0.0, abs=1e-6)

    def test_psi_detects_a_shift(self):
        assert psi([i / 100 for i in range(100)], [0.9 + i / 1000 for i in range(100)]) > 0.2

    def test_drift_alerts_use_the_threshold(self):
        monitor = Monitor(drift_alert=0.2)
        monitor.set_baseline("toxicity", [i / 100 for i in range(100)])
        for _ in range(100):
            monitor.score("toxicity", 0.95)
        assert "toxicity" in monitor.drift_alerts()

    def test_snapshot_and_prometheus(self):
        monitor = Monitor()
        monitor.request(profile="trading", language="en", latency_ms=12.0)
        monitor.error("inference")
        assert monitor.snapshot()["requests"]["trading"] == 1
        text = monitor.prometheus()
        assert text.endswith("\n")
        assert 'fintext_errors_total{kind="inference"} 1' in text
        assert 'fintext_latency_ms_bucket{le="+Inf"} 1' in text


class TestDecisionLog:
    def test_writes_jsonl(self, tmp_path):
        log = DecisionLog(tmp_path / "log.jsonl")
        log.write("score", {"id": "a"})
        log.close()
        record = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
        assert record["kind"] == "score" and record["id"] == "a"

    def test_a_null_path_disables_logging(self):
        log = DecisionLog(None)
        log.write("score", {"id": "a"})
        assert log.written == 0 and log.dropped == 0

    def test_a_full_queue_drops_rather_than_blocks(self, tmp_path):
        # A full disk must degrade the audit trail, not the trading gate.
        log = DecisionLog(tmp_path / "log.jsonl", max_queue=1)
        log._stop.set()
        for index in range(50):
            log.write("score", {"id": index})
        assert log.dropped > 0


class TestMetrics:
    def test_auroc(self):
        assert auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)

    def test_auroc_handles_ties_by_average_rank(self):
        # int8 quantisation produces ties.
        assert auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)

    def test_auroc_is_undefined_for_a_single_class(self):
        assert auroc([1, 1, 1], [0.2, 0.5, 0.9]) == 0.5

    def test_macro_f1_averages_over_declared_labels(self):
        # Averaging over observed labels only inflates the score exactly when a class is
        # missing, which is when a gate should fail.
        assert macro_f1(["a", "a"], ["a", "a"], ["a", "b", "c"]) == pytest.approx(1 / 3)

    def test_macro_f1_rejects_an_empty_split(self):
        with pytest.raises(ValueError):
            macro_f1([], [], ["a"])

    def test_confusion_matrix(self):
        assert confusion(["a", "b"], ["a", "a"], ["a", "b"])["b"]["a"] == 1

    def test_binary_scores_at_the_deployed_threshold(self):
        truth, scores = [0, 1, 1, 0], [0.2, 0.6, 0.9, 0.4]
        assert binary_scores(truth, scores, threshold=0.5)[2] == pytest.approx(1.0)
        assert binary_scores(truth, scores, threshold=0.95)[2] == pytest.approx(0.0)

    def test_threshold_tuning_respects_both_floors(self):
        # Recall-first, bounded by the review-precision floor.
        truth = [0] * 50 + [1] * 50
        scores = [0.1 + 0.004 * i for i in range(50)] + [0.5 + 0.008 * i for i in range(50)]
        _, at = tune_threshold(truth, scores, min_recall=0.90, min_precision=0.90)
        assert at["recall"] >= 0.90 and at["precision"] >= 0.90

    def test_calibration_error(self):
        assert expected_calibration_error([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.0)


class TestAcceptance:
    def test_all_gates_passing(self):
        assert evaluate_acceptance(PASSING).accepted

    def test_a_failing_gate_blocks_acceptance(self):
        report = evaluate_acceptance({**PASSING, "toxicity_auroc_ar": 0.80})
        assert not report.accepted and [f.name for f in report.failures] == ["toxicity_auroc_ar"]

    def test_the_cross_language_gap_is_an_upper_bound(self):
        report = evaluate_acceptance({**PASSING, "cross_language_f1_gap": 0.09})
        assert "cross_language_f1_gap" in [f.name for f in report.failures]

    def test_a_missing_gated_metric_fails(self):
        # "We did not measure it" is not a pass.
        assert not evaluate_acceptance(
            {k: v for k, v in PASSING.items() if k != "latency_p95_ms_cpu"}
        ).accepted

    def test_arabic_sentiment_is_reported_and_never_gates(self):
        report = evaluate_acceptance({**PASSING, "sentiment_macro_f1_ar": 0.31})
        assert report.accepted
        result = next(r for r in report.results if r.name == "sentiment_macro_f1_ar")
        assert not result.gated and result.status == "reported"

    def test_slo_gates_are_enforced(self):
        assert not evaluate_acceptance({**PASSING, "latency_p95_ms_cpu": 62.0}).accepted
        assert not evaluate_acceptance({**PASSING, "availability": 0.98}).accepted

    def test_week_three_gate_recommends_a_scope_cut_without_failing(self):
        report = evaluate_acceptance({**PASSING, "week_three_arabic_toxicity_f1": 0.62})
        assert report.accepted
        assert any("published-lexicon filtering" in note for note in report.notes)

    def test_week_three_gate_passing_is_recorded_too(self):
        report = evaluate_acceptance({**PASSING, "week_three_arabic_toxicity_f1": 0.74})
        assert any("commitment stands" in note for note in report.notes)

    def test_markdown_renders_a_verdict(self):
        assert (
            "**Result: ACCEPTED**" in evaluate_acceptance(PASSING, model_version="v1").to_markdown()
        )


class TestReplay:
    def genuine(self, item_id: str, offset: float, delay: float) -> ReplayCase:
        return ReplayCase(
            item_id=item_id,
            text="First Abu Dhabi Bank PJSC agreed to acquire a regional lender",
            observed_at=T0 + offset,
            is_genuine=True,
            expected_claim_type=ClaimType.MERGER,
            documents=(make_document(doc_id=f"doc-{item_id}", published_at=T0 + offset + delay),),
        )

    def test_genuine_items_corroborate(self):
        result = replay_provenance(
            [self.genuine("a", 0, 120), self.genuine("b", 300, 240)], windows=WINDOWS
        )
        assert result.corroboration_precision == 1.0
        assert result.claim_extraction_recall == 1.0
        assert result.false_corroborations == []

    def test_fabricated_items_never_corroborate(self):
        result = replay_provenance(
            [
                ReplayCase(
                    item_id="f1",
                    text="First Abu Dhabi Bank PJSC agreed to acquire a regional lender",
                    observed_at=T0,
                    is_genuine=False,
                    expected_claim_type=ClaimType.MERGER,
                )
            ],
            windows=WINDOWS,
        )
        assert result.counts["uncorroborated"] == 1 and result.false_corroborations == []

    def test_time_to_corroboration_is_reported_per_claim_type(self):
        result = replay_provenance([self.genuine("a", 0, 180)], windows=WINDOWS)
        assert result.median_time_to_corroboration["merger"] <= 300

    def test_claims_months_apart_each_see_their_own_filing(self):
        # Regression: claims used to be checked first and documents delivered afterwards from
        # the last claim's time, so a replay spanning months left every earlier claim
        # unconfirmed.
        day = 86400
        result = replay_provenance(
            [
                self.genuine("early", 0, 120),
                self.genuine("late", 90 * day, 120),
                self.genuine("cases-out-of-order", 30 * day, 60),
            ],
            windows=WINDOWS,
        )
        assert result.counts["corroborated"] == 3
        assert result.pending_resolves_before_expiry == 1.0
        assert result.median_time_to_corroboration["merger"] <= 180

    def test_a_filing_published_before_the_claim_confirms_it_at_once(self):
        result = replay_provenance([self.genuine("a", 0, -3600)], windows=WINDOWS)
        assert result.counts["corroborated"] == 1
        assert result.median_time_to_corroboration["merger"] == 0

    def test_missed_claims_are_named(self):
        result = replay_provenance(
            [
                ReplayCase(
                    item_id="m1",
                    text="Markets were quiet today",
                    observed_at=T0,
                    is_genuine=True,
                    expected_claim_type=ClaimType.MERGER,
                )
            ],
            windows=WINDOWS,
        )
        assert result.claim_extraction_recall == 0.0 and result.missed_claims == ["m1"]


class TestCanary:
    def test_a_perfect_detector_scores_full_recall(self):
        result = run_canary(lambda text, language: True)
        assert result.recall == 1.0 and not result.triggers_retraining_review

    def test_a_five_point_drop_triggers_a_review(self):
        seen = {"n": 0}

        def flaky(text, language):
            seen["n"] += 1
            return seen["n"] % 5 != 0

        assert run_canary(flaky).triggers_retraining_review

    def test_per_technique_recall_localises_a_regression(self):
        result = run_canary(lambda text, language: "k1ll" not in text and " " in text)
        assert result.by_technique["leetspeak"] == 0.0
        assert result.by_technique["plain"] > 0.0

    def test_the_set_has_negative_controls(self):
        # A canary that measures recall alone rewards a model that flags all.
        assert any(not case.should_flag for case in CANARY)

    def test_the_service_holds_against_the_canary(self, pipeline):
        def is_toxic(text: str, language: Language) -> bool:
            response = pipeline.score(
                ScoreRequest(
                    text=text,
                    language=language,
                    capabilities=(Capability.TOXICITY,),
                    profile=Profile.COMPLIANCE,
                    explain=False,
                )
            )
            return bool(response.toxicity and response.toxicity.toxic)

        result = run_canary(is_toxic)
        assert not result.triggers_retraining_review, f"missed: {result.missed}"
        assert result.false_positives == []
