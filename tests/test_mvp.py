from __future__ import annotations

import json
import os
import signal
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from deepresearch.agents import (
    AuditorAgent,
    BaseDeepSeekAgent,
    PlannerAgent,
    ResearchWorkerAgent,
    ReportWriterAgent,
    SourceEvaluatorAgent,
)
from deepresearch.agents.worker import _weighted_rrf_scores
from deepresearch.audit_policy import aggregate_audits, select_audit_evidence
from deepresearch.benchmark import (
    _record_evaluation_coverage,
    load_tasks,
    main as benchmark_main,
)
from deepresearch.config import Settings
from deepresearch.cli import timestamped_run_id
from deepresearch.embeddings import OpenAICompatibleEmbeddingProvider
from deepresearch.eventlog import LOGGER_NAME
from deepresearch.evidence_policy import (
    build_research_digest,
    curate_evidence,
    select_drilldown_evidence,
    select_supervisor_evidence,
    supervisor_evidence_payload,
)
from deepresearch.graph import (
    _apply_sufficiency_gates,
    _bounded_query,
    _normalize_audit,
    render_report,
    run_graph,
)
from deepresearch.local_services import managed_local_embedding_service
from deepresearch.model_gateway import model_call_scope
from deepresearch.model_gateway.router import DeterministicRouter
from deepresearch.model_gateway.schemas import (
    ModelCallSignature,
    ModelSpec,
    RoutingProfile,
)
from deepresearch.model_gateway.telemetry import (
    BudgetExceededError,
    ModelTelemetryStore,
)
from deepresearch.observability import render_model_observability
from deepresearch.providers import (
    DeepSeekModelProvider,
    FakeLanguageModelProvider,
    FakeSearchProvider,
    TavilySearchProvider,
    rank_page_chunks,
    split_page_content,
    split_retrieval_chunks,
    estimate_text_tokens,
)
from deepresearch.report_policy import normalize_report_draft, select_report_evidence
from deepresearch.runtime import ResearchRuntime
from deepresearch.schemas import (
    AuditCheck,
    AuditGap,
    AuditPlan,
    AuditResult,
    ClaimAssessment,
    ClaimStatus,
    Confidence,
    CredibilityTier,
    CoverageContract,
    EvidenceBatch,
    EvidenceStance,
    ExtractedEvidence,
    QueryIntent,
    QueryPlan,
    QuerySpec,
    ReportDraft,
    ReportStatement,
    ResearchPlan,
    ResearchTask,
    SearchResult,
    SourceAssessment,
    SourceEligibility,
    SourceHardFlag,
    SourceScoreDimensions,
    SourceDirectness,
    SourceType,
    TaskStatus,
    EvidenceRecord,
    ExtractedPage,
    PageExtractionBatch,
    WorkerDecision,
    WorkerDecisionAction,
    WorkerStopReason,
)
from deepresearch.store import EvidenceStore
from deepresearch.source_scoring import finalize_source_assessment
from deepresearch.visualization import render_run_visualization


class FailOncePlanner(FakeLanguageModelProvider):
    def __init__(self) -> None:
        self.failed = False

    def plan(
        self,
        question: str,
        max_tasks: int,
        preliminary_evidence: list[EvidenceRecord] | None = None,
    ):
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected planner failure")
        return super().plan(question, max_tasks, preliminary_evidence)


class FiveDimensionPlanner(FakeLanguageModelProvider):
    def plan(
        self,
        question: str,
        max_tasks: int,
        preliminary_evidence: list[EvidenceRecord] | None = None,
    ) -> ResearchPlan:
        dimensions = [f"dimension {index}" for index in range(1, 6)]
        tasks = [
            ResearchTask(
                task_id=f"T{index}",
                question=f"What evidence addresses dimension {index}?",
                research_role=f"dimension {index} evidence reviewer",
                objective=f"Assess evidence for dimension {index} of {question}",
                must_find=[f"evidence for dimension {index}"],
                covered_dimensions=[dimension],
                query_budget=2,
            )
            for index, dimension in enumerate(dimensions, start=1)
        ]
        return ResearchPlan(
            coverage_contract=CoverageContract(
                decision_type="five-dimensional assessment",
                required_dimensions=dimensions,
            ),
            tasks=tasks[:max_tasks],
        )


class ThreeStepAdaptiveModel(FakeLanguageModelProvider):
    def __init__(self) -> None:
        self.observations: list[tuple[int, int]] = []

    def decide_worker_next_step(
        self,
        task: ResearchTask,
        executed_queries: list[QuerySpec],
        evidence: list[EvidenceRecord],
        step_number: int,
        max_steps: int,
        search_errors: list[str],
    ) -> WorkerDecision:
        self.observations.append((len(executed_queries), len(evidence)))
        if len(executed_queries) >= 3:
            return WorkerDecision(
                action=WorkerDecisionAction.STOP,
                decision_summary="Three adaptive evidence passes are sufficient.",
                stop_reason=WorkerStopReason.SUFFICIENT,
            )
        objective = task.objective or task.question
        next_step = len(executed_queries) + 1
        return WorkerDecision(
            action=WorkerDecisionAction.SEARCH,
            decision_summary="The current evidence leaves a targeted verification gap.",
            unresolved_gaps=[f"verification gap {next_step}"],
            next_query=QuerySpec(
                intent=QueryIntent.VERIFICATION,
                query=f"{objective} adaptive verification pass {next_step}",
                expected_evidence=f"Independent verification evidence for pass {next_step}",
                priority=3,
            ),
        )


class OneWaveReplanningModel(FakeLanguageModelProvider):
    def __init__(self) -> None:
        self.replan_calls = 0

    def audit(
        self,
        question: str,
        evidence: list[EvidenceRecord],
        round_number: int,
        research_context: dict | None = None,
    ) -> AuditResult:
        audit = super().audit(
            question, evidence, round_number, research_context
        )
        if round_number == 0:
            return audit.model_copy(
                update={
                    "sufficient": False,
                    "gaps": ["Verify the central finding in one targeted wave."],
                    "follow_up_query": "central finding targeted verification",
                }
            )
        return audit

    def replan(self, *args, **kwargs):
        self.replan_calls += 1
        return super().replan(*args, **kwargs)


class HierarchicalAuditModel(FakeLanguageModelProvider):
    def __init__(self) -> None:
        self.audit_checks: list[str] = []

    def audit(
        self,
        question: str,
        evidence: list[EvidenceRecord],
        round_number: int,
        research_context: dict | None = None,
    ) -> AuditResult:
        check = (research_context or {}).get("audit_check", {})
        self.audit_checks.append(str(check.get("check_id") or "single"))
        return super().audit(question, evidence, round_number, research_context)


class OriginalSourceSearch:
    def __init__(self) -> None:
        self.call_count = 0
        self.extract_call_count = 0

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.call_count += 1
        return [
            SearchResult(
                title="Secondary overview",
                url="https://secondary.example/article",
                content="Secondary snippet",
                rank=1,
            )
        ]

    def extract_pages(self, urls: list[str]) -> PageExtractionBatch:
        self.extract_call_count += 1
        bodies = {
            "https://secondary.example/article": (
                "The secondary page attributes the measured result to the original study. "
                "https://primary.example/study"
            ),
            "https://primary.example/study": (
                "The original study directly reports the measured result."
            ),
        }
        return PageExtractionBatch(
            pages=[
                ExtractedPage(url=url, raw_content=bodies[url])
                for url in urls
                if url in bodies
            ],
            failed_urls=[url for url in urls if url not in bodies],
        )


class OriginalSourceFollowingModel(FakeLanguageModelProvider):
    def extract(
        self, task: ResearchTask, results: list[SearchResult]
    ) -> EvidenceBatch:
        result = results[0]
        if result.url == "https://secondary.example/article":
            return EvidenceBatch(
                evidence=[
                    ExtractedEvidence(
                        source_rank=1,
                        claim_candidate="A secondary page attributes a measured result.",
                        verbatim_excerpt=(
                            "The secondary page attributes the measured result to the original study."
                        ),
                        stance=EvidenceStance.SUPPORTS,
                        relevance="high",
                        source_type=SourceType.SECONDARY,
                        source_directness=SourceDirectness.ATTRIBUTED,
                        credibility_tier=CredibilityTier.D,
                        discovery_only=True,
                        attributed_source_name="Original study",
                        attributed_source_url="https://primary.example/study",
                    )
                ]
            )
        return EvidenceBatch(
            evidence=[
                ExtractedEvidence(
                    source_rank=1,
                    claim_candidate="The original study reports the result.",
                    verbatim_excerpt=(
                        "The original study directly reports the measured result."
                    ),
                    stance=EvidenceStance.SUPPORTS,
                    relevance="high",
                    source_type=SourceType.PRIMARY_RESEARCH,
                    source_directness=SourceDirectness.DIRECT,
                    credibility_tier=CredibilityTier.B,
                )
            ]
        )

    def decide_worker_next_step(
        self,
        task: ResearchTask,
        executed_queries: list[QuerySpec],
        evidence: list[EvidenceRecord],
        step_number: int,
        max_steps: int,
        search_errors: list[str],
    ) -> WorkerDecision:
        if any(
            item.source_url == "https://primary.example/study"
            for item in evidence
        ):
            return WorkerDecision(
                action=WorkerDecisionAction.STOP,
                decision_summary="The original source was retrieved.",
                stop_reason=WorkerStopReason.SUFFICIENT,
            )
        return WorkerDecision(
            action=WorkerDecisionAction.FOLLOW_SOURCE,
            decision_summary="Retrieve the explicitly attributed original.",
            unresolved_gaps=["The result still relies on secondary attribution."],
            target_source_url="https://primary.example/study",
        )


class ContextCapturingPlanner(FakeLanguageModelProvider):
    def __init__(self) -> None:
        self.preliminary_evidence_seen = 0

    def plan_with_context(
        self,
        question: str,
        max_tasks: int,
        preliminary_evidence: list[EvidenceRecord],
    ) -> ResearchPlan:
        self.preliminary_evidence_seen = len(preliminary_evidence)
        return super().plan_with_context(
            question, max_tasks, preliminary_evidence
        )


class MvpTest(unittest.TestCase):
    def test_deepresearch_benchmark_loader_filters_official_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "query.jsonl"
            dataset.write_text(
                "\n".join([
                    json.dumps({
                        "id": 1, "topic": "Science", "language": "zh",
                        "prompt": "研究问题一",
                    }, ensure_ascii=False),
                    json.dumps({
                        "id": 2, "topic": "Science", "language": "en",
                        "prompt": "Research question two",
                    }),
                    json.dumps({
                        "id": 3, "topic": "Business", "language": "en",
                        "prompt": "Research question three",
                    }),
                ]) + "\n",
                encoding="utf-8",
            )

            selected = load_tasks(
                dataset, language="en", task_ids={"2", "3"}, offset=1, limit=1,
            )

            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].task_id, 3)
            self.assertEqual(selected[0].language, "en")

    def test_deepresearch_benchmark_runs_and_exports_official_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dataset = workspace / "query.jsonl"
            dataset.write_text(
                json.dumps({
                    "id": 42,
                    "topic": "Science & Technology",
                    "language": "en",
                    "prompt": "What evidence supports the intervention?",
                }) + "\n",
                encoding="utf-8",
            )
            argv = [
                "run", "--dataset", str(dataset), "--workspace", str(workspace),
                "--model-name", "unit-test", "--limit", "1", "--fake", "--quiet",
            ]

            self.assertEqual(benchmark_main(argv), 0)
            self.assertEqual(benchmark_main(argv), 0)

            root = workspace / "benchmark_results" / "deepresearch_bench"
            articles = [
                json.loads(line)
                for line in (root / "unit-test.jsonl").read_text().splitlines()
            ]
            queries = [
                json.loads(line)
                for line in (root / "unit-test.queries.jsonl").read_text().splitlines()
            ]
            manifest = [
                json.loads(line)
                for line in (root / "unit-test.manifest.jsonl").read_text().splitlines()
            ]

            self.assertEqual(len(articles), 1)
            self.assertEqual(set(articles[0]), {"id", "prompt", "article"})
            self.assertIn("## Direct answer", articles[0]["article"])
            self.assertIn("## Findings", articles[0]["article"])
            self.assertEqual(queries[0]["language"], "en")
            self.assertEqual(manifest[0]["run_id"], "drb_unit-test_42")
            self.assertEqual(manifest[0]["status"], "completed")

    def test_deepresearch_benchmark_wraps_official_race_and_fact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_dir = root / "official"
            raw_dir = benchmark_dir / "data" / "test_data" / "raw_data"
            raw_dir.mkdir(parents=True)
            (benchmark_dir / "deepresearch_bench_race.py").write_text(
                "# official evaluator fixture\n", encoding="utf-8"
            )
            articles = root / "candidate.jsonl"
            queries = root / "candidate.queries.jsonl"
            articles.write_text(
                json.dumps({"id": 1, "prompt": "Question", "article": "Report"})
                + "\n",
                encoding="utf-8",
            )
            queries.write_text(
                json.dumps({
                    "id": 1, "topic": "Science", "language": "en",
                    "prompt": "Question",
                }) + "\n",
                encoding="utf-8",
            )
            fact_output = root / "official_eval" / "candidate" / "fact"
            fact_output.mkdir(parents=True)
            stale_extracted = fact_output / "extracted.jsonl"
            stale_extracted.write_text(
                json.dumps({"id": 999, "article": "stale"}) + "\n",
                encoding="utf-8",
            )
            (fact_output / "validated.jsonl").write_text(
                json.dumps({"id": 999, "citations_deduped": {}}) + "\n",
                encoding="utf-8",
            )

            with (
                patch.dict(
                    os.environ,
                    {"OPENROUTER_API_KEY": "test", "JINA_API_KEY": "test"},
                    clear=True,
                ),
                patch("deepresearch.benchmark.subprocess.run") as run,
            ):
                def evaluator_fixture(command, **_kwargs):
                    if "deepresearch_bench_race.py" in command:
                        output = Path(command[command.index("--output_dir") + 1])
                        output.mkdir(parents=True, exist_ok=True)
                        (output / "raw_results.jsonl").write_text(
                            json.dumps({"id": 1, "overall_score": 0.5}) + "\n",
                            encoding="utf-8",
                        )
                    if "utils.validate" in command:
                        output = Path(command[command.index("--output_path") + 1])
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text(
                            json.dumps({"id": 1, "citations_deduped": {}}) + "\n",
                            encoding="utf-8",
                        )

                run.side_effect = evaluator_fixture
                result = benchmark_main([
                    "evaluate",
                    "--benchmark-dir", str(benchmark_dir),
                    "--input", str(articles),
                    "--phase", "all",
                    "--max-workers", "2",
                ])

            self.assertEqual(result, 0)
            self.assertEqual(run.call_count, 6)
            self.assertTrue((raw_dir / "candidate.jsonl").exists())
            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn("deepresearch_bench_race.py", commands[0])
            self.assertIn("utils.stat", commands[-1])
            self.assertTrue(
                (root / "official_eval" / "candidate" / "evaluation_manifest.json").exists()
            )
            self.assertFalse(stale_extracted.exists())
            manifest = json.loads(
                (root / "official_eval" / "candidate" / "evaluation_manifest.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["phases"]["fact"]["succeeded_ids"], ["1"])
            self.assertEqual(len(manifest["input_sha256"]), 64)

    def test_official_evaluation_rejects_missing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = root / "raw_results.jsonl"
            result_path.write_text(
                json.dumps({"id": 1, "overall_score": 0.5}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "missing=2"):
                _record_evaluation_coverage(
                    root,
                    {"1", "2"},
                    "race",
                    result_path,
                )

            manifest = json.loads(
                (root / "evaluation_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["phases"]["race"]["missing_ids"], ["2"])

    @staticmethod
    def evidence_record(
        evidence_id: str,
        source_url: str,
        *,
        title: str = "Direct study",
        provenance_key: str | None = None,
    ) -> EvidenceRecord:
        domain = source_url.split("/", 3)[2]
        return EvidenceRecord(
            evidence_id=evidence_id,
            source_id=f"src_{evidence_id}",
            task_id="T1",
            source_title=title,
            source_url=source_url,
            source_domain=domain,
            claim_candidate=f"Evidence candidate {evidence_id}",
            verbatim_excerpt=f"Exact evidence excerpt {evidence_id}",
            stance=EvidenceStance.SUPPORTS,
            relevance="high",
            source_type=SourceType.PRIMARY_RESEARCH,
            source_directness=SourceDirectness.DIRECT,
            credibility_tier=CredibilityTier.B,
            provenance_key=provenance_key or source_url,
        )

    @patch("deepresearch.cli.datetime")
    def test_default_run_id_uses_utc_timestamp(self, mock_datetime: Mock) -> None:
        mock_datetime.now.return_value = datetime(
            2026, 8, 2, 19, 4, 5, 123456, tzinfo=timezone.utc
        )

        self.assertEqual(timestamped_run_id(), "run_20260802T190405_123456Z")
        mock_datetime.now.assert_called_once_with(timezone.utc)

    def make_runtime(self, root: Path):
        settings = Settings(
            workspace=root,
            tavily_api_key=None,
            deepseek_api_key=None,
        )
        settings.ensure_directories()
        store = EvidenceStore(settings.research_db)
        search = FakeSearchProvider()
        model = FakeLanguageModelProvider()
        runtime = ResearchRuntime(settings, store, search, model)
        return settings, store, search, runtime

    def test_deepseek_roles_are_separate_agents_behind_compatible_facade(self) -> None:
        settings = Settings(
            workspace=Path("/tmp"),
            tavily_api_key=None,
            deepseek_api_key="test-key",
        )
        provider = DeepSeekModelProvider(settings)

        self.assertIsInstance(provider.planner, PlannerAgent)
        self.assertIsInstance(provider.worker, ResearchWorkerAgent)
        self.assertIsInstance(provider.source_evaluator, SourceEvaluatorAgent)
        self.assertIsInstance(provider.auditor, AuditorAgent)
        self.assertIsInstance(provider.writer, ReportWriterAgent)
        self.assertIsInstance(provider.planner, BaseDeepSeekAgent)
        self.assertIsInstance(provider.worker, BaseDeepSeekAgent)
        self.assertIsInstance(provider.auditor, BaseDeepSeekAgent)
        self.assertIsInstance(provider.writer, BaseDeepSeekAgent)
        self.assertIsNot(provider.planner, provider.worker)
        self.assertIsNot(provider.worker, provider.auditor)
        self.assertIsNot(provider.worker, provider.source_evaluator)

        expected = FakeLanguageModelProvider().plan("A question", max_tasks=1)
        with patch.object(provider.planner, "plan", return_value=expected) as plan:
            actual = provider.plan("A question", max_tasks=1)
        self.assertEqual(actual, expected)
        plan.assert_called_once_with("A question", 1)

    def test_run_visualization_safely_embeds_untrusted_run_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                workspace=root,
                tavily_api_key=None,
                deepseek_api_key=None,
            )
            settings.ensure_directories()
            store = EvidenceStore(settings.research_db)
            run_id = "run_html_safety"
            question = '</script><script>window.injected = true</script>'
            store.create_run(run_id, question, {"fake": True})

            path = render_run_visualization(
                settings=settings,
                store=store,
                run_id=run_id,
            )
            document = path.read_text(encoding="utf-8")
            observability_path = path.parent / "model_observability.html"
            observability_document = observability_path.read_text(encoding="utf-8")

            self.assertNotIn(question, document)
            self.assertIn("&lt;/script&gt;", document)
            self.assertIn("<\\/script>", document)
            self.assertIn('href="model_observability.html"', document)
            self.assertNotIn(question, observability_document)
            self.assertIn("&lt;/script&gt;", observability_document)
            self.assertIn("<\\/script>", observability_document)
            self.assertIn('href="run_view.html"', observability_document)

    def test_fake_run_produces_auditable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, store, search, runtime = self.make_runtime(root)
            run_id = "run_test"
            question = "Does the intervention work, and what are its limitations?"
            store.create_run(run_id, question, {"fake": True})

            result = run_graph(
                settings=settings,
                store=store,
                runtime=runtime,
                run_id=run_id,
                question=question,
            )

            report_path = Path(result["report_path"])
            self.assertTrue(report_path.exists())
            self.assertTrue((report_path.parent / "trace.json").exists())
            self.assertTrue(
                (report_path.parent / "model_observability.html").exists()
            )
            self.assertTrue((report_path.parent / "model_calls.json").exists())
            self.assertTrue((report_path.parent / "citations.json").exists())
            self.assertTrue((report_path.parent / "report_draft.json").exists())
            self.assertTrue(
                (report_path.parent / "source_assessments.json").exists()
            )
            visualization_path = Path(result["visualization_path"])
            self.assertTrue(visualization_path.exists())
            visualization = visualization_path.read_text(encoding="utf-8")
            self.assertIn("ResearchWorker:T1", visualization)
            self.assertIn("empirical evidence reviewer", visualization)
            self.assertIn("AuditorAgent", visualization)
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("## References", report)
            self.assertRegex(report, r"\[[0-9]+(?:, [0-9]+)*\]")
            self.assertNotIn("Supporting evidence", report)
            self.assertNotIn("Counterevidence or qualification", report)
            self.assertNotIn("Exact evidence excerpt", report)
            self.assertNotIn("Sufficiency gates", report)
            self.assertNotIn("## Run trace", report)
            self.assertNotIn("source-rubric-v1", report)
            citations = json.loads(
                (report_path.parent / "citations.json").read_text(encoding="utf-8")
            )
            self.assertTrue(citations)
            self.assertEqual(citations[0]["citation_number"], 1)
            self.assertTrue(citations[0]["evidence_ids"])
            self.assertGreaterEqual(len(store.list_evidence(run_id)), 3)
            self.assertEqual(result["terminal_reason"], "evidence_sufficient")
            self.assertEqual(search.call_count, 8)
            summary = store.operation_summary(run_id)
            self.assertEqual(summary["model.formulate_queries"], 3)
            self.assertEqual(summary["model.assess_sources"], 4)
            self.assertEqual(summary["model.decide_worker_next_step"], 6)
            self.assertEqual(summary["search.tavily"], 8)
            self.assertEqual(summary["extract.tavily"], 4)
            self.assertEqual(summary["model.synthesize_report"], 1)
            with store.connect() as connection:
                contents = [
                    row["content"]
                    for row in connection.execute(
                        "SELECT content FROM sources WHERE run_id = ?", (run_id,)
                    ).fetchall()
                ]
            self.assertTrue(contents)
            self.assertTrue(all("full page" in item.casefold() for item in contents))
            self.assertTrue(store.list_source_assessments(run_id))
            self.assertTrue(
                all(
                    item.source_assessment_id
                    for item in store.list_evidence(run_id)
                )
            )

    def test_source_score_is_derived_from_subscores_not_model_total(self) -> None:
        assessment = SourceAssessment(
            source_rank=1,
            source_type=SourceType.PRIMARY_RESEARCH,
            source_directness=SourceDirectness.DIRECT,
            scores=SourceScoreDimensions(
                provenance_directness=3,
                authority_for_claim=3,
                authorship_transparency=2,
                methodology_transparency=3,
                publication_controls=2,
                citation_traceability=3,
                recency_for_claim=2,
                independence=2,
                claim_relevance=3,
            ),
            rationale=["The page exposes direct methods and traceable evidence."],
            source_quality_score=0,
            evidence_fitness_score=0,
            credibility_tier=CredibilityTier.D,
            eligibility=SourceEligibility.REJECT,
        )

        self.assertEqual(assessment.source_quality_score, 12)
        self.assertEqual(assessment.evidence_fitness_score, 11)
        self.assertEqual(assessment.credibility_tier, CredibilityTier.B)
        self.assertEqual(
            assessment.eligibility, SourceEligibility.FINAL_EVIDENCE
        )

    def test_hard_rule_downgrades_ai_synthesis_despite_high_scores(self) -> None:
        assessment = SourceAssessment(
            source_rank=1,
            source_type=SourceType.PRIMARY_RESEARCH,
            source_directness=SourceDirectness.DIRECT,
            scores=SourceScoreDimensions(
                provenance_directness=3,
                authority_for_claim=3,
                authorship_transparency=3,
                methodology_transparency=3,
                publication_controls=3,
                citation_traceability=3,
                recency_for_claim=3,
                independence=3,
                claim_relevance=3,
            ),
            rationale=["The model incorrectly awarded maximum scores."],
        )
        result = SearchResult(
            title="Who is the GOAT using OpenAI's Deep Research",
            url="https://blog.example/goat",
            content="A generated synthesis.",
        )

        finalized = finalize_source_assessment(result, assessment)

        self.assertIn(SourceHardFlag.AI_SYNTHESIS, finalized.hard_flags)
        self.assertEqual(finalized.credibility_tier, CredibilityTier.D)
        self.assertEqual(
            finalized.eligibility, SourceEligibility.DISCOVERY_ONLY
        )

    def test_audit_curation_filters_discovery_pages_and_caps_one_url(self) -> None:
        repeated = [
            self.evidence_record(
                f"ev_{index}", "https://credible.example/study"
            )
            for index in range(6)
        ]
        ai_synthesis = self.evidence_record(
            "ev_ai",
            "https://blog.example/goat",
            title="Who is the GOAT using OpenAI's Deep Research",
        )

        curated = curate_evidence([*repeated, ai_synthesis])

        self.assertEqual(len(curated), 4)
        self.assertNotIn("ev_ai", {item.evidence_id for item in curated})

    def test_one_evidence_and_one_provenance_cannot_cover_many_claims(self) -> None:
        evidence = [
            self.evidence_record(
                "ev_1", "https://one.example/page-a", provenance_key="origin-one"
            ),
            self.evidence_record(
                "ev_2", "https://one.example/page-b", provenance_key="origin-one"
            ),
            self.evidence_record(
                "ev_3", "https://two.example/page", provenance_key="origin-two"
            ),
        ]
        audit = AuditResult(
            sufficient=True,
            claims=[
                ClaimAssessment(
                    claim_id="C1",
                    claim="The first synthesized claim is supported.",
                    status=ClaimStatus.SUPPORTED,
                    confidence=Confidence.HIGH,
                    supporting_evidence_ids=["ev_1", "ev_2"],
                    reasoning="Two excerpts were proposed.",
                ),
                ClaimAssessment(
                    claim_id="C2",
                    claim="The second synthesized claim is supported.",
                    status=ClaimStatus.SUPPORTED,
                    confidence=Confidence.HIGH,
                    supporting_evidence_ids=["ev_1", "ev_3"],
                    reasoning="A reused excerpt was proposed.",
                ),
            ],
        )

        normalized = _normalize_audit(audit, evidence)

        first = normalized.claims[0].supporting_evidence_ids
        second = normalized.claims[1].supporting_evidence_ids
        self.assertEqual(first, ["ev_1"])
        self.assertEqual(second, ["ev_3"])
        self.assertFalse(set(first) & set(second))

    def test_supervisor_replans_after_intermediate_worker_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                workspace=root,
                tavily_api_key=None,
                deepseek_api_key=None,
                max_reconnaissance_queries=1,
                max_follow_up_rounds=2,
            )
            settings.ensure_directories()
            store = EvidenceStore(settings.research_db)
            model = OneWaveReplanningModel()
            runtime = ResearchRuntime(
                settings, store, FakeSearchProvider(), model
            )
            run_id = "run_replan_wave"
            question = "What does the evidence show and what remains uncertain?"
            store.create_run(run_id, question, {"fake": True})

            result = run_graph(
                settings=settings,
                store=store,
                runtime=runtime,
                run_id=run_id,
                question=question,
            )

            self.assertEqual(model.replan_calls, 1)
            self.assertEqual(result["follow_up_round"], 1)
            self.assertIn("T4", result["task_statuses"])
            self.assertEqual(
                result["task_statuses"]["T4"], TaskStatus.SUCCEEDED.value
            )
            self.assertEqual(
                store.operation_summary(run_id)["model.replan.1"], 1
            )

    def test_large_audit_uses_bounded_specialists_and_persisted_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                workspace=root,
                tavily_api_key=None,
                deepseek_api_key=None,
                max_initial_tasks=3,
                max_reconnaissance_queries=1,
                max_follow_up_rounds=0,
                single_audit_context_token_budget=1,
                audit_evidence_token_budget_per_check=5000,
                max_audit_specialists=2,
            )
            settings.ensure_directories()
            store = EvidenceStore(settings.research_db)
            model = HierarchicalAuditModel()
            runtime = ResearchRuntime(
                settings, store, FakeSearchProvider(), model
            )
            run_id = "run_hierarchical_audit"
            question = "What does the complete evidence show across dimensions?"
            store.create_run(run_id, question, {"fake": True})

            result = run_graph(
                settings=settings,
                store=store,
                runtime=runtime,
                run_id=run_id,
                question=question,
            )

            audit_plan = store.get_audit_plan(run_id, 0)
            self.assertIsNotNone(audit_plan)
            self.assertEqual(audit_plan["mode"], "hierarchical")
            self.assertEqual(len(audit_plan["checks"]), 2)
            planned_dimensions = {
                dimension
                for check in audit_plan["checks"]
                for dimension in check["dimensions"]
            }
            required_dimensions = set(
                result["plan"]["coverage_contract"]["required_dimensions"]
            )
            self.assertEqual(planned_dimensions, required_dimensions)
            self.assertEqual(set(model.audit_checks), {"A1", "A2"})
            summary = store.operation_summary(run_id)
            self.assertEqual(summary["model.audit.0.A1"], 1)
            self.assertEqual(summary["model.audit.0.A2"], 1)
            self.assertNotIn("model.audit.0.reduce", summary)
            self.assertTrue(
                (settings.runs_dir / run_id / "audit_plan.json").exists()
            )

    def test_specialist_audits_are_aggregated_without_a_model_reducer(self) -> None:
        plan = AuditPlan(
            mode="hierarchical",
            round=0,
            rationale="The audit is split across two dimensions.",
            estimated_single_audit_tokens=50_000,
            checks=[
                AuditCheck(
                    check_id="A1", dimensions=["benefits"], task_ids=["T1"],
                    priority="high",
                ),
                AuditCheck(
                    check_id="A2", dimensions=["risks"], task_ids=["T2"],
                    priority="medium",
                ),
            ],
        )
        partials = [
            AuditResult(
                sufficient=False,
                source_sufficient=False,
                gaps=["Independent verification is missing."],
                actionable_gaps=[AuditGap(
                    gap_id="A1-G1",
                    gap_type="source_quality",
                    priority="high",
                    description="Independent verification is missing.",
                    related_evidence_ids=["ev_1"],
                    missing_evidence="A second independent primary source.",
                    suggested_query="benefit independent primary source",
                )],
            ),
            AuditResult(sufficient=True),
        ]

        aggregated = aggregate_audits(plan, partials)

        self.assertFalse(aggregated.sufficient)
        self.assertEqual(len(aggregated.actionable_gaps), 1)
        gap = aggregated.actionable_gaps[0]
        self.assertEqual(gap.check_id, "A1")
        self.assertEqual(gap.dimension, "benefits")
        self.assertEqual(gap.task_ids, ["T1"])
        self.assertEqual(gap.related_evidence_ids, ["ev_1"])

    def test_specialist_claim_aggregation_preserves_every_dimension(self) -> None:
        dimensions = [
            "cultural impact",
            "championships",
            "defense",
            "advanced analytics",
            "longevity",
            "scoring",
        ]
        plan = AuditPlan(
            mode="hierarchical",
            round=1,
            rationale="Six dimensions are split across four bounded checks.",
            estimated_single_audit_tokens=50_000,
            checks=[
                AuditCheck(check_id="A1", dimensions=[dimensions[0]]),
                AuditCheck(check_id="A2", dimensions=[dimensions[1]]),
                AuditCheck(check_id="A3", dimensions=[dimensions[2]]),
                AuditCheck(check_id="A4", dimensions=dimensions[3:]),
            ],
        )

        def claim(index: int, dimension: str | None) -> ClaimAssessment:
            return ClaimAssessment(
                claim_id=f"C{index}",
                claim=f"Audited finding number {index} for {dimension}.",
                dimension=dimension,
                status=ClaimStatus.SUPPORTED,
                confidence=Confidence.HIGH,
                supporting_evidence_ids=[f"ev_{index}"],
                reasoning="The supplied evidence directly supports the finding.",
            )

        partials = [
            AuditResult(
                sufficient=True,
                claims=[claim(index, dimensions[0]) for index in range(1, 7)],
            ),
            AuditResult(
                sufficient=True,
                claims=[claim(index, dimensions[1]) for index in range(7, 13)],
            ),
            AuditResult(sufficient=True, claims=[claim(13, None)]),
            AuditResult(
                sufficient=True,
                claims=[
                    claim(14, dimensions[3]),
                    claim(15, dimensions[4]),
                    claim(16, dimensions[5]),
                ],
            ),
        ]

        aggregated = aggregate_audits(plan, partials)

        self.assertEqual(len(aggregated.claims), 12)
        self.assertEqual(
            {item.dimension for item in aggregated.claims},
            set(dimensions),
        )
        defense = next(
            item for item in aggregated.claims if item.claim == claim(13, None).claim
        )
        self.assertEqual(defense.dimension, "defense")

    def test_dimension_audit_reserves_tagged_numeric_evidence(self) -> None:
        dimension = "population projections"
        evidence = [
            self.evidence_record(
                "ev_population",
                "https://official.example/population",
            ).model_copy(update={
                "claim_candidate": "The population is projected to reach 39.5 million in 2043.",
                "verbatim_excerpt": "The population is projected to reach 39.5 million in 2043.",
                "dimension_ids": [dimension],
            }),
            self.evidence_record(
                "ev_pension",
                "https://official.example/pension",
            ).model_copy(update={
                "claim_candidate": "Pension contributions may affect future consumption.",
                "verbatim_excerpt": "Pension contributions may affect future consumption.",
                "dimension_ids": ["future consumption"],
            }),
        ]
        selected, _tokens = select_audit_evidence(
            evidence,
            AuditCheck(
                check_id="A1",
                dimensions=[dimension],
                task_ids=[],
            ),
            token_budget=5000,
        )

        self.assertEqual(
            {item.evidence_id for item in selected},
            {"ev_population"},
        )

    def test_report_context_reserves_every_required_dimension(self) -> None:
        dimensions = ["current population", "population projections"]
        plan = ResearchPlan(
            coverage_contract=CoverageContract(
                decision_type="demographic assessment",
                required_dimensions=dimensions,
            ),
            tasks=[
                ResearchTask(
                    task_id="T1",
                    question="What is the current population estimate?",
                    covered_dimensions=[dimensions[0]],
                ),
                ResearchTask(
                    task_id="T2",
                    question="What are the official population projections?",
                    covered_dimensions=[dimensions[1]],
                ),
            ],
        )
        current_one = self.evidence_record(
            "ev_current_1", "https://official.example/current-1"
        ).model_copy(update={"dimension_ids": [dimensions[0]]})
        current_two = self.evidence_record(
            "ev_current_2", "https://official.example/current-2"
        ).model_copy(update={"dimension_ids": [dimensions[0]]})
        projected = self.evidence_record(
            "ev_projected", "https://official.example/projected"
        ).model_copy(update={
            "task_id": "T2",
            "dimension_ids": [dimensions[1]],
        })
        audit = AuditResult(
            sufficient=True,
            claims=[ClaimAssessment(
                claim_id="C1",
                claim="Two current estimates are available.",
                dimension=dimensions[0],
                status=ClaimStatus.SUPPORTED,
                confidence=Confidence.HIGH,
                supporting_evidence_ids=[
                    current_one.evidence_id,
                    current_two.evidence_id,
                ],
                reasoning="Both sources directly report the current estimate.",
            )],
        )

        selected = select_report_evidence(
            plan,
            audit,
            [current_one, current_two, projected],
            max_items=2,
        )

        self.assertEqual(
            {dimension for item in selected for dimension in item.dimension_ids},
            set(dimensions),
        )

    def test_direct_evidence_provenance_uses_the_retrieved_page(self) -> None:
        source_url = "https://official.example/report.pdf"
        item = self.evidence_record("ev_direct", source_url).model_copy(update={
            "source_directness": SourceDirectness.UNCLEAR,
            "attributed_source_url": "https://official.example/index.html",
            "provenance_key": "https://official.example/index.html",
        })

        curated = curate_evidence([item])

        self.assertEqual(curated[0].provenance_key, source_url)

    def test_report_cites_retrieved_page_instead_of_attributed_lead(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                workspace=root,
                tavily_api_key=None,
                deepseek_api_key=None,
            )
            settings.ensure_directories()
            store = EvidenceStore(settings.research_db)
            run_id = "run_precise_citation"
            store.create_run(run_id, "What does the projection show?", {})
            dimension = "population projection"
            plan = ResearchPlan(
                coverage_contract=CoverageContract(
                    decision_type="projection",
                    required_dimensions=[dimension],
                ),
                tasks=[ResearchTask(
                    task_id="T1",
                    question="What does the projection show?",
                    covered_dimensions=[dimension],
                )],
            )
            evidence = self.evidence_record(
                "ev_projection",
                "https://official.example/projection.pdf",
                title="Official projection PDF",
            ).model_copy(update={
                "claim_candidate": "The population peaks at 39.53 million in 2043.",
                "verbatim_excerpt": "The population peaks at 39.53 million in 2043.",
                "source_directness": SourceDirectness.UNCLEAR,
                "attributed_source_name": "Official agency home page",
                "attributed_source_url": "https://official.example/index.html",
                "dimension_ids": [dimension],
            })
            draft = ReportDraft(
                title="Projection report",
                direct_answer=[ReportStatement(
                    text="The population peaks at 39.53 million in 2043.",
                    evidence_ids=[evidence.evidence_id],
                )],
                sections=[],
            )

            report_path = render_report(
                settings=settings,
                store=store,
                run_id=run_id,
                question="What does the projection show?",
                draft=draft,
                audit=AuditResult(sufficient=False),
                evidence=[evidence],
                plan=plan,
                task_statuses={"T1": TaskStatus.SUCCEEDED.value},
            )

            citations = json.loads(
                (report_path.parent / "citations.json").read_text(encoding="utf-8")
            )
            self.assertEqual(citations[0]["url"], evidence.source_url)
            self.assertEqual(citations[0]["title"], evidence.source_title)

    def test_report_normalization_splits_sentences_by_independent_support(self) -> None:
        dimension = "population projection"
        plan = ResearchPlan(
            coverage_contract=CoverageContract(
                decision_type="projection",
                required_dimensions=[dimension],
            ),
            tasks=[ResearchTask(
                task_id="T1",
                question="What are the population estimates?",
                covered_dimensions=[dimension],
            )],
        )
        peak = self.evidence_record(
            "ev_peak", "https://official.example/peak.pdf"
        ).model_copy(update={
            "claim_candidate": "The population peaks at 39.53 million in 2043.",
            "verbatim_excerpt": "The population peaks at 39.53 million in 2043.",
            "dimension_ids": [dimension],
        })
        current = self.evidence_record(
            "ev_current", "https://official.example/current.html"
        ).model_copy(update={
            "claim_candidate": "The population was 36.243 million in 2024.",
            "verbatim_excerpt": "The population was 36.243 million in 2024.",
            "dimension_ids": [dimension],
        })
        draft = ReportDraft(
            title="Projection report",
            direct_answer=[ReportStatement(
                text=(
                    "The population peaks at 39.53 million in 2043. "
                    "The population was 36.243 million in 2024."
                ),
                evidence_ids=[peak.evidence_id, current.evidence_id],
                confidence=Confidence.HIGH,
            )],
        )

        normalized = normalize_report_draft(
            draft,
            plan=plan,
            audit=AuditResult(sufficient=False),
            evidence=[peak, current],
        )

        self.assertEqual(len(normalized.direct_answer), 2)
        self.assertEqual(
            normalized.direct_answer[0].evidence_ids, [peak.evidence_id]
        )
        self.assertEqual(
            normalized.direct_answer[1].evidence_ids, [current.evidence_id]
        )

    def test_report_normalization_removes_partial_compound_citation(self) -> None:
        dimension = "elderly consumption"
        plan = ResearchPlan(
            coverage_contract=CoverageContract(
                decision_type="market assessment",
                required_dimensions=[dimension],
            ),
            tasks=[ResearchTask(
                task_id="T1",
                question="How much do elderly households consume?",
                covered_dimensions=[dimension],
            )],
        )
        partial = self.evidence_record(
            "ev_partial", "https://official.example/propensity.pdf"
        ).model_copy(update={
            "claim_candidate": "The average propensity to consume exceeds 100%.",
            "verbatim_excerpt": "The average propensity to consume exceeds 100%.",
            "dimension_ids": [dimension],
        })
        complete = self.evidence_record(
            "ev_complete", "https://official.example/household.pdf"
        ).model_copy(update={
            "claim_candidate": (
                "The average propensity to consume exceeded 100%, with a "
                "34,642 yen consumption deficit."
            ),
            "verbatim_excerpt": (
                "The average propensity to consume exceeded 100%, and the "
                "consumption deficit was 34,642 yen."
            ),
            "dimension_ids": [dimension],
        })
        draft = ReportDraft(
            title="Consumption report",
            direct_answer=[ReportStatement(
                text=(
                    "The average propensity to consume exceeded 100%, with a "
                    "34,642 yen consumption deficit."
                ),
                evidence_ids=[partial.evidence_id, complete.evidence_id],
            )],
        )

        normalized = normalize_report_draft(
            draft,
            plan=plan,
            audit=AuditResult(sufficient=False),
            evidence=[partial, complete],
        )

        self.assertEqual(len(normalized.direct_answer), 1)
        self.assertEqual(
            normalized.direct_answer[0].evidence_ids, [complete.evidence_id]
        )

    def test_dimension_auditor_uses_fast_specialist_profile(self) -> None:
        settings = Settings(
            workspace=Path("/tmp"),
            tavily_api_key=None,
            deepseek_api_key="test-key",
        )
        auditor = AuditorAgent(settings)
        result = AuditResult(sufficient=False, gaps=["More evidence is needed."])

        with patch.object(auditor, "_json_completion", return_value=result) as call:
            auditor.audit(
                "What does the evidence show?",
                [],
                0,
                {"audit_check": {"check_id": "A1"}},
            )

        self.assertEqual(call.call_args.kwargs["model"], settings.fast_model)
        self.assertEqual(call.call_args.kwargs["profile"], "audit_specialist")

    def test_initial_plan_is_conditioned_on_reconnaissance_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                workspace=root,
                tavily_api_key=None,
                deepseek_api_key=None,
                max_initial_tasks=1,
                max_reconnaissance_queries=1,
                max_follow_up_rounds=0,
            )
            settings.ensure_directories()
            store = EvidenceStore(settings.research_db)
            model = ContextCapturingPlanner()
            runtime = ResearchRuntime(
                settings, store, FakeSearchProvider(), model
            )
            run_id = "run_evidence_conditioned_plan"
            question = "What evidence should determine the research dimensions?"
            store.create_run(run_id, question, {"fake": True})

            run_graph(
                settings=settings,
                store=store,
                runtime=runtime,
                run_id=run_id,
                question=question,
            )

            self.assertGreater(model.preliminary_evidence_seen, 0)
            self.assertEqual(
                store.operation_summary(run_id)[
                    "model.reconnaissance_queries"
                ],
                1,
            )

    def test_worker_recursively_retrieves_an_attributed_original_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                workspace=root,
                tavily_api_key=None,
                deepseek_api_key=None,
                max_initial_tasks=1,
                max_reconnaissance_queries=0,
                max_follow_up_rounds=0,
                max_adaptive_steps=3,
            )
            settings.ensure_directories()
            store = EvidenceStore(settings.research_db)
            search = OriginalSourceSearch()
            runtime = ResearchRuntime(
                settings, store, search, OriginalSourceFollowingModel()
            )
            run_id = "run_follow_original"
            question = "What direct evidence supports the measured result?"
            store.create_run(run_id, question, {"fake": True})

            result = run_graph(
                settings=settings,
                store=store,
                runtime=runtime,
                run_id=run_id,
                question=question,
            )

            worker = result["worker_results"][0]
            self.assertEqual(worker["adaptive_steps"], 2)
            self.assertEqual(worker["search_calls_used"], 1)
            self.assertEqual(search.call_count, 1)
            self.assertEqual(search.extract_call_count, 2)
            self.assertTrue(
                any(
                    item.source_url == "https://primary.example/study"
                    and item.source_directness == SourceDirectness.DIRECT
                    for item in store.list_evidence(run_id)
                )
            )

    def test_runtime_uses_full_page_content_and_caches_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, store, search, runtime = self.make_runtime(root)
            run_id = "run_full_page"
            store.create_run(run_id, "A question", {"fake": True})
            task = ResearchTask(
                task_id="T1",
                question="What complete source text supports the finding?",
            )
            store.upsert_task(run_id, task)
            query = QuerySpec(
                intent=QueryIntent.PRIMARY_EVIDENCE,
                query="complete primary source evidence",
                expected_evidence="Full primary-source text",
                priority=3,
            )
            snippets = runtime.search(run_id, task, query)

            first = runtime.extract_pages(run_id, task, snippets)
            second = runtime.extract_pages(run_id, task, snippets)

            self.assertEqual(first, second)
            self.assertEqual(search.extract_call_count, 1)
            self.assertTrue(all(item.content_source == "full_page" for item in first))
            self.assertTrue(all(item.snippet for item in first))
            self.assertTrue(all("full page" in item.content.casefold() for item in first))
            summary = store.operation_summary(run_id)
            self.assertEqual(summary["extract.tavily"], 1)

    def test_page_chunking_processes_every_character(self) -> None:
        content = (
            "First paragraph contains evidence.\n\n"
            "Second paragraph contains limitations.\n\n"
            "Third paragraph contains replication details."
        )
        chunks = split_page_content(content, max_chars=48)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), content)
        self.assertTrue(all(len(chunk) <= 48 for chunk in chunks))

    def test_local_chunk_ranker_selects_task_relevant_text(self) -> None:
        task = ResearchTask(
            task_id="T1",
            question="What are Michael Jordan's championship achievements?",
            objective="Verify championship evidence",
            must_find=["NBA championships", "Finals MVP"],
            covered_dimensions=["career achievements"],
        )
        content = "\n\n".join(
            [
                "Navigation and unrelated advertising " * 8,
                "Weather forecasts and unrelated baseball coverage " * 8,
                "Michael Jordan won six NBA championships and six Finals MVP awards. "
                * 6,
                "More unrelated footer and subscription information " * 8,
            ]
        )

        ranked = rank_page_chunks(
            content,
            task=task,
            title="Michael Jordan career",
            search_query="Michael Jordan NBA championships Finals MVP",
            max_chars=350,
            max_chunks=1,
        )

        self.assertEqual(len(ranked), 1)
        self.assertIn("six NBA championships", ranked[0].content)
        self.assertGreater(ranked[0].relevance_score, 0)

    def test_local_chunk_ranker_keeps_short_pages_intact(self) -> None:
        task = ResearchTask(
            task_id="T1",
            question="What evidence answers this research question?",
        )
        ranked = rank_page_chunks(
            "A short complete source.",
            task=task,
            max_chars=2000,
            max_chunks=3,
        )

        self.assertEqual([item.content for item in ranked], ["A short complete source."])
        self.assertEqual(ranked[0].original_index, 1)
        self.assertEqual(ranked[0].total_chunks, 1)

    def test_hybrid_ranker_recovers_a_semantic_passage(self) -> None:
        task = ResearchTask(
            task_id="T1",
            question="What evidence evaluates automobile safety?",
        )
        unrelated = "Recipes describe simmering vegetables and seasoning soup. " * 8
        semantic_match = (
            "The vehicle protected occupants during a frontal crash test. " * 8
        )
        content = f"{unrelated}\n\n{semantic_match}"

        def fake_embeddings(texts: list[str]) -> list[list[float]]:
            return [
                [1.0, 0.0]
                if "Query:" in text or "protected occupants" in text
                else [0.0, 1.0]
                for text in texts
            ]

        ranked = rank_page_chunks(
            content,
            task=task,
            max_chunks=1,
            retrieval_chunk_tokens=80,
            retrieval_chunk_overlap_tokens=0,
            extraction_window_tokens=100,
            max_windows=1,
            embed_texts=fake_embeddings,
        )

        self.assertEqual(len(ranked), 1)
        self.assertIn("protected occupants", ranked[0].content)
        self.assertIsNotNone(ranked[0].embedding_score)
        self.assertGreater(ranked[0].relevance_score, 0)

    def test_weighted_rrf_fuses_ranks_instead_of_raw_score_magnitudes(self) -> None:
        scores = _weighted_rrf_scores(
            [1000.0, 1.0, 0.0],
            [0.1, 0.9, 0.8],
            embedding_weight=0.5,
            rrf_k=60,
        )

        expected_second = 0.5 / (60 + 2) + 0.5 / (60 + 1)
        self.assertAlmostEqual(scores[1], expected_second, places=7)
        self.assertGreater(scores[1], scores[0])

    def test_weighted_rrf_does_not_break_bm25_ties_arbitrarily(self) -> None:
        scores = _weighted_rrf_scores(
            [0.0, 0.0, 0.0],
            [0.1, 0.9, 0.8],
            embedding_weight=0.35,
            rrf_k=60,
        )

        self.assertGreater(scores[1], scores[2])
        self.assertGreater(scores[2], scores[0])

    def test_qwen_compatible_embedder_batches_and_orders_vectors(self) -> None:
        first = Mock()
        first.raise_for_status.return_value = None
        first.json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }
        second = Mock()
        second.raise_for_status.return_value = None
        second.json.return_value = {
            "data": [{"index": 0, "embedding": [0.5, 0.5]}]
        }
        provider = OpenAICompatibleEmbeddingProvider(
            base_url="https://embedding.example/v1",
            api_key="test-key",
            model="text-embedding-v4",
            dimensions=2,
            batch_size=2,
        )

        with patch(
            "deepresearch.embeddings.httpx.post", side_effect=[first, second]
        ) as post:
            vectors = provider.embed(["one", "two", "three"])

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].args[0], (
            "https://embedding.example/v1/embeddings"
        ))
        self.assertEqual(
            post.call_args_list[0].kwargs["json"]["dimensions"], 2
        )

    def test_local_ollama_is_started_and_stopped_for_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                workspace=Path(directory),
                tavily_api_key=None,
                deepseek_api_key=None,
                embedding_base_url="http://localhost:11434/v1",
            )
            process = Mock()
            process.pid = 4321
            process.poll.side_effect = [None, None]
            process.wait.return_value = 0

            with (
                patch(
                    "deepresearch.local_services._service_is_ready",
                    side_effect=[False, True, True],
                ),
                patch(
                    "deepresearch.local_services.shutil.which",
                    return_value="/opt/homebrew/bin/ollama",
                ),
                patch(
                    "deepresearch.local_services.subprocess.Popen",
                    return_value=process,
                ) as popen,
                patch("deepresearch.local_services.os.killpg") as killpg,
            ):
                with managed_local_embedding_service(settings):
                    popen.assert_called_once()

            killpg.assert_called_once_with(4321, signal.SIGTERM)
            self.assertTrue((settings.data_dir / "ollama.log").exists())

    def test_worker_merges_short_page_into_one_extraction_window(self) -> None:
        settings = Settings(
            workspace=Path("/tmp"),
            tavily_api_key=None,
            deepseek_api_key="test-key",
            page_chunk_chars=100,
            max_relevant_chunks_per_source=2,
        )
        worker = ResearchWorkerAgent(settings)
        task = ResearchTask(
            task_id="T1",
            question="What primary evidence supports the target finding?",
            must_find=["target finding"],
        )
        result = SearchResult(
            title="Long source",
            url="https://example.com/long",
            content=("unrelated material " * 45) + ("target finding evidence " * 10),
            content_source="full_page",
            query="target finding evidence",
        )

        with patch.object(
            worker,
            "_json_completion",
            return_value=EvidenceBatch(),
        ) as completion:
            worker.extract(task, [result])

        self.assertEqual(completion.call_count, 1)

    def test_worker_falls_back_to_bm25_when_embedding_fails(self) -> None:
        settings = Settings(
            workspace=Path("/tmp"),
            tavily_api_key=None,
            deepseek_api_key="test-key",
        )
        embedding_provider = Mock()
        embedding_provider.embed.side_effect = RuntimeError("endpoint unavailable")
        worker = ResearchWorkerAgent(
            settings, embedding_provider=embedding_provider
        )
        task = ResearchTask(
            task_id="T1",
            question="What supports the target finding?",
            must_find=["target finding"],
        )
        result = SearchResult(
            title="Source",
            url="https://example.com/source",
            content="The source contains target finding evidence.",
            content_source="full_page",
            query="target finding",
        )

        with patch.object(
            worker,
            "_json_completion",
            return_value=EvidenceBatch(),
        ) as completion:
            worker.extract(task, [result])

        embedding_provider.embed.assert_called_once()
        self.assertEqual(completion.call_count, 1)

    def test_retrieval_chunks_are_bounded_near_500_tokens_with_overlap(self) -> None:
        content = "\n\n".join(
            f"Section {index}. " + ("research evidence and context " * 90)
            for index in range(6)
        )

        chunks = split_retrieval_chunks(
            content, max_tokens=500, overlap_tokens=75
        )

        self.assertGreater(len(chunks), 2)
        self.assertTrue(
            all(estimate_text_tokens(chunk.content) <= 500 for chunk in chunks)
        )
        self.assertEqual(chunks[0].start_offset, 0)
        self.assertEqual(chunks[-1].end_offset, len(content))
        self.assertTrue(
            any(
                current.start_offset < previous.end_offset
                for previous, current in zip(chunks, chunks[1:])
            )
        )

    def test_ranker_expands_hits_but_caps_context_to_two_windows(self) -> None:
        sections = []
        for index in range(18):
            marker = (
                "pace adjusted cross era target finding "
                if index in {4, 14}
                else "unrelated navigation and background material "
            )
            sections.append(f"Section {index}. " + marker * 100)
        content = "\n\n".join(sections)
        task = ResearchTask(
            task_id="T1",
            question="How does pace adjustment affect cross-era comparison?",
            must_find=["pace adjusted cross era target finding"],
        )

        windows = rank_page_chunks(
            content,
            task=task,
            search_query="pace adjusted cross era",
            max_chunks=4,
            retrieval_chunk_tokens=500,
            retrieval_chunk_overlap_tokens=75,
            extraction_window_tokens=1800,
            max_windows=2,
        )

        self.assertLessEqual(len(windows), 2)
        self.assertTrue(windows)
        self.assertTrue(
            all(estimate_text_tokens(window.content) <= 1800 for window in windows)
        )
        self.assertTrue(
            all("pace adjusted cross era" in window.content for window in windows)
        )
        self.assertLess(
            sum(estimate_text_tokens(window.content) for window in windows),
            estimate_text_tokens(content),
        )

    def test_tavily_extract_endpoint_returns_full_markdown(self) -> None:
        settings = Settings(
            workspace=Path("/tmp"),
            tavily_api_key="test-key",
            deepseek_api_key=None,
        )
        provider = TavilySearchProvider(settings)
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {
                    "url": "https://example.com/article",
                    "raw_content": "# Complete article\n\nFull body text.",
                }
            ],
            "failed_results": [],
            "usage": {"credits": 1},
        }

        with patch("deepresearch.providers.httpx.post", return_value=response) as post:
            batch = provider.extract_pages(["https://example.com/article"])

        self.assertEqual(batch.pages[0].raw_content, "# Complete article\n\nFull body text.")
        self.assertEqual(batch.credits_used, 1)
        request = post.call_args.kwargs
        self.assertEqual(request["json"]["extract_depth"], "basic")
        self.assertEqual(request["json"]["format"], "markdown")
        self.assertTrue(str(post.call_args.args[0]).endswith("/extract"))

    def test_worker_recursively_searches_from_updated_evidence_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                workspace=root,
                tavily_api_key=None,
                deepseek_api_key=None,
                max_initial_tasks=1,
                max_follow_up_rounds=0,
                max_adaptive_steps=4,
            )
            settings.ensure_directories()
            store = EvidenceStore(settings.research_db)
            search = FakeSearchProvider()
            model = ThreeStepAdaptiveModel()
            runtime = ResearchRuntime(settings, store, search, model)
            run_id = "run_adaptive_worker"
            question = "What does the evidence show and where is verification needed?"
            store.create_run(run_id, question, {"fake": True})

            with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
                result = run_graph(
                    settings=settings,
                    store=store,
                    runtime=runtime,
                    run_id=run_id,
                    question=question,
                )

            worker = result["worker_results"][0]
            self.assertEqual(search.call_count, 5)
            self.assertEqual(worker["adaptive_steps"], 3)
            self.assertEqual(worker["stop_reason"], "sufficient")
            self.assertEqual(len(worker["executed_queries"]), 3)
            self.assertEqual([item[0] for item in model.observations], [1, 2, 3])
            self.assertTrue(all(evidence_count > 0 for _, evidence_count in model.observations))
            summary = store.operation_summary(run_id)
            self.assertEqual(summary["model.decide_worker_next_step"], 3)
            output = "\n".join(captured.output)
            self.assertIn("[ResearchWorker:T1] adaptive.step.started", output)
            self.assertIn("[ResearchWorker:T1] adaptive.decision", output)

    def test_search_operation_is_idempotently_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, store, search, runtime = self.make_runtime(root)
            run_id = "run_cache"
            store.create_run(run_id, "A question", {"fake": True})
            task = ResearchTask(
                task_id="T1",
                question="What evidence answers the research question?",
            )
            store.upsert_task(run_id, task)
            query = QuerySpec(
                intent=QueryIntent.PRIMARY_EVIDENCE,
                query="research question evidence",
                expected_evidence="Direct empirical evidence",
                priority=3,
            )

            first = runtime.search(run_id, task, query)
            second = runtime.search(run_id, task, query)

            self.assertEqual(first, second)
            self.assertEqual(search.call_count, 1)

    def test_planner_can_spawn_five_workers_without_global_search_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                workspace=root,
                tavily_api_key=None,
                deepseek_api_key=None,
                max_initial_tasks=5,
                max_follow_up_rounds=0,
            )
            settings.ensure_directories()
            store = EvidenceStore(settings.research_db)
            search = FakeSearchProvider()
            runtime = ResearchRuntime(
                settings,
                store,
                search,
                FiveDimensionPlanner(),
            )
            run_id = "run_five_workers"
            question = "Assess five independent dimensions of the intervention."
            store.create_run(run_id, question, {"fake": True})

            result = run_graph(
                settings=settings,
                store=store,
                runtime=runtime,
                run_id=run_id,
                question=question,
            )

            self.assertEqual(len(result["task_statuses"]), 5)
            self.assertEqual(search.call_count, 12)
            summary = store.operation_summary(run_id)
            self.assertEqual(summary["model.formulate_queries"], 5)
            self.assertEqual(summary["model.decide_worker_next_step"], 10)
            self.assertEqual(summary["search.tavily"], 12)
            self.assertEqual(summary["extract.tavily"], 6)

    def test_runtime_logs_agent_and_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, store, _search, runtime = self.make_runtime(root)
            run_id = "run_logging"
            store.create_run(run_id, "A question", {"fake": True})
            task = ResearchTask(
                task_id="T1",
                question="What evidence answers the research question?",
            )
            store.upsert_task(run_id, task)
            query = QuerySpec(
                intent=QueryIntent.PRIMARY_EVIDENCE,
                query="research question evidence",
                expected_evidence="Direct empirical evidence",
                priority=3,
            )

            with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
                runtime.search(run_id, task, query)
                runtime.search(run_id, task, query)

            output = "\n".join(captured.output)
            self.assertIn("[ResearchWorker:T1] search.tavily.started", output)
            self.assertIn("[ResearchWorker:T1] search.tavily.completed", output)
            self.assertIn("[ResearchWorker:T1] search.tavily.cache_hit", output)

    def test_non_verbatim_quote_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, store, _search, _runtime = self.make_runtime(root)
            run_id = "run_quote_guard"
            store.create_run(run_id, "A question", {"fake": True})
            task = ResearchTask(
                task_id="T1",
                question="What evidence answers the research question?",
            )
            store.upsert_task(run_id, task)
            results = [
                SearchResult(
                    title="Source",
                    url="https://example.com/source",
                    content="The source contains a modest finding.",
                    score=0.9,
                    rank=1,
                    query="research question evidence",
                )
            ]
            batch = EvidenceBatch(
                evidence=[
                    ExtractedEvidence(
                        source_rank=1,
                        claim_candidate="The source proves a large effect.",
                        verbatim_excerpt="The source proves a large effect.",
                        stance=EvidenceStance.SUPPORTS,
                        relevance="high",
                    )
                ]
            )

            _sources, evidence_ids = store.persist_worker_artifacts(
                run_id, task, results, batch
            )
            self.assertEqual(evidence_ids, [])

    def test_same_excerpt_is_owned_separately_by_each_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, store, _search, _runtime = self.make_runtime(root)
            run_id = "run_task_evidence_identity"
            store.create_run(run_id, "A question", {"fake": True})
            first_task = ResearchTask(
                task_id="T1",
                question="What benefits does this evidence establish?",
            )
            second_task = ResearchTask(
                task_id="T2",
                question="What limitations does this evidence establish?",
            )
            store.upsert_task(run_id, first_task)
            store.upsert_task(run_id, second_task)
            results = [
                SearchResult(
                    title="Shared source",
                    url="https://example.com/shared",
                    content="The shared source reports a qualified result.",
                    score=0.9,
                    rank=1,
                    query="shared source evidence",
                )
            ]
            batch = EvidenceBatch(
                evidence=[
                    ExtractedEvidence(
                        source_rank=1,
                        claim_candidate="The result is qualified.",
                        verbatim_excerpt="The shared source reports a qualified result.",
                        stance=EvidenceStance.NEUTRAL,
                        relevance="high",
                    )
                ]
            )

            _sources, first_ids = store.persist_worker_artifacts(
                run_id, first_task, results, batch
            )
            _sources, second_ids = store.persist_worker_artifacts(
                run_id, second_task, results, batch
            )

            self.assertNotEqual(first_ids, second_ids)
            records = store.list_evidence(run_id)
            self.assertEqual({item.task_id for item in records}, {"T1", "T2"})

    def test_model_output_is_normalized_at_schema_boundary(self) -> None:
        extracted = [
            {
                "source_rank": index + 1,
                "claim_candidate": f"Atomic evidence claim {index + 1}",
                "verbatim_excerpt": f"Exact source excerpt number {index + 1}",
                "stance": "supports",
                "relevance": "high",
            }
            for index in range(14)
        ]
        batch = EvidenceBatch.model_validate({"evidence": extracted})
        self.assertEqual(len(batch.evidence), 14)

        claims = [
            {
                "claim_id": f"C{index + 1}",
                "claim": f"Synthesized atomic claim {index + 1}",
                "status": "supported",
                "confidence": "medium",
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
                "reasoning": "The supplied evidence supports this conclusion.",
            }
            for index in range(18)
        ]
        audit = AuditResult.model_validate(
            {
                "sufficient": False,
                "claims": claims,
                "gaps": [
                    {"gap_id": "G1", "description": "Long-term effects are unknown."},
                    {"gap_id": "G2", "reason": "Independent replication is sparse."},
                ],
            }
        )
        self.assertEqual(len(audit.claims), 12)
        self.assertEqual(
            audit.gaps,
            ["Long-term effects are unknown.", "Independent replication is sparse."],
        )

    def test_deepseek_json_retry_repairs_invalid_output(self) -> None:
        settings = Settings(
            workspace=Path("/tmp"),
            tavily_api_key=None,
            deepseek_api_key="test-key",
        )
        provider = DeepSeekModelProvider(settings)
        invalid = Mock()
        invalid.raise_for_status.return_value = None
        invalid.json.return_value = {
            "choices": [{"message": {"content": '{"queries": []}'}}]
        }
        valid = Mock()
        valid.raise_for_status.return_value = None
        valid.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"queries":[{"intent":"overview",'
                            '"query":"valid research query",'
                            '"expected_evidence":"relevant primary evidence",'
                            '"priority":3,"provider":"tavily"}]}'
                        )
                    }
                }
            ]
        }

        with patch(
            "deepresearch.model_gateway.providers.httpx.post",
            side_effect=[invalid, valid],
        ) as post:
            result = provider._json_completion(
                model="test-model",
                system_prompt="Return valid JSON.",
                user_prompt="Create a query.",
                schema=QueryPlan,
                max_tokens=200,
            )

        self.assertEqual(result.queries[0].query, "valid research query")
        retry_messages = post.call_args_list[1].kwargs["json"]["messages"]
        self.assertEqual(
            [item["role"] for item in retry_messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertIn("failed schema validation", retry_messages[-1]["content"])

    def test_deepseek_json_retry_regenerates_truncated_output_with_more_tokens(self) -> None:
        settings = Settings(
            workspace=Path("/tmp"),
            tavily_api_key=None,
            deepseek_api_key="test-key",
        )
        provider = DeepSeekModelProvider(settings)
        truncated = Mock()
        truncated.raise_for_status.return_value = None
        truncated.json.return_value = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": '{"queries":[{"intent":"overview"'},
            }]
        }
        valid = Mock()
        valid.raise_for_status.return_value = None
        valid.json.return_value = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": (
                    '{"queries":[{"intent":"overview",'
                    '"query":"valid research query",'
                    '"expected_evidence":"relevant primary evidence",'
                    '"priority":3,"provider":"tavily"}]}'
                )},
            }]
        }

        with patch(
            "deepresearch.model_gateway.providers.httpx.post",
            side_effect=[truncated, valid],
        ) as post:
            result = provider._json_completion(
                model="test-model",
                system_prompt="Return valid JSON.",
                user_prompt="Create a query.",
                schema=QueryPlan,
                max_tokens=200,
            )

        self.assertEqual(result.queries[0].query, "valid research query")
        retry_payload = post.call_args_list[1].kwargs["json"]
        self.assertEqual(retry_payload["max_tokens"], 1200)
        self.assertEqual(
            [item["role"] for item in retry_payload["messages"]],
            ["system", "user", "user"],
        )
        self.assertIn("truncated", retry_payload["messages"][-1]["content"])

    def test_planner_plan_uses_5000_output_tokens(self) -> None:
        settings = Settings(
            workspace=Path("/tmp"),
            tavily_api_key=None,
            deepseek_api_key="test-key",
        )
        planner = PlannerAgent(settings)
        expected = Mock(spec=ResearchPlan)

        with patch.object(planner, "_json_completion", return_value=expected) as call:
            planner.plan("Compare A and B", max_tasks=6)

        self.assertEqual(call.call_args.kwargs["max_tokens"], 5000)

    def test_gateway_profile_owns_model_choice_and_budget_downgrades(self) -> None:
        models = {
            "pro": ModelSpec(
                name="pro", provider="test", premium=True, context_window=100_000
            ),
            "flash": ModelSpec(
                name="flash", provider="test", premium=False, context_window=100_000
            ),
        }
        router = DeterministicRouter(
            models=models,
            profiles={
                "deep_reasoning": RoutingProfile(
                    "deep_reasoning", ("pro", "flash"), 0.85
                ),
                "standard_research": RoutingProfile(
                    "standard_research", ("flash", "pro"), 0.85
                ),
            },
            budget_usd=10.0,
        )

        selected, _ = router.route(
            profile_name="deep_reasoning",
            preferred_model="flash",
            estimated_context_tokens=1000,
            current_spend_usd=1.0,
        )
        self.assertEqual(selected.name, "pro")

        selected, reason = router.route(
            profile_name="deep_reasoning",
            preferred_model="pro",
            estimated_context_tokens=1000,
            current_spend_usd=9.0,
        )
        self.assertEqual(selected.name, "flash")
        self.assertIn("premium=blocked", reason)

    def test_gateway_records_runtime_signature_tokens_and_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                workspace=Path(tmp),
                tavily_api_key=None,
                deepseek_api_key="test-key",
                planner_model="pro",
                fast_model="flash",
                audit_model="pro",
                model_gateway_pricing={
                    "pro": {"input": 2.0, "output": 4.0},
                    "flash": {"input": 0.5, "output": 1.0},
                },
            )
            store = EvidenceStore(settings.research_db)
            store.create_run(
                "run_gateway_test",
                "Observe this run",
                {"model_gateway_budget_usd": None},
            )
            provider = DeepSeekModelProvider(settings)
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "model": "flash",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"content": (
                        '{"queries":[{"intent":"overview",'
                        '"query":"valid research query",'
                        '"expected_evidence":"relevant primary evidence",'
                        '"priority":3,"provider":"tavily"}]}'
                    )},
                }],
                "usage": {"prompt_tokens": 120, "completion_tokens": 30},
            }

            with patch(
                "deepresearch.model_gateway.providers.httpx.post",
                return_value=response,
            ), model_call_scope(
                run_id="run_gateway_test",
                task_id="T7",
                operation="model.formulate_queries",
            ):
                provider._json_completion(
                    model="flash",
                    system_prompt="Return valid JSON.",
                    user_prompt="Create a query.",
                    schema=QueryPlan,
                    max_tokens=200,
                    profile="standard_research",
                )

            rows = provider.gateway.telemetry.summary("run_gateway_test")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["agent_id"], "ResearchWorkerAgent")
            self.assertEqual(rows[0]["task_id"], "T7")
            self.assertEqual(rows[0]["operation"], "model.formulate_queries")
            self.assertEqual(rows[0]["input_tokens"], 120)
            self.assertEqual(rows[0]["output_tokens"], 30)
            self.assertGreater(rows[0]["cost_usd"], 0)
            report_path = render_model_observability(
                settings=settings,
                store=store,
                run_id="run_gateway_test",
            )
            report_payload = json.loads(
                (report_path.parent / "model_calls.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report_payload["totals"]["calls"], 1)
            self.assertEqual(report_payload["by_agent"][0]["task_id"], "T7")

    def test_gateway_reservation_enforces_hard_run_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            telemetry = ModelTelemetryStore(Path(tmp) / "gateway.sqlite")
            signature = ModelCallSignature(
                run_id="run_budget_test",
                task_id="T1",
                agent_id="ResearchWorkerAgent",
                operation="model.extract",
                profile="standard_research",
            )
            spec = ModelSpec(
                name="priced-model",
                provider="test",
                premium=False,
                context_window=10_000,
            )

            with self.assertRaises(BudgetExceededError):
                telemetry.reserve(
                    call_id="mc_over_budget",
                    signature=signature,
                    spec=spec,
                    estimated_input_tokens=100,
                    max_output_tokens=100,
                    estimated_cost_usd=1.01,
                    routing_reason="test",
                    budget_usd=1.00,
                )

    def test_single_domain_claim_is_not_treated_as_robust_support(self) -> None:
        evidence = [
            EvidenceRecord(
                evidence_id="ev_1", source_id="src_1", task_id="T1",
                source_title="One study", source_url="https://one.example/study",
                source_domain="one.example", claim_candidate="A measured effect exists.",
                verbatim_excerpt="A measured effect exists.", stance=EvidenceStance.SUPPORTS,
                relevance="high",
            )
        ]
        audit = AuditResult(
            sufficient=True,
            claims=[ClaimAssessment(
                claim_id="C1", claim="A measured effect exists.",
                status=ClaimStatus.SUPPORTED, confidence=Confidence.HIGH,
                supporting_evidence_ids=["ev_1"],
                reasoning="One source reports the effect.",
            )],
        )

        normalized = _normalize_audit(audit, evidence)

        self.assertEqual(normalized.claims[0].status, ClaimStatus.UNRESOLVED)
        self.assertEqual(normalized.claims[0].confidence, Confidence.LOW)

    def test_supervisor_evidence_context_is_bounded_and_keeps_audit_citations(self) -> None:
        evidence = [
            EvidenceRecord(
                evidence_id=f"ev_{index}",
                source_id=f"src_{index}",
                task_id="T1",
                source_title=f"Source {index}",
                source_url=f"https://source{index}.example/study",
                source_domain=f"source{index}.example",
                claim_candidate="A relevant claim " + "x" * 500,
                verbatim_excerpt="A verbatim excerpt " + "y" * 1800,
                stance=EvidenceStance.SUPPORTS,
                relevance="high",
            )
            for index in range(100)
        ]
        audit = AuditResult(
            sufficient=False,
            claims=[
                ClaimAssessment(
                    claim_id="C1",
                    claim="The final source contains report-changing evidence.",
                    status=ClaimStatus.SUPPORTED,
                    confidence=Confidence.MEDIUM,
                    supporting_evidence_ids=["ev_99"],
                    reasoning="The Auditor cited it.",
                )
            ],
            gaps=["Independent verification remains missing."],
        )

        selected = select_supervisor_evidence(evidence, audit)
        payload = [supervisor_evidence_payload(item) for item in selected]

        self.assertLessEqual(len(selected), 48)
        self.assertIn("ev_99", {item.evidence_id for item in selected})
        self.assertLess(len(json.dumps(payload)), 100_000)

    def test_research_digest_maps_workers_without_copying_excerpts(self) -> None:
        plan = ResearchPlan(
            coverage_contract=CoverageContract(
                required_dimensions=["benefits"],
            ),
            tasks=[ResearchTask(
                task_id="T1",
                question="What benefits are supported by primary evidence?",
                covered_dimensions=["benefits"],
            )],
        )
        evidence = [self.evidence_record(f"ev_{index}", f"https://s{index}.example")
                    for index in range(20)]
        audit = AuditResult(sufficient=False, gaps=["Independent verification"])

        digest = build_research_digest(
            plan, audit, evidence, {"T1": "succeeded"},
            [{"task_id": "T1", "unresolved_gaps": ["long-term effects"]}],
            {"total_evidence_items": 20}, 1,
        )
        payload = digest.model_dump_json()

        self.assertEqual(digest.evidence_count, 20)
        self.assertEqual(len(digest.workers[0].claims), 6)
        self.assertIn("long-term effects", digest.workers[0].unresolved_gaps)
        self.assertNotIn("Exact evidence excerpt", payload)

    def test_drilldown_uses_token_budget_and_keeps_cited_evidence_first(self) -> None:
        evidence = [self.evidence_record(f"ev_{index}", f"https://s{index}.example")
                    for index in range(30)]
        audit = AuditResult(
            sufficient=False,
            gaps=["Independent verification remains missing"],
            claims=[ClaimAssessment(
                claim_id="C1", claim="A cited finding", status=ClaimStatus.MIXED,
                confidence=Confidence.MEDIUM, supporting_evidence_ids=["ev_29"],
                reasoning="The cited evidence needs verification.",
            )],
        )

        selected = select_drilldown_evidence(
            evidence, audit, "What evidence is reliable?", token_budget=900,
        )

        self.assertTrue(selected)
        self.assertEqual(selected[0].evidence_id, "ev_29")
        estimated = sum(
            max(1, (len(json.dumps(supervisor_evidence_payload(item))) + 2) // 3)
            for item in selected
        )
        self.assertLessEqual(estimated, 900)

    def test_drilldown_is_scoped_to_each_actionable_gap(self) -> None:
        evidence = [
            EvidenceRecord(
                evidence_id=f"ev_t1_{index}", source_id=f"src_t1_{index}",
                task_id="T1", source_title="Benefit evidence",
                source_url=f"https://benefit{index}.example/study",
                source_domain=f"benefit{index}.example",
                claim_candidate="A benefit was measured.",
                verbatim_excerpt="A benefit was measured in the study.",
                stance=EvidenceStance.SUPPORTS, relevance="high",
            )
            for index in range(3)
        ] + [
            EvidenceRecord(
                evidence_id=f"ev_t2_{index}", source_id=f"src_t2_{index}",
                task_id="T2", source_title="Risk evidence",
                source_url=f"https://risk{index}.example/study",
                source_domain=f"risk{index}.example",
                claim_candidate="A long-term risk remains unresolved.",
                verbatim_excerpt="The long-term risk requires verification.",
                stance=EvidenceStance.CONTRADICTS, relevance="high",
            )
            for index in range(3)
        ]
        audit = AuditResult(
            sufficient=False,
            gaps=["Long-term risk verification is missing."],
            actionable_gaps=[AuditGap(
                gap_id="A2-G1", check_id="A2", dimension="risks",
                task_ids=["T2"], gap_type="verification", priority="high",
                description="Long-term risk verification is missing.",
                related_evidence_ids=["ev_t2_2"],
                missing_evidence="Independent long-term risk evidence.",
                suggested_query="long-term risk independent verification",
            )],
        )

        selected = select_drilldown_evidence(
            evidence, audit, "What are the benefits and risks?", token_budget=1200,
        )

        self.assertTrue(selected)
        self.assertEqual(selected[0].evidence_id, "ev_t2_2")
        self.assertEqual({item.task_id for item in selected}, {"T2"})

    def test_research_digest_is_persisted_by_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "research.sqlite")
            store.create_run("run_digest", "A sufficiently long question?", {})
            store.save_research_digest("run_digest", 1, {"version": 1, "round": 1})

            self.assertEqual(
                store.get_research_digest("run_digest", 1),
                {"version": 1, "round": 1},
            )

    def test_partial_required_dimension_overrides_semantic_sufficiency(self) -> None:
        tasks = [
            ResearchTask(task_id="T1", question="What benefits are demonstrated?",
                         covered_dimensions=["benefits"]),
            ResearchTask(task_id="T2", question="What risks are demonstrated?",
                         covered_dimensions=["risks"]),
        ]
        plan = ResearchPlan(
            coverage_contract=CoverageContract(
                decision_type="benefit-risk assessment",
                required_dimensions=["benefits", "risks"],
            ),
            tasks=tasks,
        )
        evidence = [
            EvidenceRecord(
                evidence_id=f"ev_{i}", source_id=f"src_{i}", task_id="T1",
                source_title=f"Study {i}", source_url=f"https://source{i}.example/study",
                source_domain=f"source{i}.example", claim_candidate="A benefit was measured.",
                verbatim_excerpt="A benefit was measured.", stance=EvidenceStance.SUPPORTS,
                relevance="high",
            ) for i in (1, 2)
        ]
        audit = AuditResult(
            sufficient=True,
            claims=[ClaimAssessment(
                claim_id="C1", claim="A benefit was measured.",
                status=ClaimStatus.SUPPORTED, confidence=Confidence.HIGH,
                supporting_evidence_ids=["ev_1", "ev_2"],
                reasoning="Two independent domains report it.",
            )],
        )

        gated = _apply_sufficiency_gates(
            audit, evidence, plan,
            {"T1": TaskStatus.SUCCEEDED.value, "T2": TaskStatus.PARTIAL.value},
            "What are the risks and benefits?",
        )

        self.assertFalse(gated.sufficient)
        self.assertFalse(gated.coverage_sufficient)
        self.assertFalse(gated.task_execution_sufficient)
        self.assertTrue(any("risks" in gap for gap in gated.gaps))
        self.assertTrue(gated.actionable_gaps)

    def test_generated_follow_up_query_is_bounded_and_revalidated(self) -> None:
        task = ResearchTask(
            task_id="T1",
            question="What evidence addresses the required dimension?",
            covered_dimensions=["missing dimension"],
        )
        plan = ResearchPlan(
            coverage_contract=CoverageContract(
                decision_type="evidence assessment",
                required_dimensions=["missing dimension"],
            ),
            tasks=[task],
        )
        audit = AuditResult(sufficient=False, claims=[], gaps=[])

        gated = _apply_sufficiency_gates(
            audit,
            [],
            plan,
            {"T1": TaskStatus.PARTIAL.value},
            "A very long research question " * 30,
        )

        self.assertIsInstance(gated, AuditResult)
        self.assertIsNotNone(gated.follow_up_query)
        self.assertLessEqual(len(gated.follow_up_query or ""), 400)
        self.assertEqual(_bounded_query("  compact   query  "), "compact query")

    def test_required_dimension_needs_an_audited_conclusion(self) -> None:
        task = ResearchTask(
            task_id="T1",
            question="What benefits are supported by direct evidence?",
            covered_dimensions=["benefits"],
        )
        plan = ResearchPlan(
            coverage_contract=CoverageContract(
                decision_type="benefit assessment",
                required_dimensions=["benefits"],
            ),
            tasks=[task],
        )
        evidence = [
            self.evidence_record("ev_1", "https://one.example/study"),
            self.evidence_record("ev_2", "https://two.example/study"),
        ]
        audit = AuditResult(
            sufficient=True,
            claims=[
                ClaimAssessment(
                    claim_id="C1",
                    claim="A finding was reported without a mapped dimension.",
                    status=ClaimStatus.SUPPORTED,
                    confidence=Confidence.MEDIUM,
                    supporting_evidence_ids=["ev_1", "ev_2"],
                    reasoning="Two sources support the finding.",
                )
            ],
        )

        gated = _apply_sufficiency_gates(
            audit,
            evidence,
            plan,
            {"T1": TaskStatus.SUCCEEDED.value},
            "What benefits are supported?",
        )

        self.assertFalse(gated.sufficient)
        self.assertFalse(gated.coverage_sufficient)
        self.assertTrue(
            any("audited conclusion" in gap for gap in gated.gaps)
        )

    def test_checkpoint_resume_retries_failed_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                workspace=root,
                tavily_api_key=None,
                deepseek_api_key=None,
            )
            settings.ensure_directories()
            store = EvidenceStore(settings.research_db)
            model = FailOncePlanner()
            runtime = ResearchRuntime(settings, store, FakeSearchProvider(), model)
            run_id = "run_resume"
            question = "What does the evidence support?"
            store.create_run(run_id, question, {"fake": True})

            with self.assertRaises(RuntimeError):
                run_graph(
                    settings=settings,
                    store=store,
                    runtime=runtime,
                    run_id=run_id,
                    question=question,
                )

            result = run_graph(
                settings=settings,
                store=store,
                runtime=runtime,
                run_id=run_id,
                question=question,
                resume=True,
            )
            self.assertEqual(result["terminal_reason"], "evidence_sufficient")
            self.assertTrue(Path(result["report_path"]).exists())


if __name__ == "__main__":
    unittest.main()
