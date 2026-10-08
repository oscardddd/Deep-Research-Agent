from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .schemas import (
    AuditResult,
    CredibilityTier,
    EvidenceBatch,
    EvidenceRecord,
    ResearchTask,
    SearchResult,
    SourceAssessment,
    SourceAssessmentBatch,
    SourceDirectness,
    SourceEligibility,
    SourceHardFlag,
    SourceType,
    TaskStatus,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_id(prefix: str, *parts: str) -> str:
    raw = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:20]}"


def canonicalize_url(url: str) -> str:
    split = urlsplit(url.strip())
    ignored = {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "source",
    }
    query = [
        (key, value)
        for key, value in parse_qsl(split.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in ignored
    ]
    path = split.path.rstrip("/") or "/"
    return urlunsplit(
        (split.scheme.lower(), split.netloc.lower(), path, urlencode(query), "")
    )


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


class EvidenceStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    report_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, task_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS operations (
                    operation_key TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT,
                    operation_type TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    canonical_url TEXT NOT NULL,
                    content TEXT NOT NULL,
                    score REAL NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    UNIQUE (run_id, canonical_url),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    claim_candidate TEXT NOT NULL,
                    verbatim_excerpt TEXT NOT NULL,
                    stance TEXT NOT NULL,
                    relevance TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'secondary',
                    source_directness TEXT NOT NULL DEFAULT 'unclear',
                    credibility_tier TEXT NOT NULL DEFAULT 'C',
                    discovery_only INTEGER NOT NULL DEFAULT 0,
                    attributed_source_name TEXT,
                    attributed_source_url TEXT,
                    follow_up_query TEXT,
                    source_assessment_id TEXT,
                    source_quality_score INTEGER,
                    evidence_fitness_score INTEGER,
                    source_assessment_rationale TEXT,
                    source_eligibility TEXT,
                    source_hard_flags TEXT,
                    source_score_dimensions TEXT,
                    dimension_ids TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    UNIQUE (run_id, task_id, source_id, verbatim_excerpt),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id),
                    FOREIGN KEY (source_id) REFERENCES sources(source_id)
                );

                CREATE TABLE IF NOT EXISTS source_assessments (
                    assessment_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    rubric_version TEXT NOT NULL,
                    source_quality_score INTEGER NOT NULL,
                    evidence_fitness_score INTEGER NOT NULL,
                    credibility_tier TEXT NOT NULL,
                    eligibility TEXT NOT NULL,
                    assessment_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (run_id, task_id, source_id, rubric_version),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id),
                    FOREIGN KEY (source_id) REFERENCES sources(source_id)
                );

                CREATE TABLE IF NOT EXISTS audits (
                    run_id TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    audit_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, round),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS research_digests (
                    run_id TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    digest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, round),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS audit_plans (
                    run_id TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, round),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id);
                CREATE INDEX IF NOT EXISTS idx_source_assessments_run
                    ON source_assessments(run_id);
                CREATE INDEX IF NOT EXISTS idx_operations_run ON operations(run_id);
                """
            )
            existing_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(evidence)").fetchall()
            }
            migrations = {
                "source_type": "TEXT NOT NULL DEFAULT 'secondary'",
                "source_directness": "TEXT NOT NULL DEFAULT 'unclear'",
                "credibility_tier": "TEXT NOT NULL DEFAULT 'C'",
                "discovery_only": "INTEGER NOT NULL DEFAULT 0",
                "attributed_source_name": "TEXT",
                "attributed_source_url": "TEXT",
                "follow_up_query": "TEXT",
                "source_assessment_id": "TEXT",
                "source_quality_score": "INTEGER",
                "evidence_fitness_score": "INTEGER",
                "source_assessment_rationale": "TEXT",
                "source_eligibility": "TEXT",
                "source_hard_flags": "TEXT",
                "source_score_dimensions": "TEXT",
                "dimension_ids": "TEXT NOT NULL DEFAULT '[]'",
            }
            for column, definition in migrations.items():
                if column not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE evidence ADD COLUMN {column} {definition}"
                    )

    def create_run(self, run_id: str, question: str, config: dict[str, Any]) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO runs
                    (run_id, question, status, config_json, created_at, updated_at)
                VALUES (?, ?, 'running', ?, ?, ?)
                """,
                (run_id, question, json.dumps(config, sort_keys=True), now, now),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def set_run_status(
        self, run_id: str, status: str, report_path: str | None = None
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, report_path = COALESCE(?, report_path), updated_at = ?
                WHERE run_id = ?
                """,
                (status, report_path, utc_now(), run_id),
            )

    def upsert_task(self, run_id: str, task: ResearchTask) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (run_id, task_id, status, spec_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, task_id) DO UPDATE SET
                    spec_json = excluded.spec_json,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    task.task_id,
                    TaskStatus.PENDING.value,
                    task.model_dump_json(),
                    utc_now(),
                ),
            )

    def set_task_status(
        self,
        run_id: str,
        task_id: str,
        status: TaskStatus,
        error: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE tasks SET status = ?, error = ?, updated_at = ?
                WHERE run_id = ? AND task_id = ?
                """,
                (status.value, error, utc_now(), run_id, task_id),
            )

    def list_tasks(self, run_id: str) -> list[dict[str, Any]]:
        """Return task specifications together with their durable run status."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT task_id, status, spec_json, error, updated_at
                FROM tasks
                WHERE run_id = ?
                ORDER BY task_id
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "task_id": row["task_id"],
                "status": row["status"],
                "spec": json.loads(row["spec_json"]),
                "error": row["error"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def get_successful_operation(self, operation_key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT response_json FROM operations
                WHERE operation_key = ? AND status = 'succeeded'
                """,
                (operation_key,),
            ).fetchone()
        if not row or not row["response_json"]:
            return None
        return json.loads(row["response_json"])

    def start_operation(
        self,
        operation_key: str,
        run_id: str,
        task_id: str | None,
        operation_type: str,
        request: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO operations
                    (operation_key, run_id, task_id, operation_type,
                     request_json, status, started_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?)
                ON CONFLICT(operation_key) DO UPDATE SET
                    status = CASE
                        WHEN operations.status = 'succeeded' THEN 'succeeded'
                        ELSE 'running'
                    END,
                    error = NULL,
                    started_at = CASE
                        WHEN operations.status = 'succeeded' THEN operations.started_at
                        ELSE excluded.started_at
                    END
                """,
                (
                    operation_key,
                    run_id,
                    task_id,
                    operation_type,
                    json.dumps(request, sort_keys=True),
                    utc_now(),
                ),
            )

    def complete_operation(
        self, operation_key: str, response: dict[str, Any]
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE operations
                SET status = 'succeeded', response_json = ?, completed_at = ?
                WHERE operation_key = ?
                """,
                (json.dumps(response, sort_keys=True), utc_now(), operation_key),
            )

    def fail_operation(self, operation_key: str, error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE operations
                SET status = 'failed', error = ?, completed_at = ?
                WHERE operation_key = ?
                """,
                (error[:2000], utc_now(), operation_key),
            )

    def persist_worker_artifacts(
        self,
        run_id: str,
        task: ResearchTask,
        results: list[SearchResult],
        batch: EvidenceBatch,
    ) -> tuple[list[str], list[str]]:
        source_by_rank: dict[int, tuple[str, SearchResult]] = {}
        source_ids: list[str] = []
        evidence_ids: list[str] = []

        with self.connect() as connection:
            for index, result in enumerate(results, start=1):
                canonical_url = canonicalize_url(result.url)
                source_id = stable_id("src", run_id, canonical_url)
                source_by_rank[index] = (source_id, result)
                source_ids.append(source_id)
                connection.execute(
                    """
                    INSERT INTO sources
                        (source_id, run_id, task_id, query, rank, title, url,
                         canonical_url, content, score, retrieved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, canonical_url) DO UPDATE SET
                        score = MAX(sources.score, excluded.score),
                        content = CASE
                            WHEN length(excluded.content) > length(sources.content)
                            THEN excluded.content ELSE sources.content END
                    """,
                    (
                        source_id,
                        run_id,
                        task.task_id,
                        result.query or task.query_hint or task.question,
                        index,
                        result.title,
                        result.url,
                        canonical_url,
                        result.content,
                        result.score,
                        utc_now(),
                    ),
                )

            assessment_rows = connection.execute(
                """
                SELECT source_id, assessment_id, assessment_json
                FROM source_assessments
                WHERE run_id = ? AND task_id = ?
                """,
                (run_id, task.task_id),
            ).fetchall()
            assessments = {
                row["source_id"]: (
                    row["assessment_id"],
                    SourceAssessment.model_validate_json(row["assessment_json"]),
                )
                for row in assessment_rows
            }

            for item in batch.evidence:
                source_entry = source_by_rank.get(item.source_rank)
                if source_entry is None:
                    continue
                source_id, source = source_entry
                assessment_entry = assessments.get(source_id)
                assessment_id = None
                assessment = None
                if assessment_entry:
                    assessment_id, assessment = assessment_entry

                source_type = (
                    assessment.source_type if assessment else item.source_type
                )
                source_directness = item.source_directness
                if assessment:
                    source_directness = assessment.source_directness
                    if item.source_directness == SourceDirectness.ATTRIBUTED:
                        source_directness = SourceDirectness.ATTRIBUTED
                credibility_tier = (
                    assessment.credibility_tier
                    if assessment
                    else item.credibility_tier
                )
                discovery_only = item.discovery_only or bool(
                    assessment
                    and assessment.eligibility
                    != SourceEligibility.FINAL_EVIDENCE
                )
                source_eligibility = (
                    assessment.eligibility if assessment else None
                )
                source_hard_flags = (
                    list(assessment.hard_flags) if assessment else []
                )
                if source_directness == SourceDirectness.ATTRIBUTED:
                    discovery_only = True
                    if SourceHardFlag.ATTRIBUTED_ONLY not in source_hard_flags:
                        source_hard_flags.append(SourceHardFlag.ATTRIBUTED_ONLY)
                    if credibility_tier in {
                        CredibilityTier.A,
                        CredibilityTier.B,
                    }:
                        credibility_tier = CredibilityTier.C
                if item.source_type in {
                    SourceType.AI_SYNTHESIS,
                    SourceType.SOCIAL,
                    SourceType.AGGREGATOR,
                }:
                    source_type = item.source_type
                    credibility_tier = CredibilityTier.D
                    discovery_only = True
                    hard_flag = {
                        SourceType.AI_SYNTHESIS: SourceHardFlag.AI_SYNTHESIS,
                        SourceType.SOCIAL: SourceHardFlag.SOCIAL_CONTENT,
                        SourceType.AGGREGATOR: SourceHardFlag.CONTENT_FARM,
                    }[item.source_type]
                    if hard_flag not in source_hard_flags:
                        source_hard_flags.append(hard_flag)
                if discovery_only and source_eligibility != SourceEligibility.REJECT:
                    source_eligibility = SourceEligibility.DISCOVERY_ONLY
                attributed_source_name = (
                    (assessment.attributed_source_name if assessment else None)
                    or item.attributed_source_name
                )
                attributed_source_url = (
                    (assessment.attributed_source_url if assessment else None)
                    or item.attributed_source_url
                )
                follow_up_query = (
                    (assessment.follow_up_query if assessment else None)
                    or item.follow_up_query
                )

                # Search snippets are evidence only when the quoted excerpt is
                # actually present. This prevents the extraction model from
                # inventing a plausible-looking quotation.
                if normalized_text(item.verbatim_excerpt) not in normalized_text(
                    source.content
                ):
                    continue

                evidence_id = stable_id(
                    "ev",
                    run_id,
                    task.task_id,
                    source_id,
                    normalized_text(item.verbatim_excerpt),
                )
                evidence_ids.append(evidence_id)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO evidence
                        (evidence_id, run_id, task_id, source_id,
                         claim_candidate, verbatim_excerpt, stance,
                         relevance, source_type, source_directness,
                         credibility_tier, discovery_only,
                         attributed_source_name, attributed_source_url,
                         follow_up_query, source_assessment_id,
                         source_quality_score, evidence_fitness_score,
                         source_assessment_rationale, source_eligibility,
                         source_hard_flags, source_score_dimensions,
                         dimension_ids, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence_id,
                        run_id,
                        task.task_id,
                        source_id,
                        item.claim_candidate,
                        item.verbatim_excerpt,
                        item.stance.value,
                        item.relevance,
                        source_type.value,
                        source_directness.value,
                        credibility_tier.value,
                        int(discovery_only),
                        attributed_source_name,
                        attributed_source_url,
                        follow_up_query,
                        assessment_id,
                        assessment.source_quality_score if assessment else None,
                        assessment.evidence_fitness_score if assessment else None,
                        (
                            json.dumps(assessment.rationale, ensure_ascii=False)
                            if assessment
                            else None
                        ),
                        source_eligibility.value if source_eligibility else None,
                        (
                            json.dumps(
                                [item.value for item in source_hard_flags],
                                ensure_ascii=False,
                            )
                            if assessment
                            else None
                        ),
                        (
                            assessment.scores.model_dump_json()
                            if assessment
                            else None
                        ),
                        json.dumps(item.dimension_ids, ensure_ascii=False),
                        utc_now(),
                    ),
                )

        return list(dict.fromkeys(source_ids)), list(dict.fromkeys(evidence_ids))

    def persist_source_assessments(
        self,
        run_id: str,
        task: ResearchTask,
        results: list[SearchResult],
        batch: SourceAssessmentBatch,
    ) -> list[str]:
        assessment_ids: list[str] = []
        by_rank = {item.source_rank: item for item in batch.assessments}
        with self.connect() as connection:
            for rank, result in enumerate(results, start=1):
                assessment = by_rank.get(rank)
                if assessment is None:
                    continue
                source_id = stable_id(
                    "src", run_id, canonicalize_url(result.url)
                )
                assessment_id = stable_id(
                    "sa",
                    run_id,
                    task.task_id,
                    source_id,
                    assessment.rubric_version,
                )
                assessment_ids.append(assessment_id)
                connection.execute(
                    """
                    INSERT INTO source_assessments
                        (assessment_id, run_id, task_id, source_id,
                         rubric_version, source_quality_score,
                         evidence_fitness_score, credibility_tier,
                         eligibility, assessment_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, task_id, source_id, rubric_version)
                    DO UPDATE SET
                        source_quality_score = excluded.source_quality_score,
                        evidence_fitness_score = excluded.evidence_fitness_score,
                        credibility_tier = excluded.credibility_tier,
                        eligibility = excluded.eligibility,
                        assessment_json = excluded.assessment_json
                    """,
                    (
                        assessment_id,
                        run_id,
                        task.task_id,
                        source_id,
                        assessment.rubric_version,
                        assessment.source_quality_score,
                        assessment.evidence_fitness_score,
                        assessment.credibility_tier.value,
                        assessment.eligibility.value,
                        assessment.model_dump_json(),
                        utc_now(),
                    ),
                )
        return assessment_ids

    def list_source_assessments(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT sa.assessment_id, sa.task_id, sa.source_id,
                       s.title AS source_title, s.url AS source_url,
                       sa.rubric_version, sa.source_quality_score,
                       sa.evidence_fitness_score, sa.credibility_tier,
                       sa.eligibility, sa.assessment_json, sa.created_at
                FROM source_assessments sa
                JOIN sources s ON s.source_id = sa.source_id
                WHERE sa.run_id = ?
                ORDER BY sa.created_at, sa.assessment_id
                """,
                (run_id,),
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            raw_assessment = record.pop("assessment_json")
            record["assessment"] = json.loads(raw_assessment)
            records.append(record)
        return records

    def list_evidence(self, run_id: str) -> list[EvidenceRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT e.evidence_id, e.source_id, e.task_id,
                       s.title AS source_title, s.url AS source_url,
                       s.canonical_url,
                       e.claim_candidate, e.verbatim_excerpt,
                       e.stance, e.relevance, e.source_type,
                       e.source_directness, e.credibility_tier,
                       e.discovery_only, e.attributed_source_name,
                       e.attributed_source_url, e.follow_up_query,
                       e.source_assessment_id, e.source_quality_score,
                       e.evidence_fitness_score,
                       e.source_assessment_rationale, e.source_eligibility,
                       e.source_hard_flags, e.source_score_dimensions
                       , e.dimension_ids
                FROM evidence e
                JOIN sources s ON s.source_id = e.source_id
                WHERE e.run_id = ?
                ORDER BY e.created_at, e.evidence_id
                """,
                (run_id,),
            ).fetchall()

        records: list[EvidenceRecord] = []
        for row in rows:
            domain = urlsplit(row["canonical_url"]).netloc
            records.append(
                EvidenceRecord(
                    evidence_id=row["evidence_id"],
                    source_id=row["source_id"],
                    task_id=row["task_id"],
                    source_title=row["source_title"],
                    source_url=row["source_url"],
                    source_domain=domain,
                    claim_candidate=row["claim_candidate"],
                    verbatim_excerpt=row["verbatim_excerpt"],
                    stance=row["stance"],
                    relevance=row["relevance"],
                    source_type=row["source_type"],
                    source_directness=row["source_directness"],
                    credibility_tier=row["credibility_tier"],
                    discovery_only=bool(row["discovery_only"]),
                    attributed_source_name=row["attributed_source_name"],
                    attributed_source_url=row["attributed_source_url"],
                    follow_up_query=row["follow_up_query"],
                    provenance_key=(
                        canonicalize_url(row["attributed_source_url"])
                        if (
                            row["source_directness"]
                            == SourceDirectness.ATTRIBUTED.value
                            and row["attributed_source_url"]
                        )
                        else row["canonical_url"]
                    ),
                    source_assessment_id=row["source_assessment_id"],
                    source_quality_score=row["source_quality_score"],
                    evidence_fitness_score=row["evidence_fitness_score"],
                    source_assessment_rationale=(
                        json.loads(row["source_assessment_rationale"])
                        if row["source_assessment_rationale"]
                        else []
                    ),
                    source_eligibility=row["source_eligibility"],
                    source_hard_flags=(
                        json.loads(row["source_hard_flags"])
                        if row["source_hard_flags"]
                        else []
                    ),
                    source_score_dimensions=(
                        json.loads(row["source_score_dimensions"])
                        if row["source_score_dimensions"]
                        else None
                    ),
                    dimension_ids=(
                        json.loads(row["dimension_ids"])
                        if row["dimension_ids"]
                        else []
                    ),
                )
            )
        return records

    def save_research_digest(
        self, run_id: str, round_number: int, digest: dict[str, Any]
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO research_digests
                    (run_id, round, digest_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id, round) DO UPDATE SET
                    digest_json = excluded.digest_json,
                    created_at = excluded.created_at
                """,
                (run_id, round_number, json.dumps(digest, sort_keys=True), utc_now()),
            )

    def get_research_digest(
        self, run_id: str, round_number: int
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT digest_json FROM research_digests WHERE run_id=? AND round=?",
                (run_id, round_number),
            ).fetchone()
        return json.loads(row["digest_json"]) if row else None

    def save_audit_plan(
        self, run_id: str, round_number: int, plan: dict[str, Any]
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_plans (run_id, round, plan_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id, round) DO UPDATE SET
                    plan_json = excluded.plan_json,
                    created_at = excluded.created_at
                """,
                (run_id, round_number, json.dumps(plan, sort_keys=True), utc_now()),
            )

    def get_audit_plan(
        self, run_id: str, round_number: int
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT plan_json FROM audit_plans WHERE run_id=? AND round=?",
                (run_id, round_number),
            ).fetchone()
        return json.loads(row["plan_json"]) if row else None

    def get_latest_audit_plan(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT plan_json FROM audit_plans
                WHERE run_id=? ORDER BY round DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return json.loads(row["plan_json"]) if row else None

    def list_sources(self, run_id: str) -> list[dict[str, Any]]:
        """Return source metadata without embedding complete page bodies."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT source_id, task_id, query, rank, title, url,
                       canonical_url, score, length(content) AS content_chars,
                       retrieved_at
                FROM sources
                WHERE run_id = ?
                ORDER BY retrieved_at, source_id
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_audit(self, run_id: str, round_number: int, audit: AuditResult) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO audits (run_id, round, audit_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id, round) DO UPDATE SET
                    audit_json = excluded.audit_json,
                    created_at = excluded.created_at
                """,
                (run_id, round_number, audit.model_dump_json(), utc_now()),
            )

    def get_latest_audit(self, run_id: str) -> AuditResult | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT audit_json
                FROM audits
                WHERE run_id = ?
                ORDER BY round DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return AuditResult.model_validate_json(row["audit_json"]) if row else None

    def operation_summary(self, run_id: str) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT operation_type, COUNT(*) AS count
                FROM operations
                WHERE run_id = ? AND status = 'succeeded'
                GROUP BY operation_type
                """,
                (run_id,),
            ).fetchall()
        return {row["operation_type"]: row["count"] for row in rows}

    def list_operations(
        self, run_id: str, *, include_response: bool = False
    ) -> list[dict[str, Any]]:
        response_column = ", response_json" if include_response else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT operation_key, task_id, operation_type, request_json,
                       status, error, started_at, completed_at{response_column}
                FROM operations
                WHERE run_id = ?
                ORDER BY started_at, operation_key
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]
