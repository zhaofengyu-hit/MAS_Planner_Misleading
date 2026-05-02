"""
Core analysis engine.

- Single database: data/agents.db (agents table + agent_analysis table).
- File exports (CSV, LaTeX, …) go to  data/study/.
"""
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Type, TypeVar

import yaml
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_HERE = Path(__file__).parent
_CONFIG_PATH = _HERE / "config.yaml"

AGENTS_DB = Path(__file__).parent / "data" / "agents.db"

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
from typing import Callable
_TASK_REGISTRY: dict[str, Callable] = {}


def task(name: str):
    """Decorator: register a function as a CLI-callable task.

    The decorated function must accept (analyzer, *, platform, limit) where
    *analyzer* is an Analyzer instance.  It can live anywhere (task modules,
    inside the class, …) — the registry holds the callable directly.
    """
    def decorator(fn):
        _TASK_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class Analyzer:
    def __init__(self, config_path: Path = _CONFIG_PATH) -> None:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        llm_cfg = cfg["llm"]
        self._model: str       = llm_cfg["model"]
        self._temperature: float = llm_cfg.get("temperature", 0.2)
        self._max_tokens: int  = llm_cfg.get("max_tokens", 4096)
        self._max_retries: int = cfg.get("max_retries", 3)

        self._client = OpenAI(
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg.get("base_url"),
        )

        self._conn = sqlite3.connect(str(AGENTS_DB), check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    def _call_structured(self, system: str, user: str, model_cls: Type[T]) -> T:
        """Call the LLM with json_schema structured output; retry on failure."""
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": model_cls.__name__,
                            "schema": model_cls.model_json_schema(),
                        },
                    },
                )
                return model_cls.model_validate_json(response.choices[0].message.content)
            except Exception as exc:
                last_exc = exc
                logger.warning("Attempt %d/%d failed: %s", attempt, self._max_retries, exc)
        raise RuntimeError(f"All {self._max_retries} attempts failed") from last_exc

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------
    def _ensure_table(self, table: str, columns: list[tuple[str, str]]) -> None:
        """Create a result table in study.db if it doesn't already exist.

        Args:
            table:   Table name (e.g. "category").
            columns: (col_name, col_type) pairs; agent_id + analyzed_at are added automatically.
        """
        col_defs = "".join(f"\n    {n}  {t}," for n, t in columns)
        self._conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                agent_id    INTEGER PRIMARY KEY,{col_defs}
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    def _get_unprocessed(
        self,
        table: str,
        platform: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Return agents (from agents.db) that have no row in *table* yet."""
        where = "WHERE s.agent_id IS NULL"
        params: list = []
        if platform:
            where += " AND a.platform = ?"
            params.append(platform)

        sql = f"""
            SELECT a.id, a.platform, a.name, a.description
            FROM agents a
            LEFT JOIN {table} s ON a.id = s.agent_id
            {where}
            ORDER BY RANDOM()
        """
        if limit:
            sql += f" LIMIT {limit}"

        rows = self._conn.execute(sql, params).fetchall()
        return [{"id": r[0], "platform": r[1], "name": r[2], "description": r[3]}
                for r in rows]

    def _save_result(self, table: str, agent_id: int, data: dict) -> None:
        """Upsert one result row into *table*."""
        cols = list(data.keys())
        col_list   = ", ".join(cols)
        placeholders = ", ".join("?" * len(cols))
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} (agent_id, {col_list}) VALUES (?, {placeholders})",
            [agent_id, *data.values()],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Unified analysis table helpers
    # ------------------------------------------------------------------
    def _ensure_analysis_table(self) -> None:
        """Create the shared agent_analysis table in study.db if needed."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS agent_analysis (
                agent_id        INTEGER PRIMARY KEY,
                has_description INTEGER,   -- 0/1, SQL-computed by fill_desc
                is_meaningful   INTEGER,   -- NULL=not evaluated / 0=gibberish / 1=meaningful
                has_capability  INTEGER,   -- NULL=not evaluated / 0=no / 1=yes
                has_input_spec  INTEGER,   -- NULL=not evaluated / 0=no / 1=yes
                has_output_spec INTEGER,   -- NULL=not evaluated / 0=no / 1=yes
                has_constraints INTEGER,   -- NULL=not evaluated / 0=no / 1=yes
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    def _upsert_analysis(self, agent_id: int, **fields) -> None:
        """Insert or update specific columns in agent_analysis.

        Only the provided keyword arguments are written; other columns are
        left untouched if the row already exists.
        """
        col_list     = ", ".join(fields.keys())
        placeholders = ", ".join("?" * len(fields))
        updates      = ", ".join(f"{k} = excluded.{k}" for k in fields)
        self._conn.execute(
            f"""INSERT INTO agent_analysis (agent_id, {col_list})
                VALUES (?, {placeholders})
                ON CONFLICT(agent_id) DO UPDATE SET
                    {updates},
                    updated_at = CURRENT_TIMESTAMP""",
            [agent_id, *fields.values()],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Convenience: paths for file exports
    # ------------------------------------------------------------------
    @staticmethod
    def export_dir() -> Path:
        """Directory for file outputs (CSVs, LaTeX tables, …)."""
        p = AGENTS_DB.parent / "study"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Analysis tasks — add new task methods below this line
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def run(self, task_name: str, platform: str | None, limit: int | None) -> None:
        if task_name not in _TASK_REGISTRY:
            available = ", ".join(sorted(_TASK_REGISTRY)) or "(none registered yet)"
            logger.error("Unknown task %r. Available: %s", task_name, available)
            sys.exit(1)
        fn = _TASK_REGISTRY[task_name]
        fn(self, platform=platform, limit=limit)

    def close(self) -> None:
        self._conn.close()
