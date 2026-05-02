# MAS Planner Misleading

## Project Structure

```
.
├── Benchmark/              # GAIA benchmark dataset
├── Company/                # MAS implementation
│   ├── Component/          # Tool components (web, document, database, etc.)
│   ├── Defense/            # Defense mechanisms (DescriptionNormalizer)
│   ├── Docker/             # Docker shared folder
│   ├── Roles/              # Agent role definitions
│   └── Teams/              # Team configurations
├── analysis/               # Marketplace agent description analysis
│   ├── data/               # SQLite database of collected agent descriptions
│   └── tasks/              # Analysis tasks and prompts (analyze.py, report.py)
├── scripts/                # Experiment runner scripts
├── statistics/             # Metrics computation and table generation
├── test/AttackTest/        # Execution and plan-only test entry points
├── utils/                  # Shared utilities
├── env/                    # Compressed virtual environment (split archive)
├── tmp/                    # Runtime outputs (logs, plans, table data)
├── config.yaml             # LLM configuration
├── .env                    # API keys and workspace path
└── playwright_config.json  # Playwright MCP server configuration
```

---

## 1. Environment Setup

### 1.1 Prerequisites

- Python **3.11.x**
- Node.js (for Playwright MCP server)

### 1.2 Restore and Fix the Virtual Environment

The virtual environment is stored as a split compressed archive under `env/`.

```bash
# Merge parts, extract, then clean up
cat env/venv.tar.gz.part* > venv.tar.gz
tar -xzf venv.tar.gz
rm venv.tar.gz

# Fix absolute paths inside the venv to match the current machine
python3.11 -m venv --upgrade .venv
```

### 1.3 Activate

```bash
source .venv/bin/activate
```

### 1.4 Configuration Files

Copy the backup templates and fill in your API keys:

```bash
cp config.yaml.backup config.yaml
cp .env.backup .env
cp playwright_config.json.backup playwright_config.json
```

**`config.yaml`** — LLM used by the statistics module (plan metric computation):
```yaml
llm:
  api_key: <ali-bailian-api-key>
  model: "qwen3-max"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  temperature: 0.7
  max_tokens: 10000
```

**`.env`** — Multimodal LLM keys and workspace path:
```
MY_WORKSPACE="<absolute_path_to_project_root>/tmp"

AUDIO_LLM_API_KEY=<ali-bailian-api-key>
AUDIO_LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1
AUDIO_LLM_MODEL_NAME=qwen3-asr-flash
AUDIO_LLM_PROTOCOL=dashscope

IMAGE_LLM_API_KEY=<ali-bailian-api-key>
IMAGE_LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
IMAGE_LLM_MODEL_NAME=qwen3-vl-plus
IMAGE_LLM_TEMPERATURE=1.0
IMAGE_LLM_PROTOCOL=dashscope

VIDEO_LLM_API_KEY=<ali-bailian-api-key>
VIDEO_LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1
VIDEO_LLM_MODEL_NAME=qwen3-vl-plus
VIDEO_LLM_TEMPERATURE=1.0
VIDEO_LLM_PROTOCOL=dashscope

RAGFLOW_API=<ragflow-api-key>
```

**`playwright_config.json`** — Update `outputDir` to your project root:
```json
{
  "server": { "port": 61002 },
  "outputDir": "<absolute_path_to_project_root>/tmp/browser_wp/",
  "imageResponses": "omit",
  "browser": {
    "browserName": "chromium",
    "launchOptions": {
      "headless": false,
      "args": ["--window-size=2560,1600", "--window-position=0,0"]
    }
  }
}
```

---

## 2. Docker Services

### 2.1 Core Services (PostgreSQL, etc.)

> TODO: document the docker-compose file and how to start it.

```bash
docker compose -f <compose-file> up -d
```

### 2.2 RAGFlow

> TODO: document RAGFlow container setup and how to obtain the API key for `.env`.

---

## 3. Playwright Setup

Playwright is required for the web browsing capability of the MAS execution agent. This step is only needed when running full MAS execution (Section 4.1).

### 3.1 Install Playwright and Browser

```bash
pip install playwright
playwright install chromium
```

### 3.2 Install Playwright MCP

```bash
npm install -g @playwright/mcp
```

### 3.3 Start the Playwright MCP Server

The MCP server must be running before executing MAS. Start it with the provided config:

```bash
npx @playwright/mcp --config playwright_config.json
```

Keep this process running in a separate terminal throughout the experiment.

---

## 4. Running Experiments

### 4.1 Run MAS Execution

Runs the full multi-agent system on the GAIA benchmark and records results.

```bash
python test/AttackTest/ExecutionTest.py \
    --provided_desc "<agent description string>" \
    --content_dir "<output subdirectory name>" \
    [--use_defense]
```

- `--provided_desc`: Description injected into the PostgresManager agent. Use a benign description for baseline, or a crafted one for attack variants.
- `--content_dir`: Output directory name under `tmp/logs/` (e.g. `baseline`, `over_fragmentation`).
- `--use_defense`: Enable DescriptionNormalizer defense on the provided description.

Results are saved to:
- `tmp/logs/<content_dir>/task_cache.json` — answers and judgements per task
- `tmp/logs/<content_dir>/<task_id>_llm_call_log.json` — per-task LLM call traces

To limit execution to a subset of tasks (e.g. for testing), edit `utils/task_filter.py`:
```python
TASK_IDS: list[str] | None = ["task-id-1", "task-id-2", ...]  # set to None to run all
```

### 4.2 Fetch Plans Only

Runs only the planning stage to collect structured task plans from the Planner agent. Does not require Playwright or Docker.

```bash
python scripts/fetch_plans.py \
    --name <variant_name> \
    --desc "<agent description string>" \
    [--planner <planner_key>] \
    [--use_defense] \
    [--replace_worker_descs] \
    [--overwrite]
```

- `--name`: Output file name, saved to `tmp/plans/<name>.json`.
- `--planner`: LLM to use as the planner. Options: `deepseek_ali` (default), `deepseek`, `gpt`, `gpt_mini`, `kimi`, `qwen`.
- `--use_defense`: Apply DescriptionNormalizer to the provided description.
- `--replace_worker_descs`: Replace non-DB worker agent descriptions with real-world descriptions from `scripts/agent_descriptions_replace.json`.
- `--overwrite`: Re-run even if the output file already exists.

Supports checkpoint resume: interrupted runs automatically continue from where they left off.

---

## 5. Computing Metrics

### 5.1 From Pre-computed Data (Recommended)

If `tmp/plans/` already contains pre-computed plan metrics and execution metrics (as provided in the release), seed the checkpoints first to skip LLM-based recomputation:

```bash
python scripts/seed_checkpoints.py
```

This writes all three checkpoint files to `tmp/table_data/` and requires no API calls.

### 5.2 Full Recomputation

To recompute all metrics from scratch (requires LLM API access for plan metrics):

```bash
python statistics/compute_metrics.py
```

Reads from `tmp/plans/` and `tmp/logs/`. Supports resume — already-completed variants are skipped automatically.

---

## 6. Generating Tables

All scripts output LaTeX source to stdout.

### Marketplace Agent Description Statistics

Generates a summary table of agent descriptions collected from the marketplace:

```bash
python -m analysis --task report
```

The prompts used for analyzing agent descriptions are in `analysis/tasks/analyze.py`.

### Tables 1–4: Main Experiment Results

```bash
python statistics/gen_table.py
```

### Table 5: LLM Comparison

```bash
python statistics/llm_comparison.py table
```

To recompute plan metrics for LLM comparison variants before generating the table:
```bash
python statistics/llm_comparison.py
```

### Table 6: Real Agent Descriptions

```bash
python statistics/real_agent_desc.py table
```

To recompute:
```bash
python statistics/real_agent_desc.py
```
