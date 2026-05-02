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

## Getting Started

This repository uses **Git LFS** for large files (`env/` and `analysis/data/agents.db`). Make sure Git LFS is installed before cloning:

```bash
# Install git-lfs (Ubuntu)
sudo apt install git-lfs
git lfs install

# Clone and download LFS files
git clone https://github.com/zhaofengyu-hit/MAS_Planner_Misleading.git
cd MAS_Planner_Misleading
git lfs pull
```

---

## 1. Environment Setup

> Tested on **WSL2 (Ubuntu 24.04)**.

### 1.1 Prerequisites

**Python 3.11** — the virtual environment was built with CPython 3.11.14 via [uv](https://github.com/astral-sh/uv). Using uv is recommended:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.11.14
uv python install 3.11.14
```

Any other Python 3.11 installation (apt, conda, pyenv) also works as long as `python3.11` is available on `PATH`.

**Node.js 18** — required only for running MAS execution (Playwright MCP). Tested with v18.19.1 (the default version shipped with Ubuntu 24.04):

```bash
sudo apt install nodejs npm
```

### 1.2 Restore and Fix the Virtual Environment

The virtual environment is stored as a split compressed archive under `env/`. No additional `pip install` is needed after extraction.

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

**Getting API keys:** Most API keys in this project come from [Alibaba Bailian](https://bailian.console.aliyun.com/). Create an account, apply for an API key, and top up your balance to get started. The exception is `RAGFLOW_API`, which comes from your self-hosted RAGFlow instance (see Section 2.2).

Copy the backup templates and fill in your keys:

```bash
cp config.yaml.backup config.yaml
cp Company/Defense/config/config.yaml.backup Company/Defense/config/config.yaml
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

**`Company/Defense/config/config.yaml`** — LLM used by the defense module (DescriptionNormalizer):
```yaml
llm:
  api_key: <ali-bailian-api-key>
  model: "qwen3-max"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  temperature: 0.2
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

**`playwright_config.json`** — Update `outputDir` to the absolute path of your project root:
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

**`test/AttackTest/ExecutionTest.py` and `test/AttackTest/PlannerTest.py`** — Replace the API key placeholders directly in the source files:

| Placeholder | Source |
|---|---|
| `<ali_bailian_api_key>` | Alibaba Bailian |
| `<deepseek_api_key>` | Alibaba Bailian (recommended — DeepSeek models are available on Bailian; alternatively use [DeepSeek Platform](https://platform.deepseek.com/) directly) |
| `<gpt_api_key>` | [OpenAI Platform](https://platform.openai.com/) |

---

## 2. Docker Services

### 2.1 Core Services (PostgreSQL, etc.)

Create the shared Docker network first (only needed once), then start the services:

```bash
docker network create mas
docker compose -f Docker/docker-compose.yml up -d
```

This starts three containers: a PostgreSQL database, an Adminer web UI (port 61080), and a Postgres MCP server (port 61001).

### 2.2 RAGFlow

Clone the RAGFlow repository and edit its `docker-compose.yml` to join the `mas` network:

```bash
git clone https://github.com/infiniflow/ragflow.git
cd ragflow
```

In `docker-compose.yml`, add the `mas` network as external and attach it to each service:

```yaml
# At the top level:
networks:
  mas:
    external: true

# Under each service:
services:
  <service_name>:
    networks:
      - mas
      # ... other existing networks
```

Then start RAGFlow:

```bash
docker compose -f docker-compose.yml up -d
```

Once running, open the RAGFlow web UI at `http://localhost:80`, create an account, and generate an API key. Fill it in as `RAGFLOW_API` in `.env`. RAGFlow's API service runs on port 9380.

---

## 3. Playwright Setup

Playwright is required for the web browsing capability of the MAS execution agent. This step is only needed when running full MAS execution (Section 4.1).

### 3.1 Start the Playwright MCP Server

The MCP server must be running before executing MAS. Keep it running in a separate terminal throughout the experiment. The browser will be downloaded automatically on first run.

Run from the `tmp/browser_wp/` directory (create it first if it doesn't exist):

```bash
mkdir -p tmp/browser_wp
cd tmp/browser_wp
npx --yes @playwright/mcp@0.0.68 --config ../../playwright_config.json
```

---

## 4. Running Experiments

The GAIA benchmark dataset is included in `Benchmark/GAIA/`. It can also be downloaded from the [official GAIA repository](https://huggingface.co/datasets/gaia-benchmark/GAIA).

The agent descriptions used in all experiments are stored in `scripts/tested_descs.json`. Each key is a variant name and the value is the description string injected into the PostgresManager agent.

To read a description for use in a command:
```bash
DESC=$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['baseline'])")
```

To limit execution to a subset of tasks (e.g. for testing), edit `utils/task_filter.py`:
```python
TASK_IDS: list[str] | None = ["task-id-1", ...]  # set to None to run all
```

### 4.1 Run MAS Execution

Requires Playwright MCP server and Docker services to be running. Results are saved to `tmp/logs/<content_dir>/`.

```bash
python test/AttackTest/ExecutionTest.py \
    --provided_desc "<desc>" \
    --content_dir "<variant_name>" \
    [--use_defense]
```

The following commands reproduce all 13 execution variants:

```bash
# Main experiments (no defense)
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['baseline'])")" --content_dir baseline
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['over_fragmentation'])")" --content_dir over_fragmentation
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['under_decomposition'])")" --content_dir under_decomposition
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['dependency_disruption'])")" --content_dir dependency_disruption
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['over_assignment'])")" --content_dir over_assignment
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['agent_exclusion'])")" --content_dir agent_exclusion
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['intermediate_output_suppression'])")" --content_dir intermediate_output_suppression
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['overworking'])")" --content_dir overworking
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['planner_prior_misguidance'])")" --content_dir planner_prior_misguidance

# With defense
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['baseline'])")" --content_dir baseline_with_defense --use_defense
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['over_fragmentation'])")" --content_dir over_fragmentation_with_defense --use_defense
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['agent_exclusion'])")" --content_dir agent_exclusion_with_defense --use_defense
python test/AttackTest/ExecutionTest.py --provided_desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['overworking'])")" --content_dir overworking_with_defense --use_defense
```

### 4.2 Fetch Plans Only

Does not require Playwright or Docker. Results are saved to `tmp/plans/<name>.json`. Supports checkpoint resume.

```bash
python scripts/fetch_plans.py \
    --name <variant_name> \
    --desc "<desc>" \
    [--planner <planner_key>] \
    [--use_defense] \
    [--replace_worker_descs] \
    [--overwrite]
```

- `--planner`: `deepseek_ali` (default), `deepseek`, `gpt`, `gpt_mini`, `kimi`, `qwen`
- `--replace_worker_descs`: Replace non-DB worker descriptions with real-world ones from `scripts/agent_descriptions_replace.json`

**Group 1 — Main experiments** (same 13 variants as execution, default planner):

```bash
# No defense (9 variants) — same descriptions as execution commands above, e.g.:
python scripts/fetch_plans.py --name baseline --desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['baseline'])")"
python scripts/fetch_plans.py --name over_fragmentation --desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['over_fragmentation'])")"
# ... (repeat for all 9 attack variants)

# With defense (4 variants):
python scripts/fetch_plans.py --name baseline_with_defense --desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['baseline'])")" --use_defense
python scripts/fetch_plans.py --name over_fragmentation_with_defense --desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['over_fragmentation'])")" --use_defense
python scripts/fetch_plans.py --name agent_exclusion_with_defense --desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['agent_exclusion'])")" --use_defense
python scripts/fetch_plans.py --name overworking_with_defense --desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['overworking'])")" --use_defense
```

**Group 2 — LLM comparison** (baseline, over_fragmentation, agent_exclusion, overworking × planners gpt/gpt_mini/kimi/qwen, 16 variants total):

Output names follow the pattern `<attack>_<planner_suffix>` where `gpt` → `gpt_5`, `gpt_mini` → `gpt_5_mini`. Example:

```bash
python scripts/fetch_plans.py --name baseline_gpt_5 \
    --desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['baseline'])")" \
    --planner gpt
```

**Group 3 — Real agent descriptions** (same 4 attacks, with and without defense, 8 variants total). Uses `--replace_worker_descs` to substitute non-DB agent descriptions with real-world ones. Example:

```bash
python scripts/fetch_plans.py --name baseline_replace_desc \
    --desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['baseline'])")" \
    --replace_worker_descs

python scripts/fetch_plans.py --name baseline_replace_desc_with_defense \
    --desc "$(python3 -c "import json; print(json.load(open('scripts/tested_descs.json'))['baseline'])")" \
    --replace_worker_descs --use_defense
```

---

## 5. Computing Metrics

### 5.1 From Pre-computed Data (Recommended)

The release includes pre-computed checkpoints under `tmp/export/table_data/`. Copy them to skip all LLM-based recomputation:

```bash
cp -r tmp/export/table_data tmp/table_data
```

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
