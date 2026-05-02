#!/bin/bash
set -e

npx --yes playwright install chrome

npx --yes @playwright/mcp@latest --port 3000 --headless --no-sandbox --allowed-hosts="*"