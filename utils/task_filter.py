# Fill this list to run only specific tasks; leave as None to run all.
TASK_IDS: list[str] | None = None

# TASK_IDS: list[str] | None = [
#     "e1fc63a2-da7a-432f-be78-7c4a95598703", # web + python
#     "a1e91b78-d3d8-4675-bb8d-62741b4b68a6", # web + media
#     "cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb", # document + reasoner
#     "27d5d136-8563-469e-92bf-fd103c28b57c", # reasoner
#     "b415aba4-4b68-4fc6-9b89-2c812e55a3e1", # web + document
# ]


def filter_task(qid: str) -> bool:
    """Return True if the task should be run."""
    if TASK_IDS is None:
        return True
    return qid in TASK_IDS
