# BFCL v3 (subset)

Test cases from the Berkeley Function Calling Leaderboard, v3:
https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
(Apache License 2.0, © the Gorilla LLM authors). Downloaded 2026-10-02.

`<category>.json` holds the questions and function definitions and
`<category>_answers.json` the accepted answers (`possible_answer/` upstream).
`irrelevance` has no answers file: the correct behavior is to call no tool.
