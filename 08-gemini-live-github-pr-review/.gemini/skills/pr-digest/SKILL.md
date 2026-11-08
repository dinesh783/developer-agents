---
name: pr-digest
description: Summarize an already-reviewed PR using only findings and the
  published review from the saved session. Use for digest, recap, or review
  summary requests.
---

Write a digest for someone who will not open the PR.

1. Use only the saved `review_evidence` for this PR: findings successfully
   submitted with `submit_finding` and the result from `publish_review`. Do not
   fetch the PR again or run another review.
2. Write 3-5 plain-English sentences. Lead with the verdict, then explain only
   the one or two things that matter most. Do not produce a list of findings.
3. Never invent findings. If there are none, say so plainly.
4. End with exactly one line counting findings by severity, for example:
   `1 blocking, 0 warning, 2 nit.`
5. If there is no saved review evidence, state that a digest cannot be grounded
   in a completed review rather than guessing.
