---
name: changelog-entry
description: Draft one user-facing Keep a Changelog entry from a PR digest
  already present in the current saved conversation.
---

Draft exactly one Keep a Changelog-style entry for end users.

1. Use only the PR digest already produced in the current request or saved
   conversation. Do not re-read or re-review the PR.
2. Choose one category heading: Added, Changed, Fixed, Removed, or Security.
   Choose based on the user-visible change, not the review severity.
3. Write one bullet with one plain-language sentence. Do not include file paths,
   severity labels, or internal review terminology.
4. If the digest does not establish a user-visible change, say that no
   user-facing changelog entry can be written from the available information;
   do not invent impact.
