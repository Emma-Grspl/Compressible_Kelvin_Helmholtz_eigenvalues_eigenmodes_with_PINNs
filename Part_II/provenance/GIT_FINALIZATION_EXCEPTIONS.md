# Git finalization exceptions

Audit date: 2026-08-28.

## Large historical objects

The staged tree reuses 21 local-history blobs above 100 MiB and three further
local-history blobs between 50 and 100 MiB. None is a newly created staged
blob. The 21 large blobs are not reachable from the currently fetched `origin`
refs, so publication must be confirmed by the actual non-force push.

Per the migration policy, historical blobs are not converted to Git LFS merely
because of their size and are not deleted locally. The two genuinely new modal
CSV blobs were handled separately: they remain local-only and are excluded by
exact `.gitignore` rules documented in `provenance/LARGE_FILE_POLICY.md`.

`NEW STAGED BLOBS >=100 MiB = 0`.

## Whitespace check

`git diff --cached --check` reports pre-existing CRLF/trailing whitespace in
114 migrated data, SVG, log, provenance, and legacy files. They are not
normalized because doing so would change archived/scientific outputs solely to
silence Git's whitespace heuristic.

This exception does not concern executable Python or active launcher syntax,
which passed their dedicated checks.
