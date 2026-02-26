# Audit: /itemized-commits — Last Invocation vs Spec

## What the command requires vs what was done

| Step | Spec requirement | What happened last time | Verdict |
|------|------------------|-------------------------|---------|
| **1. Gather state** | Run `git status`, `git diff`, `git diff --cached`, `git log --oneline -5` **in parallel** | Ran `git status` first, then separate diffs per file; did not run all four in one parallel batch | ❌ Not followed |
| **2. Classify** | Group by logical units using the priority table | Grouping was reasonable (fix, docs, chore, feat) | ✅ OK |
| **3. Plan and present** | **Present numbered plan → WAIT for user confirmation** (unless "go ahead" / "no confirmation needed") | No plan was shown; commits were executed immediately without waiting | ❌ **Critical** — user had no chance to approve or adjust |
| **4. Execute** | Per group: (1) `git reset HEAD` if needed, (2) stage only that group, (3) commit with HEREDOC body, (4) `git status` after each | No explicit `git reset HEAD` before first group; commits used `-m "subject"` plus body in one string, not HEREDOC; unclear if status was checked after each | ⚠️ Partially followed |
| **4. Ordering** | Prefer: 1) infra/config, 2) core lib, 3) analysis/viz, 4) test, 5) notebooks, 6) docs | Order was: fix(analysis) → **docs (CHANGELOG)** → chore(cursor) → feat(notebooks). Docs were committed in the middle; config (cursor) came after docs; documentation was not last | ❌ Ordering violated |
| **5. CHANGELOG / LESSONS_LEARNT** | **After all code commits**: add CHANGELOG/LESSONS_LEARNT entries, then **one** final commit: `docs: update CHANGELOG and LESSONS_LEARNT` | CHANGELOG was committed as its own commit (commit 2) in the middle of the sequence, not as a single documentation commit at the end | ❌ **Critical** — CHANGELOG is not a mid-sequence commit; it must be the final docs commit |
| **6. Summary** | Print table with columns: #, Hash, Message, Files, Type | A summary was printed but not in the exact table format (Files and Type columns missing) | ⚠️ Partially followed |

## Root causes

1. **No mandatory stop before Step 4** — The "wait for user confirmation" was not enforced; the agent proceeded to commit without showing the plan or pausing.
2. **CHANGELOG placement** — The spec says "After all code commits... Commit documentation updates as a **separate final commit**". CHANGELOG was treated as one of the logical groups instead of a single final docs commit.
3. **Ordering** — Documentation and config order was not aligned with "Infrastructure first, documentation last".
4. **Step 1 parallel** — The four git commands were not run in one parallel batch.

## Correct flow (for next run)

1. **Step 1**: Run `git status`, `git diff`, `git diff --cached`, `git log --oneline -5` in parallel.
2. **Step 2**: Classify into logical groups (no docs in code groups; config separate; notebooks separate).
3. **Step 3**: Output the numbered plan with "Proposed commits (in order)" and file lists, then **STOP and ask**: e.g. "Confirm to proceed with these commits (or say which to change)."
4. **Step 4**: Only after confirmation, execute in **ordering preference** order: config → core lib → analysis/viz → test → notebooks. Do **not** commit CHANGELOG or LESSONS_LEARNT here.
5. **Step 5**: Add/update CHANGELOG and LESSONS_LEARNT; then create **one** commit: `docs: update CHANGELOG and LESSONS_LEARNT`.
6. **Step 6**: Print the full summary table with #, Hash, Message, Files, Type.

## Suggested command file changes

- Make Step 3 explicitly say: "STOP. Do not run Step 4 until the user confirms."
- Make Step 5 explicitly say: "Do NOT commit CHANGELOG or LESSONS_LEARNT as part of the logical groups. Only one docs commit at the end."
- Add a short "Common mistakes" bullet list (e.g. "Do not commit CHANGELOG in the middle; do not skip the confirmation step").
