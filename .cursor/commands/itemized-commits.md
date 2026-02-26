# Itemized Commits

Analyse all uncommitted changes in the working tree and create separate, focused commits for each logical unit of work. Follow this procedure exactly.

## Step 1: Gather State

Run these commands **in parallel**:

1. `git status` — identify all modified, staged, and untracked files
2. `git diff` — see unstaged changes
3. `git diff --cached` — see staged changes
4. `git log --oneline -5` — recent commit style reference

## Step 2: Classify Changes into Logical Groups

Read every changed file's diff and group them into **logical units** — each group becomes one commit. Grouping criteria (in priority order):

| Priority | Grouping Rule | Example |
|----------|--------------|---------|
| 1 | Same feature across files | New model + its tests + `__init__.py` export |
| 2 | Same module, same concern | Two fixes in `calibration/` |
| 3 | Single file, single concern | Docstring update in one file |
| 4 | Cross-cutting infrastructure | `pyproject.toml` + dependency changes |
| 5 | Documentation-only | README, CHANGELOG, LESSONS_LEARNT |

**Rules:**
- Never mix feature code with documentation-only changes
- Never mix unrelated modules in one commit
- Keep notebook `.py` scripts separate from library code unless tightly coupled
- Configuration files (`pyproject.toml`, `.cursor/rules/`) get their own commit unless they are part of a feature
- Exclude `__pycache__/`, `.pyc`, `.ipynb_checkpoints/` — never commit these

## Step 3: Plan and Present

Present a numbered plan showing:

```
Proposed commits (in order):

1. type(scope): subject
   Files: file1.py, file2.py

2. type(scope): subject
   Files: file3.py
```

Use the project's conventional commit format: `type(scope): subject`

Types: feat, fix, docs, refactor, test, perf, style, chore, ci
Scopes: models, calibration, routing, analysis, visualization, data, notebooks, benchmark, examples

**STOP. Do not run Step 4 until the user confirms.** Ask: "Confirm to proceed with these commits (or say which to change)." Exception: if the user already said "go ahead", "no confirmation needed", or "just do it", proceed to Step 4 without waiting.

## Step 4: Execute Commits Sequentially

**Do not include CHANGELOG.md or LESSONS_LEARNT.md in any of these groups** — they are committed only in Step 5.

For each group, in **ordering preference** order (see Ordering Preference below): config → core lib → analysis/viz → test → notebooks. Within that, dependency order (foundations first).

1. **Reset staging area**: `git reset HEAD` (so only the next `git add` is staged)
2. **Stage only the group's files**: `git add <file1> <file2> ...`
3. **Commit** with a well-formed message using HEREDOC:

```bash
git commit -m "$(cat <<'EOF'
type(scope): subject line (≤50 chars)

Body explaining what and why (wrap at 72 chars).
EOF
)"
```

4. **Verify**: `git status` after each commit to confirm clean staging

## Step 5: Update CHANGELOG and LESSONS_LEARNT (one final commit only)

**Do not commit CHANGELOG or LESSONS_LEARNT in Step 4.** Only after all code/config/notebook commits are done:

- If any commit was `feat`, `fix`, `perf`, or breaking `refactor`, add entries to `CHANGELOG.md` under `[Unreleased]`
- If any lesson-worthy insight was encountered, add to `LESSONS_LEARNT.md`
- Stage only these docs: `git add CHANGELOG.md LESSONS_LEARNT.md` (omit if unchanged)
- Create **one** final commit: `docs: update CHANGELOG and LESSONS_LEARNT` (or `docs: update CHANGELOG` / `docs: update LESSONS_LEARNT` if only one changed)

## Step 6: Summary

Print a summary table with exactly these columns: **#**, **Hash** (short), **Message** (subject line), **Files** (count), **Type**:

```
| # | Hash    | Message                        | Files | Type |
|---|---------|--------------------------------|-------|------|
| 1 | abc1234 | type(scope): subject           |     3 | feat |
| 2 | def5678 | type(scope): subject           |     1 | fix  |
```

## Ordering Preference

Commit in this order when possible:

1. Infrastructure / config changes
2. Core library changes (models, calibration, data)
3. Analysis / visualization changes
4. Test additions or updates
5. Notebook updates
6. Documentation (README, CHANGELOG, LESSONS_LEARNT, rules)

## Common mistakes to avoid

- **Do not skip Step 3 confirmation** — present the plan and wait unless the user already said to proceed.
- **Do not commit CHANGELOG/LESSONS_LEARNT in the middle** — they are one final docs commit only (Step 5).
- **Do not mix documentation-only changes** into code commits (group docs separately and commit last).

## Safety

- **NEVER** force push or amend existing remote commits
- **NEVER** commit files containing secrets (`.env`, credentials)
- **NEVER** commit `__pycache__/`, `.pyc`, or `.ipynb_checkpoints/`
- **ALWAYS** verify each commit with `git status`
- If a commit fails (e.g., pre-commit hook), fix the issue and create a **new** commit
