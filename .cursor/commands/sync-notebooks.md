# Sync Notebooks

Sync Jupytext-paired Python scripts in `notebooks/` to their `.ipynb` counterparts. The `.py` scripts are the source of truth — this pushes changes **one-way** from `.py` → `.ipynb`.

## Default: Sync Only the Notebook in Context

**Prefer syncing a single notebook** — the one that is in context for this chat:

1. **Identify the target notebook** using (in order of preference):
   - The file currently open in the editor (if it’s in `notebooks/` and is a `.py` or `.ipynb`)
   - A notebook explicitly mentioned or edited in this chat
   - Recently viewed files in this session that are under `notebooks/`

2. **Resolve to the `.py` script**:
   - If the context file is `notebooks/XX_name.ipynb`, sync `notebooks/XX_name.py`
   - If the context file is `notebooks/XX_name.py`, sync that file

3. **Sync only that script** (activate conda first):

```bash
conda activate pyrrm && jupytext --to notebook --update notebooks/XX_name.py
```

4. **If no single notebook can be determined**, ask the user which notebook to sync (e.g. “Which notebook should I sync? You can give the name or number, e.g. 06_algorithm_comparison.”).

**If the user explicitly asks to “sync all notebooks” or “sync every notebook”**, use the full sync procedure below instead.

---

## Full Sync (All Notebooks)

Use only when the user explicitly requests syncing **all** notebooks.

### Step 1: Discover Scripts

Find all `.py` files in `notebooks/` that use the Jupytext percent format (contain `# %%` cell delimiters).

### Step 2: Sync All Notebooks

**CRITICAL**: Always activate the conda environment first.

```bash
conda activate pyrrm && jupytext --to notebook --update notebooks/*.py
```

If the glob approach fails (e.g., too many arguments), fall back to syncing each file individually in a loop:

```bash
conda activate pyrrm && for f in notebooks/*.py; do jupytext --to notebook --update "$f"; done
```

### Step 3: Verify

After syncing, list the notebooks and confirm each `.py` has a matching `.ipynb`:

```bash
ls -1 notebooks/*.py notebooks/*.ipynb | sort
```

Report any `.py` scripts that are missing a paired `.ipynb` (these may be new notebooks that need `jupytext --to notebook` without `--update`).

### Step 4: Report

Print a summary (for single sync: “Synced 1 notebook: …”. For full sync: list all).

## Important Rules

- **NEVER use `jupytext --sync`** — it is bidirectional and timestamp-based, which can overwrite `.py` edits
- **ALWAYS use `--to notebook`** direction (`.py` → `.ipynb`)
- **Use `--update`** to preserve cached cell outputs unless the user explicitly asks to clear outputs
- **Always activate `conda activate pyrrm`** before running jupytext
