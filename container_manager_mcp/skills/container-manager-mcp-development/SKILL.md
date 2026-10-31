---
name: container-manager-mcp-development
skill_type: skill
description: >-
  Develop, test, and release container-manager-mcp changes safely. Use this skill for
  connector code, specs, workflow wiring, packaged skills, release metadata, and the
  focused gates that prove the Docker, Podman, Swarm, and Kubernetes MCP surfaces.
license: MIT
tags: [container-manager-mcp, development, connectors, mcp, specs, gates]
metadata:
  author: Repository Maintainers
  version: '0.1.0'
---
# container-manager-mcp development

This skill is the repository contract for the container-manager-mcp connector. The
package owns the Docker, Podman, Swarm, and Kubernetes calls and their MCP content;
transport, release rendering, generated ontology artifacts, and shared validation
come from `scion-sdk` and `pipelines`.

## Repository workflow

1. Start from a real worktree based on `origin/main`; keep the canonical checkout and
   other worktrees untouched.
2. Find the owning row with `git grep -n "<term>" -- specs` before changing behavior.
3. Keep changes in the connector boundary. Use the SDK for MCP construction, governed
   HTTP, credentials, TLS, and packaged content discovery.
4. Run the narrow test or gate that proves the changed behavior, then run
   `uvx pre-commit run --all-files` before opening a PR.
5. Let the shared pipelines render Docker/compose files and generate connector
   ontology artifacts during release. Those generated files are not committed here.

## Container-manager checks

Use the locked project environment (`uv sync --locked`) when it is already available.
Focused tests can select one backend or tool family, for example:

```bash
uv run --frozen pytest -q tests/test_call_context.py
uv run --frozen pytest -q tests/test_mcp_server.py
```

Do not call a real container engine or Kubernetes cluster from a unit test. Mock the
manager boundary and prove context selection, refusal, and serialization. Keep
credentials, hostnames, kubeconfigs, and private paths out of fixtures and receipts.

## Rapid delivery: the 5-minute contract

Specs are already designed. The work is implementation, delivered in very small, continuously landed slices.

**Extend first (anti-sprawl)**
- Before any code change, find the spec row that owns the behavior: `git grep -n "<term>" -- specs`. Cite its ID in the commit `Spec:` trailer.
- Before any new spec text, extend the owning spec: a child row (`R012.4`), a new rollup at the end of that spec, or a dated entry in the `plan.md` section "Amendments". Create a new spec only for a capability that no spec owns, and record the search in "Why a new spec".
- Audits, reviews, gap analyses and carry-overs never get a spec directory. Their findings land in the spec that owns the row; a finding against another repository lands in that repository's spec.
- Before any new function, module, planner, selector or store, search for the existing one and reuse it. A second implementation of an owned capability is a defect, not a delivery.
- The rules are in `specs/README.md`. The `spec-standard` gate enforces the names, file set, title and index.

**Time and size**
- One agent delivers one PR in 5 minutes or less. No agent or sub-agent runs longer than 5 minutes; an orchestrator dispatches the next slice to a fresh agent.
- Self-timer (hard): run `date +%s > /tmp/agent-start-$$` as your FIRST command, and before every later step check elapsed with `echo $(( $(date +%s) - $(cat /tmp/agent-start-*) ))`. At 300 s, stop starting new work. At 480 s, commit what passes, push it, open the PR, report and STOP. Never exceed 600 s. Never wait on CI or background jobs; use `git commit --no-verify` and `git push --no-verify` after running pre-commit on the changed files yourself.
- CI turns around in 5 minutes or less. PR CI runs the fast subset: lint, type check, the spec check, and only the tests and crates the diff touches. Pushes to main run the full suite.
- CI timer (hard): every PR-path job carries `timeout-minutes: ${{ github.event_name == 'pull_request' && 10 || 60 }}`, so a PR gate is killed at 10 minutes and a slow gate shows up as a failure instead of a silent queue.
- Never spawn sub-agents from a delivery agent.

**Investigation budget**
- At most 2 minutes and about 10 tool calls of reading before the first edit, and at most about 25 tool calls per PR.
- If the change is not clear by then, ship the `.1` slice (typed model plus refusal test) or skip the row with a one-line note. Do not write deferral essays.
- Trust the orchestrator's evidence and ID list; do not re-verify it.

**Sizing (deterministic)**
- Split a requirement when its size score is above 6, it names more than 2 code roots, or it is cross-repo. Children are `<ID>.<n>`, producer first.
- Net-new work is sliced, never skipped: `.1` typed model plus validation and refusal tests, `.2` the one entry point that uses it, `.3`+ each further behavior.
- Cross-repo moves split into one child per repo: the destination copies the behavior first, then the source switches importers and deletes.

**Specs update themselves (land VERIFIED, not just LANDED)**
- Every commit that delivers a requirement ends with the trailer `Spec: <ID>[, <ID>...]`. A commit that touches code without delivering one says `Spec: none (ci|deps|refactor|fix-forward)`.
- Every delivered ID gets at least one test bound to it in the same PR. Python: `@pytest.mark.spec("<ID>")`. Rust/TypeScript: a `// spec: <ID>` line directly above the test.
- Never hand-edit `status.json`; it is generated (`spec-status`). Never write placeholder SHAs. If the work is only part of a requirement, decompose it into `<ID>.<n>` children first and deliver whole children.
- Write the marker with the literal ID, never a variable. Put `Spec:` and `Co-Authored-By:` on consecutive lines in the last paragraph of the commit message, or `Spec:` is not read as a trailer.
- Generated status depends on the commit: run `pipelines-hook spec-status --write` again after committing and commit the result.
- The generator applies the rules. Run `pipelines-hook spec-status` to see each row's state.

**Delivered means the row's acceptance is met**
- Never delete or stub out behaviour whose replacement is not wired. Never make a caller "fail closed" in place of repointing it. Never substitute a scope, weaken an assertion or add an allowlist entry to pass a gate.
- If the replacement lives in another repository, do not deliver: record the blocker in the row, name the missing capability, and stop. The orchestrator adds the row in the owning repository.
- Never re-add a path that another requirement retired; report the conflict.
- Valid outcomes are DELIVERED (with a bound test), SPLIT, BLOCKED and GAP (a "landed" row that is not really delivered). A half-delivery is not an outcome.

**Where progress shows**
- The repository's spec-delivery page is rebuilt from `specs/*/status.json` on each push to main. SPECIFIED is written, LANDED is on main, VERIFIED is on main with a bound test.

**Before every push**
- Work in a real `git worktree add` from `origin/main`. Never edit a shared checkout, never `git stash`, never `git add -A`, never force-push.
- Run the repo's own pre-commit on the changed files: `uvx --from pre-commit==4.6.0 pre-commit run --files $(git diff --name-only origin/main...HEAD)`. Fix every failure. No `noqa`, `type: ignore`, skip, xfail or whitelist entries to pass a gate.
- Run the targeted tests only, never a full suite. Run the type checker on every changed file, and keep each function at cyclomatic complexity 10 or less.

**Landing**
- Every PR lands within minutes of going green; no PR sits idle. Mechanical conflicts (generated files, Markdown, `status.json`) are resolved automatically by the merge-train tool. Real conflicts are additive in most cases and are resolved, not deferred.
- Many open PRs land together as a merge train: one integration branch and one CI run.
- Full CI on main catches what the fast subset missed; regressions are fixed forward immediately.
- Landing in this repo: open one reviewable PR; the orchestrator merges only after independent review and green hosted checks.

**Report**: at most 8 lines: PR URL, requirement IDs, test result, and any `ID:<main sha>` proof for rows already on main.

## Release boundary

The package version lives only in `pyproject.toml`. The central Python workflow is
called with `connector: true`; the central container workflow is called with
`render_connector: true`. A package-content change requires a normal patch version
bump and a matching `uv.lock` update before publication. Existing 3.3.6 runtime
images are deployment evidence and are never rewritten by this skill.
