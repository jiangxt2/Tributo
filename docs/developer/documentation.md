# Contribute documentation

Tributo follows the organization and writing conventions of the Ray project
while keeping Tributo's component boundaries and maturity levels explicit.

## Organize by component and task

Each first-class component provides a landing page, a shortest successful path,
key concepts, task guides, examples or troubleshooting, and an API entry. Do
not make users understand the source directory layout before choosing a task.

Keep architecture decisions and migration history outside the user path. A
landing page describes implemented behavior and links to the support matrix for
verification boundaries.

## Write in Ray style

- Use active voice, present tense, second person, and imperative task steps.
- Use sentence case for headings and omit numeric heading prefixes.
- Use the full names Ray Data, Ray Train, Ray Tune, Ray Serve, and Ray Jobs.
- Avoid time-relative words such as `currently`, `recently`, `new`, and `now`.
- Avoid em dashes, excessive parentheses, semicolons, and marketing claims.
- Use MyST admonitions and angle brackets for placeholders.
- Declare a language on every code fence.

The repository style gate enforces objective heading rules. Spelling and strict
Sphinx builds cover the remaining automated rules. Review prose for
voice, tense, precision, and unsupported claims.

See the upstream [Ray documentation guide](https://docs.ray.io/en/latest/ray-contribute/docs.html),
[writing style guide](https://docs.ray.io/en/latest/ray-contribute/writing-style.html),
[API policy](https://docs.ray.io/en/latest/ray-contribute/api-policy.html), and
[code snippet guide](https://docs.ray.io/en/latest/ray-contribute/writing-code-snippets.html).

## Document every public API

Every top-level object annotated Stable, Beta, or Alpha must appear in the API
reference. Do not hand-edit generated component pages.

```bash
python tools/generate_public_api_reference.py
python tools/generate_public_api_reference.py --check
```

The generator reads source annotations with the Python AST, routes objects to a
component, and emits explicit autodoc directives. `tools/check_docs.py` rejects
missing, duplicated, unexpected, stale, or runtime-stability-mismatched
targets.

Deprecated behavior needs a replacement and the runtime removal facts that
actually exist. If a warning conflicts with a stability annotation, document
the conflict and request a separate API decision. Do not change stability in a
documentation-only contribution.

## Make examples verifiable

Store runnable examples in `docs/examples/doc_code/` and include them with
`literalinclude`. Add a corresponding test that executes local deterministic
behavior. For external infrastructure, state the prerequisite and identify the
existing integration Gate. Do not hide executable code in an untested copied
fence.

## Run the documentation gates

```bash
python tools/check_docs.py --static-only
make strict SPHINXBUILD=.docs-venv/bin/sphinx-build
make spelling SPHINXBUILD=.docs-venv/bin/sphinx-build
SPHINX_REAL_IMPORTS=1 make strict \
  SPHINXBUILD=.venv/bin/sphinx-build \
  BUILDDIR=docs/_build-real \
  HTMLDIR=docs/_build-real/html
python tools/generate_public_api_reference.py --check
```

The lightweight Read the Docs build mocks third-party packages only. The
real-import CI build installs the documented optional import profiles and
imports every generated target.
