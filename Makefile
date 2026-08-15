SPHINXBUILD ?= sphinx-build
PYTHON ?= python3
SOURCEDIR ?= docs
BUILDDIR ?= docs/_build
HTMLDIR ?= $(BUILDDIR)/html
SPHINXOPTS ?=

.PHONY: help docs-check html strict spelling linkcheck api-smoke api-reference rtd clean

help:
	@echo "html        Build the Sphinx site"
	@echo "docs-check  Validate generated references, navigation, CLI, style, and examples"
	@echo "strict      Rebuild with warnings and missing references as errors"
	@echo "spelling    Check documentation spelling"
	@echo "linkcheck   Check external links"
	@echo "api-smoke   Import documented APIs and inspect the CLI tree"
	@echo "api-reference Check that generated PublicAPI pages match source annotations"
	@echo "rtd         Build into Read the Docs' requested output directory"
	@echo "clean       Remove only Sphinx-generated documentation output"

# The dirhtml builder keeps directory-style URLs (``quickstart/``) identical
# to the previous MkDocs site so transitional GitHub Pages links stay valid.
html:
	$(SPHINXBUILD) -b dirhtml "$(SOURCEDIR)" "$(HTMLDIR)" \
		-j auto $(SPHINXOPTS)

docs-check:
	$(PYTHON) tools/check_docs.py --static-only

strict:
	$(SPHINXBUILD) -b dirhtml -a -E -n -W --keep-going -j auto \
		"$(SOURCEDIR)" "$(HTMLDIR)" $(SPHINXOPTS)

spelling:
	TRIBUTO_DOCS_SKIP_INTERSPHINX=1 $(SPHINXBUILD) \
		-b spelling -a -E -W --keep-going \
		"$(SOURCEDIR)" "$(BUILDDIR)/spelling" $(SPHINXOPTS)

# External sites are outside the repository's control. This target still
# returns a failure for broken links; CI runs it in a separately visible,
# non-blocking job so all failures are collected without weakening strict.
linkcheck:
	$(SPHINXBUILD) -b linkcheck -a -E --keep-going \
		"$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

api-smoke:
	$(PYTHON) tools/check_docs.py

api-reference:
	$(PYTHON) tools/generate_public_api_reference.py --check

rtd: docs-check
	@test -n "$(HTMLDIR)" || \
		{ echo "HTMLDIR must point to the Read the Docs output directory"; exit 2; }
	$(SPHINXBUILD) -b dirhtml -a -E -n -W --keep-going -j auto \
		"$(SOURCEDIR)" "$(HTMLDIR)" $(SPHINXOPTS)

clean:
	$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"
