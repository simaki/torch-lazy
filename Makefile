.PHONY: install
install:
	@poetry install

.PHONY: test
test:
	@pytest --doctest-modules torch_lazy
	@pytest --doctest-modules tests

.PHONY: lint
lint:
	@black --check --quiet .
	@isort --check --force-single-line-imports --quiet .

.PHONY: format
format:
	@black --quiet .
	@isort --force-single-line-imports --quiet .

.PHONY: publish
publish:
	@gh workflow run publish.yml