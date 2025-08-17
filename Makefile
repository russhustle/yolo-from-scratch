clean:
	find . -name "__pycache__" -exec rm -rf {} \;
	find . -name ".DS_Store" -exec rm -rf {} \;

pre-commit:
	uv run pre-commit autoupdate
	uv run pre-commit install
	uv run pre-commit run --all-files