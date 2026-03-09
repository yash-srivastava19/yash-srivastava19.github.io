.PHONY: serve build sync install

# Start local dev server with live reload
serve:
	bundle exec jekyll serve --livereload

# One-off build (output goes to _site/)
build:
	bundle exec jekyll build

# Install Ruby gems (run once after cloning)
install:
	bundle install

# Fetch fresh Letterboxd data locally before serving
sync:
	pip install -q feedparser pyyaml
	python scripts/fetch_letterboxd.py
