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
# Uses the canonical script from the letterboxd-widget package
sync:
	pip install -q feedparser pyyaml
	curl -fsSL https://raw.githubusercontent.com/yash-srivastava19/letterboxd-widget/master/jekyll/fetch_letterboxd.py | python
