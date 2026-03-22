## Source Files for yash-sri.xyz

Personal blog, diary, and project notes. Built with Jekyll + GitHub Pages using the `aterenin/minima-reboot` theme.

---

### Running locally

**One-time setup:**
```bash
bundle install
```

**Pull your Letterboxd data** (needed to preview the /films page):
```bash
make sync
```

**Start the dev server** (live-reloads on file save):
```bash
make serve
# → http://localhost:4000
```

Or chain them:
```bash
make sync && make serve
```

**All make targets:**

| Command | What it does |
|---------|-------------|
| `make install` | Install Ruby gems (run once) |
| `make sync` | Fetch latest Letterboxd diary into `_data/letterboxd.yml` |
| `make serve` | Start local server at http://localhost:4000 with live reload |
| `make build` | One-off build into `_site/` |

---

### Films page / Letterboxd sync

Powered by [`letterboxd-widget`](https://github.com/yash-srivastava19/letterboxd-widget) — a drop-in widget that works on Jekyll, PHP, or plain HTML sites.

The GitHub Actions workflow (`.github/workflows/sync-letterboxd.yml`) runs daily at 9am UTC and auto-commits updated film data. You can also trigger it manually from the Actions tab.
