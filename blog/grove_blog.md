
Note: I had Obsidian open next to my terminal for a year. Every time I needed to write something down, I'd leave the terminal, wait for Electron, write two sentences, and come back. One day I got annoyed enough to just build what I wanted. This is that.

# grove: I built Obsidian for the terminal

- **Project Home:** [github.com/yash-srivastava19/grove](https://github.com/yash-srivastava19/grove)
- **Language:** Go, Bubble Tea, Glamour

## I live in the terminal

I have a confession: I'm the kind of person who has `tmux` sessions named after projects, who writes `grep` commands before reaching for a search bar, and who genuinely prefers `vim` over anything that requires a mouse. The terminal is where I live. It's where I think.

So when I started keeping notes in [Obsidian](https://obsidian.md/), something felt off. Don't get me wrong - Obsidian is great. But every time I wanted to jot something down, I'd have to stop what I was doing, switch context entirely, wait for the Electron app to load, and then come back to my terminal. That friction is small in isolation, but over a day of coding it accumulates into something that genuinely kills flow.

My hypothesis was simple: if I live in the terminal anyway, why can't my notes live there too?

**grove** is my answer. A terminal knowledge garden - Obsidian-style note-taking for the CLI, built in Go with [Bubble Tea](https://github.com/charmbracelet/bubbletea).

## Why another note-taking tool?

I know, I know. The world has enough note-taking apps. But hear me out - none of them solve the specific problem I had, which is: *I want to write and read notes without leaving my terminal session*.

Plain text files in `~/notes/` work, but they have no structure. Vim with some plugins gets you part of the way there. But what I wanted was something that:

1. Stays entirely in the terminal, no browser, no Electron
2. Stores notes as plain `.md` files - no lock-in, no proprietary format
3. Has `[[wiki-links]]` so I can build a knowledge graph over time
4. Has a fuzzy search TUI so I can find things fast
5. Has a vault-wide AI search for when I vaguely remember something but can't recall the exact note
6. Has zero-friction capture - `grove add "thought"` and done

The closest thing I found was `zettelkasten` scripts people post on HN, but those are usually Bash + `fzf` and have no TUI worth speaking of. I wanted something that felt complete.

## Architecture - Go, Bubble Tea, and how it fits together

The stack is deliberately simple.

**Go** because: single binary, fast startup, great standard library for file I/O, and I wanted to learn more Go beyond small scripts. If you `go install`, you get one binary. That's it.

**[Bubble Tea](https://github.com/charmbracelet/bubbletea)** for the TUI. Bubble Tea is a Go framework built on the Elm architecture - you have a `Model`, a `Update` function, and a `View` function. Your entire app state lives in the model, events come in through `Update`, and `View` renders the current state to the terminal. It's a genuinely pleasant way to build TUIs because the state management is explicit and there are no surprises.

**[Glamour](https://github.com/charmbracelet/glamour)** for markdown rendering inside the TUI. When you open a note in grove, it renders the markdown with syntax highlighting, proper headings, and styled code blocks - right in the terminal.

**[Lip Gloss](https://github.com/charmbracelet/lipgloss)** for layout and styling. This is the CSS-for-terminals library from Charm - it lets you compose styled components with borders, padding, and color.

The notes themselves live at `~/.local/share/grove/notes/` as plain `.md` files with YAML frontmatter. This is a deliberate choice. If grove disappears tomorrow, your notes are just files. You can open them in vim, push them to git, sync them with rsync - whatever you want. No lock-in.

The internal structure looks like this:

```
internal/
  config/     - config loading, reads pairy's Gemini key
  notes/      - Note struct, frontmatter parsing, Store (CRUD)
  ai/         - Gemini client for vault-wide AI search
  ui/         - Bubble Tea app: list view, note viewer, search, AI panel, help
main.go       - CLI entry: TUI + subcommands (new, today, add, list)
```

The `Store` in `internal/notes/` is the heart of it. It handles creating, reading, updating, and listing notes, parses frontmatter (title, tags, date), and builds the in-memory index that the search uses.

## The glamour OSC hang - a genuinely painful bug

I want to talk about this one because it took me an embarrassingly long time to debug.

When I first integrated Glamour to render markdown, I used `glamour.WithAutoStyle()`. This is Glamour's "detect whether the terminal is dark or light and pick the appropriate style" option. Very convenient in theory.

In practice: it hangs forever inside Bubble Tea's alt-screen mode.

Here's why. `glamour.WithAutoStyle()` uses an OSC terminal query - it sends an escape sequence to the terminal and waits for a response that tells it the background color. The problem is that Bubble Tea, when it starts with `tea.NewProgram()`, takes ownership of stdin to process key events. The OSC response from the terminal never gets read by Glamour - Bubble Tea is sitting on stdin waiting for the next keypress, and the response just disappears into the void.

The fix is straightforward once you understand the cause: detect the terminal theme *before* calling `tea.NewProgram()`. I call the color query, read the response, stash the result in the config, and then hand control to Bubble Tea. From that point on Bubble Tea owns stdin and the Glamour renderer uses the pre-detected theme.

```go
// Detect theme before Bubble Tea takes over stdin
theme := detectTerminalTheme()  // does the OSC query here

// Now start Bubble Tea - it owns stdin from this point
p := tea.NewProgram(
    ui.New(store, theme),
    tea.WithAltScreen(),
    tea.WithMouseCellMotion(),
)
```

The second Glamour-related bug was subtler. When I tried to compose Glamour's rendered output with Lip Gloss styled borders, I was getting garbled output - weird characters showing up, styling bleeding across lines.

The cause: Glamour outputs its own ANSI escape codes for colors and styling. Lip Gloss also outputs ANSI escape codes. When you try to nest them naively - passing Glamour's output into a `lipgloss.NewStyle().Border(...).Render()` call - the two ANSI streams interfere with each other.

The fix was to keep the rendering pipelines separate. The note content area gets rendered purely by Glamour. The chrome around it (borders, status bar, sidebar) gets rendered purely by Lip Gloss. They never touch each other's output. This required restructuring the `View()` function to assemble the final screen in a specific order and concatenate the outputs rather than nesting them.

Debugging terminal rendering issues is a special kind of pain because your debugging tool (the terminal) is also the broken thing.

## Features walkthrough

### Zero-friction capture

```bash
grove add "look into B+ tree index scan costs for the postgres query"
```

This appends a timestamped bullet to today's daily note, creating it if it doesn't exist. I use this constantly. It's the thing I miss most when I'm not using grove.

### Note templates

```bash
grove new --template meeting
grove new --template brainstorm
grove new --template research
```

Each template pre-fills the frontmatter and a basic structure. The meeting template gives you a Participants / Agenda / Notes / Action Items layout. The research template gives you a hypothesis block. Small thing, but it means I actually use consistent structure rather than starting from a blank file every time.

### Wiki-links

Inside any note, `[[another-note]]` creates a link to another note in the vault. The TUI renders these as clickable links - press Enter on one and it opens the linked note. The Store builds a bidirectional link index on startup, so you can also see which notes link to any given note (backlinks), which is the feature that makes a knowledge garden actually useful over time.

### Vault-wide AI search

```bash
grove search "what did I write about the OSC terminal query bug"
```

This sends the query to Gemini along with the full text of all your notes and returns the relevant passages with note titles. Useful when you vaguely remember writing something but can't recall the exact filename or keyword.

grove reuses the Gemini API key from [pairy](https://github.com/yash-srivastava19/pairy), my Neovim AI pair programmer. Both tools read from `~/.config/pairy/config.json`. This is one of those small plumbing decisions that I'm genuinely happy about - one API key configuration, two tools that benefit from it.

### Full TUI with vim keybindings

The TUI has four views: the note list, the note viewer, the search panel, and the AI chat panel. Navigation is `j/k` for up/down, `Enter` to open, `q` to go back, `/` to search, `n` to create a new note, `e` to open in `$EDITOR`. The whole thing feels like a mini Neovim for notes.

## What's next

A few things I want to add:

- **Graph view**: render the wiki-link graph in the terminal using box-drawing characters. This is purely for the aesthetic of it but I think it'd be genuinely useful for understanding the shape of your vault.
- **Git sync**: `grove sync` that commits and pushes your notes vault to a git remote. Version history for notes is something Obsidian charges for.
- **Tag filtering**: the frontmatter supports tags already, but the TUI doesn't have a proper tag browser yet.
- **Better diff on note updates**: right now editing a note just overwrites it. I want to store a git-style history of changes so you can see how your thinking evolved.

## Try it

```bash
go install github.com/yash-srivastava19/grove@latest
grove  # opens the TUI
```

Or just clone the repo and `go run main.go`. The only external dependency at runtime is a Gemini API key for the AI search feature - everything else works without it.

If you have ideas, find bugs, or want to add a template - [PRs are open](https://github.com/yash-srivastava19/grove/pulls). I'd genuinely love to see what other people's note-taking workflows look like and whether grove can fit into them.

**Note:** If you find this kind of work interesting and your organization does something similar, consider hiring me? I'm on the job market and would love to [chat](mailto:ysrivastava82@gmail.com).
