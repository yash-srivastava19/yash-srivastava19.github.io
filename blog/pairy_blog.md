---
layout: post
title: "pairy — AI pair programming the way I like it"
---

Note: Get nerd sniped for the love of god. Also, if you like this kind of work and your organization does something similar, consider hiring me? I'm on the job market and would love to [chat](mailto:ysrivastava82@gmail.com).

# pairy — AI pair programming the way I like it

- **Project Home:** [github.com/yash-srivastava19/pairy](https://github.com/yash-srivastava19/pairy)
- **Language:** Lua (Neovim plugin), Gemini API

## Own your tools

There's a principle I keep coming back to: **own your tools**. Not in some vague philosophical sense, but literally — if a tool is important enough to be in your daily workflow, you should understand how it works, be able to modify it, and ideally not be paying a SaaS subscription for the privilege of using it.

AI coding assistants are in my daily workflow now. They've been genuinely useful — not as a replacement for thinking, but as a rubber duck that can also write code. When I'm stuck on an API I've never used, or want to see three different ways to approach a refactor, having a capable model available is valuable.

But the existing options all had something that bothered me.

**Copilot** is great, but it's GitHub's SaaS product. Monthly subscription. I don't fully control what gets sent where. The completions are good but I have no visibility into the model being used or how to tune the behavior.

**Cursor** is genuinely impressive — the context awareness is miles ahead of Copilot. But it's a whole separate editor. I've spent years getting my Neovim configuration exactly right. I know every keybinding, every plugin interaction, every quirk. Switching to Cursor means starting over with a VSCode fork that I don't control.

**Codeium, Continue, and the rest** are mostly fine but they either require accounts, phone home constantly, or have a plugin architecture that feels heavy.

My requirements were simpler:

1. Works inside Neovim — no context switching
2. No subscription — use an API key I control
3. No proxy server — talk to the model directly
4. Minimal, non-invasive UI — doesn't break my existing workflow
5. Open source — I can read and modify every line of it

**pairy** is what I built to satisfy all five.

## Why Gemini + Neovim

The model choice came down to the free tier. Gemini has a genuinely generous free tier — enough for serious daily use without touching a credit card. For a personal tool that I'm running constantly, "free until you seriously need to scale" is the right price point. The API is clean, the context windows are large, and the code quality is good.

Neovim was always going to be the environment. I'm not switching editors. The Lua plugin ecosystem is mature enough now that you can build non-trivial plugins without fighting the API, and the performance is miles better than VimScript for anything complex.

The combination of Gemini's free tier + Neovim's Lua API means pairy has zero operating cost for solo use and integrates cleanly into the environment where I actually write code.

## How it works

pairy is a Neovim plugin written in Lua. It stores its configuration at `~/.config/pairy/config.json`, which looks like this:

```json
{
  "gemini_api_key": "YOUR_KEY_HERE",
  "model": "gemini-2.0-flash",
  "temperature": 0.2,
  "max_tokens": 2048
}
```

The core of the plugin is a thin HTTP client that talks directly to the Gemini REST API. No SDK, no intermediary — just `curl`-equivalent requests from Lua using Neovim's built-in `vim.loop` (libuv) for async I/O. This keeps the dependency list empty.

```lua
-- Simplified version of the API call
local function call_gemini(prompt, callback)
  local config = require("pairy.config").load()
  local body = vim.json.encode({
    contents = {{ parts = {{ text = prompt }} }}
  })

  local handle = vim.loop.new_tcp()
  -- ... async HTTP POST to generativelanguage.googleapis.com
  -- response gets parsed and passed to callback
end
```

The plugin exposes a small set of commands, each of which grabs context from the current buffer and constructs a prompt:

- `:PairyExplain` — takes the visual selection (or current function) and asks Gemini to explain it in plain English
- `:PairyRefactor` — asks Gemini to suggest a refactored version, shows the diff in a split
- `:PairySuggest` — autocomplete at cursor position based on the current file context
- `:PairyChat` — opens a scratch buffer for a freeform chat about the current file
- `:PairyTest` — generates unit tests for the selected function

The context construction is the interesting part. For `:PairyExplain`, the prompt includes the visual selection plus a window of surrounding code (configurable, default 50 lines above/below). For `:PairyRefactor`, it includes the full function and the file type so the model knows which language idioms to use. For `:PairyChat`, the full buffer content goes in as context.

This is all explicit — you know exactly what gets sent to the model because you can read the 200 lines of Lua that construct the prompts.

## Usage — what it looks like day to day

Here's a real example from last week. I was working on a Go function that was doing too much — reading a config, validating it, and applying defaults all in one place. I selected the function, ran `:PairyRefactor`, and got back a suggestion to split it into three functions with clear responsibilities.

The diff appeared in a vertical split:

```
" pairy: suggested refactor (press <CR> to apply, q to discard)
─────────────────────────────────────────────────────
- func loadConfig(path string) (*Config, error) {
-     data, err := os.ReadFile(path)
-     if err != nil {
-         return nil, err
-     }
-     var cfg Config
-     if err := json.Unmarshal(data, &cfg); err != nil {
-         return nil, err
-     }
-     if cfg.MaxRetries == 0 {
-         cfg.MaxRetries = 3
-     }
-     if cfg.Timeout == 0 {
-         cfg.Timeout = 30 * time.Second
-     }
-     return &cfg, nil
- }
+ func readConfig(path string) ([]byte, error) {
+     return os.ReadFile(path)
+ }
+
+ func parseConfig(data []byte) (*Config, error) {
+     var cfg Config
+     return &cfg, json.Unmarshal(data, &cfg)
+ }
+
+ func applyDefaults(cfg *Config) *Config {
+     if cfg.MaxRetries == 0 { cfg.MaxRetries = 3 }
+     if cfg.Timeout == 0 { cfg.Timeout = 30 * time.Second }
+     return cfg
+ }
```

Press `<CR>` and it applies the change to the buffer. Press `q` and it discards. No magic, no automatic file modification — you stay in control.

`:PairyExplain` is something I use a lot when reading unfamiliar codebases. Select a block of code I don't understand, run the command, get a plain-English explanation in a floating window. It's quicker than reading docs and usually more accurate for explaining what *this specific code* does as opposed to what the library *can* do in general.

`:PairyChat` is for longer conversations. I'll often open it at the start of a debugging session: "I have a bug where X happens when I do Y. Here's the relevant code. What could cause this?" The full buffer context means the model usually has enough to give a useful starting point.

## The config lives at `~/.config/pairy/config.json`

One thing I like about this setup: the config file is simple JSON that you edit directly. No wizard, no GUI, no ``:Pairy setup`` command. You know where it is, you know what's in it.

This same config is read by [grove](https://github.com/yash-srivastava19/grove), my terminal note-taking tool. Both tools share the Gemini API key. If you use both (and I do), you configure once.

```json
{
  "gemini_api_key": "AIza...",
  "model": "gemini-2.0-flash-exp",
  "temperature": 0.1
}
```

Lower temperature for code tasks. The default is 0.2, which gives you mostly deterministic output with a little variation. Crank it up if you want more creative refactor suggestions, dial it down if you want precise completions.

## Limitations — being honest about what it's not

pairy is not Cursor. It doesn't have a continuous awareness of your entire codebase — it works on what you explicitly give it as context. For large projects where you need cross-file intelligence, Cursor or a proper LSP-backed tool will be better.

The async I/O is functional but not battle-hardened. If you trigger multiple commands quickly, there can be race conditions in the response handling. I've worked around the worst cases but haven't made it bulletproof.

Streaming responses aren't implemented yet. The model generates the full response and then displays it, which means there's a perceptible pause for long outputs. Streaming would make it feel much faster even for the same total latency.

I'd also like to add a project-level context feature — something like a `.pairy/context.md` file in your project root that always gets included in prompts. This would let you add "this codebase uses X convention, prefer Y pattern" instructions that persist across sessions.

## What's next

- Streaming responses — show output as it's generated
- Project context files — persistent instructions per repo
- A `:PairyCommit` command that generates a commit message from staged diffs
- Better test generation — right now it's generic, I want it to pick up existing test patterns from the repo

## Try it

Installation is the standard Neovim plugin manager flow. With [lazy.nvim](https://github.com/folke/lazy.nvim):

```lua
{
  "yash-srivastava19/pairy",
  config = function()
    require("pairy").setup({
      config_path = vim.fn.expand("~/.config/pairy/config.json")
    })
  end
}
```

Then create your config file with your Gemini API key (free tier from [Google AI Studio](https://aistudio.google.com/)) and you're done.

If you use Neovim and have thoughts on the UX, I'd love feedback. What commands do you actually reach for? What does the AI coding workflow feel like when it's integrated directly into your editor vs. a chat interface? Hit me up at [ysrivastava82@gmail.com](mailto:ysrivastava82@gmail.com) or drop a [PR](https://github.com/yash-srivastava19/pairy/pulls).

**Note:** If you find this kind of work interesting and your organization does something similar, consider hiring me? I'm on the job market and would love to [chat](mailto:ysrivastava82@gmail.com).
