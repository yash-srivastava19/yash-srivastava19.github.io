---
layout: post
title: "I built Git from scratch. Here's what I learned."
---

Note: Someone told me once that you don't really understand Git until you've built it. I spent a weekend finding out if that was true. It is. If you've ever stared at a detached HEAD state at 11pm and felt genuinely confused, this project is for you.

# I built Git from scratch. Here's what I learned.

- **Project Home:** [github.com/yash-srivastava19/verizon](https://github.com/yash-srivastava19/verizon)
- **Language:** Python

> *"Git is magic until you understand it. Then it's elegant."*

I've been using Git daily for years. `git add`, `git commit`, `git push` - the muscle memory is so deep I do it without thinking. But for a long time I had this uneasy feeling about it. I knew how to use it but I didn't understand it. There's a difference. You can drive a car without understanding internal combustion. That's fine for most purposes. But if you want to debug when things go wrong - when you're staring at a detached HEAD state at 11pm - you need the mental model.

So I built one from scratch. **Verizon** - a version control system written in Python, from the ground up.

(The name is a pun: version + control = Verizon, like the telecom. Yes, I'm aware this is a terrible joke. No, I don't regret it.)

## Why build it? What do you actually learn?

The honest answer is: you learn that Git is a surprisingly small, elegant data structure wrapped in a lot of UI.

The core of Git is just a **content-addressable store** - a key-value database where the keys are SHA-1 hashes of the values. That's it. Everything else - branches, history, merges, the DAG of commits - is built on top of that simple primitive. Once that clicks, a lot of previously mysterious Git behavior becomes obvious.

You also learn why certain things in Git are the way they are. Why does `git rebase` rewrite history? Because commits are immutable - the SHA is computed from the content, including the parent hash. Change the parent, you get a new hash, which is effectively a new commit. Why does a "force push" feel so dangerous? Same reason - you're replacing references to immutable objects on the remote.

I couldn't learn this from reading the [Pro Git book](https://git-scm.com/book/en/v2) (good as it is). I had to implement it. When you implement a concept, you find all the places where your mental model was wrong, because the code tells you.

## How Git really works - the mental model

Before getting to Verizon, let me explain the model that building it gave me, because this is the thing I wish someone had explained clearly when I started.

**Everything in Git is an object.** There are four kinds:

- **blob**: the contents of a single file at a point in time
- **tree**: a directory listing - maps filenames to blob/tree SHAs
- **commit**: a snapshot of the entire repo - points to a tree (the root), has parent commit SHAs, has author/timestamp metadata
- **tag**: a named pointer to a commit

All of these are stored in `.git/objects/`, named by their SHA-1 hash. The hash is computed from the content of the object. This is why objects are immutable - if you change the content, you get a different hash, which is a different object.

**Branches are just files containing a commit SHA.** `.git/refs/heads/main` is a text file containing a 40-character hash. When you `git commit`, Git writes a new commit object, then updates that file to point to the new commit. That's it. Branches are cheap because they're just files.

**The staging area (index) is a binary file** at `.git/index` that tracks which blobs should go into the next commit. When you `git add`, Git writes the file content as a blob to the object store and records it in the index. When you `git commit`, Git takes the index, builds a tree from it, and creates a commit pointing to that tree.

The **DAG (directed acyclic graph)** of commits emerges naturally from commits pointing to their parents. Merge commits have two parents. That's where the graph structure comes from.

Once you have this model, `git log --graph` stops being magic and becomes "traverse the commit DAG and draw lines". `git diff HEAD~3` stops being magic and becomes "get the tree for HEAD, get the tree for HEAD~3, compare them". Everything becomes traversal over a graph of immutable objects stored by hash.

## What Verizon implements

Verizon implements the core of this model in Python. No external dependencies - just the standard library.

**The object store:**

```python
import hashlib
import zlib
import os

class ObjectStore:
    def __init__(self, repo_path):
        self.objects_dir = os.path.join(repo_path, ".verizon", "objects")
        os.makedirs(self.objects_dir, exist_ok=True)

    def write(self, obj_type: str, data: bytes) -> str:
        """Write an object, return its SHA."""
        header = f"{obj_type} {len(data)}\0".encode()
        full = header + data
        sha = hashlib.sha1(full).hexdigest()

        path = os.path.join(self.objects_dir, sha[:2], sha[2:])
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            f.write(zlib.compress(full))

        return sha

    def read(self, sha: str) -> tuple[str, bytes]:
        """Read an object by SHA, return (type, data)."""
        path = os.path.join(self.objects_dir, sha[:2], sha[2:])
        with open(path, "rb") as f:
            raw = zlib.decompress(f.read())

        null_idx = raw.index(b"\0")
        header = raw[:null_idx].decode()
        obj_type, _ = header.split(" ")
        data = raw[null_idx + 1:]

        return obj_type, data
```

This is the real Git object format - header, null byte, content, compressed with zlib, stored at a path derived from the first two characters of the SHA. Verizon's object store is binary-compatible with Git's.

**The commit model:**

```python
import time

class Commit:
    def __init__(self, tree_sha, parent_sha, message, author):
        self.tree = tree_sha
        self.parent = parent_sha  # None for the first commit
        self.message = message
        self.author = author
        self.timestamp = int(time.time())

    def serialize(self) -> bytes:
        lines = [f"tree {self.tree}"]
        if self.parent:
            lines.append(f"parent {self.parent}")
        lines.append(f"author {self.author} {self.timestamp} +0000")
        lines.append(f"committer {self.author} {self.timestamp} +0000")
        lines.append("")
        lines.append(self.message)
        return "\n".join(lines).encode()

    @classmethod
    def deserialize(cls, data: bytes) -> "Commit":
        text = data.decode()
        lines = text.split("\n")
        fields = {}
        i = 0
        while lines[i]:  # read until blank line
            key, _, val = lines[i].partition(" ")
            fields[key] = val
            i += 1
        message = "\n".join(lines[i+1:])

        c = cls.__new__(cls)
        c.tree = fields["tree"]
        c.parent = fields.get("parent")
        c.message = message
        c.author = fields["author"].rsplit(" ", 2)[0]
        return c
```

**Branching:**

```python
class Refs:
    def __init__(self, repo_path):
        self.refs_dir = os.path.join(repo_path, ".verizon", "refs", "heads")
        self.head_path = os.path.join(repo_path, ".verizon", "HEAD")
        os.makedirs(self.refs_dir, exist_ok=True)

    def get_head(self) -> str:
        """Return the SHA of the current HEAD commit."""
        with open(self.head_path) as f:
            content = f.read().strip()
        if content.startswith("ref: "):
            branch = content[5:]
            branch_path = os.path.join(
                os.path.dirname(self.refs_dir), branch
            )
            with open(branch_path) as f:
                return f.read().strip()
        return content  # detached HEAD

    def update_branch(self, branch_name: str, sha: str):
        path = os.path.join(self.refs_dir, branch_name)
        with open(path, "w") as f:
            f.write(sha)
```

Verizon implements `init`, `add`, `commit`, `log`, `branch`, `checkout`, `diff`, and `status`. The diff implementation uses Python's `difflib` to compute the unified diff between two blobs. The `log` command traverses the commit DAG by following parent pointers.

## What was hard / what surprised me

**The index format.** Git's actual binary index format is complex - it has file stat caching, extension sections, a hash at the end. I simplified Verizon's staging area to a plain JSON file that maps paths to blob SHAs. This gets the semantics right at the cost of not being binary-compatible with Git's index.

**Trees are recursive.** A tree object maps names to SHAs - but those SHAs can be blobs (files) or other trees (subdirectories). Building a tree from a directory means recursively building subtrees. This means a commit is really a hash of a root tree that contains hashes of subtrees that contain hashes of blobs. It's a Merkle tree. You get content integrity at every level for free.

**Diff is harder than it looks.** A diff between two commits isn't just "find changed files" - it's "compare two trees recursively, find added/removed/modified files and directories, then diff the changed blobs." The tree traversal logic has a lot of edge cases (renamed directories, files that become directories, etc.).

**The "detached HEAD" state** finally made sense. HEAD normally points to a branch (a ref), and the branch points to a commit SHA. "Detached HEAD" just means HEAD points directly to a commit SHA, not to a branch. When you make a new commit in detached HEAD state, the branch doesn't update - only HEAD does. If you then checkout a branch, that commit becomes unreachable from any ref. It's not deleted - objects in the store are only removed by garbage collection - but you can't easily find it. That's why Git warns you.

## What I'd do differently

**Use bytes consistently.** I mixed bytes and strings in early versions and paid for it with constant `encode()`/`decode()` calls. A real VCS lives in byte-land - filenames, content, everything.

**Implement GC from the start.** Currently Verizon's object store only grows. A real implementation needs garbage collection - find all objects reachable from refs, delete everything else. Not hard but I didn't build it.

**Better tree diffing.** The current diff is naive - compare flat listings of files, match by path. A smarter diff would detect renames by comparing content hashes of blobs, which is closer to what Git does.

If I started over, I'd also implement pack files - the way Git compresses multiple objects into a single binary file for efficiency. The per-object zlib compression works but pack files are where the real storage efficiency comes from, and implementing them would be a great follow-up project.

## The thing I'm taking away from this

Building Verizon made me a better Git user in a way that reading documentation never did. Now when something weird happens - a rebase goes sideways, a merge has unexpected conflicts, a push gets rejected - I have an actual mental model of what's happening, not just a bag of memorized commands.

More broadly, this is the most reliable way I know to learn how something works: build it. Not a perfect implementation, not production-ready, but enough to make the key design decisions yourself and understand why they were made the way they were.

The source is at [github.com/yash-srivastava19/verizon](https://github.com/yash-srivastava19/verizon). If you're learning Git internals, building something like this is a better investment than reading any book. Try adding a feature - remote sync, packed refs, a simple merge algorithm. Every new feature forces you to understand the data model more deeply.

**Note:** If you find this kind of work interesting and your organization does something similar, consider hiring me? I'm on the job market and would love to [chat](mailto:ysrivastava82@gmail.com).
