Note: I once watched someone run an UPDATE without a WHERE clause on a shared staging database. It wasn't intentional. It affected every row in the table. It was not a good afternoon. I built Safe SQL because the tooling around this problem is either "be more careful" or "restore from backup," and neither of those is satisfying.

# Safe SQL: A Seatbelt for Your Database

- **Project Home:** [github.com/yash-srivastava19/safe_sql](https://github.com/yash-srivastava19/safe_sql)
- **Install:** `pip install safe-sql`
- **Downloads:** 300+ monthly

---

## The problem is old. The solutions are bad.

Every developer who works with a database long enough runs into the same category of mistake. A query that affects more rows than intended.

```sql
-- The classic: forget the WHERE clause entirely
DELETE FROM users;

-- The subtler version: WHERE clause that matches more than you expect
UPDATE orders SET status = 'cancelled' WHERE created_at < '2024-01-01';
-- (you're thinking in 2023 timelines but it's 2024 in the database)
```

The standard responses to this class of mistake:

- **"Be more careful."** Good advice. Doesn't scale. Doesn't help at 2am.
- **"Restore from backup."** Works, but restore-from-backup means the data is already gone and you're racing against business impact while someone is asking you questions.
- **"Use transactions."** Necessary, not sufficient. You still have to realize you need to rollback, and you have to do it fast, before you commit.

What I wanted was something that catches these mistakes *before* they hit the database, not after.

---

## What Safe SQL actually does

Safe SQL wraps your database connection and adds seven protection layers around UPDATE and DELETE operations. Use it as a Python library or as a CLI.

### Layer 1: Query preview

Before executing any UPDATE or DELETE, Safe SQL constructs the equivalent SELECT query and shows you which rows will be affected:

```bash
safe_sql execute \
  --connection-string "postgresql://localhost/mydb" \
  --mode write \
  --query "DELETE FROM orders WHERE status = 'pending'"
```

Output before execution:
```
Preview: 847 rows will be affected by this operation.
  id=1234, status=pending, created_at=2024-01-15
  id=1235, status=pending, created_at=2024-01-16
  ... (845 more)

Proceed? [y/N]:
```

847 rows. You typed that query expecting maybe 20. You press N.

### Layer 2: Missing WHERE clause detection

If your query has no WHERE clause on an UPDATE or DELETE, Safe SQL stops before doing anything else:

```
Warning: DELETE without WHERE clause detected.
This will affect ALL rows in 'orders' (current count: 847,293).

Proceed? [y/N]:
```

This catches the most dangerous case before it starts.

### Layer 3: Schema validation

Every column referenced in the query is verified against the actual table schema before execution. Catches typos before they become errors halfway through a long operation:

```
Error: Column 'statuss' does not exist in table 'orders'.
Did you mean: 'status'?
```

### Layer 4: Automatic backup

Before executing the actual query, Safe SQL backs up the affected rows into a separate table:

```sql
-- Created automatically before your DELETE runs:
CREATE TABLE orders_backup_20240815_143022 AS
SELECT * FROM orders WHERE status = 'pending';
```

If something still goes wrong after you confirmed, you have a queryable snapshot of exactly what was there.

### Layer 5: Transaction wrapping

Every query runs inside a transaction. If anything fails mid-execution, it rolls back automatically. No partial deletes, no half-updated tables.

### Layer 6: Result caching

Frequent SELECT queries within a session are cached, reducing redundant database hits during preview and validation phases.

### Layer 7: Impact reporting

After execution, you get a count of what was actually changed:

```
Operation complete. 23 rows deleted.
Backup table: orders_backup_20240815_143022
```

---

## The design philosophy: friction as a feature

These safety checks add friction. That's the point.

The instinct in software is usually to make things faster and more automated. But for destructive operations on production data, a small amount of mandatory friction is protective. The preview step takes 5 seconds. Those 5 seconds are worth it when your DELETE would have hit 847 rows instead of the 20 you expected.

This is the same reasoning behind `git push --force` requiring an explicit flag, or macOS asking for your password before installing software. The friction is load-bearing. Removing it removes the protection.

---

## Usage

**Python library:**

```python
from safe_sql import SafeConnection

conn = SafeConnection(
    connection_string="postgresql://localhost/mydb",
    mode="write"
)

# Previews affected rows, requires confirmation before executing
conn.execute("DELETE FROM sessions WHERE expires_at < NOW()")
```

**CLI:**

```bash
pip install safe-sql

safe_sql execute \
  --connection-string "postgresql://localhost/mydb" \
  --mode write \
  --query "UPDATE users SET tier='free' WHERE last_active < '2024-01-01'"
```

The connection string accepts any SQLAlchemy-compatible URL, which means PostgreSQL, MySQL, and SQLite work out of the box.

---

## The three modes

**Read mode**: Query-only. Any attempted write raises an error before it touches the database. Useful for analytics scripts, monitoring jobs, or any context where you want to enforce a read-only guarantee at the library level.

**Write mode**: Full protection stack enabled. Every UPDATE and DELETE goes through all seven layers.

**Admin mode**: Elevated permissions for schema changes and administrative operations. Not wrapped with the same guardrails, because schema changes are intentional by definition.

---

## Honest limitations

Safe SQL is a layer on top of your database, not a replacement for proper access controls. If someone connects directly and bypasses the library, the protections don't apply. For real data security guarantees, you want database-level permissions as the foundation, with tools like this as an additional layer on top.

The backup mechanism creates tables, not transaction log entries. This is portable across databases and makes backups queryable, but it's less efficient than WAL-based approaches for very large tables. If you're running this against a table with 50 million rows, the backup step will be slow.

300+ monthly PyPI downloads means people are using it. Zero open issues might mean it's handling the common cases well, or might mean the edge cases just haven't been reported yet. If you hit something it doesn't handle, [file an issue](https://github.com/yash-srivastava19/safe_sql/issues).

---

## Install

```bash
pip install safe-sql
```

Run it on your next production UPDATE and see how it feels to know exactly what you're about to change before you change it.
