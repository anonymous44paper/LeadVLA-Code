"""Audit the local distribution tree for common accidental disclosures.

Heuristic checks cover working-tree files, not Git history. They complement
manual review of commit identities, account metadata and external resources.
"""

from pathlib import Path
import hashlib
import re
import sys

PATTERNS = {
    "machine-specific path": re.compile(r"(?:/(?:home|Users|mnt|media|tmp|root)/[^\s]+|(?<!\w)[A-Za-z]:[\\/])"),
    "email address": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "credential token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"),
    "remote telemetry": re.compile(r"(?:wandb\.ai|api\.wandb|mlflow\.set_tracking_uri|googletagmanager\.com)"),
}
FORBIDDEN_PARTS = {".env", ".ssh", ".github-anonymous", "__pycache__", "node_modules", "checkpoints"}
ALLOWED_SUFFIXES = {".py", ".md", ".json", ".toml"}
REVIEWED_FIGURES = {
    "docs/figures/method.webp": "fe48731dd6b14394d38a2fb66d39a2bc6eb4d83e44e3cdaa05e0d27fe9955fb6",
    "docs/figures/leadinfra-assets.webp": "30c07ee8cab46021816614f04eec293c39a8c62b3ac894ce93e24356149510a5",
    "docs/figures/leadinfra-generation.webp": "ebfab1e0414ebd6efe221491779e217d8cf4eec058bea8b70838475f8b752871",
}


def audit(root):
    issues, count = [], 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            issues.append((str(relative), "symlink"))
            continue
        if any(part in FORBIDDEN_PARTS or part.endswith(".egg-info") for part in relative.parts):
            issues.append((str(relative), "non-distribution artifact"))
            continue
        if not path.is_file():
            continue
        count += 1
        if relative.as_posix() in REVIEWED_FIGURES:
            # These exact files have had visual and embedded-metadata review.
            # New or changed images require a new review, not a blanket bypass.
            if hashlib.sha256(path.read_bytes()).hexdigest() != REVIEWED_FIGURES[relative.as_posix()]:
                issues.append((str(relative), "figure changed; anonymity review required"))
            continue
        if path.name != ".gitignore" and path.suffix not in ALLOWED_SUFFIXES:
            issues.append((str(relative), "unexpected file type"))
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            issues.append((str(relative), "binary content"))
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(content):
                issues.append((str(relative), name))
    return count, issues


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    count, issues = audit(root)
    for relative, reason in issues:
        print(f"{relative}: {reason}")
    print(f"Audited {count} files; {len(issues)} findings.")
    sys.exit(bool(issues))
