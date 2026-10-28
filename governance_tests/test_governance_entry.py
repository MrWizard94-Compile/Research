"""Read-only entry checks; no scientific module, model or provider is executed."""

import ast
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

ROOT = Path(os.environ.get("RECURRSIVE_GOV_ROOT", Path(__file__).resolve().parents[1]))
SOURCE = "https://github.com/MrWizard94-Compile/Agents-Constitution"


def parse_sources(root):
    """Parse only immediate source files, refusing links before reading bytes."""
    result = {}
    for path in sorted(root.glob("*.py")):
        if path.is_symlink() or path.resolve().parent != root.resolve():
            raise ValueError("Linked or out-of-root source requires separate review")
        result[path.name] = ast.parse(path.read_text(encoding="utf-8-sig"), str(path))
    return result


class GovernanceTests(unittest.TestCase):
    def test_agent_entry_uses_one_verified_frozen_source(self):
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8-sig")
        for value in (
            SOURCE,
            "wpai-constitution",
            "full commit SHA",
            "SOURCE",
            "same frozen revision",
            "not authority",
            "verified",
            "offline",
        ):
            with self.subTest(value=value):
                self.assertIn(value, text)
        self.assertNotRegex(text, r"(?i)binding\s+pack|pack\s+path")

    def test_provider_entry_routes_to_project_not_a_second_constitution(self):
        path = ROOT / ".claude/AGENTS.md"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8-sig")
        for value in (
            SOURCE,
            "../AGENTS.md",
            "wpai-constitution",
            "same frozen revision",
            "not constitutional law",
            "permissions",
        ):
            with self.subTest(value=value):
                self.assertIn(value, text)
        self.assertNotRegex(text, r"(?i)binding\s+pack|pack\s+path")

    def test_research_evidence_boundaries_are_preserved(self):
        text = " ".join((ROOT / "AGENTS.md").read_text(encoding="utf-8-sig").split())
        for value in (
            "Iterations 10–15",
            "retracted",
            "random-weight",
            "real-checkpoint",
            "GPU-board",
            "wall-plug",
            "dense attention",
            "not fresh",
            "thresholds",
            "unpublished",
        ):
            with self.subTest(value=value):
                self.assertIn(value, text)

    def test_data_and_execution_approval_boundaries_are_explicit(self):
        text = " ".join((ROOT / "AGENTS.md").read_text(encoding="utf-8-sig").split())
        for value in (
            "Do not run",
            "results/",
            "plots/",
            "models/",
            "tools/",
            "--force-full-gpu",
            "director",
            "first publication",
            "deletions",
        ):
            with self.subTest(value=value):
                self.assertIn(value, text)

    def test_retired_working_artifacts_are_absent(self):
        for relative in (
            "SOUL.md",
            "SOUL.md.legacy-pre-pack-20260727-205752",
            ".claude/SOUL.md",
            ".claude/SOUL.md.legacy-pre-pack-20260727-205752",
        ):
            with self.subTest(path=relative):
                self.assertFalse((ROOT / relative).exists())

    def test_local_document_links_resolve(self):
        for relative in (
            "AGENTS.md",
            ".claude/AGENTS.md",
            "README.md",
            "docs/GOVERNANCE-ADOPTION.md",
        ):
            path = ROOT / relative
            with self.subTest(path=relative):
                self.assertTrue(path.is_file())
                if not path.is_file():
                    continue
                for target in re.findall(
                    r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8-sig")
                ):
                    if not target.startswith(("https://", "http://", "#")):
                        self.assertTrue(
                            (path.parent / unquote(target)).is_file(), target
                        )

    def test_readme_keeps_retractions_but_repairs_dead_onboarding(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8-sig")
        self.assertIn("Iterations 10–15 are retracted", text)
        self.assertIn("2026-07-03, Iteration 19", text)
        self.assertIn("[AGENTS.md](AGENTS.md)", text)
        self.assertIn("governance_tests", text)
        self.assertIn("not a new benchmark", text)
        self.assertNotIn("CLAUDE.md §15", text)
        self.assertNotIn("`CLAUDE.md` — the working rules", text)

    def test_workflow_pins_source_and_does_not_execute_benchmarks(self):
        path = ROOT / ".github/workflows/constitution.yml"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        for value in (
            "0010a8353b00754fad31a30ed24327d3b6d05425",
            "windows-2025",
            "ubuntu-24.04",
            "macos-15",
            "'3.10'",
            "'3.12'",
            "governance_tests",
            "-W error",
            "contents: read",
            "persist-credentials: false",
        ):
            with self.subTest(value=value):
                self.assertIn(value, text)
        for action in re.findall(r"uses:\s+([^\s]+)", text):
            self.assertRegex(action, r"@[0-9a-f]{40}$")
        self.assertNotRegex(
            text, r"python\s+iteration\d|pip\s+install|continue-on-error"
        )

    def test_first_party_source_parses_without_imports(self):
        trees = parse_sources(ROOT)
        self.assertTrue(trees)
        for name in (
            "iteration16_corrected_streaming.py",
            "iteration16_quality_mqar.py",
            "iteration19_opt_unpack.py",
            "iteration19_combined_moe.py",
        ):
            self.assertIn(name, trees)
        self.assertTrue(all(isinstance(tree, ast.Module) for tree in trees.values()))

    def test_parser_does_not_execute_a_side_effectful_source(self):
        with tempfile.TemporaryDirectory(prefix="recurrsive-parse-") as temporary:
            root = Path(temporary)
            marker = root / "executed.txt"
            (root / "dangerous.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n"
                "import torch\nraise RuntimeError('must not execute')\n",
                encoding="utf-8",
            )
            with mock.patch(
                "subprocess.Popen", side_effect=AssertionError("No process allowed")
            ):
                trees = parse_sources(root)
            self.assertIn("dangerous.py", trees)
            self.assertFalse(marker.exists())

    def test_parser_rejects_invalid_source_instead_of_skipping_it(self):
        with tempfile.TemporaryDirectory(prefix="recurrsive-syntax-") as temporary:
            root = Path(temporary)
            (root / "broken.py").write_text("def broken(:\n", encoding="utf-8")
            with self.assertRaises(SyntaxError):
                parse_sources(root)

    def test_parser_refuses_a_link_before_reading_source(self):
        with tempfile.TemporaryDirectory(prefix="recurrsive-link-") as temporary:
            root = Path(temporary)
            (root / "candidate.py").write_text("pass\n", encoding="utf-8")
            with (
                mock.patch.object(Path, "is_symlink", return_value=True),
                mock.patch.object(
                    Path, "read_text", side_effect=AssertionError("Do not read a link")
                ),
            ):
                with self.assertRaisesRegex(ValueError, "Linked"):
                    parse_sources(root)


if __name__ == "__main__":
    unittest.main()
