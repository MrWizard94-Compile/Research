import argparse
import os
import subprocess
import sys
from pathlib import Path

COMMON_PYTHON_PACKAGES = [
    "requests",
    "numpy",
    "pandas",
    "matplotlib",
    "python-dotenv",
    "openai",
    "pytest",
    "black",
    "flake8",
    "rich",
    "click",
    "httpx",
    "beautifulsoup4",
    "pillow",
    "scikit-learn",
    "fastapi",
    "uvicorn",
    "pydantic",
]


def run_command(command, description, cwd=None):
    """Execute a shell command and fail with a helpful message when needed."""
    print(f"Executing: {description}...")
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
        if result.stdout:
            print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        print(f"Error during: {description}")
        if stderr:
            print(stderr)
        sys.exit(1)


def build_remote_url(repo_name=None, repo_url=None, owner="MrWizard94-Compile"):
    """Create a GitHub remote URL from a repository slug or a full URL."""
    if repo_url:
        return repo_url

    if not repo_name:
        raise ValueError("A repository name or remote URL is required.")

    slug = repo_name.strip()
    if slug.startswith(("https://", "http://", "git@")):
        return slug

    if "/" in slug:
        return f"https://github.com/{slug}.git"

    return f"https://github.com/{owner}/{slug}.git"


def get_venv_python_path(venv_dir):
    """Return the Python executable path for the virtual environment."""
    venv_path = Path(venv_dir)
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def ensure_virtual_environment(target_dir, venv_name=".venv"):
    """Create a virtual environment if it does not already exist."""
    venv_path = Path(target_dir) / venv_name
    if not venv_path.exists():
        run_command(
            [sys.executable, "-m", "venv", str(venv_path)],
            f"Creating virtual environment at {venv_path}",
            cwd=target_dir,
        )
    else:
        print(f"Virtual environment already exists at {venv_path}")
    return get_venv_python_path(venv_path)


def write_requirements_file(target_dir, requirements_path="requirements.txt"):
    """Write a common Python dependency requirements file."""
    req_path = Path(target_dir) / requirements_path
    req_path.write_text("\n".join(COMMON_PYTHON_PACKAGES) + "\n", encoding="utf-8")
    return req_path


def install_dependencies(python_executable, target_dir, requirements_path):
    """Upgrade pip and install the standard dependency set."""
    run_command(
        [str(python_executable), "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip",
        cwd=target_dir,
    )
    run_command(
        [str(python_executable), "-m", "pip", "install", "-r", str(requirements_path)],
        "Installing common Python dependencies",
        cwd=target_dir,
    )


def write_gitignore(target_dir):
    """Create a useful .gitignore for Python projects."""
    gitignore_path = Path(target_dir) / ".gitignore"
    gitignore_path.write_text(
        "\n".join(
            [
                ".venv/",
                "__pycache__/",
                "*.py[cod]",
                ".pytest_cache/",
                ".mypy_cache/",
                ".ruff_cache/",
                ".idea/",
                ".vscode/",
                "*.log",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def initialize_git_repository(target_dir, remote_url):
    """Initialize git, commit the project files, and connect the remote."""
    commands = [
        (["git", "init"], "Initializing local Git repository"),
        (["git", "branch", "-M", "main"], "Renaming primary branch to 'main'"),
        (["git", "add", "."], "Staging files for the initial commit"),
        (["git", "commit", "-m", "Initial commit"], "Creating the initial commit"),
        (["git", "remote", "add", "origin", remote_url], f"Linking remote URL: {remote_url}"),
    ]

    for command, description in commands:
        run_command(command, description, cwd=target_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Set up a directory as a Git repository, create a virtual environment, and install common Python dependencies."
    )
    parser.add_argument(
        "repo_name",
        nargs="?",
        type=str,
        help="GitHub repository slug such as demo-repo or a full remote URL.",
    )
    parser.add_argument(
        "--repo-url",
        dest="repo_url",
        type=str,
        help="Full remote Git URL to use instead of building one from a repo slug.",
    )
    parser.add_argument(
        "--directory",
        "-d",
        default=".",
        help="Directory to initialize (defaults to the current working directory).",
    )
    parser.add_argument(
        "--venv-name",
        default=".venv",
        help="Name of the virtual environment directory to create.",
    )
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="Skip creating the virtual environment.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installing the common Python dependencies.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_dir = Path(args.directory).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    remote_url = build_remote_url(args.repo_name, args.repo_url)

    write_gitignore(target_dir)

    if not args.skip_venv:
        python_executable = ensure_virtual_environment(target_dir, args.venv_name)
        requirements_path = write_requirements_file(target_dir)
        if not args.skip_install:
            install_dependencies(python_executable, target_dir, requirements_path)
    else:
        print("Skipping virtual environment creation.")

    initialize_git_repository(target_dir, remote_url)

    print(f"\n[SUCCESS] Setup complete for {target_dir}")
    print(f"Remote: {remote_url}")
    if not args.skip_venv:
        print(f"Virtual environment: {target_dir / args.venv_name}")


if __name__ == "__main__":
    main()
