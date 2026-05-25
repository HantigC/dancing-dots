import subprocess
import tomllib


def get_project_version():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    pkg_version = data["project"]["version"].replace(".", "-")
    short_commit = (
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .strip()
    )
    return f"{pkg_version}-{short_commit}"


def get_project_name() -> str:
    return "dancing-dots"


def get_wheels_name() -> str:
    return f"wheels-{get_project_name()}"
