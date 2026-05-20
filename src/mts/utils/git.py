import subprocess


def get_git_commit():
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        .decode("utf-8")
        .strip()
    )
