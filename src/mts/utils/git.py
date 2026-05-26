import subprocess


class NotAGitRepositoryError(Exception):
    pass


def get_git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError as e:
        raise NotAGitRepositoryError("Not inside a git repository") from e
