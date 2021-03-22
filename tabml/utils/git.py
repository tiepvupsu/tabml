import git


def get_git_repo_dir():
    """Gets path to the current git repo.  """
    repo = git.Repo(".", search_parent_directories=True)
    return repo.working_tree_dir
