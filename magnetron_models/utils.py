# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from rich.console import Console

console = Console()


def download_or_ensure_resource(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    console.print(f'Downloading {filename}...', style='dim')
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type='model',
    )


def find_snapshot_file(repo_id: str, dtype_short_name: str) -> str:
    from huggingface_hub import HfApi
    mags = sorted(f for f in HfApi().list_repo_files(repo_id, repo_type='model') if f.endswith('.mag'))
    if not mags:
        raise FileNotFoundError(f'{repo_id} holds no .mag snapshot')
    if len(mags) == 1:
        return mags[0]
    matching = [f for f in mags if f.endswith(f'-{dtype_short_name}.mag')]  # The converter names files <model>-<dtype>.mag
    if len(matching) == 1:
        return matching[0]
    raise FileNotFoundError(f'{repo_id} holds {len(mags)} snapshots ({", ".join(mags)}), pass one with --snapshot after downloading it')
