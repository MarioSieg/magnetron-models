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
