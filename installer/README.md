# Files of the one-click installer

The release zip is this repository plus a portable Python and Git. This folder holds the parts of the zip
that are not Python code, so that they are versioned too.

```
trellis2-stableprojectorz_vNN/
    HOW TO USE - INSTRUCTIONS.txt                    from this folder
    README - LICENSE.txt
    run-browser/run-gradio.bat                       from this folder
    run-browser/support & donations.txt
    run-stableprojectorz/run-stableprojectorz.bat    from this folder
    run-stableprojectorz/support & donations.txt
    tools/*.bat                                      from this folder's tools
    tools/python/                                    portable Python 3.11 (not tracked)
    tools/git/                                       portable MinGit (not tracked)
    code/                                            the repository's files, as committed
    code/whl/flash_attn-2.8.3+cu128torch2.8.0cxx11abiTRUE-cp311-cp311-win_amd64.whl
                                                     (409 MB, over GitHub's file limit: not tracked)
```

## Building `code` for a release

Take the files from the commit, not from the working folder: the working folder also holds `venv`, `MODELS`,
`build`, `tmp` and editor folders, and may hold uncommitted experiments.

```
git archive --format=zip --output=code.zip HEAD
```

Unpack `code.zip` into an empty `code` folder, then add the flash_attn wheel to `code/whl`.

## How users get a new version

`tools/update.bat` fetches `main` into `code` and then copies `code/installer/tools/*.bat` over the launchers
in `tools`, so a fix to a launcher reaches existing installs too. The `run-*` bat files are left alone:
people edit them to add `--host` or `--port`.

The `.bat` files must stay plain ASCII with Windows (CRLF) line endings: `cmd` misreads anything else.
