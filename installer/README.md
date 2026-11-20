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

A folder that was run once for a test also holds `codeenv`, `code\models`, `__pycache__` folders and
`code	rellis2_init_done.txt`. None of them belongs in the zip, and the last one must not be there: with it the
installer believes it has already run, and installs nothing.

## The wheels in `whl`, and CPUs without AVX2

The wheels in `whl` are prebuilt, so that nobody needs a compiler. They must also run on CPUs that have AVX but no
AVX2 (Intel Sandy / Ivy Bridge): a wheel built with MSVC's `/arch:AVX2` stops there with "Illegal instruction" as
soon as it is imported. `nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl` was such a wheel;
`nvdiffrast-0.4.0+spellcast3d.1-...` is the same source built without `/arch` (`whl/nvdiffrast.json` says from
what and how, and where the recipe is). Its version is new on purpose: pip keeps an installed 0.4.0 otherwise.

Whenever a wheel is added or replaced, and before a release, ask Spellcast3D's release script whether every binary
would run on such a CPU (it needs Visual Studio's C++ tools; give it the release folder, or a tested install's
`venv` to include what pip brings):

```
python <spellcast3d>\prepare_release.py scan-cpu <release folder>
```

## How users get a new version

`tools/update.bat` fetches `main` into `code` and then copies `code/installer/tools/*.bat` over the launchers
in `tools`, so a fix to a launcher reaches existing installs too. The `run-*` bat files are left alone:
people edit them to add `--host` or `--port`.

The `.bat` files must stay plain ASCII with Windows (CRLF) line endings: `cmd` misreads anything else.
