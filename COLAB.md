# Running notebooks on Google Colab

For every member. Use this when a run is too slow on your laptop. Everything in
[CONTRIBUTING.md](CONTRIBUTING.md) still applies — Colab only changes *where* a notebook runs, never
how it is written.

1. [How Colab fits in](#1-how-colab-fits-in)
2. [Do you need a Google Drive project? No](#2-do-you-need-a-google-drive-project-no)
3. [Your own Colab, or a shared one? Your own](#3-your-own-colab-or-a-shared-one-your-own)
4. [One-time setup](#4-one-time-setup)
5. [Path A — hand a development run over from VS Code](#5-path-a--hand-a-development-run-over-from-vs-code)
6. [Path B — an official run in the browser](#6-path-b--an-official-run-in-the-browser)
7. [Which notebooks to run where](#7-which-notebooks-to-run-where)
8. [Rules](#8-rules)
9. [Troubleshooting](#9-troubleshooting)
10. [Optional — keep a long run's checkpoints in Drive](#10-optional--keep-a-long-runs-checkpoints-in-drive)

---

## 1. How Colab fits in

**GitHub holds the project. Colab lends you a GPU machine for a while.**

Every notebook's cell 1 recognises when it is running on Colab. It then clones the repository onto
that machine, installs it, and prints the exact commit it is running. From there the notebook behaves
exactly as it does on your laptop. The Colab machine is wiped when the session ends, so anything that
must survive goes back through git.

There are two ways in:

| | **Path A** — VS Code + the Colab extension | **Path B** — Colab in the browser |
|---|---|---|
| **Use it for** | development: `synthetic` and `debug` runs | **official** runs |
| **Where the notebook lives** | on your laptop, as always | on GitHub; Colab opens it from there |
| **Outputs are saved** | into your local notebook, as always | back to GitHub with *File → Save a copy in GitHub* |
| **Result files** | none — debug runs write none | pushed from the Colab machine by the notebook's last cell |
| **Which fork / branch runs** | you set it in cell 1 for the session | your `A03_REPO` / `A03_BRANCH` Colab Secrets |
| **Token needed** | no | yes, to push the results ([or skip it](#no-token-download-instead)) |

**The Colab machine only sees what you have pushed.** Cell 1 clones from GitHub, so commit and push
before every hand-over. Work that exists only on your laptop does not exist on Colab.

## 2. Do you need a Google Drive project? No

Nothing in this project lives in Google Drive, on purpose:

- **Copies drift.** Our notebooks are opened from GitHub (path B) or from your laptop (path A). A copy
  kept in Drive is disconnected from the repository: nobody else sees your edits, and they silently
  go stale.
- **Drive is slow for this data.** EuroSAT is 27,000 small images, and reading them through Drive
  makes every epoch crawl. Cell 1 keeps everything on the Colab machine's fast local disk; EuroSAT
  downloads again in about a minute each session.
- **Git and Drive don't mix well.** A repository working copy on Drive is slow and fragile, and Drive
  cannot hold the links git and the pipeline expect.

The one place Drive helps is keeping a *long* run's checkpoints alive across a disconnect. Our runs
take minutes on a T4, so you will probably never need it — see [§10](#10-optional--keep-a-long-runs-checkpoints-in-drive).

> Colab's default save — **Ctrl+S / *Save a copy in Drive*** — creates exactly the disconnected copy
> described above. For our notebooks, always use ***File → Save a copy in GitHub*** instead.

## 3. Your own Colab, or a shared one? Your own

**Each member uses Colab with their own Google account and their own secrets.** The only thing shared
is the repository. That is better for four reasons:

- **GPU time is per Google account.** On one shared account, four people would drain a single free
  allowance and block each other's sessions.
- **Your pushes carry your name.** Results pushed from Colab are committed under the name and token
  in *your* secrets. The graded commit history then shows who did what; a shared token would put
  everyone's work under one person.
- **Colab does not merge edits.** Two people in one notebook overwrite each other.
- **There is nothing else to share.** The "project" is the repository. Anyone can open any notebook
  from GitHub with the same link.

Sharing an open-in-Colab *link* is fine. Sharing an account, a token or a Drive copy is not.

## 4. One-time setup

### 4.1 For path A — the Colab extension in VS Code

In VS Code: **Extensions** (`Ctrl+Shift+X`) → search **Google Colab** → install the official
extension from Google. That's all; you sign in the first time you connect.

### 4.2 For path B — Colab Secrets

Colab Secrets are Colab's version of environment variables. They belong to your Google account, so
you set them once and every notebook can read them. Open any notebook at
[colab.research.google.com](https://colab.research.google.com), click the **key icon (Secrets)** in
the left sidebar, and add:

| Name | Value | Used for |
|---|---|---|
| `A03_REPO` | your fork, e.g. `yourname/EN3150-Assignment-03-CNN` | where cell 1 gets the code ([§4.3](#43-repo-and-branch--where-colab-gets-the-code)) |
| `A03_BRANCH` | the branch to run — optional, defaults to `main` | the same |
| `GH_TOKEN` | a GitHub token — see below | pushing an official run's results |
| `GIT_NAME` | your name, as it should appear on commits | the same |
| `GIT_EMAIL` | your GitHub email (the `…@users.noreply.github.com` address is fine) | the same |

The same five names, with placeholder values, are listed in [`.env.example`](.env.example). If you
keep your own values in a copy called `.env`, it stays out of git — but leave the token out of it.

The first time a notebook asks, allow it access to your secrets.

**Why a token at all, when you own your fork?** Owning the fork gives *your GitHub account*
permission to push to it. The Colab machine, though, isn't logged in as you. Your laptop remembers
your GitHub login, which is why you never type a token there; a fresh Colab machine has no such
login. The token is how the Colab machine proves it is acting for you. Cloning needs no token,
because the repositories are public.

**Creating the token:** GitHub → **Settings → Developer settings → Personal access tokens →
Fine-grained tokens → Generate new token**:

- **Repository access:** *Only select repositories* → your fork.
- **Permissions:** *Contents* → **Read and write**.
- **Expiration:** a date after the submission deadline.

Copy it into the `GH_TOKEN` secret straight away — GitHub shows it only once. **Never paste it into a
cell.** The repositories are public and notebook outputs are committed, so anything a cell shows is
published.

<a id="no-token-download-instead"></a>
**No token? Download instead.** You can skip the token entirely:

1. Skip the notebook's last cell.
2. From Colab's file browser (the folder icon in the left sidebar), download the run's folders:
   `results/metrics/<run_id>/` and `results/figures/<run_id>/`.
3. Copy them into the same places in your clone, and commit from your laptop.

The notebook itself still comes back with *File → Save a copy in GitHub*, which uses Colab's own
GitHub sign-in rather than a token.

### 4.3 `REPO` and `BRANCH` — where Colab gets the code

Cell 1 runs whatever is at `REPO` @ `BRANCH`:

```python
REPO = _secret("A03_REPO", "ThejithaR/EN3150-Assignment-03-CNN")
BRANCH = _secret("A03_BRANCH", "main")
```

- **In the browser** it uses your `A03_REPO` / `A03_BRANCH` Secrets. You never edit the notebook.
- **In VS Code** Secrets can't be read, so the defaults are used. To run your own work there, edit
  the defaults in cell 1 for the session, e.g. `_secret("A03_REPO", "yourname/your-fork")`, and **put
  them back before you commit.** `tests/test_notebooks.py` fails if the eight notebooks' defaults
  disagree, as a reminder.
- **Without either**, cell 1 runs the defaults above.

## 5. Path A — hand a development run over from VS Code

The notebook stays on your laptop. Only the running moves to Colab's GPU.

1. **Push your work:** `git push`. Cell 1 on the Colab machine clones from GitHub.
2. **Point cell 1 at it:** VS Code can't read Colab Secrets, so edit the two defaults in cell 1 to
   your fork and branch ([§4.3](#43-repo-and-branch--where-colab-gets-the-code)).
3. **Switch the kernel:** in the notebook, **Select Kernel** (top right; **Select Another Kernel…** if
   one is already chosen) → **Colab** → **New Colab Server** → sign in with Google → **GPU → T4** →
   pick the server's **Python 3** kernel. Until you have done this once, the extension's Colab panel
   says *No assigned Colab servers*. That only means no server exists yet.
4. **Run cell 1.** The first time it clones and installs, which takes about a minute. It then prints
   the commit it is running — check that it is yours.
5. **Run the rest step by step** with `MODE = "debug"` (or `"synthetic"`). Outputs land in your local
   notebook as usual.
6. **Fix and repeat.** Edit locally → commit → push → **re-run cell 1** (it fetches your new commit) →
   re-run the cells that *build* things (loaders, model, trainer). Objects built before the update
   keep the old code.
7. **Hand back:** **Select Kernel** → `.venv`, and put cell 1's defaults back. Then release the
   machine: `Ctrl+Shift+P` → **Colab: Remove Server**. An idle server keeps using your free GPU time
   until Colab times it out.

Path A is for runs that write nothing. Official runs push result files from the Colab machine with
the token in your Colab Secrets, which VS Code cannot read — do those with path B.

## 6. Path B — an official run in the browser

1. **Push** your code and your notebook to your fork.
2. **Open the notebook straight from GitHub**, with the same fork and branch as your `A03_REPO` /
   `A03_BRANCH` secrets:

   ```
   https://colab.research.google.com/github/<your fork>/blob/<branch>/notebooks/<notebook>.ipynb
   ```

   For example: `https://colab.research.google.com/github/ThejithaR/EN3150-Assignment-03-CNN/blob/main/notebooks/03_optimizer_study.ipynb`
3. **Runtime → Change runtime type → T4 GPU → Save.**
4. **Cell 2:** `MODE = "official"`. Cell 1 needs no edits — it reads your fork and branch from your
   secrets.
5. **Runtime → Run all.** Allow access to your secrets the first time you are asked.
6. **Check the commit cell 1 prints** — it should be your latest push.
7. **The last cell pushes this run's result files** to your fork and branch.
8. **Save the notebook itself:** **File → Save a copy in GitHub** → same repository, same branch, same
   path → a short commit message → OK. This is what brings the outputs back. The first time, authorise
   Colab to use your GitHub account.
9. **On your laptop:** `git pull`.

Do steps 7 and 8 before you close the tab — a Colab session is wiped when it ends.

## 7. Which notebooks to run where

| Notebook | Colab? | Why |
|---|---|---|
| `00_setup` | optional | a quick check that a new runtime works |
| `01_data_preparation` | not needed | no training. Its official run writes the split — do that on your laptop |
| `02_custom_architectures` | not needed | no data, no training |
| `03_optimizer_study` | **good fit** | three trainings. Its official run must share a runtime with `04` |
| `04_custom_training_evaluation` | **good fit** | same runtime as `03` for the official run |
| `05_pretrained_finetuning` | **best fit** | the heaviest training. Run both backbones on the same runtime |
| `06_resource_benchmark` | **never** | measures CPU latency; Colab's CPU changes between sessions. Run it on your laptop |
| `07_final_comparison` | not needed | reads committed JSON only |

**What must exist before Colab can run `debug` or `official`:**

- your code, pushed to `REPO` @ `BRANCH`;
- the committed split — Member 1's official run of `01`;
- Member 1's `ensure_images_present`. A fresh Colab machine has the committed split but no images;
  until this lands, Colab can only run `MODE = "synthetic"`.

## 8. Rules

- **Only pushed code exists on Colab.** Push first, then re-run cell 1.
- **Don't edit code on Colab.** Its copy is wiped when the session ends. Edit on your laptop, push,
  re-run cell 1.
- **Never paste or print a token.** It stays in Colab Secrets.
- **Edited cell 1's defaults in VS Code? Put them back before committing.** In the browser, use your
  `A03_REPO` / `A03_BRANCH` secrets instead of editing.
- **Keep hardware consistent:** `03` and `04` official runs share a runtime; `05`'s two backbones share
  a runtime; `06` runs on your laptop's CPU.
- **One place per notebook at a time.** After *Save a copy in GitHub*, run `git pull` before editing
  that notebook locally, or you will create a notebook conflict.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| VS Code: the Colab panel says *No assigned Colab servers* | no Colab server created yet — normal before the first connection | **Select Kernel → Colab → New Colab Server** ([§5](#5-path-a--hand-a-development-run-over-from-vs-code), step 3) |
| Cell 1: `Remote branch … not found` or `… is not a commit` | the branch isn't pushed, or its name is misspelled | push it; check `A03_BRANCH` |
| Cell 1: `could not read Username` | the repository name is misspelled, or it is private | check `A03_REPO` (or the default in VS Code) |
| Cell 1 prints a commit that isn't your latest | not pushed yet, or cell 1 is using the defaults | push; set `A03_REPO` / `A03_BRANCH` (browser) or edit the defaults (VS Code); re-run cell 1 |
| `ModuleNotFoundError: No module named 'edgecnn'` | the kernel restarted | re-run cell 1 — needed after every restart |
| a fix you pushed makes no difference | objects were built before the update | re-run cell 1, then the cells that build the loader, model and trainer |
| `NotImplementedError: Member N …` | that member's part hasn't landed yet | same on Colab as locally: use `synthetic`, skip the cell, or ask them |
| "Cannot connect to GPU backend" / usage limit | free-tier GPU time is used up | try later, or run `debug` on a CPU runtime |
| last cell: `No GitHub token` | `GH_TOKEN` is missing, or the notebook was not allowed to read it | [§4.2](#42-for-path-b--colab-secrets), then allow access |
| last cell: `push` rejected with a permission error | the token can't write to your fork | re-create it for your fork, with *Contents: Read and write* |
| session disconnected during a run | Colab idle / time limits | run it again; for very long runs see [§10](#10-optional--keep-a-long-runs-checkpoints-in-drive) |
| *Save a copy in GitHub* can't see the repository | Colab isn't authorised on GitHub yet | follow the authorisation prompt, then retry |

## 10. Optional — keep a long run's checkpoints in Drive

Only needed if a single run could outlast your Colab session (unlikely — our runs take minutes on a
T4). **Browser only:** the VS Code extension cannot mount Drive. This keeps that run's checkpoints on
Drive, so a restarted session can resume. Run it in a temporary cell after the config cell and
before training, then delete the cell before saving the notebook:

```python
from google.colab import drive

drive.mount("/content/drive")
target = f"/content/drive/MyDrive/en3150-a03-checkpoints/{cfg.run_id}"
os.makedirs(target, exist_ok=True)
link = paths.checkpoint_dir(cfg.run_id, official=cfg.is_official)
link.parent.mkdir(parents=True, exist_ok=True)
if not link.exists():
    os.symlink(target, link)
```

After a disconnect: open the notebook again, run up to the config cell, run this cell again, then
train with `training.resume` switched on:

```python
cfg = load_config("configs/experiments/<run>.yaml", mode=MODE, **{"training.resume": True})
```

Only the checkpoints go to Drive. Code, data and results stay where they always are.
