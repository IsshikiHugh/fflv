---
name: fflv
description: >
  Use fflv / LVF layered video (.lvd) to debug vision or ML output: frames, predictions, masks and
  heat maps as synchronized layers, viewed and toggled in a web player. Installs it when missing.
  Triggers: "fflv", ".lvd", "layered video", "put these in layers", "debug video with layers".
---

# fflv

fflv writes, reads, edits and plays **layered videos** (`.lvd`, the LVF format): several video
layers, stills and audio in one file, strictly frame-synchronized, each layer shown or hidden in the
player. Typical use: dump inputs, predictions, masks and heat maps of a model as layers, then step
through them frame by frame.

This skill holds **no commands or versions of its own**. Installation, API and options change between
releases; read them from the sources below every time instead of from memory.

---

## Sources (the only places these facts live)

| What | Where |
|------|-------|
| Installing, supported systems, building from source | README → *Install*: https://github.com/IsshikiHugh/fflv#install (raw: https://raw.githubusercontent.com/IsshikiHugh/fflv/main/README.md) |
| What the installer runs, its settings | https://github.com/IsshikiHugh/fflv/blob/main/scripts/install.sh |
| Releases and what changed | https://github.com/IsshikiHugh/fflv/releases, `CHANGELOG.md` in the repo |
| Python API and command-line overview | README → *Usage* |
| Exact options of the **installed** version | `fflv --help`, `fflv <command> --help`, `python -c "import fflv; help(fflv.Writer)"` (also `fflv.open`, `fflv.set_layer`, …) |
| Viewing on a remote server | README → *On a remote Linux server* |
| File format, invariants, player behaviour | `LVF_SPEC.md` in the repo |

The README on `main` describes the latest release. For a pinned release read it at that tag:
`https://raw.githubusercontent.com/IsshikiHugh/fflv/<tag>/README.md`.

---

## Workflow

### Step 1: Find the environment and check for fflv

Use the Python environment the project uses (ask the user if it is not clear — never the system
Python by default). Then check:

```bash
fflv --version
python -c "import fflv; print(fflv.__version__)"
```

If both work, go to Step 3.

### Step 2: Install (only with the user's consent)

1. Tell the user fflv is missing and that you want to install it into that environment; wait for a yes.
2. Fetch the README and read its *Install* section (WebFetch the raw URL above, or `curl -fsSL` it).
3. Run exactly what it says, in the chosen environment. Do not reconstruct commands from memory
   or from this file.
4. Re-run the Step 1 checks. If the installer reports no wheel for the system, the README's
   supported-systems list and *From source* section say what to do; show them to the user.

### Step 3: Use it

1. Read README → *Usage* for the shape of the Python API and the commands.
2. Before writing code or command lines, confirm the details against the installed version with
   `--help` and `help(...)` — they are authoritative for what is installed.
3. For format questions (layer rules, alpha, timing, what the validator checks), read `LVF_SPEC.md`.
4. To look at a result: on a desktop, `fflv view FILE`; on a server, follow README → *On a remote
   Linux server* and give the user the port-forwarding command and the URL.

### Step 4: Verify

- `fflv check FILE` (or `fflv.validate(path).ok`) passes for every file you wrote.
- `fflv info FILE` lists the layers you intended, with the expected frame ranges.

---

## Error Handling

| Error | Detection | Action |
|-------|-----------|--------|
| No wheel for this system | installer fails with "no wheel for this system" / pip "No matching distribution" | Read README supported systems and *From source*; tell the user which applies |
| An option or function is missing | `--help` / `help()` differs from the README | The installed version differs from `main`; follow the installed version, or read the README at its tag |
| Importing media, audio or writing video fails | error mentions FFmpeg | README → *Install* says which features need the FFmpeg command line; ask before installing it |
| Player shows nothing on a server | no desktop browser there | Use README → *On a remote Linux server* |

---

## Quality Checklist

- [ ] Commands and versions came from the sources above, not from memory or this file.
- [ ] Installed into the project's environment, with the user's consent.
- [ ] Every written `.lvd` passes `fflv check`.
- [ ] The user got a way to view the result (local `fflv view`, or port forwarding on a server).
