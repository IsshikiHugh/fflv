//! The web player (player/) is compiled into the binary by `include_dir!("$OUT_DIR/viewer")`
//! (src/view.rs); this script fills that directory. `FFLV_PLAYER` chooses how:
//!
//! - unset or `auto`: in a checkout (player/ present) with npm on PATH, build the player whenever
//!   its sources change (`npm ci` first when player/node_modules is missing or older than
//!   package-lock.json). Without npm, use crates/fflv/viewer when it holds a build (a source
//!   package ships one; `npm run build` in player/ writes one); otherwise build fflv without the
//!   player (`fflv view` then says so).
//! - `build`: like `auto`, but npm is required (CI: never build without the player by accident).
//! - `prebuilt`: use crates/fflv/viewer as it is; it must hold a build (release builds, which
//!   build the player once and pass it to every platform).
//! - `skip`: build without the player.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The player's sources (relative to player/): a change rebuilds it.
const SOURCES: &[&str] = &["src", "index.html", "vite.config.ts", "tsconfig.json", "package.json", "package-lock.json"];

fn main() {
    println!("cargo:rerun-if-env-changed=FFLV_PLAYER");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("viewer");
    let player = manifest.join("../../player");
    let prebuilt = manifest.join("viewer");
    let cleared =
        fs::remove_dir_all(&out).or_else(|e| if e.kind() == io::ErrorKind::NotFound { Ok(()) } else { Err(e) });
    cleared.unwrap_or_else(|e| panic!("cannot clear {}: {e}", out.display()));
    fs::create_dir_all(&out).unwrap_or_else(|e| panic!("cannot create {}: {e}", out.display()));

    let mode = env::var("FFLV_PLAYER").unwrap_or_default();
    let result = match mode.as_str() {
        "" | "auto" => auto(&player, &prebuilt, &out, false),
        "build" => auto(&player, &prebuilt, &out, true),
        "prebuilt" => use_prebuilt(&prebuilt, &out),
        "skip" => Ok(()),
        other => Err(format!("FFLV_PLAYER={other:?}: expected auto, build, prebuilt or skip")),
    };
    if let Err(e) = result {
        panic!("{e}");
    }
}

fn auto(player: &Path, prebuilt: &Path, out: &Path, required: bool) -> Result<(), String> {
    let checkout = player.join("package.json").is_file();
    if checkout {
        for s in SOURCES {
            println!("cargo:rerun-if-changed={}", player.join(s).display());
        }
        if let Some(npm) = find_npm() {
            return build(&npm, player, out);
        }
        if required {
            return Err("FFLV_PLAYER=build: npm was not found (install Node.js ≥ 20)".into());
        }
        // let installing Node take effect without touching the player's sources
        println!("cargo:rerun-if-env-changed=PATH");
    } else if required {
        return Err(format!("FFLV_PLAYER=build: no player sources at {}", player.display()));
    }
    if prebuilt.join("index.html").is_file() {
        if checkout {
            warn("npm was not found: the web player is taken from crates/fflv/viewer, which may be older than player/src");
        }
        return use_prebuilt(prebuilt, out);
    }
    if checkout {
        warn(
            "npm was not found: fflv is built without the web player (`fflv view` will not work). Install Node.js \
             ≥ 20 and build again, or set FFLV_PLAYER=skip to silence this.",
        );
    } else {
        warn("no built player in crates/fflv/viewer: fflv is built without the web player (`fflv view` will not work)");
    }
    Ok(())
}

/// `vite build` into `out` (`npm ci` first when the dependencies are not installed, or were
/// installed from another package-lock.json).
fn build(npm: &str, player: &Path, out: &Path) -> Result<(), String> {
    // npm writes node_modules/.package-lock.json on every install: older than package-lock.json
    // means the lock file changed since (e.g. a pulled dependency update)
    let modified = |p: &str| fs::metadata(player.join(p)).and_then(|m| m.modified()).ok();
    let installed = modified("node_modules/.package-lock.json");
    if installed.is_none() || modified("package-lock.json") > installed {
        run(Command::new(npm).arg("ci").current_dir(player), "npm ci")?;
    }
    let mut vite = Command::new(npm);
    vite.args(["exec", "--", "vite", "build", "--emptyOutDir", "--outDir"]).arg(out).current_dir(player);
    run(&mut vite, "vite build")
}

fn run(cmd: &mut Command, what: &str) -> Result<(), String> {
    // npm's output goes to stderr: cargo reads the build script's stdout for its directives
    let status = cmd.stdout(io::stderr()).status().map_err(|e| format!("{what}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{what} failed in player/ ({status}, output above); FFLV_PLAYER=skip builds fflv without the player"
        ))
    }
}

fn find_npm() -> Option<String> {
    let npm = if cfg!(windows) { "npm.cmd" } else { "npm" };
    let ok = Command::new(npm).arg("--version").output().is_ok_and(|o| o.status.success());
    ok.then(|| npm.to_string())
}

fn use_prebuilt(prebuilt: &Path, out: &Path) -> Result<(), String> {
    println!("cargo:rerun-if-changed={}", prebuilt.display());
    if !prebuilt.join("index.html").is_file() {
        return Err(format!("no built player in {} (run `npm run build` in player/)", prebuilt.display()));
    }
    copy_dir(prebuilt, out).map_err(|e| format!("cannot copy {}: {e}", prebuilt.display()))
}

fn copy_dir(from: &Path, to: &Path) -> io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

fn warn(msg: &str) {
    println!("cargo:warning={msg}");
}
