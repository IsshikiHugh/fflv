//! The built player (viewer/) is compiled into the binary by `include_dir!`, which cannot tell
//! cargo about the files it reads: rebuild whenever the directory changes (`npm run build`).

fn main() {
    println!("cargo:rerun-if-changed=viewer");
}
