//! Print the validation result of each file as one JSON line (used to compare implementations).
fn main() {
    for path in std::env::args().skip(1) {
        let rep = lvf::validate(&path);
        let mut frames: Vec<(String, u32)> =
            rep.errors().iter().filter_map(|i| i.frame.map(|f| (i.code.clone(), f))).collect();
        frames.sort();
        frames.dedup();
        println!(
            "{}",
            serde_json::json!({"file": path, "ok": rep.ok(), "codes": rep.error_codes(), "frames": frames, "raps": rep.rap_frames.len()})
        );
    }
}
