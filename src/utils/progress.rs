use std::time::Duration;

use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle};

pub fn progress(len: u64, msg: &'static str) -> ProgressBar {
    ProgressBar::new(len)
    .with_message(msg)
    .with_style(ProgressStyle::with_template("ETA {eta_precise:<20} {pos:>10}/{len:<10} {msg:<13} [{per_sec}]").unwrap())
    .with_finish(ProgressFinish::Abandon)
}

pub fn progress_usize(len: usize, msg: &'static str) -> ProgressBar {
    progress(len.try_into().unwrap(), msg)
}

pub fn make_multiprogress<const N: usize>(bars: [&ProgressBar; N]) {
    let mp = MultiProgress::new();
    for bar in bars {
        mp.add((*bar).clone());
        bar.enable_steady_tick(Duration::from_secs(1) / 60);
    }
    // dropping `mp` is OK because indicatif uses Arc internally
}
