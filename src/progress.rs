use indicatif::{ProgressBar, ProgressStyle};

pub fn progress(len: u64, msg: &'static str) -> ProgressBar {
    ProgressBar::new(len).with_message(msg).with_style(ProgressStyle::with_template("ETA {eta_precise:<20} {pos:>10}/{len:<10} {msg:<10} [{per_sec}]").unwrap())
}

#[macro_export]
macro_rules! make_multiprogress {
    ($($progress:ident$(,)?)*) => {
        let multi = ::indicatif::MultiProgress::new();
        $(multi.add($progress.clone()); $progress.tick();)*
    };
}
