use std::{sync::Mutex, time::Duration};

use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle, WeakProgressBar};

static BARS: Mutex<Vec<WeakProgressBar>> = Mutex::new(Vec::new());

pub fn register_panic_hook() {
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        for bar in BARS.lock().unwrap().iter().filter_map(|b| b.upgrade()) {
            bar.disable_steady_tick();
            bar.finish();
        }
        default_panic(panic_info)
    }));
}

pub fn progress(len: u64, msg: &'static str) -> ProgressBar {
    let bar = ProgressBar::new(len)
        .with_message(msg)
        .with_style(ProgressStyle::with_template("ETA {eta_precise:<20} {pos:>10}/{len:<10} {msg:<13} [{per_sec}]").unwrap())
        .with_finish(ProgressFinish::Abandon);
    BARS.lock().unwrap().push(bar.downgrade());
    bar
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

pub trait IncIfNotFinished {
    fn add(&self, inc: u64);
}

impl IncIfNotFinished for ProgressBar {
    fn add(&self, inc: u64) {
        if !self.is_finished() {
            self.inc(inc);
        }
    }
}
