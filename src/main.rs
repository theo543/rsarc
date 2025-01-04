#![warn(clippy::all)]

mod math;
mod utils;
mod encoder;

use std::fs::OpenOptions;

use encoder::{EncodeOptions, encode};
use utils::{gen_test_file, ShareModeExt};

/*
Block = contiguous, contains one symbol from each RS code.
RS-code = not contiguous, spread across all blocks.
Loss of one block cause one erasure per RS-code.
Metadata contains hash of each block to detect corruption.
Metadata also contains a few bytes from the beginning of each block to allow scanning the file for blocks to locate them,
to be able to locate blocks in case of deletion, insertion, or reordering of data (instead of simple bit-flipping).


Format structure:
For now, parity will be in a separate file from data (like with PAR2 files).
Later: Implement single-file archive, multiple input files, optional separate parity file.
Parity file will contain a header with block metadata, then parity blocks follow.
Later: Implement metadata redundancy to recover from header corruption by interleaving meta-parity blocks with parity blocks.
       This will require some magic string prepended to each block, i.e. "HEADERRECOVERY", and a hash, to locate meta-parity blocks.
*/

const TEST_FILE_NAME: &str = "test.txt";
const OUTPUT_FILE_NAME: &str = "out.rsarc";

fn main() {
    math::gf64::check_cpu_support_for_carryless_multiply();

    let input_size = 1024 * 1024 * 1024 * 15;
    let block_bytes = 1024 * 1024 * 150;
    let parity_blocks = 50;

    gen_test_file(input_size, TEST_FILE_NAME).expect("generating test file");

    let mut input = OpenOptions::new().read(true).share_mode_lock().open(TEST_FILE_NAME).unwrap();
    let mut output = OpenOptions::new().read(true).write(true).create(true).truncate(true).share_mode_lock().open(OUTPUT_FILE_NAME).unwrap();

    let start_time = std::time::Instant::now();
    encode(&mut input, &mut output, EncodeOptions{block_bytes, parity_blocks});
    println!("Time: {:?}", start_time.elapsed());

    math::gf64::print_stats();
}
