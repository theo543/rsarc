#![warn(clippy::all)]

use std::{fs::OpenOptions, io::Write};

mod gf64;
mod polynomials;
mod encoder;

use encoder::{EncodeOptions, encode};

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

trait ShareModeExt {
    fn share_mode_lock(&mut self) -> &mut Self;
}

#[cfg(windows)]
impl ShareModeExt for OpenOptions {
    fn share_mode_lock(&mut self) -> &mut Self {
        // don't allow other processes to delete or write to the file
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 0x00000001;
        self.share_mode(FILE_SHARE_READ);
        self
    }
}

#[cfg(not(windows))]
impl ShareModeExt for OpenOptions {
    // not supported
    fn share_mode_lock(&mut self) -> &mut Self { self }
}

fn main() {
    gf64::check_cpu_support_for_carryless_multiply();
    let start_time = std::time::Instant::now();
    {
        // generate test file
        const TEST_FILE_SIZE: usize = 1024;
        const REPEATING_SEQUENCE: [u8; 33] = *b"\0\0\0\0 TEST FILE FOR RSARC ENCODER\n";
        let mut test_file = REPEATING_SEQUENCE.iter().cycle().take(TEST_FILE_SIZE).copied().collect::<Vec<_>>();
        for (mut i, chunk) in test_file.chunks_exact_mut(REPEATING_SEQUENCE.len()).enumerate() {
            for out in chunk.iter_mut().take(4).rev() {
                *out = (i % 10) as u8 + b'0';
                i /= 10;
            }
            assert_eq!(i, 0);
        }
        let mut input = OpenOptions::new().read(true).write(true).truncate(true).create(true).open("test.txt").unwrap();
        input.write_all(&test_file).unwrap();
    }
    let input = OpenOptions::new().read(true).share_mode_lock().open("test.txt").unwrap();
    let mut output = OpenOptions::new().read(true).write(true).create(true).truncate(true).share_mode_lock().open("out.rsarc").unwrap();
    const SIZE: usize = 32;
    encode(&input, &mut output, EncodeOptions {
        block_bytes: SIZE,
        parity_blocks: 14,
    });

    println!("Time: {:?}", start_time.elapsed());
    gf64::print_stats();
}
