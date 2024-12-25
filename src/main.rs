#![warn(clippy::all)]

use std::fs::OpenOptions;

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

const TEST_FILE_NAME: &str = "test.txt";
const OUTPUT_FILE_NAME: &str = "out.rsarc";
const TEST_FILE_SIZE: usize = 1024 * 1024 * 10;
const OPT: EncodeOptions = EncodeOptions {
    block_bytes: 1024 * 100,
    parity_blocks: 100,
};

fn gen_test_file() {
    const TEXT: &[u8] = b" TEST FILE FOR RSARC ENCODER\n";
    const DIGITS: usize = (TEST_FILE_SIZE / TEXT.len()).ilog10() as usize + 1;
    const CHUNK_SIZE: usize = DIGITS + TEXT.len();
    const TRAILING: usize = TEST_FILE_SIZE - TEST_FILE_SIZE % CHUNK_SIZE;

    let test_file = OpenOptions::new().read(true).write(true).truncate(false).create(true).open(TEST_FILE_NAME).unwrap();
    test_file.set_len(TEST_FILE_SIZE as u64).unwrap();
    let mut test_file = unsafe { memmap2::MmapMut::map_mut(&test_file).unwrap() };
    for (mut i, chunk) in test_file.chunks_exact_mut(CHUNK_SIZE).enumerate() {
        for out in chunk.iter_mut().take(DIGITS).rev() {
            *out = (i % 10) as u8 + b'0';
            i /= 10;
        }
        assert_eq!(i, 0);
        chunk[DIGITS..].copy_from_slice(TEXT);
    }
    test_file[TRAILING..].fill(b'\n');
    test_file.flush().unwrap();
}

fn main() {
    gf64::check_cpu_support_for_carryless_multiply();

    gen_test_file();

    let input = OpenOptions::new().read(true).share_mode_lock().open(TEST_FILE_NAME).unwrap();
    let mut output = OpenOptions::new().read(true).write(true).create(true).truncate(true).share_mode_lock().open(OUTPUT_FILE_NAME).unwrap();

    let start_time = std::time::Instant::now();
    encode(&input, &mut output, OPT);
    println!("Time: {:?}", start_time.elapsed());

    gf64::print_stats();
}
