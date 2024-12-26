#![warn(clippy::all)]

use std::{fs::OpenOptions, io::{self, BufWriter, Write}};

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
const TEXT: &str = " TEST FILE FOR RSARC ENCODER\n";

fn gen_test_file(file_size: u64) -> io::Result<()> {
    if std::fs::metadata(TEST_FILE_NAME).map(|m| m.len() == file_size).unwrap_or(false) { return Ok(()); }
    let test_file = OpenOptions::new().read(false).write(true).truncate(false).create(true).open(TEST_FILE_NAME)?;
    test_file.set_len(file_size)?;
    let mut test_file = BufWriter::new(test_file);

    println!("Generating test data...");

    let digits = (file_size / TEXT.len() as u64).ilog10() as usize + 1;
    let chunk_size = digits + TEXT.len();
    let chunks = file_size / chunk_size as u64;
    for i in 0..chunks {
        write!(test_file, "{i:0d$}{TEXT}", d = digits)?;
    }
    test_file.write_all(&vec![b'\n'; (file_size % chunk_size as u64) as usize])?;

    let test_file = test_file.into_inner()?;
    test_file.sync_all()?;
    Ok(())
}

fn main() {
    gf64::check_cpu_support_for_carryless_multiply();

    let input_size = 1024 * 1024 * 10;
    let block_bytes = 1024 * 100;
    let parity_blocks = 100;

    gen_test_file(input_size).expect("generating test file");

    let input = OpenOptions::new().read(true).share_mode_lock().open(TEST_FILE_NAME).unwrap();
    let mut output = OpenOptions::new().read(true).write(true).create(true).truncate(true).share_mode_lock().open(OUTPUT_FILE_NAME).unwrap();

    let start_time = std::time::Instant::now();
    encode(&input, &mut output, EncodeOptions{block_bytes, parity_blocks});
    println!("Time: {:?}", start_time.elapsed());

    gf64::print_stats();
}
