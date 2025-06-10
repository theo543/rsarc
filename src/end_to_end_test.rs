use std::{fs::OpenOptions, io::{self, BufWriter, Write}};

use positioned_io::WriteAt;

use crate::{encoder::{encode, repair, EncodeOptions}, header::HEADER_LEN, utils::{progress::progress, IntoU64Ext, ShareModeExt}, verifier::{verify, VerifyResult}};

const TEXT: &str = " TEST FILE FOR RSARC ENCODER\n";

pub fn gen_test_file(file_size: u64, name: &str) -> std::io::Result<()> {
    let test_file = OpenOptions::new().read(false).write(true).truncate(false).create(true).open(name)?;
    test_file.set_len(file_size)?;
    let mut test_file = BufWriter::new(test_file);

    let p = progress(file_size, "test file");
    let digits = file_size.div_ceil(TEXT.len() as u64).ilog10() as usize + 1;
    let chunk_size = digits + TEXT.len();
    let chunks = file_size / chunk_size as u64;
    for i in 0..chunks {
        write!(test_file, "{i:0d$}{TEXT}", d = digits)?;
        p.inc(chunk_size as u64);
    }
    test_file.write_all(&vec![b'\n'; (file_size % chunk_size as u64) as usize])?;

    let test_file = test_file.into_inner()?;
    test_file.sync_all()?;
    p.finish_and_clear();
    Ok(())
}

fn hash_file(file: &str) -> io::Result<blake3::Hash> {
    Ok(blake3::hash(std::fs::read(file)?.as_slice()))
}

pub fn test() -> io::Result<()> {
    const TEST_FILE_NAME: &str = "test.txt";
    const OUTPUT_FILE_NAME: &str = "out.rsarc";

    println!("Current RNG seed: {}", fastrand::get_seed());

    let input_size = fastrand::u64(1024 * 1024..1024 * 1024 * 2);
    let block_bytes = fastrand::usize(1024..4096).next_multiple_of(8);
    let parity_blocks = fastrand::usize(200..300);

    gen_test_file(input_size, TEST_FILE_NAME).expect("generating test file");

    let test_file_hash = hash_file(TEST_FILE_NAME)?;

    let mut input = OpenOptions::new().read(true).share_mode_lock().open(TEST_FILE_NAME)?;
    let mut output = OpenOptions::new().read(true).write(true).create(true).truncate(true).share_mode_lock().open(OUTPUT_FILE_NAME)?;

    let start_time = std::time::Instant::now();
    encode(&mut input, &mut output, EncodeOptions{block_bytes, parity_blocks})?;
    println!("Time: {:?}", start_time.elapsed());

    let output_file_hash = hash_file(OUTPUT_FILE_NAME)?;

    verify(&mut input, &mut output)?.report_corruption(true);

    drop(input);
    let mut input = OpenOptions::new().read(true).write(true).share_mode_lock().open(TEST_FILE_NAME)?;

    let input_len = input.metadata()?.len();
    let output_len = output.metadata()?.len();
    let metadata_bytes = HEADER_LEN.as_u64() + (input_len.div_ceil(block_bytes.as_u64()) + parity_blocks.as_u64()) * 40;

    for _ in 0..parity_blocks.div_ceil(2) {
        let corrupt = fastrand::u64(0..input_len);
        input.write_all_at(corrupt, &[0; 1])?;
    }

    for _ in 0..parity_blocks / 2 {
        let corrupt = fastrand::u64(metadata_bytes..output_len);
        output.write_all_at(corrupt, &[0; 1])?;
    }

    let corruption = verify(&mut input, &mut output)?;
    corruption.report_corruption(false);
    let VerifyResult::Ok{data_errors: data_file, parity_errors: parity_file, header} = corruption else { panic!("metadata should not be corrupted") };
    repair(&mut input, &mut output, header, data_file, parity_file)?;

    verify(&mut input, &mut output)?.report_corruption(true);

    assert_eq!(test_file_hash, hash_file(TEST_FILE_NAME)?);
    assert_eq!(output_file_hash, hash_file(OUTPUT_FILE_NAME)?);
    Ok(())
}
