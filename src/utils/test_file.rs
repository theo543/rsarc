use std::{fs::OpenOptions, io::{self, BufWriter, Write}};
use super::progress::progress;

const TEXT: &str = " TEST FILE FOR RSARC ENCODER\n";

pub fn gen_test_file(file_size: u64, name: &str) -> io::Result<()> {
    if std::fs::metadata(name).map(|m| m.len() == file_size).unwrap_or(false) { return Ok(()); }
    let test_file = OpenOptions::new().read(false).write(true).truncate(false).create(true).open(name)?;
    test_file.set_len(file_size)?;
    let mut test_file = BufWriter::new(test_file);

    let p = progress(file_size, "test file");
    let digits = (file_size / TEXT.len() as u64).ilog10() as usize + 1;
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
