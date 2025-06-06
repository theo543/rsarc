use std::{fs::File, io::{self, Read, Seek}};

use crate::{header::{get_meta_hash, read_header, Header, HEADER_LEN, HEADER_STRING}, utils::IntoU64Ext};

pub enum VerifyResult {
    Ok{data_errors: Vec<u64>, parity_errors: Vec<u64>, header: Header},
    MetadataCorrupted(String),
}

impl VerifyResult {
    pub fn report_corruption(&self, panic_on_corruption: bool) {
        match self {
            VerifyResult::Ok { data_errors: data_file, parity_errors: parity_file, .. } => {
                if !data_file.is_empty() { println!("Data file corrupted") }
                if !parity_file.is_empty() { println!("Parity file corrupted") }
                if !data_file.is_empty() && !parity_file.is_empty() { println!("No corruption detected") } else if panic_on_corruption { panic!() }
            }
            VerifyResult::MetadataCorrupted(msg) => { println!("Metadata corrupted: {}", msg); if panic_on_corruption { panic!() } }
        }
    }
}

pub fn verify_file(file: &mut File, hashes: &[u8], block_bytes: usize, blocks: usize, mut len: u64) -> io::Result<Vec<u64>> {
    assert_eq!(hashes.len(), blocks * 40);
    assert_eq!(block_bytes % 8, 0);

    let mut errors = vec![];
    let mut block_buf = vec![0; block_bytes];
    for (expected, i) in hashes.chunks_exact(40).zip(0_u64..) {
        let expected_block_header: [u8; 8] = expected[..8].try_into().unwrap();
        let expected_block_hash: [u8; 32] = expected[8..].try_into().unwrap();

        if len >= block_bytes.as_u64() {
            len -= block_bytes.as_u64();
            file.read_exact(&mut block_buf)?;
        } else {
            assert!(len > 0);
            file.read_exact(&mut block_buf[..len as usize])?;
            block_buf[len as usize..].fill(0);
            len = 0;
        }

        let block_header: [u8; 8] = block_buf[..8].try_into().unwrap();
        if block_header != expected_block_header || *blake3::hash(&block_buf).as_bytes() != expected_block_hash {
            errors.push(i);
        }
    }
    assert_eq!(len, 0);
    Ok(errors)
}

pub fn verify(input: &mut File, parity: &mut File) -> io::Result<VerifyResult> {
    parity.seek(io::SeekFrom::Start(0))?;

    let par_len = parity.metadata()?.len();
    if par_len < HEADER_LEN as u64 {
        return Ok(VerifyResult::MetadataCorrupted("header missing (file too short)".into()));
    }

    let mut metadata = vec![0; HEADER_LEN];
    parity.read_exact(&mut metadata)?;
    let Some(header) = read_header(&metadata) else {
        return Ok(VerifyResult::MetadataCorrupted("header invalid".into()));
    };

    let input_len = input.metadata()?.len();
    if header.file_len != input_len {
        return Ok(VerifyResult::MetadataCorrupted(format!("input file length mismatch: {input_len} bytes instead of {}", header.file_len)));
    }

    let hashes_bytes = (header.data_blocks + header.parity_blocks) * 40;
    let parity_bytes = header.parity_blocks.as_u64() * header.block_bytes.as_u64();
    let expected_bytes = HEADER_LEN.as_u64() + hashes_bytes.as_u64() + parity_bytes;
    if par_len != expected_bytes {
        return Ok(VerifyResult::MetadataCorrupted(format!("parity file length mismatch: {par_len} bytes instead of {expected_bytes}")));
    }

    metadata.resize(HEADER_LEN + hashes_bytes, 0);
    parity.read_exact(&mut metadata[HEADER_LEN..])?;

    if get_meta_hash(&metadata) != *blake3::hash(&metadata[HEADER_STRING.len() + blake3::OUT_LEN..]).as_bytes() {
        return Ok(VerifyResult::MetadataCorrupted("metadata corrupted, hash invalid".into()));
    }

    let (data_hashes, par_hashes) = metadata[HEADER_LEN..].split_at(header.data_blocks * 40);

    input.seek(io::SeekFrom::Start(0))?;
    let data_file = verify_file(input, data_hashes, header.block_bytes, header.data_blocks, input_len)?;

    parity.seek(io::SeekFrom::Start(HEADER_LEN as u64 + hashes_bytes as u64))?;
    let parity_file = verify_file(parity, par_hashes, header.block_bytes, header.parity_blocks, parity_bytes)?;

    Ok(VerifyResult::Ok{data_errors: data_file, parity_errors: parity_file, header})
}
