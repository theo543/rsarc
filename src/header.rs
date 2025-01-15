use crate::utils::IntoU64Ext;

pub const HEADER_STRING: [u8; 24] = *b"RSARC PARITY FILE\0\0\0\0\0\0\0";
pub const HEADER_LEN: usize = HEADER_STRING.len() + blake3::OUT_LEN + 8 + 8 + 8 + 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub block_bytes: usize,
    pub data_blocks: usize,
    pub parity_blocks: usize,
    pub file_len: u64,
}

pub fn format_header(header: Header) -> [u8; HEADER_LEN] {
    const { assert!(HEADER_LEN % 8 == 0) }
    HEADER_STRING.into_iter()
    .chain([0_u8; blake3::OUT_LEN])
    .chain(header.block_bytes.as_u64().to_le_bytes())
    .chain(header.data_blocks.as_u64().to_le_bytes())
    .chain(header.parity_blocks.as_u64().to_le_bytes())
    .chain(header.file_len.to_le_bytes())
    .collect::<Vec<_>>().try_into().unwrap()
}

pub fn read_header(header: &[u8]) -> Option<Header> {
    if header[..HEADER_STRING.len()] != HEADER_STRING { return None }

    let mut pos = HEADER_STRING.len() + blake3::OUT_LEN;
    let mut read_u64 = || {
        pos += 8;
        u64::from_le_bytes(header[pos - 8..pos].try_into().unwrap())
    };
    let mut read_usize = || {
        let val = read_u64();
        usize::try_from(val).unwrap_or_else(|_| panic!("{val} doesn't fit in {} bits", usize::BITS))
    };

    Some(Header {
        block_bytes: read_usize(),
        data_blocks: read_usize(),
        parity_blocks: read_usize(),
        file_len: read_u64(),
    })
}

pub fn set_meta_hash(header: &mut [u8], hash: [u8; blake3::OUT_LEN]) {
    header[HEADER_STRING.len()..HEADER_STRING.len() + blake3::OUT_LEN].copy_from_slice(&hash);
}

pub fn get_meta_hash(header: &[u8]) -> [u8; blake3::OUT_LEN] {
    header[HEADER_STRING.len()..HEADER_STRING.len() + blake3::OUT_LEN].try_into().unwrap()
}

#[test]
fn header_encode_decode() {
    for _ in 0..10 {
        let header = Header {
            block_bytes: fastrand::usize(0..),
            data_blocks: fastrand::usize(0..),
            parity_blocks: fastrand::usize(0..),
            file_len: fastrand::u64(0..),
        };
        let encoded = format_header(header);
        let decoded = read_header(&encoded).unwrap();
        assert_eq!(header, decoded);
    }
}

#[test]
fn bad_header() {
    assert!(read_header(&[0; HEADER_LEN]).is_none());
}

#[test]
fn get_set_hash() {
    let hash: [u8; blake3::OUT_LEN] = std::array::from_fn(|_| fastrand::u8(0..));
    let mut header = [0; HEADER_LEN];
    set_meta_hash(&mut header, hash);
    assert_eq!(get_meta_hash(&header), hash);
}
