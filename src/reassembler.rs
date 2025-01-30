use std::{collections::HashMap, fs::File, io};

use memmap2::{Mmap, MmapMut};

use crate::{header, utils::IntoU64Ext};

type Prefix = [u8; 8];
type Hash = [u8; blake3::OUT_LEN];

#[derive(Clone, Copy)]
struct Block {
    offset: usize,
    from_parity_file: bool, // will need to be changed once multiple input files are supported
}

enum HashBlocks {
    SingleBlock(Block),

    MultipleIdenticalBlocks {
        already_found_one: bool,
        blocks: Vec<Block>,
    }
}

enum PrefixData {
    Single(Hash, Block),

    // it's possible for multiple blocks have the same prefix
    MultipleBlocks(HashMap<Hash, HashBlocks>)
}

struct PrefixLookup {
    normal_len: usize,
    blocks: HashMap<Prefix, PrefixData>,

    // the last block of the input file can have a different length
    last: Option<(Prefix, Hash, Block, usize)>
}

fn add_to_lookup(lookup: &mut PrefixLookup, offset: usize, from_parity_file: bool, hash_prefix: &[u8]) {
    let prefix: Prefix = hash_prefix[..8].try_into().unwrap();
    let hash: Hash = hash_prefix[8..].try_into().unwrap();
    let block = Block { offset, from_parity_file };
    let entry = lookup.blocks.entry(prefix);

    use PrefixData::*;
    use HashBlocks::*;
    entry.and_modify(|data| {
        match data {
            &mut Single(hash_, block_) => {
                if hash != hash_ {
                    *data = MultipleBlocks(HashMap::from([(hash, SingleBlock(block)), (hash_, SingleBlock(block_))]));
                } else {
                    *data = MultipleBlocks(HashMap::from([(hash, MultipleIdenticalBlocks { already_found_one: false, blocks: vec![block, block_] })]));
                }
            },
            MultipleBlocks(map) => {
                map.entry(hash).and_modify(|blocks| {
                    match blocks {
                        &mut SingleBlock(block_) => {
                            *blocks = MultipleIdenticalBlocks { already_found_one: false, blocks: vec![block, block_] };
                        },
                        MultipleIdenticalBlocks { already_found_one: _, blocks } => {
                            blocks.push(block);
                        }
                    }
                }).or_insert(SingleBlock(block));
            },
        }
    }).or_insert(Single(hash, block));
}

fn prepare_lookup(header: header::Header, hashes_prefixes: &[u8]) -> (PrefixLookup, Option<(usize, &[u8])>) {
    assert!(hashes_prefixes.len() == (header.data_blocks + header.parity_blocks) * 40);
    let file_len: usize = header.file_len.try_into().unwrap();

    let mut lookup = PrefixLookup {
        normal_len: header.block_bytes,
        blocks: HashMap::new(),
        last: None
    };

    // if the last block is 8 bytes or less, the prefix contains the entire block padded with zeroes
    // return the offset and slice for the caller to write into the output file directly
    let mut tiny_last_block = None;

    for (i, hash_prefix) in hashes_prefixes[..header.data_blocks * 40].chunks_exact(40).enumerate() {
        let offset = i * header.block_bytes;
        if offset + header.block_bytes > file_len {
            assert_eq!(i, header.data_blocks - 1);
            if file_len - offset <= 8 {
                tiny_last_block = Some((offset, &hash_prefix[..file_len - offset]));
            } else {
                lookup.last = Some((hash_prefix[..8].try_into().unwrap(), hash_prefix[8..].try_into().unwrap(), Block { offset, from_parity_file: false }, file_len - offset));
            }
        } else {
            add_to_lookup(&mut lookup, offset, false, hash_prefix);
        }
    }

    for (i, hash_prefix) in hashes_prefixes[header.data_blocks * 40..].chunks_exact(40).enumerate() {
        add_to_lookup(&mut lookup, i * header.block_bytes, true, hash_prefix);
    }

    (lookup, tiny_last_block)
}

fn reassemble_from_file(file: &[u8], out_data: &mut [u8], out_parity: &mut [u8], prefix_lookup: &mut PrefixLookup) -> u64 {
    let mut offset = 0;
    let mut recovered = 0;

    let mut copy_block = |offset: usize, len: usize, block_info: Block| {
        let out = if block_info.from_parity_file { &mut *out_parity } else { &mut *out_data };
        let destination = &mut out[block_info.offset..block_info.offset + len];
        destination.copy_from_slice(&file[offset..offset + len]);
        recovered += 1;
    };

    let mut last_block_buf = if prefix_lookup.last.is_some() { vec![0_u8; prefix_lookup.normal_len] } else { vec![] };
    while offset + 8 <= file.len() {
        let block_prefix: Prefix = file[offset..offset + 8].try_into().unwrap();

        if let Some((prefix, hash, block, len)) = prefix_lookup.last {
            if block_prefix == prefix && offset + len <= file.len() {
                last_block_buf[..len].copy_from_slice(&file[offset..offset + len]);
                if blake3::hash(&last_block_buf) == hash {
                    copy_block(offset, len, block);
                    offset += len;
                    continue;
                }
            } else if offset + len > file.len() {
                break;
            }
        } else if offset + prefix_lookup.normal_len > file.len() {
            break;
        }

        let len = prefix_lookup.normal_len;
        if offset + len > file.len() {
            offset += 1;
            continue;
        }

        match prefix_lookup.blocks.get_mut(&block_prefix) {
            Some(PrefixData::Single(hash, block)) => {
                if blake3::hash(&file[offset..offset + len]) == *hash {
                    copy_block(offset, len, *block);
                    offset += len;
                    continue;
                }
            }
            Some(PrefixData::MultipleBlocks(blocks)) => {
                let hash = *blake3::hash(&file[offset..offset + len]).as_bytes();
                match blocks.get_mut(&hash) {
                    Some(HashBlocks::SingleBlock(block)) => {
                        copy_block(offset, len, *block);
                        offset += len;
                        continue;
                    }
                    Some(HashBlocks::MultipleIdenticalBlocks { already_found_one, blocks }) => {
                        if *already_found_one { continue; }
                        for block in blocks {
                            copy_block(offset, len, *block);
                        }
                        offset += len;
                        *already_found_one = true;
                        continue;
                    }
                    None => {}
                }
            }
            None => {}
        }

        offset += 1;
    }

    recovered
}

pub fn reassemble(data: &File, parity: &File, out_data: &mut File, out_parity: &mut File) -> io::Result<(u64, u64)> {
    let data_map = unsafe { Mmap::map(data)? };
    let parity_map = unsafe { Mmap::map(parity)? };

    let Some(header) = header::read_header(parity_map.as_ref()) else {
        return Err(io::Error::new(io::ErrorKind::Other, "Invalid header"));
    };

    let hashes_bytes = (header.data_blocks + header.parity_blocks) * 40;
    if header::HEADER_LEN + hashes_bytes > parity_map.len() {
        return Err(io::Error::new(io::ErrorKind::Other, "Metadata corrupted - file too short"));
    }

    let meta_hash = header::get_meta_hash(parity_map.as_ref());
    if meta_hash != *blake3::hash(&parity_map[header::HEADER_STRING.len() + blake3::OUT_LEN..header::HEADER_LEN + hashes_bytes]).as_bytes() {
        return Err(io::Error::new(io::ErrorKind::Other, "Metadata corrupted - hash invalid"));
    }

    let (mut prefix, tiny_last_block) = prepare_lookup(header, &parity_map[header::HEADER_LEN..header::HEADER_LEN + hashes_bytes]);

    out_data.set_len(header.file_len)?;
    out_parity.set_len(header::HEADER_LEN.as_u64() + hashes_bytes.as_u64() + header.parity_blocks.as_u64() * header.block_bytes.as_u64())?;
    let mut out_data_map = unsafe { MmapMut::map_mut(&*out_data)? };
    let mut out_parity_map = unsafe { MmapMut::map_mut(&*out_parity)? };

    if let Some((offset, last_block)) = tiny_last_block {
        out_data_map[offset..].copy_from_slice(last_block);
    }

    let (out_par_meta, out_par) = out_parity_map.split_at_mut(header::HEADER_LEN + hashes_bytes);
    out_par_meta.copy_from_slice(&parity_map[..header::HEADER_LEN + hashes_bytes]);

    let recovered_data = reassemble_from_file(data_map.as_ref(), out_data_map.as_mut(), out_par, &mut prefix);
    let recovered_parity = reassemble_from_file(&parity_map[header::HEADER_LEN + hashes_bytes..], out_data_map.as_mut(), out_par, &mut prefix);

    drop(out_data_map);
    drop(out_parity_map);
    out_data.sync_all()?;
    out_parity.sync_all()?;

    Ok((recovered_data, recovered_parity + tiny_last_block.is_some() as u64))
}
