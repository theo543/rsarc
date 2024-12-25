use std::fs::File;
use std::io::Write;
use std::sync::mpsc::{channel, Sender, Receiver};

use memmap2::MmapOptions;

use crate::gf64::{u64_as_gf64, u64_as_gf64_mut, GF64};
use crate::polynomials::{evaluate_poly, newton_interpolation};

type Buf = Box<[u64]>;

fn read_data(mapped: &[u8], recv_data_buffer: Receiver<Buf>, send_data: Sender<Option<(Buf, usize)>>, block_bytes: usize) {
    // Safety: u8 to u64 conversion is valid.
    let (mapped_prefix, mapped_u64, mapped_trailing) = unsafe { mapped.align_to::<u64>() };
    assert_eq!(mapped_prefix.len(), 0, "mmap should be 8-byte aligned");
    let block_size_in_u64 = block_bytes / 8;
    for code_idx in 0..block_size_in_u64 {
        let mut buf = recv_data_buffer.recv().unwrap();
        let mut buf_idx = 0;
        let mut mapped_idx = code_idx;
        while mapped_idx < mapped_u64.len() {
            buf[buf_idx] = mapped_u64[mapped_idx].to_le();
            buf_idx += 1;
            mapped_idx += block_size_in_u64;
        }
        if buf_idx < buf.len() && mapped_idx == mapped_u64.len() {
            let mut last = [0_u8; 8];
            last[..mapped_trailing.len()].copy_from_slice(mapped_trailing);
            buf[buf_idx] = u64::from_le_bytes(last);
            buf_idx += 1;
        }
        if buf_idx < buf.len() {
            buf[buf_idx + 1..].fill(0);
        }
        send_data.send(Some((buf, code_idx))).unwrap();
    }
    send_data.send(None).unwrap();
}

fn process_codes(recv_data: Receiver<Option<(Buf, usize)>>, return_data_buf: Sender<Buf>, recv_parity_buf: Receiver<Buf>, send_parity: Sender<Option<(Buf, usize)>>,
                 data_blocks: usize, parity_blocks: usize) {
    let mut poly = vec![GF64(0); data_blocks];
    let mut memory = vec![GF64(0); data_blocks * 2];
    loop {
        let Some((data, code_index)) = recv_data.recv().unwrap() else {
            send_parity.send(None).unwrap();
            return;
        };
        assert_eq!(data.len(), data_blocks);
        let mut parity_buf = recv_parity_buf.recv().unwrap();
        assert_eq!(parity_buf.len(), parity_blocks);
        newton_interpolation(u64_as_gf64(&data), None, &mut poly, &mut memory);
        let _ = return_data_buf.send(data); // will fail after reader thread shut down
        for (x, y) in u64_as_gf64_mut(&mut parity_buf).iter_mut().enumerate() {
            *y = evaluate_poly(&poly, GF64((data_blocks + x) as u64));
        }
        send_parity.send(Some((parity_buf, code_index))).unwrap();
    }
}

fn write_data(mapped: &mut [u8], recv_parity: Receiver<Option<(Buf, usize)>>, return_parity_buf: Sender<Buf>, parity_blocks: usize, block_bytes: usize) {
    // Safety: u8 to u64 conversion is valid.
    let (mapped_prefix, mapped, mapped_trailing) = unsafe { mapped.align_to_mut::<u64>() };
    assert_eq!(mapped_prefix.len(), 0, "mmap should be 8-byte aligned");
    assert_eq!(mapped_trailing.len(), 0, "there should be no trailing bytes in output mmap");
    let block_size_in_u64 = block_bytes / 8;
    let mut i: u64 = 0;
    let mut stdout = std::io::stdout().lock();
    loop {
        write!(stdout, "\x1b[100D\x1b[2K{} / {} = {}%", i, block_size_in_u64, (i as f64 / block_size_in_u64 as f64) * 100_f64).unwrap();
        stdout.flush().unwrap();
        i += 1;
        let Some((parity, code_index)) = recv_parity.recv().unwrap() else {
            stdout.write_all(b"\n").unwrap();
            return;
        };
        assert!(parity.len() == parity_blocks);
        let mut mapped_idx = code_index;
        for p in &parity {
            mapped[mapped_idx] = p.to_le();
            mapped_idx += block_size_in_u64;
        }
        let _ = return_parity_buf.send(parity); // will fail after processor thread shut down
    }
}

fn hash_blocks(mapped: &[u8], output: &mut [u8], block_bytes: usize) {
    assert_eq!(mapped.len().div_ceil(block_bytes), output.len() / 40,
              "amount of blocks in input should match output");
    const { assert!(blake3::OUT_LEN == 32, "blake3 hash length is 32 bytes"); }
    let (exact_blocks, final_block) = mapped.split_at(mapped.len() - mapped.len() % block_bytes);
    assert_eq!(exact_blocks.len() / block_bytes + usize::from(!final_block.is_empty()), output.len() / 40,
              "amount of whole blocks plus final partial block (if there is one) should match output");

    let mut output_chunks = output.chunks_exact_mut(40);
    for (block, out) in exact_blocks.chunks_exact(block_bytes).zip(&mut output_chunks) {
        out[..8].copy_from_slice(&block[..8]);
        out[8..40].copy_from_slice(blake3::hash(block).as_bytes());
    }

    if final_block.is_empty() { return; }

    // if there is a final partial block, there is one output chunk left in the iterator
    let out = output_chunks.next().unwrap(); 
    if final_block.len() >= 8 {
        // copy first 8 bytes of final block
        out[..8].copy_from_slice(&final_block[..8]);
    } else {
        // TODO: maybe if the final block is smaller than 8 bytes, it should be stored in the file header (once metadata redundancy is implemented)
        // pad final block with zero bytes to 8 bytes
        out[..8].fill(0);
        out[..final_block.len()].copy_from_slice(final_block);
    }
    out[8..40].copy_from_slice(blake3::hash(final_block).as_bytes());
}

pub struct EncodeOptions {
    pub block_bytes: usize,
    pub parity_blocks: usize,
}

const HEADER_STRING: [u8; 24] = *b"RSARC PARITY FILE\0\0\0\0\0\0\0";

fn format_header(opt: &EncodeOptions, data_blocks: usize, file_size: u64) -> Vec<u8> {
    HEADER_STRING.into_iter()
    .chain((opt.block_bytes as u64).to_le_bytes())
    .chain((data_blocks as u64).to_le_bytes())
    .chain((opt.parity_blocks as u64).to_le_bytes())
    .chain(file_size.to_le_bytes())
    .collect()
}

pub fn encode(input: &File, output: &mut File, opt: EncodeOptions) {
    assert!(opt.block_bytes % 8 == 0, "block bytes must be divisible by 8");
    let metadata = input.metadata().unwrap();
    let file_len_usize = usize::try_from(metadata.len()).unwrap_or_else(|_| panic!("file size must fit in {} bits", std::mem::size_of::<usize>() * 8));
    let data_blocks = file_len_usize.div_ceil(opt.block_bytes);
    println!("{data_blocks} data blocks");
    let header = format_header(&opt, data_blocks, metadata.len());
    let header_bytes = header.len();
    assert!(header_bytes % 8 == 0);
    // 32 bytes per hash, and first u64 from each block
    let hashes_bytes = 40 * (data_blocks + opt.parity_blocks);
    let parity_blocks_bytes = opt.block_bytes * opt.parity_blocks;
    let full_size_bytes = header_bytes + hashes_bytes + parity_blocks_bytes;
    output.set_len(full_size_bytes as u64).unwrap();
    output.write_all(&header).unwrap();

    // Safety:
    // Maps are of separate files, so no aliasing.
    // On Windows, the caller should use share_mode_lock to prevent other processes from modifying the input file.
    // Nothing can really be done on Unix about other processes modifying the input file (without copying it first or root access), as Unix has no mandatory file locking.
    // TODO: Use advisory file locking on Unix, to at least prevent processes which check for locks from modifying the file, and other instances of the encoder.
    // TODO: Implement a check of mtime to error out if the file was modified during encoding.
    // The mtime check would be after UB has occurred, but this UB is unlikely to cause the check to pass incorrectly.
    let input_map = unsafe { MmapOptions::new().map(input).unwrap() };
    let mut output_map = unsafe { MmapOptions::new().offset((header_bytes) as u64).map_mut(&*output).unwrap() };
    let (hashes_map, parity_map) = output_map.split_at_mut(hashes_bytes);

    std::thread::scope(|s| {
        // create channels and allocate buffers
        let data = channel::<Option<(Buf, usize)>>();
        let return_data = channel::<Buf>();
        let parity = channel::<Option<(Buf, usize)>>();
        let return_parity = channel::<Buf>();
        for _ in 0..5 {
            let data_buf = vec![0; data_blocks].into_boxed_slice();
            let parity_buf = vec![0; opt.parity_blocks].into_boxed_slice();
            return_data.0.send(data_buf).unwrap();
            return_parity.0.send(parity_buf).unwrap();
        }

        s.spawn(|| read_data(&input_map, return_data.1, data.0, opt.block_bytes));
        s.spawn(|| process_codes(data.1, return_data.0, return_parity.1, parity.0, data_blocks, opt.parity_blocks));
        s.spawn(|| write_data(parity_map, parity.1, return_parity.0, opt.parity_blocks, opt.block_bytes));
    });

    std::thread::scope(|s| {
        let (data_hashes, parity_hashes) = hashes_map.split_at_mut(40 * data_blocks);
        s.spawn(|| hash_blocks(&input_map, data_hashes, opt.block_bytes));
        s.spawn(|| hash_blocks(parity_map, parity_hashes, opt.block_bytes));
    });

    output_map.flush().unwrap();
    output.flush().unwrap();

}
