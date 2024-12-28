use std::fs::File;
use std::io::{Read, Seek, Write};
use std::ops::Div;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, RwLock};
use crossbeam_channel::{unbounded as channel, Sender, Receiver};

use indicatif::ProgressBar;
use memmap2::MmapOptions;
use positioned_io::{RandomAccessFile, ReadAt};

use crate::gf64::{u64_as_gf64, u64_as_gf64_mut, GF64};
use crate::make_multiprogress;
use crate::polynomials::{evaluate_poly, newton_interpolation};
use crate::progress::progress;

trait IntoU64Ext {
    fn as_u64(&self) -> u64;
}

impl IntoU64Ext for usize {
    fn as_u64(&self) -> u64 { u64::try_from(*self).unwrap() }
}

// Data symbols from multiple codes are read in at once, to try to minimize read overhead
// RwLock is only used as a thread-safe RefCell, not for blocking
type BigBuf = Arc<RwLock<Box<[u64]>>>;

// Writes seem to be fast enough without BigBuf
type Buf = Box<[u64]>;

fn u64_buf_as_u8(buf: &mut [u64]) -> &mut [u8] {
    // u64 to u8 conversion is safe
    let (head, mid, tail) = unsafe { buf.align_to_mut::<u8>() };
    // u8 has alignment 1
    assert!(head.is_empty());
    assert!(tail.is_empty());
    mid
}

struct ReadDataMsg {
    buf_rw: BigBuf,
    first_code: usize,
    codes: usize // last message may have less codes
}

#[allow(clippy::too_many_arguments)]
fn read_data(input: &mut File, recv_buf: &Receiver<BigBuf>, send_buf: &Sender<Option<ReadDataMsg>>, block_symbols: usize, codes_per_full_read: usize, file_size: u64,
             progress: &ProgressBar, single_progress: &ProgressBar) {

    let file = RandomAccessFile::try_new(input.try_clone().unwrap()).unwrap();

    for code_idx in (0..block_symbols).step_by(codes_per_full_read) {
        let buf_rw = recv_buf.recv().unwrap();
        let mut buf_guard = buf_rw.try_write().unwrap();
        let buf = &mut *buf_guard;
        assert_eq!(buf.len().as_u64(), codes_per_full_read.as_u64() * file_size.div_ceil(block_symbols.as_u64() * 8));
        let buf_u8 = u64_buf_as_u8(buf);

        let mut buf_idx = 0;
        let mut file_idx = code_idx.as_u64();
        let codes_in_read = codes_per_full_read.min(block_symbols - code_idx); // last read may be smaller
        let bytes_in_read = codes_in_read * 8;

        single_progress.reset();
        while file_idx + bytes_in_read.as_u64() <= file_size {
            file.read_exact_at(file_idx, &mut buf_u8[buf_idx..buf_idx + bytes_in_read]).unwrap();
            buf_idx += bytes_in_read;
            file_idx += block_symbols.as_u64() * 8;
            single_progress.inc(codes_in_read.as_u64());
        }
        if buf_idx < buf_u8.len() {
            assert_eq!(code_idx, block_symbols - 1); // should only happen on last iteration
            if file_idx < file_size {
                let remaining_in_file = usize::try_from(file_size - file_idx).unwrap();
                file.read_exact_at(file_idx, &mut buf_u8[buf_idx..buf_idx + remaining_in_file]).unwrap();
                buf_u8[buf_idx + remaining_in_file..buf_idx + bytes_in_read].fill(0);
            } else {
                buf_u8[buf_idx..].fill(0);
            }
        } else {
            assert_eq!(codes_in_read, codes_per_full_read);
            assert_eq!(buf_idx, buf_u8.len());
        }
        single_progress.finish();

        for x in buf.iter_mut() {
            *x = x.to_le();
        }

        drop(buf_guard);
        send_buf.send(Some(ReadDataMsg{buf_rw, first_code: code_idx, codes: codes_in_read})).unwrap();
        progress.inc(codes_in_read.as_u64());
    }

    send_buf.send(None).unwrap(); // shut down read_to_processors
}

struct ProcessTaskMsg {
    buf_rw: BigBuf,
    reader_count: Arc<AtomicUsize>,
    offset: usize,
    code_idx: usize,
    codes: usize
}

// connects reader to processor worker threads
fn read_to_processors(recv_data: &Receiver<Option<ReadDataMsg>>, to_processors: &Sender<Option<ProcessTaskMsg>>) {
    while let Some(ReadDataMsg{buf_rw, first_code, codes}) = recv_data.recv().unwrap() {
        let reader_count = Arc::new(AtomicUsize::new(codes));
        for offset in 0..codes {
            to_processors.send(Some(ProcessTaskMsg{buf_rw: buf_rw.clone(), reader_count: reader_count.clone(), offset, code_idx: offset + first_code, codes})).unwrap();
        }
    }
}

fn process_codes(recv_parity_buf: &Receiver<Buf>, send_parity: &Sender<Option<(Buf, usize)>>,
                 recv_data: &Receiver<Option<ProcessTaskMsg>>, return_data_buf: &Sender<BigBuf>, data_blocks: usize, parity_blocks: usize,
                 progress: &ProgressBar) {

    let mut memory = vec![GF64(0); data_blocks * 3];
    let (poly, memory) = memory.split_at_mut(data_blocks);

    while let Some(ProcessTaskMsg{buf_rw, reader_count, offset, code_idx, codes}) = recv_data.recv().unwrap() {
        let locked_buf = buf_rw.try_read().unwrap();
        newton_interpolation(u64_as_gf64(&locked_buf), offset, codes, None, poly, memory);
        drop(locked_buf);
        let is_last = reader_count.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) == 1;
        if is_last {
            assert!(buf_rw.try_write().is_ok());
            return_data_buf.send(buf_rw).unwrap();
        }
        let mut parity_buf = recv_parity_buf.recv().unwrap();
        assert_eq!(parity_buf.len(), parity_blocks);
        for (x, y) in u64_as_gf64_mut(&mut parity_buf).iter_mut().enumerate() {
            *y = evaluate_poly(poly, GF64((data_blocks + x).as_u64()));
        }
        send_parity.send(Some((parity_buf, code_idx))).unwrap();
        progress.inc(1);
    }
}

fn write_data(mapped: &mut [u8], recv_parity: &Receiver<Option<(Buf, usize)>>, return_parity_buf: &Sender<Buf>, parity_blocks: usize, block_symbols: usize, progress: &ProgressBar) {
    assert_eq!(mapped.len(), block_symbols * 8 * parity_blocks);

    // u8 to u64 conversion is safe
    let mapped = unsafe {
        let (mapped_prefix, mapped, mapped_trailing) = mapped.align_to_mut::<u64>();
        assert!(mapped_prefix.is_empty(), "mmap should be 8-byte aligned");
        assert!(mapped_trailing.is_empty(), "there should be no trailing bytes in output mmap");
        mapped
    };

    loop {
        let Some((parity, code_index)) = recv_parity.recv().unwrap() else { return; };
        assert!(parity.len() == parity_blocks);
        let mut mapped_idx = code_index;
        for p in &parity {
            mapped[mapped_idx] = p.to_le();
            mapped_idx += block_symbols;
        }
        return_parity_buf.send(parity).unwrap();
        progress.inc(1);
    }
}

pub struct EncodeOptions {
    pub block_bytes: usize,
    pub parity_blocks: usize,
}

const HEADER_STRING: [u8; 24] = *b"RSARC PARITY FILE\0\0\0\0\0\0\0";

fn format_header(opt: &EncodeOptions, data_blocks: usize, file_size: u64) -> Vec<u8> {
    HEADER_STRING.into_iter()
    .chain([0_u8; blake3::OUT_LEN])
    .chain(opt.block_bytes.as_u64().to_le_bytes())
    .chain(data_blocks.as_u64().to_le_bytes())
    .chain(opt.parity_blocks.as_u64().to_le_bytes())
    .chain(file_size.to_le_bytes())
    .collect()
}

pub fn encode(input: &mut File, output: &mut File, opt: EncodeOptions) {

    assert!(opt.block_bytes % 8 == 0, "block bytes must be divisible by 8");
    let block_symbols = opt.block_bytes / 8;
    let file_len = input.metadata().unwrap().len();

    let data_blocks = usize::try_from(file_len.div_ceil(opt.block_bytes.as_u64())).expect("data blocks number must fit in usize");
    println!("{data_blocks} data blocks");

    let header = format_header(&opt, data_blocks, file_len);
    let header_bytes = header.len();
    assert!(header_bytes % 8 == 0);

    // 32 bytes per hash, and first u64 from each block
    let hashes_bytes = 40 * (data_blocks + opt.parity_blocks);
    assert!(hashes_bytes % 8 == 0);

    let parity_blocks_bytes = opt.block_bytes * opt.parity_blocks;

    output.set_len((header_bytes + hashes_bytes + parity_blocks_bytes).as_u64()).unwrap();
    output.write_all(&header).unwrap();

    let mut output_map = unsafe { MmapOptions::new().map_mut(&*output).unwrap() };
    let (hashes_map, parity_map) = output_map[header_bytes..].split_at_mut(hashes_bytes);
    assert_eq!(hashes_map.len(), hashes_bytes);
    assert_eq!(parity_map.len(), parity_blocks_bytes);

    let cpus = num_cpus::get();

    let (buf_amount, symbols_read_per_code) = 'a: {
        const MEM_USE_PERCENT: u64 = 90;
        const MAX_SYM_PER_CODE: u64 = 16284;
        let available_memory = {
            use sysinfo::{System, RefreshKind, MemoryRefreshKind};
            let sys_mem = System::new_with_specifics(RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram())).available_memory();
            let mem = (sys_mem * MEM_USE_PERCENT) / 100;
            mem.checked_sub((data_blocks * 3 * 8 * cpus).as_u64()) // substract memory used by processor threads (interpolation memory, polynomial)
               .and_then(|x| x.checked_sub((opt.parity_blocks * 8).as_u64())) // substract memory used for output buffers
               .unwrap_or(0)
        };

        // Maybe use a direct mmap in this case?
        if available_memory >= file_len { break 'a (1, block_symbols); }

        let sym_per_code = available_memory / data_blocks.as_u64() * 8 * 2;
        if sym_per_code > MAX_SYM_PER_CODE { ((sym_per_code / MAX_SYM_PER_CODE).clamp(1, 2) as usize, MAX_SYM_PER_CODE as usize) } else { ((sym_per_code / 2) as usize, 2) }
    };
    println!("{symbols_read_per_code} {buf_amount}");
    let mem_per_buf = symbols_read_per_code * data_blocks;

    {
        // create channels
        let data_to_adapter = channel();
        let adapter_to_processors = channel();
        let return_data = channel();
        let parity = channel();
        let return_parity = channel();

        // allocate BigBuf for reading
        for _ in 0..buf_amount {
            return_data.0.send(Arc::new(RwLock::new(vec![0_u64; mem_per_buf].into_boxed_slice()))).unwrap();
        }

        // allocate Buf for writing
        for _ in 0..cpus * 2 {
            return_parity.0.send(vec![0; opt.parity_blocks].into_boxed_slice()).unwrap();
        }

        let single = progress(data_blocks.as_u64() * symbols_read_per_code.as_u64(), "one read pass");
        let pb = |msg| progress(opt.block_bytes.as_u64() / 8, msg);
        let read_prog = pb("read");
        let process_prog = pb("process");
        let write_prog = pb("write");
        make_multiprogress!(single, read_prog, process_prog, write_prog);

        // spawn threads
        std::thread::scope(|s| {
            let reader = s.spawn(|| read_data(input, &return_data.1, &data_to_adapter.0, block_symbols, symbols_read_per_code, file_len, &read_prog, &single));
            let read_to_processors = s.spawn(|| read_to_processors(&data_to_adapter.1, &adapter_to_processors.0));
            let processor_threads = (0..cpus).map(|_| {
                s.spawn(|| process_codes(&return_parity.1, &parity.0, &adapter_to_processors.1, &return_data.0, data_blocks, opt.parity_blocks, &process_prog))
            }).collect::<Vec<_>>();
            let writer = s.spawn(|| write_data(parity_map, &parity.1, &return_parity.0, opt.parity_blocks, block_symbols, &write_prog));

            // wait for all data to be read
            reader.join().unwrap();
            read_to_processors.join().unwrap();

            // stop processor threads
            for _ in 0..cpus {
                adapter_to_processors.0.send(None).unwrap();
            }
            for t in processor_threads {
                t.join().unwrap();
            }

            // stop writer thread
            parity.0.send(None).unwrap();
            writer.join().unwrap();
        });

    }

    {
        let (data_hashes, parity_hashes) = hashes_map.split_at_mut(40 * data_blocks);
        let data_prog = progress(data_hashes.len().div(40).as_u64(), "hash data");
        let par_prog = progress(parity_hashes.len().div(40).as_u64(), "hash parity");
        make_multiprogress!(data_prog, par_prog);
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut buf = vec![0_u8; opt.block_bytes];
                let last_block_rem = (file_len % opt.block_bytes.as_u64()) as usize;
                input.seek(std::io::SeekFrom::Start(0)).unwrap();
                let last_hash = data_hashes.len() / 40 - 1;
                for (i, out) in data_hashes.chunks_exact_mut(40).enumerate() {
                    if i == last_hash && last_block_rem != 0 {
                        buf.fill(0);
                        input.read_exact(&mut buf[0..last_block_rem]).unwrap();
                    } else {
                        input.read_exact(&mut buf).unwrap();
                    }
                    out[..8].copy_from_slice(&buf[..8]);
                    out[8..40].copy_from_slice(blake3::hash(&buf).as_bytes());
                    data_prog.inc(1);
                }
            });
            s.spawn(|| {
                for (block, out) in parity_map.chunks_exact(opt.block_bytes).zip(parity_hashes.chunks_exact_mut(40)) {
                    out[..8].copy_from_slice(&block[..8]);
                    out[8..40].copy_from_slice(blake3::hash(block).as_bytes());
                    par_prog.inc(1);
                }
            });
        });
    }

    // hash all the metadata expect the header string and the placeholder meta-hash zeroes
    let metadata_map = &mut output_map[HEADER_STRING.len()..header_bytes + hashes_bytes];
    let metadata_hash = blake3::hash(&metadata_map[blake3::OUT_LEN..]);
    output_map[..blake3::OUT_LEN].copy_from_slice(metadata_hash.as_bytes());

    println!("Flush output...");
    output_map.flush().unwrap();
    drop(output_map);
    output.flush().unwrap();

}
