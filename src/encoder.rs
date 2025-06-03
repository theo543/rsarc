#![allow(clippy::too_many_arguments)]

use std::fs::File;
use std::io::{self, Read, Seek, Write};
use std::iter::StepBy;
use std::ops::Range;
use std::slice::Iter;
use std::sync::{atomic::{AtomicUsize, Ordering}, Arc, RwLock};
use crossbeam_channel::{unbounded as channel, Sender, Receiver};

use indicatif::ProgressBar;
use memmap2::MmapOptions;
use positioned_io::{RandomAccessFile, ReadAt};

use crate::header::{format_header, set_meta_hash, Header, HEADER_LEN, HEADER_STRING};
use crate::math::gf64::{u64_as_gf64, u64_as_gf64_mut, GF64};
use crate::math::polynomials::{evaluate_poly, newton_interpolation};
use crate::math::novelpoly::{formal_derivative, forward_transform, inverse_transform, DerivativeFactors, TransformFactors};
use crate::utils::progress::{progress_usize as progress, make_multiprogress};
use crate::utils::{IntoU64Ext, IntoUSizeExt};

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
    buf: BigBuf,
    first_code: usize,
    codes: usize // last message may have less codes
}

// Iterator for position of blocks in file. Can be none, a range, or a list of indices.
#[derive(Clone)]
enum BlockIndices<'a> {
    Empty,
    Range(StepBy<Range<u64>>),
    Slice(Iter<'a, u64>),
    //Range{initial_idx: u64, exclusive_end_idx: u64, step: u64},
    //Some(Vec<u64>),
}

impl Iterator for BlockIndices<'_> {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            BlockIndices::Empty => None,
            BlockIndices::Range(iter) => iter.next(),
            BlockIndices::Slice(iter) => iter.next().copied(),
        }
    }
}

impl From<StepBy<Range<u64>>> for BlockIndices<'_> {
    fn from(iter: StepBy<Range<u64>>) -> Self {
        BlockIndices::Range(iter)
    }
}

impl<'a> From<Iter<'a, u64>> for BlockIndices<'a> {
    fn from(iter: Iter<'a, u64>) -> Self {
        BlockIndices::Slice(iter)
    }
}

fn read_symbols_from_file(file: &RandomAccessFile, file_len: u64, offset: u64, indices: BlockIndices, symbols_bytes: usize, buf_idx: &mut usize, buf: &mut [u8]) -> io::Result<bool> {
    let mut incomplete_read_occurred = false;
    for base_file_idx in indices {
        let file_idx = base_file_idx + offset;
        if file_idx + symbols_bytes.as_u64() <= file_len {
            file.read_exact_at(file_idx, &mut buf[*buf_idx..*buf_idx + symbols_bytes])?;
        } else {
            assert!(!incomplete_read_occurred, "an incomplete read should only happen once at the end of the file");
            incomplete_read_occurred = true;

            if file_idx < file_len {
                let remaining_in_file = usize::try_from(file_len - file_idx).unwrap();
                file.read_exact_at(file_idx, &mut buf[*buf_idx..*buf_idx + remaining_in_file])?;
                buf[*buf_idx + remaining_in_file..*buf_idx + symbols_bytes].fill(0);
            } else {
                buf[*buf_idx..*buf_idx + symbols_bytes].fill(0);
            }
        }
        *buf_idx += symbols_bytes;
    }
    Ok(incomplete_read_occurred)
}

fn read_data(recv_buf: &Receiver<BigBuf>, send_buf: &Sender<Option<ReadDataMsg>>,
             block_symbols: usize, codes_per_full_read: usize,
             input: &mut File, output: &mut File,
             input_file_size: u64, output_file_size: u64,
             good_indices: (BlockIndices, BlockIndices),
             progress: &ProgressBar, single_progress: &ProgressBar) -> io::Result<()> {

    let input = RandomAccessFile::try_new(input.try_clone()?)?;
    let output = RandomAccessFile::try_new(output.try_clone()?)?;

    for code_idx in (0..block_symbols).step_by(codes_per_full_read) {
        let buf = recv_buf.recv().unwrap();
        let mut buf_lock = buf.try_write().unwrap();
        let buf_u8 = u64_buf_as_u8(&mut buf_lock);

        let codes_in_read = codes_per_full_read.min(block_symbols - code_idx); // last read may be smaller
        let bytes_in_read = codes_in_read * 8;

        single_progress.reset();
        let mut buf_idx = 0;
        let offset = code_idx.as_u64() * 8;
        read_symbols_from_file(&input, input_file_size, offset, good_indices.0.clone(), bytes_in_read, &mut buf_idx, buf_u8)?;
        let incomplete_output_read = read_symbols_from_file(&output, output_file_size, offset, good_indices.1.clone(), bytes_in_read, &mut buf_idx, buf_u8)?;
        assert!(!incomplete_output_read, "output file contains only full blocks");
        single_progress.finish();

        for x in buf_u8.iter_mut() {
            *x = x.to_le();
        }

        drop(buf_lock);
        send_buf.send(Some(ReadDataMsg{buf, first_code: code_idx, codes: codes_in_read})).unwrap();
        progress.inc(codes_in_read.as_u64());
    }

    send_buf.send(None).unwrap(); // shut down read_to_processors
    Ok(())
}

struct ProcessTaskMsg {
    buf: BigBuf,
    reader_count: Arc<AtomicUsize>,
    offset: usize,
    code_idx: usize,
    codes: usize
}

// connects reader to processor worker threads
fn read_to_processors(recv_data: &Receiver<Option<ReadDataMsg>>, to_processors: &Sender<Option<ProcessTaskMsg>>) {
    while let Some(ReadDataMsg{buf, first_code, codes}) = recv_data.recv().unwrap() {
        let reader_count = Arc::new(AtomicUsize::new(codes));
        for offset in 0..codes {
            to_processors.send(Some(ProcessTaskMsg{buf: buf.clone(), reader_count: reader_count.clone(), offset, code_idx: offset + first_code, codes})).unwrap();
        }
    }
}

fn return_if_last(buf: BigBuf, reader_count: Arc<AtomicUsize>, return_data_buf: &Sender<BigBuf>) {
    if reader_count.fetch_sub(1, Ordering::SeqCst) == 1 {
        assert!(buf.try_write().is_ok());
        return_data_buf.send(buf).unwrap();
    }
}

fn oversample(recv_parity_buf: &Receiver<Buf>, send_parity: &Sender<Option<(Buf, usize)>>,
              recv_data: &Receiver<Option<ProcessTaskMsg>>, return_data_buf: &Sender<BigBuf>,
              data_symbols: usize, parity_symbols: usize,
              factors: &[TransformFactors],
              progress: &ProgressBar) {

    assert_eq!(factors.len(), data_symbols.div_ceil(parity_symbols));
    for (f, expected_offset) in factors.iter().zip((0..).step_by(data_symbols)) {
        assert_eq!(f.offset().0, expected_offset);
    }

    let mut memory = vec![GF64(0); data_symbols];

    while let Some(ProcessTaskMsg{buf, reader_count, offset, code_idx, codes}) = recv_data.recv().unwrap() {
        for (x, m) in buf.try_read().unwrap()[offset..].iter().step_by(codes).copied().map(GF64).zip(memory.iter_mut()) {
            *m = x;
        }
        return_if_last(buf, reader_count, return_data_buf);

        inverse_transform(&mut memory, &factors[0]);

        let mut parity_buf = recv_parity_buf.recv().unwrap();
        assert_eq!(parity_buf.len(), parity_symbols);

        let mut iter = u64_as_gf64_mut(&mut parity_buf).chunks_exact_mut(data_symbols);
        assert_eq!(iter.len(), factors.len() - 1);
        for (chunk, chunk_factors) in (&mut iter).zip(&factors[1..]) {
            chunk.copy_from_slice(&memory);
            forward_transform(chunk, chunk_factors);
        }

        let last = iter.into_remainder();
        forward_transform(&mut memory, factors.last().unwrap());
        last.copy_from_slice(&memory[..last.len()]);

        send_parity.send(Some((parity_buf, code_idx))).unwrap();
        progress.inc(1);
    }
}

fn recovery(recv_parity_buf: &Receiver<Buf>, send_parity: &Sender<Option<(Buf, usize)>>,
            recv_data: &Receiver<Option<ProcessTaskMsg>>, return_data_buf: &Sender<BigBuf>,
            data_symbols: usize, parity_symbols: usize,
            t_factors: &TransformFactors,
            d_factors: &DerivativeFactors,
            error_indices: &[usize], valid_indices: &[usize],
            err_locator_values: &[GF64], err_locator_derivative_inverses: &[GF64],
            progress: &ProgressBar) {

    assert_eq!(error_indices.len(), err_locator_values.len());
    assert_eq!(valid_indices.len(), err_locator_derivative_inverses.len());

    let mut memory = vec![GF64(0); data_symbols + parity_symbols];

    while let Some(ProcessTaskMsg{buf, reader_count, offset, code_idx, codes}) = recv_data.recv().unwrap() {
        let locked_buf = buf.try_read().unwrap();
        let block_iter = u64_as_gf64(&locked_buf)[offset..].iter().step_by(codes).copied();
        assert_eq!(block_iter.len(), valid_indices.len());
        assert_eq!(block_iter.len(), err_locator_values.len());
        for ((x, i), e) in block_iter.zip(valid_indices.iter().copied()).zip(err_locator_values.iter().copied()) {
            // Extract block values from buffer of interleaved blocks and multiply by error locator
            memory[i] = x * e;
        }
        for i in error_indices.iter().copied() {
            memory[i] = GF64(0);
        }
        drop(locked_buf);
        return_if_last(buf, reader_count, return_data_buf);

        inverse_transform(&mut memory, t_factors);
        formal_derivative(&mut memory, d_factors);
        forward_transform(&mut memory, t_factors);

        let mut recovered_data_buf = recv_parity_buf.recv().unwrap(); // TODO: rename parity to 'output' or 'recovered' or something
        assert_eq!(recovered_data_buf.len(), error_indices.len());
        assert_eq!(recovered_data_buf.len(), err_locator_derivative_inverses.len());
        for ((x, i), inverse) in u64_as_gf64_mut(&mut recovered_data_buf).iter_mut().zip(error_indices.iter().copied()).zip(err_locator_derivative_inverses.iter().copied()) {
            *x = memory[i] * inverse;
        }
        send_parity.send(Some((recovered_data_buf, code_idx))).unwrap();
        progress.inc(1);
    }
}

fn process_codes(recv_parity_buf: &Receiver<Buf>, send_parity: &Sender<Option<(Buf, usize)>>,
                 recv_data: &Receiver<Option<ProcessTaskMsg>>, return_data_buf: &Sender<BigBuf>,
                 data_blocks: usize, parity_blocks: usize,
                 block_x_values: Option<&Vec<u64>>, output_x_values: Option<&Vec<u64>>,
                 progress: &ProgressBar) {

    let mut memory = vec![GF64(0); data_blocks * 3];
    let (poly, memory) = memory.split_at_mut(data_blocks);

    let block_x_values = block_x_values.as_ref().map(|x| u64_as_gf64(x.as_slice()));

    while let Some(ProcessTaskMsg{buf, reader_count, offset, code_idx, codes}) = recv_data.recv().unwrap() {
        let locked_buf = buf.try_read().unwrap();
        newton_interpolation(u64_as_gf64(&locked_buf[0..codes * data_blocks]), offset, codes, block_x_values, poly, memory);
        drop(locked_buf);
        let is_last = reader_count.fetch_sub(1, Ordering::SeqCst) == 1;
        if is_last {
            assert!(buf.try_write().is_ok());
            return_data_buf.send(buf).unwrap();
        }
        let mut parity_buf = recv_parity_buf.recv().unwrap();
        assert_eq!(parity_buf.len(), parity_blocks);
        if let Some(output_x_values) = output_x_values {
            for (x, y) in u64_as_gf64(output_x_values).iter().zip(u64_as_gf64_mut(&mut parity_buf)) {
                *y = evaluate_poly(poly, *x);
            }
        } else {
            for (x, y) in u64_as_gf64_mut(&mut parity_buf).iter_mut().enumerate() {
                *y = evaluate_poly(poly, GF64((data_blocks + x).as_u64()));
            }
        }
        send_parity.send(Some((parity_buf, code_idx))).unwrap();
        progress.inc(1);
    }
}

fn write_data(recv_parity: &Receiver<Option<(Buf, usize)>>, return_parity_buf: &Sender<Buf>,
              mapped_input: &mut [u8], mapped_output: &mut [u8],
              output_blocks: usize, output_indices: (BlockIndices, BlockIndices),
              progress: &ProgressBar) -> io::Result<()> {

    loop {
        let Some((parity, code_index)) = recv_parity.recv().unwrap() else { return Ok(()); };
        assert!(parity.len() == output_blocks);

        let offset = code_index * 8;
        let mut parity_idx = 0;
        for base_file_idx in output_indices.0.clone() {
            let file_idx = base_file_idx.as_usize() + offset;
            if file_idx + 8 <= mapped_input.len() {
                mapped_input[file_idx..file_idx + 8].copy_from_slice(&parity[parity_idx].to_le_bytes());
            } else {
                assert_eq!(parity[parity_idx], 0, "values outside the file must be zero");
            }
            parity_idx += 1;
        };
        for base_file_idx in output_indices.1.clone() {
            let file_idx = base_file_idx.as_usize() + offset;
            mapped_output[file_idx..file_idx + 8].copy_from_slice(&parity[parity_idx].to_le_bytes());
            parity_idx += 1;
        };
        
        return_parity_buf.send(parity).unwrap();
        progress.inc(1);
    }
}

pub struct EncodeOptions {
    pub block_bytes: usize,
    pub parity_blocks: usize,
}

struct PerfParams {
    cpus: usize,
    buf_amount: usize,
    mem_per_buf: usize,
    symbols_read_per_code: usize,
}
fn choose_cpus_and_mem(input_blocks: usize, output_blocks: usize, block_symbols: usize) -> PerfParams {
    let cpus = num_cpus::get();

    let (buf_amount, symbols_read_per_code) = 'a: {
        const MEM_USE_PERCENT: u64 = 90;
        const MAX_SYM_PER_CODE: u64 = 16284;
        let available_memory = {
            use sysinfo::{System, RefreshKind, MemoryRefreshKind};
            let sys_mem = System::new_with_specifics(RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram())).available_memory();
            let mem = (sys_mem * MEM_USE_PERCENT) / 100;
            mem.checked_sub((input_blocks * 3 * 8 * cpus).as_u64()) // substract memory used by processor threads (interpolation memory, polynomial)
               .and_then(|x| x.checked_sub((output_blocks * 8).as_u64())) // substract memory used for output buffers
               .unwrap_or(0)
        };

        if available_memory >= (input_blocks * block_symbols * 8).as_u64() { break 'a (1, block_symbols); }

        let sym_per_code = available_memory / input_blocks.as_u64() * 8 * 2;
        if sym_per_code > MAX_SYM_PER_CODE { ((sym_per_code / MAX_SYM_PER_CODE).clamp(1, 2) as usize, MAX_SYM_PER_CODE as usize) } else { ((sym_per_code / 2) as usize, 2) }
    };
    println!("Using {cpus} CPUs, {buf_amount} buffers, reading {} bytes per reading pass ({symbols_read_per_code} symbols per code)", symbols_read_per_code * 8 * input_blocks);
    let mem_per_buf = symbols_read_per_code * input_blocks;
    PerfParams {cpus, buf_amount, mem_per_buf, symbols_read_per_code}
}

fn internal_encode_pipeline(
        input: &mut File, output: &mut File, file_len: u64, output_file_len: u64,
        good_blocks: usize, bad_blocks: usize, block_symbols: usize,
        input_block_indices: (BlockIndices, BlockIndices),
        output_block_indices: (BlockIndices, BlockIndices),
        input_x_values: Option<Vec<u64>>, output_x_values: Option<Vec<u64>>,
        input_map: &mut [u8], parity_map: &mut [u8]
    ) -> io::Result<()> {

    let PerfParams{cpus, buf_amount, mem_per_buf, symbols_read_per_code} = choose_cpus_and_mem(good_blocks, bad_blocks, block_symbols);

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
        return_parity.0.send(vec![0; bad_blocks].into_boxed_slice()).unwrap();
    }

    let single = progress(good_blocks * symbols_read_per_code, "one read pass");
    let read_prog = progress(block_symbols, "read");
    let process_prog = progress(block_symbols, "process");
    let write_prog = progress(block_symbols, "write");
    make_multiprogress([&single, &read_prog, &process_prog, &write_prog]);

    // spawn threads
    std::thread::scope(|s| {
        let reader = s.spawn(|| read_data(
            &return_data.1, &data_to_adapter.0,
            block_symbols, symbols_read_per_code,
            input, output,
            file_len, output_file_len,
            input_block_indices,
            &read_prog, &single
        ));
        let read_to_processors = s.spawn(|| read_to_processors(&data_to_adapter.1, &adapter_to_processors.0));
        let processor_threads = (0..cpus).map(|_| {
            s.spawn(|| process_codes(
                &return_parity.1, &parity.0,
                &adapter_to_processors.1, &return_data.0,
                good_blocks, bad_blocks,
                input_x_values.as_ref(), output_x_values.as_ref(),
                &process_prog
            ))
        }).collect::<Vec<_>>();
        let writer = s.spawn(|| write_data(
            &parity.1, &return_parity.0,
            input_map, parity_map,
            bad_blocks, output_block_indices,
            &write_prog
        ));

        // wait for all data to be read
        reader.join().unwrap()?; // TODO: make threads shutdown in case of error
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
        writer.join().unwrap()?;
        io::Result::Ok(())
    })?;
    Ok(())
}

pub fn encode(input: &mut File, output: &mut File, opt: EncodeOptions) -> io::Result<()> {

    assert!(opt.block_bytes % 8 == 0, "block bytes must be divisible by 8");
    let block_symbols = opt.block_bytes / 8;
    let input_file_len = input.metadata()?.len();

    let data_blocks = usize::try_from(input_file_len.div_ceil(opt.block_bytes.as_u64())).expect("data blocks number must fit in usize");
    println!("{data_blocks} data blocks");

    let header = format_header(Header{data_blocks, parity_blocks: opt.parity_blocks, block_bytes: opt.block_bytes, file_len: input_file_len});

    // 32 bytes per hash, and first u64 from each block
    let hashes_bytes = 40 * (data_blocks + opt.parity_blocks);
    assert!(hashes_bytes % 8 == 0);

    let parity_blocks_bytes = opt.block_bytes * opt.parity_blocks;

    let output_file_len = (HEADER_LEN + hashes_bytes + parity_blocks_bytes).as_u64();
    output.set_len(output_file_len)?;
    output.write_all(&header)?;

    let mut output_map = unsafe { MmapOptions::new().map_mut(&*output)? };
    let (hashes_map, parity_map) = output_map[HEADER_LEN..].split_at_mut(hashes_bytes);
    assert_eq!(hashes_map.len(), hashes_bytes);
    assert_eq!(parity_map.len(), parity_blocks_bytes);

    internal_encode_pipeline(
        input, output, input_file_len, output_file_len,
        data_blocks, opt.parity_blocks, block_symbols,
        ((0..data_blocks.as_u64()).step_by(opt.block_bytes).into(), BlockIndices::Empty),
        (BlockIndices::Empty, (0..opt.parity_blocks.as_u64()).step_by(opt.block_bytes).into()),
        None, None,
        &mut [], parity_map,
    )?;

    let (data_hashes, parity_hashes) = hashes_map.split_at_mut(40 * data_blocks);
    std::thread::scope(|s| {
        let data_prog = progress(data_hashes.len() / 40, "hash data");
        let par_prog = progress(parity_hashes.len() / 40, "hash parity");
        make_multiprogress([&data_prog, &par_prog]);
        let t = s.spawn(|| {
            let data_prog = data_prog;
            let mut buf = vec![0_u8; opt.block_bytes];
            let last_block_rem = (input_file_len % opt.block_bytes.as_u64()) as usize;
            input.seek(std::io::SeekFrom::Start(0))?;
            let last_hash = data_hashes.len() / 40 - 1;
            for (i, out) in data_hashes.chunks_exact_mut(40).enumerate() {
                if i == last_hash && last_block_rem != 0 {
                    buf.fill(0);
                    input.read_exact(&mut buf[0..last_block_rem])?;
                } else {
                    input.read_exact(&mut buf)?;
                }
                out[..8].copy_from_slice(&buf[..8]);
                out[8..40].copy_from_slice(blake3::hash(&buf).as_bytes());
                data_prog.inc(1);
            }
            io::Result::Ok(())
        });
        s.spawn(|| {
            let par_prog = par_prog;
            for (block, out) in parity_map.chunks_exact(opt.block_bytes).zip(parity_hashes.chunks_exact_mut(40)) {
                out[..8].copy_from_slice(&block[..8]);
                out[8..40].copy_from_slice(blake3::hash(block).as_bytes());
                par_prog.inc(1);
            }
        });
        t.join().unwrap()?;
        io::Result::Ok(())
    })?;

    // hash all the metadata expect the header string and the placeholder meta-hash zeros
    assert_eq!(output_map[0..HEADER_STRING.len()], HEADER_STRING);
    let metadata_hash = blake3::hash(&output_map[HEADER_STRING.len() + blake3::OUT_LEN..HEADER_LEN + hashes_bytes]);
    set_meta_hash(&mut output_map, *metadata_hash.as_bytes());

    println!("Flush output...");
    output_map.flush()?;
    drop(output_map);
    output.flush()?;

    Ok(())
}

fn bools_to_indices(b: &Option<Box<[bool]>>, default_len: usize, invert: bool) -> Vec<u64> {
    if let Some(b) = b {
        b.iter().enumerate().filter_map(|(i, &x)| if x ^ invert { Some(i.as_u64()) } else { None }).collect()
    } else if invert {
        (0..default_len.as_u64()).collect()
    } else {
        vec![]
    }
}

pub fn repair(input: &mut File, output: &mut File, header: Header, input_corruption: Option<Box<[bool]>>, output_corruption: Option<Box<[bool]>>) -> io::Result<()> {
    let block_bytes = header.block_bytes;
    let parity_blocks = header.parity_blocks;
    let data_blocks = header.data_blocks;

    let block_symbols = block_bytes / 8;

    let input_len = input.metadata()?.len();
    let output_len = output.metadata()?.len();

    let mut input_map = unsafe { MmapOptions::new().map_mut(&*input)? };
    let mut output_map = unsafe { MmapOptions::new().map_mut(&*output)? };

    let mut good_input_blocks = bools_to_indices(&input_corruption, data_blocks, true);
    let mut good_output_blocks = bools_to_indices(&output_corruption, parity_blocks, true);
    let mut bad_input_blocks = bools_to_indices(&input_corruption, data_blocks, false);
    let mut bad_output_blocks = bools_to_indices(&output_corruption, parity_blocks, false);

    let mut input_x_values = good_input_blocks.clone();
    input_x_values.extend(good_output_blocks.iter().map(|&x| x + data_blocks.as_u64()));

    let mut output_x_values = bad_input_blocks.clone();
    output_x_values.extend(bad_output_blocks.iter().map(|&x| x + data_blocks.as_u64()));

    for x in good_input_blocks.iter_mut().chain(bad_input_blocks.iter_mut()) {
        *x *= block_bytes.as_u64();
    }

    let metadata_bytes = HEADER_LEN + 40 * (data_blocks + parity_blocks);
    for x in good_output_blocks.iter_mut().chain(bad_output_blocks.iter_mut()) {
        *x = metadata_bytes.as_u64() + *x * block_bytes.as_u64();
    }

    let good_blocks = input_x_values.len();
    let bad_blocks = output_x_values.len();

    println!("{good_blocks} good blocks, {data_blocks} required for repair, {bad_blocks} bad blocks");
    if good_blocks < data_blocks {
        println!("Not enough good blocks to repair");
        return Ok(());
    }

    internal_encode_pipeline(
        input, output, input_len, output_len,
        good_blocks, bad_blocks, block_symbols,
        (good_input_blocks.iter().into(), good_output_blocks.iter().into()),
        (bad_input_blocks.iter().into(), bad_output_blocks.iter().into()),
        Some(input_x_values), Some(output_x_values),
        &mut input_map, &mut output_map,
    )?;

    drop(input_map);
    drop(output_map);
    input.sync_all()?;
    output.sync_all()?;

    Ok(())
}
