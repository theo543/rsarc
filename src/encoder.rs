#![allow(clippy::too_many_arguments)]

use std::fs::File;
use std::io::{self, Read, Seek, Write};
use std::sync::{atomic::{AtomicUsize, Ordering}, Arc, RwLock};
use crossbeam_channel::{unbounded, Sender, Receiver};

use indicatif::ProgressBar;
use memmap2::MmapOptions;
use positioned_io::{RandomAccessFile, ReadAt};

use crate::header::{format_header, set_meta_hash, Header, HEADER_LEN, HEADER_STRING};
use crate::math::gf64::{u64_as_gf64, u64_as_gf64_mut, GF64};
use crate::math::novelpoly::{compute_error_locator_poly, formal_derivative, forward_transform, inverse_transform, precompute_derivative_factors, precompute_transform_factors, DerivativeFactors, TransformFactors};
use crate::utils::progress::{progress_usize as progress, make_multiprogress, IncIfNotFinished};
use crate::utils::{get_available_memory, IntoU64Ext, IntoUSizeExt, ZipEqExt};

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

trait BlockIndices<'a>: ExactSizeIterator<Item = u64> + Send + Clone {}
impl<'a, T: ExactSizeIterator<Item = u64> + Send + Clone + 'a> BlockIndices<'a> for T {}

fn read_symbols_from_file<'a>(file: &RandomAccessFile, file_len: u64, offset: u64, indices: impl BlockIndices<'a>, symbols_bytes: usize, buf_idx: &mut usize, buf: &mut [u8], progress: &ProgressBar) -> io::Result<bool> {
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
        progress.inc(1);
    }
    Ok(incomplete_read_occurred)
}

fn read_data<'a>(recv_buf: &Receiver<BigBuf>, send_buf: &Sender<Option<ReadDataMsg>>,
             block_symbols: usize, codes_per_full_read: usize,
             input: &mut File, output: &mut File,
             good_indices: (impl BlockIndices<'a>, impl BlockIndices<'a>),
             progress: &ProgressBar, single_progress: &ProgressBar) -> io::Result<()> {

    let input_file_size = input.metadata()?.len();
    let output_file_size = output.metadata()?.len();
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
        read_symbols_from_file(&input, input_file_size, offset, good_indices.0.clone(), bytes_in_read, &mut buf_idx, buf_u8, single_progress)?;
        let incomplete_output_read = read_symbols_from_file(&output, output_file_size, offset, good_indices.1.clone(), bytes_in_read, &mut buf_idx, buf_u8, single_progress)?;
        assert!(!incomplete_output_read, "output file contains only full blocks");
        single_progress.finish();

        for x in buf_u8.iter_mut() {
            *x = x.to_le();
        }

        drop(buf_lock);
        send_buf.send(Some(ReadDataMsg{buf, first_code: code_idx, codes: codes_in_read})).unwrap();
        progress.add(codes_in_read.as_u64());
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

struct WorkerChans<'a> {
    recv_result_buf: &'a Receiver<Buf>,
    send_result: &'a Sender<Option<(Buf, usize)>>,
    recv_data: &'a Receiver<Option<ProcessTaskMsg>>,
    return_data_buf: &'a Sender<BigBuf>,
}

fn oversample(chans: WorkerChans,
              data_symbols: usize, padded_data_symbols: usize,
              factors: &[TransformFactors],
              progress: &ProgressBar) {

    for (f, expected_offset) in factors.iter().zip((0..).step_by(padded_data_symbols)) {
        assert_eq!(f.offset().0, expected_offset);
    }

    let mut memory = vec![GF64(0); padded_data_symbols];

    while let Some(ProcessTaskMsg{buf, reader_count, offset, code_idx, codes}) = chans.recv_data.recv().unwrap() {
        let (mem_for_data, mem_padding) = memory.split_at_mut(data_symbols);
        for (x, m) in buf.try_read().unwrap()[offset..codes * data_symbols].iter().step_by(codes).zip_eq(mem_for_data) {
            *m = GF64(*x);
        }
        return_if_last(buf, reader_count, chans.return_data_buf);
        mem_padding.fill(GF64(0));

        inverse_transform(&mut memory, &factors[0]);

        let mut parity_buf = chans.recv_result_buf.recv().unwrap();

        let extra_chunk = parity_buf.len() % padded_data_symbols != 0;
        let mut iter = u64_as_gf64_mut(&mut parity_buf).chunks_exact_mut(padded_data_symbols);
        for (chunk, chunk_factors) in (&mut iter).zip_eq(&factors[1..factors.len() - extra_chunk as usize]) {
            chunk.copy_from_slice(&memory);
            forward_transform(chunk, chunk_factors);
        }

        if extra_chunk {
            forward_transform(&mut memory, factors.last().unwrap());
            let last = iter.into_remainder();
            last.copy_from_slice(&memory[..last.len()]);
        }

        chans.send_result.send(Some((parity_buf, code_idx))).unwrap();
        progress.add(1);
    }
}

fn recovery(chans: WorkerChans,
            padded_codeword_len: usize,
            t_factors: &TransformFactors,
            d_factors: &DerivativeFactors,
            error_indices: &[u64], valid_indices: &[u64],
            err_locator_values: &[GF64], err_locator_derivative_inverses: &[GF64],
            progress: &ProgressBar) {

    assert_eq!(padded_codeword_len, err_locator_values.len());
    assert_eq!(padded_codeword_len, err_locator_derivative_inverses.len());
    assert_eq!(t_factors.offset().0, 0);

    let mut memory = vec![GF64(0); padded_codeword_len];

    while let Some(ProcessTaskMsg{buf, reader_count, offset, code_idx, codes}) = chans.recv_data.recv().unwrap() {
        let locked_buf = buf.try_read().unwrap();
        for (x, i) in u64_as_gf64(&locked_buf)[offset..codes * valid_indices.len()].iter().step_by(codes).zip_eq(valid_indices) {
            // Extract block values from buffer of interleaved blocks and multiply by error locator
            memory[i.as_usize()] = *x * err_locator_values[i.as_usize()];
        }
        drop(locked_buf);
        return_if_last(buf, reader_count, chans.return_data_buf);

        inverse_transform(&mut memory, t_factors);
        formal_derivative(&mut memory, d_factors);
        forward_transform(&mut memory, t_factors);

        let mut recovered_data_buf = chans.recv_result_buf.recv().unwrap();
        for (x, i) in u64_as_gf64_mut(&mut recovered_data_buf).iter_mut().zip_eq(error_indices) {
            *x = memory[i.as_usize()] * err_locator_derivative_inverses[i.as_usize()];
        }
        chans.send_result.send(Some((recovered_data_buf, code_idx))).unwrap();
        progress.add(1);
        memory.fill(GF64(0));
    }
}

fn write_data<'a>(recv_data: &Receiver<Option<(Buf, usize)>>, return_buf: &Sender<Buf>,
              mapped_data_file: &mut [u8], mapped_parity_file: &mut [u8],
              output_indices: (impl BlockIndices<'a>, impl BlockIndices<'a>),
              progress: &ProgressBar) -> io::Result<()> {

    loop {
        let Some((data, code_index)) = recv_data.recv().unwrap() else { return Ok(()); };
        assert_eq!(data.len(), output_indices.0.len() + output_indices.1.len());

        let offset = code_index * 8;
        let mut parity_idx = 0;
        for base_file_idx in output_indices.0.clone() {
            let file_idx = base_file_idx.as_usize() + offset;
            if file_idx + 8 <= mapped_data_file.len() {
                mapped_data_file[file_idx..file_idx + 8].copy_from_slice(&data[parity_idx].to_le_bytes());
            } else if file_idx < mapped_data_file.len() {
                // Final value is less than 8 bytes
                let final_bytes_len = mapped_data_file.len() - file_idx;
                let final_bytes = data[parity_idx].to_le_bytes();
                let (final_bytes, padding) = final_bytes.split_at(final_bytes_len);
                mapped_data_file[file_idx..].copy_from_slice(final_bytes);
                assert_eq!(padding, &[0; 8][final_bytes_len..], "implicit padding bytes right after file end must be zero");
            } else {
                assert_eq!(data[parity_idx], 0, "implicit padding value outside the file must be zero");
            }
            parity_idx += 1;
        };
        for base_file_idx in output_indices.1.clone() {
            let file_idx = base_file_idx.as_usize() + offset;
            mapped_parity_file[file_idx..file_idx + 8].copy_from_slice(&data[parity_idx].to_le_bytes());
            parity_idx += 1;
        };
        
        return_buf.send(data).unwrap();
        progress.add(1);
    }
}

pub struct EncodeOptions {
    pub block_bytes: usize,
    pub parity_blocks: usize,
}

struct BufParams {
    two_bigbuf: bool,
    codes_read_at_once: usize,
}

fn choose_buf_size(threads: usize, good_blocks: usize, bad_blocks: usize, block_symbols: usize, memory_per_thread: usize) -> BufParams {
    const MEM_USE_PERCENT: u64 = 90;
    const MAX_CODES: u64 = 65536;

    let available_memory = ((get_available_memory() * MEM_USE_PERCENT) / 100)
    .saturating_sub((memory_per_thread * threads).as_u64()) // substract memory used by processor threads (interpolation memory, polynomial)
    .saturating_sub((bad_blocks * 8).as_u64()) // substract memory used for output buffers
    .max(1);

    let codes_read_at_once = available_memory.div_ceil(good_blocks.as_u64() * 8 * 2);

    let (two_bigbuf, codes_read_at_once) =
        if available_memory >= (good_blocks * block_symbols * 8).as_u64() { (false, block_symbols) } // entire file fits in memory
        else if codes_read_at_once > MAX_CODES { (codes_read_at_once >= MAX_CODES * 2, MAX_CODES.as_usize()) } // read MAX_CODES per pass, if there's enough memory create two read buffers to allow reading while processing
        else { (false, codes_read_at_once as usize) }; // just read as many codes as possible per pass

    println!("Using {threads} CPUs, reading {} bytes per reading pass ({codes_read_at_once} symbols per block)", codes_read_at_once * 8 * good_blocks);
    BufParams {two_bigbuf, codes_read_at_once}
}

fn internal_encode_pipeline<'a>(
        input: &mut File, output: &mut File,
        good_blocks: usize, bad_blocks: usize, block_symbols: usize,
        input_block_indices: (impl BlockIndices<'a>, impl BlockIndices<'a>),
        output_block_indices: (impl BlockIndices<'a>, impl BlockIndices<'a>),
        input_map: &mut [u8], parity_map: &mut [u8],
        spawn_processor_thread: &(dyn Fn(WorkerChans, &ProgressBar) + Sync),
        memory_per_thread: usize,
    ) -> io::Result<()> {

    let cpus = num_cpus::get();
    let BufParams{two_bigbuf, codes_read_at_once} = choose_buf_size(cpus, good_blocks, bad_blocks, block_symbols, memory_per_thread);

    // create channels
    let data_to_adapter = unbounded();
    let adapter_to_processors = unbounded();
    let return_data_buf = unbounded();
    let result = unbounded();
    let return_result_buf = unbounded();

    // allocate BigBuf for reading
    for _ in 0..1 + two_bigbuf as usize {
        return_data_buf.0.send(Arc::new(RwLock::new(vec![0_u64; codes_read_at_once * good_blocks].into_boxed_slice()))).unwrap();
    }

    // allocate Buf for writing
    for _ in 0..cpus * 2 {
        return_result_buf.0.send(vec![0; bad_blocks].into_boxed_slice()).unwrap();
    }

    let single = progress(good_blocks, "one read pass");
    let read_prog = progress(block_symbols, "read");
    let process_prog = progress(block_symbols, "process");
    let write_prog = progress(block_symbols, "write");
    make_multiprogress([&single, &read_prog, &process_prog, &write_prog]);

    // spawn threads
    std::thread::scope(|s| {
        let reader = s.spawn(|| read_data(
            &return_data_buf.1, &data_to_adapter.0,
            block_symbols, codes_read_at_once,
            input, output,
            input_block_indices,
            &read_prog, &single
        ));
        let read_to_processors = s.spawn(|| read_to_processors(&data_to_adapter.1, &adapter_to_processors.0));
        let processor_threads = (0..cpus).map(|_| s.spawn(|| spawn_processor_thread(WorkerChans {
            recv_result_buf: &return_result_buf.1,
            send_result: &result.0,
            recv_data: &adapter_to_processors.1,
            return_data_buf: &return_data_buf.0,
        }, &process_prog))).collect::<Vec<_>>();
        let writer = s.spawn(|| write_data(
            &result.1, &return_result_buf.0,
            input_map, parity_map,
            output_block_indices,
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
        result.0.send(None).unwrap();
        writer.join().unwrap()?;
        io::Result::Ok(())
    })?;
    Ok(())
}

fn precompute_oversample_factors(block_symbols: usize, factors_len: usize) -> Vec<TransformFactors> {
    assert!(block_symbols.is_power_of_two());
    let pow = block_symbols.ilog2();
    let threads = num_cpus::get().min(factors_len);

    let mut factors: Vec<Option<TransformFactors>> = vec![None; factors_len];

    std::thread::scope(|s| {
        let chan = unbounded::<(u64, &mut Option<TransformFactors>)>();

        for _ in 0..threads {
            let recv = chan.1.clone();
            s.spawn(move || {
                while let Ok((offset, out)) = recv.recv() {
                    *out = Some(precompute_transform_factors(pow, GF64(offset)));
                }
            });
        }

        for (offset, out) in (0..).step_by(block_symbols).zip(factors.iter_mut()) {
            chan.0.send((offset, out)).unwrap();
        }
    });

    factors.into_iter().map(Option::unwrap).collect()
}

pub fn encode(input: &mut File, output: &mut File, EncodeOptions { block_bytes, parity_blocks }: EncodeOptions) -> io::Result<()> {

    assert!(block_bytes % 8 == 0, "block bytes must be divisible by 8");
    let block_symbols = block_bytes / 8;
    let input_file_len = input.metadata()?.len();

    let data_blocks = input_file_len.div_ceil(block_bytes.as_u64()).as_usize();
    println!("{data_blocks} data blocks");

    let padded_data_blocks = data_blocks.next_power_of_two(); // next_power_of_two means "smallest power of two greater than or equal to data_blocks", not "next power of two after data_blocks"
    let factors_len = 1 + parity_blocks.div_ceil(padded_data_blocks);
    let factors = precompute_oversample_factors(padded_data_blocks, factors_len);

    let header = format_header(Header{data_blocks, parity_blocks, block_bytes, file_len: input_file_len});

    // 32 bytes per hash, and first u64 from each block
    let hashes_bytes = 40 * (data_blocks + parity_blocks);
    assert!(hashes_bytes % 8 == 0);

    let parity_blocks_bytes = block_bytes * parity_blocks;

    output.set_len((HEADER_LEN + hashes_bytes + parity_blocks_bytes).as_u64())?;
    output.write_all(&header)?;

    let mut output_map = unsafe { MmapOptions::new().map_mut(&*output)? };
    let (hashes_map, parity_map) = output_map[HEADER_LEN..].split_at_mut(hashes_bytes);
    assert_eq!(hashes_map.len(), hashes_bytes);
    assert_eq!(parity_map.len(), parity_blocks_bytes);

    internal_encode_pipeline(
        input, output,
        data_blocks, parity_blocks, block_symbols,
        ((0..data_blocks).map(|x| x.as_u64() * block_bytes.as_u64()), std::iter::empty()),
        (std::iter::empty(), (0..parity_blocks).map(|x| x.as_u64() * block_bytes.as_u64())),
        &mut [], parity_map,
        &|chans, progress| oversample(chans, data_blocks, padded_data_blocks, &factors, progress),
        padded_data_blocks,
    )?;

    let (data_hashes, parity_hashes) = hashes_map.split_at_mut(40 * data_blocks);
    std::thread::scope(|s| {
        let data_prog = progress(data_hashes.len() / 40, "hash data");
        let par_prog = progress(parity_hashes.len() / 40, "hash parity");
        make_multiprogress([&data_prog, &par_prog]);
        let t = s.spawn(|| {
            let data_prog = data_prog;
            let mut buf = vec![0_u8; block_bytes];
            let last_block_rem = (input_file_len % block_bytes.as_u64()) as usize;
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
                data_prog.add(1);
            }
            io::Result::Ok(())
        });
        s.spawn(|| {
            let par_prog = par_prog;
            for (block, out) in parity_map.chunks_exact(block_bytes).zip(parity_hashes.chunks_exact_mut(40)) {
                out[..8].copy_from_slice(&block[..8]);
                out[8..40].copy_from_slice(blake3::hash(block).as_bytes());
                par_prog.add(1);
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

fn collect_good_indices(corrupt: &[u64], len: usize) -> Vec<u64> {
    if corrupt.is_empty() {
        return (0..len.as_u64()).collect();
    }
    let mut good_indices = Vec::with_capacity(len - corrupt.len());
    let mut corrupt_idx = 0;
    for idx in 0..len.as_u64() {
        if corrupt[corrupt_idx] == idx {
            corrupt_idx += 1;
            if corrupt_idx == corrupt.len() {
                good_indices.extend(idx + 1..len.as_u64());
                break;
            }
        } else {
            good_indices.push(idx);
        }
    }
    assert_eq!(corrupt_idx, corrupt.len());
    assert_eq!(good_indices.len(), len - corrupt.len());
    good_indices
}

pub fn repair(input: &mut File, output: &mut File, Header { block_bytes, data_blocks, parity_blocks, file_len: _ }: Header, data_corruption: Vec<u64>, parity_corruption: Vec<u64>) -> io::Result<()> {
    let bad_blocks = data_corruption.len() + parity_corruption.len();
    let good_blocks = data_blocks + parity_blocks - bad_blocks;

    let padded_codeword_len = (data_blocks.next_power_of_two() + parity_blocks).next_power_of_two();

    let (t_factors, d_factors) = std::thread::scope(|s| {
        let t_factors = s.spawn(|| precompute_transform_factors(padded_codeword_len.ilog2(), GF64(0)));
        let d_factors = s.spawn(|| precompute_derivative_factors(padded_codeword_len.ilog2()));
        (t_factors.join().unwrap(), d_factors.join().unwrap())
    });

    println!("{good_blocks} good blocks, {data_blocks} required for repair, {bad_blocks} bad blocks");
    if good_blocks < data_blocks {
        println!("Not enough good blocks to repair");
        return Ok(());
    }

    let metadata_bytes = HEADER_LEN + 40 * (data_blocks + parity_blocks);

    let mut input_map = unsafe { MmapOptions::new().map_mut(&*input)? };
    let mut output_map = unsafe { MmapOptions::new().map_mut(&*output)? };

    let good_data_blocks = collect_good_indices(&data_corruption, data_blocks);
    let good_parity_blocks = collect_good_indices(&parity_corruption, parity_blocks);

    let mut error_indices = data_corruption.iter().copied().chain(parity_corruption.iter().map(|x| *x + data_blocks.as_u64().next_power_of_two()))
        // implicit parity zero padding must be included in the error locator, since conceptually the padding is corrupted (zeroed instead of oversampled values)
        .chain(((data_blocks.next_power_of_two() + parity_blocks)..padded_codeword_len).map(|x| x.as_u64()))
        .collect::<Vec<_>>();
    let (err_locator, mut err_locator_derivative) = compute_error_locator_poly(&error_indices, padded_codeword_len, &t_factors, &d_factors);
    for x in err_locator_derivative.iter_mut() {
        *x = x.invert();
    }

    // remove padding from error indices so that the recovery function doesn't attempt to actually repair the padding
    error_indices.truncate(data_corruption.len() + parity_corruption.len());

    let valid_indices = good_data_blocks.iter().copied().chain(good_parity_blocks.iter().map(|x| *x + data_blocks.as_u64().next_power_of_two())).collect::<Vec<_>>();

    let bb = block_bytes.as_u64();

    internal_encode_pipeline(
        input, output,
        good_blocks, bad_blocks, block_bytes / 8,
        // good_parity_blocks is offset by metadata bytes because reading does not use the memory maps, so slicing output_map by metadata_bytes does not help good_parity_blocks skip the metadata
        (good_data_blocks.iter().map(|x| x * bb), good_parity_blocks.iter().map(|x| x * bb + metadata_bytes.as_u64())),
        (data_corruption.iter().map(|x| x * bb), parity_corruption.iter().map(|x| x * bb)),
        &mut input_map, &mut output_map[metadata_bytes..],
        &|chans, progress| recovery(chans, padded_codeword_len, &t_factors, &d_factors, &error_indices, &valid_indices, &err_locator, &err_locator_derivative, progress),
        padded_codeword_len,
    )?;

    input_map.flush()?;
    output_map.flush()?;
    drop(input_map);
    drop(output_map);
    input.sync_all()?;
    output.sync_all()?;

    Ok(())
}
