#![warn(clippy::all)]

mod math;
mod utils;
mod header;
mod encoder;
mod verifier;
mod reassembler;
mod end_to_end_test;

use std::fs::OpenOptions;

use encoder::{encode, repair, EncodeOptions};
use utils::ShareModeExt;

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

fn main() {
    math::gf64::check_cpu_support_for_carryless_multiply();
    utils::progress::register_panic_hook();

    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|x| x.as_str()).unwrap_or("test") {
        "test" => end_to_end_test::test(),

        "encode" => {
            if args.len() != 6 {
                println!("Usage: encode <input file> <output file> <block bytes> <parity blocks>");
                return;
            }
            let input_file = &args[2];
            let output_file = &args[3];
            let block_bytes = args[4].parse().unwrap();
            let parity_blocks = args[5].parse().unwrap();

            let mut input = OpenOptions::new().read(true).share_mode_lock().open(input_file).unwrap();
            let mut output = OpenOptions::new().read(true).write(true).create(true).truncate(true).share_mode_lock().open(output_file).unwrap();

            encode(&mut input, &mut output, EncodeOptions{block_bytes, parity_blocks}).unwrap();
        }

        "verify" => {
            if args.len() != 4 {
                println!("Usage: verify <input file> <output file>");
                return;
            }
            let input_file = &args[2];
            let output_file = &args[3];

            let mut input = OpenOptions::new().read(true).share_mode_lock().open(input_file).unwrap();
            let mut output = OpenOptions::new().read(true).share_mode_lock().open(output_file).unwrap();

            verifier::verify(&mut input, &mut output).unwrap().report_corruption(false);
        }

        "repair" => {
            if args.len() != 4 {
                println!("Usage: repair <input file> <output file>");
                return;
            }
            let input_file = &args[2];
            let output_file = &args[3];

            let mut input = OpenOptions::new().read(true).write(true).share_mode_lock().open(input_file).unwrap();
            let mut output = OpenOptions::new().read(true).write(true).share_mode_lock().open(output_file).unwrap();

            let corruption = verifier::verify(&mut input, &mut output).unwrap();

            match corruption {
                verifier::VerifyResult::Ok { data_file, parity_file, header } => {
                    if data_file.is_none() && parity_file.is_none() {
                        println!("No corruption detected");
                        return;
                    }
                    repair(&mut input, &mut output, header, data_file, parity_file).unwrap();
                }
                verifier::VerifyResult::MetadataCorrupted(msg) => { println!("Metadata corrupted: {}", msg); }
            }
        }

        "reassemble" => {
            if args.len() != 6 {
                println!("Usage: reassemble <data file> <parity file> <output data file> <output parity file>");
                return;
            }
            let data_file = &args[2];
            let parity_file = &args[3];
            let output_data_file = &args[4];
            let output_parity_file = &args[5];

            let data = OpenOptions::new().read(true).share_mode_lock().open(data_file).unwrap();
            let parity = OpenOptions::new().read(true).share_mode_lock().open(parity_file).unwrap();
            let mut output_data = OpenOptions::new().read(true).write(true).create(true).truncate(true).share_mode_lock().open(output_data_file).unwrap();
            let mut output_parity = OpenOptions::new().read(true).write(true).create(true).truncate(true).share_mode_lock().open(output_parity_file).unwrap();

            let (recovered_data, recovered_parity) = reassembler::reassemble(&data, &parity, &mut output_data, &mut output_parity).unwrap();
            println!("Found {} data blocks and {} parity blocks", recovered_data, recovered_parity);
        }

        _ => println!("Unknown command"),
    }

    math::gf64::print_stats();
}
