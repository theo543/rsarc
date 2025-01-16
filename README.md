# RSARC: Reed-Solomon Archive

This project is a work in progress.

## Commands

- `test` (default): Runs an end-to-end test of the program.
- `encode <input file> <output file> <block bytes> <parity blocks>`: Generates parity data for the input file.
- `verify <input file> <parity file>`: Check for corruption in the input and parity files.
- `repair <input file> <parity file>`: Attempt to find and repair corruption in the input and parity files, if there is enough parity data and the metadata is not corrupt.
