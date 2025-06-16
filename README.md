# RSARC: Reed-Solomon Archive

Implementation of Reed-Solomon erasure coding for file corruption detection and repair using $O(n \log n)$ algorithms introduced in the paper [Novel Polynomial Basis and Its Application to Reed-Solomon Erasure Codes](https://arxiv.org/pdf/1404.3458).

Project for [undergraduate thesis](/thesis/thesis.pdf) at University of Bucharest, Faculty of Mathematics and Computer Science.

## Commands

- `test`: Runs an end-to-end test of the program.
- `encode <input file> <output file> <block bytes> <parity blocks>`: Generates parity data for the input file.
- `verify <input file> <parity file>`: Check for corruption in the input and parity files.
- `repair <input file> <parity file>`: Attempt to find and repair corruption in the input and parity files, if there is enough parity data and the metadata is not corrupt.
- `reassemble <input file> <parity file> <reassembled input file> <reassembled parity file>`: Reassemble the input and parity files after an insertion/deletion error (very unlikely to happen).
