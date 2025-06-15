from os import system, environ

assert system("cargo test") == 0
assert system("cargo test --release") == 0
environ["RUSTFLAGS"] = "-C target-feature=+pclmulqdq"
assert system("cargo test") == 0
assert system("cargo test --release") == 0
