FROM circleci/rust

RUN rustup default nightly

RUN rustup component add rustfmt clippy-preview

# Add WASM target
RUN rustup target add wasm32-unknown-unknown --toolchain nightly

RUN cargo +nightly install wasm-bindgen-cli

# install add-apt-repository
RUN sudo apt-get install software-properties-common

# debian stretch uses LLVM-3.8, but we want 7.0
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - && \
    sudo add-apt-repository "deb http://apt.llvm.org/stretch/ llvm-toolchain-stretch-7 main" && \
    sudo apt-get update && \
    # `llvm-config` is in `llvm`
    sudo apt-get install llvm-7 \
                         lld-7 \
                         # `lld/Common/Driver.h`
                         liblld-7-dev

ENV LLVM_SYS_70_PREFIX /usr/lib/llvm-7
ENV PATH "$LLVM_SYS_70_PREFIX/bin:$PATH"
ENV CPPFLAGS "-I$LLVM_SYS_70_PREFIX/include"
ENV LDFLAGS "-L$LLVM_SYS_70_PREFIX/lib"
