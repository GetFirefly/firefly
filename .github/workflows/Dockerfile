FROM rust:latest

RUN cd /tmp \
  && wget https://github.com/lumen/llvm-project/releases/download/lumen-10.0.0-dev_2020-04-16/clang+llvm-10.0.0-x86_64-linux-gnu.tar.gz \
  && tar -xvzf clang+llvm-10.0.0-x86_64-linux-gnu.tar.gz \
  && rm -f clang+llvm-10.0.0-x86_64-linux-gnu.tar.gz \
  && mv clang+llvm-10.0.0-x86_64-linux-gnu /opt/llvm

ENV PATH="/opt/llvm/bin:${PATH}"
ENV LLVM_PREFIX="/opt/llvm"
ENV CC=/opt/llvm/bin/clang
ENV CXX=/opt/llvm/bin/clang++

RUN apt-get update \
  && apt-get install -y cmake jq libc++-dev libc++abi-dev libunwind-dev ninja-build rsync

RUN rustup default nightly-2020-03-10

RUN rustup component add rustfmt

# Add WASM target
RUN rustup target add wasm32-unknown-unknown --toolchain nightly-2020-03-10

RUN cargo +nightly-2020-03-10 install wasm-bindgen-cli

RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# install firefox

ENV FIREFOX_VERSION=69.0

RUN cd /tmp \
  && wget https://download-installer.cdn.mozilla.net/pub/firefox/releases/${FIREFOX_VERSION}/linux-x86_64/en-US/firefox-${FIREFOX_VERSION}.tar.bz2 \
  && tar -xvjf /tmp/firefox-${FIREFOX_VERSION}.tar.bz2 -C /opt \
  && ln -s /opt/firefox/firefox /usr/local/bin/firefox \
  && rm -rf /tmp/firefox.* \
  && apt-get update \
  && apt-get install -y libgtk-3-0 libx11-xcb1 libdbus-glib-1-2 \
  && firefox --version

# install geckodriver

ENV GECKO_DRIVER_VERSION=0.24.0

RUN cd /tmp \
     && wget https://github.com/mozilla/geckodriver/releases/download/v${GECKO_DRIVER_VERSION}/geckodriver-v${GECKO_DRIVER_VERSION}-linux64.tar.gz \
     && tar xf geckodriver-v${GECKO_DRIVER_VERSION}-linux64.tar.gz \
     && rm -f geckodriver-v${GECKO_DRIVER_VERISON}-linux64.tar.gz \
     && mv geckodriver /usr/local/bin/geckodriver \
     && chmod +x /usr/local/bin/geckodriver \
     && geckodriver --version

# install chrome

RUN curl --silent --show-error --location --fail --retry 3 --output /tmp/google-chrome-stable_current_amd64.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && (dpkg -i /tmp/google-chrome-stable_current_amd64.deb || apt-get -fy install)  \
    && rm -rf /tmp/google-chrome-stable_current_amd64.deb \
    && sed -i 's|HERE/chrome"|HERE/chrome" --disable-setuid-sandbox --no-sandbox|g' \
        "/opt/google/chrome/google-chrome" \
    && google-chrome --version

RUN CHROME_VERSION="$(google-chrome --version)" \
    && export CHROMEDRIVER_RELEASE="$(echo $CHROME_VERSION | sed 's/^Google Chrome //')" && export CHROMEDRIVER_RELEASE=${CHROMEDRIVER_RELEASE%%.*} \
    && CHROMEDRIVER_VERSION=$(curl --silent --show-error --location --fail --retry 4 --retry-delay 5 http://chromedriver.storage.googleapis.com/LATEST_RELEASE_${CHROMEDRIVER_RELEASE}) \
    && curl --silent --show-error --location --fail --retry 4 --retry-delay 5 --output /tmp/chromedriver_linux64.zip "http://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip" \
    && cd /tmp \
    && unzip chromedriver_linux64.zip \
    && rm -rf chromedriver_linux64.zip \
    && mv chromedriver /usr/local/bin/chromedriver \
    && chmod +x /usr/local/bin/chromedriver \
    && chromedriver --version

ENV NODE_VERSION 12.10.0
ENV ARCH x64

RUN curl -fsSLO --compressed "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-$ARCH.tar.xz" \
  && tar -xJf "node-v$NODE_VERSION-linux-$ARCH.tar.xz" -C /usr/local --strip-components=1 --no-same-owner \
  && rm "node-v$NODE_VERSION-linux-$ARCH.tar.xz" \
  && ln -s /usr/local/bin/node /usr/local/bin/nodejs

ENV YARN_VERSION 1.17.3

RUN curl -fsSLO --compressed "https://yarnpkg.com/downloads/$YARN_VERSION/yarn-v$YARN_VERSION.tar.gz" \
  && mkdir -p /opt \
  && tar -xzf yarn-v$YARN_VERSION.tar.gz -C /opt/ \
  && ln -s /opt/yarn-v$YARN_VERSION/bin/yarn /usr/local/bin/yarn \
  && ln -s /opt/yarn-v$YARN_VERSION/bin/yarnpkg /usr/local/bin/yarnpkg \
  && rm yarn-v$YARN_VERSION.tar.gz
