#!/bin/bash

set -e

SCALE_DIR="$1"
SCALE_GPU_ARCH="$2"
EXAMPLE="$3"

source "${SCALE_DIR}/bin/scaleenv" "${SCALE_GPU_ARCH}"

case "${EXAMPLE}" in

    "basic" | "blas" | "ptx")
        rm -rf "src/${EXAMPLE}/build"

        cmake \
            -DCMAKE_CUDA_ARCHITECTURES="${CUDAARCHS}" \
            -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -B "src/${EXAMPLE}/build" \
            "src/${EXAMPLE}"

        make \
            -C "src/${EXAMPLE}/build"

        export SCALE_EXCEPTIONS=1

        "src/${EXAMPLE}/build/example_${EXAMPLE}"
    ;;

    *)
        echo "Usage: $0 {PATH_TO_SCALE} {basic|blas|ptx}"
    ;;

esac
