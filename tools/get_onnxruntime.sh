#!/bin/bash
# Downloads the prebuilt ONNX Runtime C/C++ library into third_party/onnxruntime
set -e
VER=1.20.1
cd "$(dirname "$0")/.."
mkdir -p third_party
cd third_party
wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${VER}/onnxruntime-linux-x64-${VER}.tgz"
tar xzf "onnxruntime-linux-x64-${VER}.tgz"
rm -rf onnxruntime
mv "onnxruntime-linux-x64-${VER}" onnxruntime
rm "onnxruntime-linux-x64-${VER}.tgz"
ls onnxruntime/include/onnxruntime_cxx_api.h && echo "ONNX Runtime ${VER} ready in third_party/onnxruntime"
