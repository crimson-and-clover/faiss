#!/bin/bash
set -e

# 安装依赖
# sudo apt install -y swig
# pip install wheel setuptools

# 配置 CMake
cmake -B build . \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_C_API=ON \
    -DCMAKE_BUILD_TYPE=Release

# 编译 faiss 库
make -C build -j faiss

# 编译 Python 绑定（SWIG 包装）
echo "Building Python bindings..."
make -C build -j swigfaiss

# 构建 Python wheel 包
echo "Building Python wheel..."
cd build/faiss/python
python setup.py bdist_wheel

# 显示生成的 wheel 文件
echo "Wheel package created:"
ls -lh dist/*.whl
