#!/bin/bash
# 编译 protobuf 生成 Python 代码
#
# 必须在源码树根（src/）运行 protoc：protoc 的 python 生成器以
# proto 文件相对 --proto_path 的完整路径作为模块名
# （tributo.serving.proto.inference_pb2），生成文件也落在同名路径下——
# 模块名与文件位置自洽，__module__ 无需 sed 修补，pickle/序列化直接可用。
# 默认生成回源码树；设置 PROTO_OUTPUT_DIR 时可生成到临时目录供 CI
# 比较，而不会修改 checkout 中的文件。
#
# 历史教训：此前在 proto/ 目录内生成（模块名裸 "inference_pb2"），
# 再用 sed 修补 BuildTopDescriptorsAndMessages 的字符串，但修补值与
# 实际 import 路径不一致，导致 protobuf 对象无法 pickle。

set -e

SRC_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
PROTO_REL="tributo/serving/proto/inference.proto"
OUTPUT_DIR="${PROTO_OUTPUT_DIR:-.}"

cd "${SRC_DIR}"
mkdir -p "${OUTPUT_DIR}"

uv run python -m grpc_tools.protoc \
    -I . \
    --python_out="${OUTPUT_DIR}" \
    --grpc_python_out="${OUTPUT_DIR}" \
    "${PROTO_REL}"

echo "Generated:"
ls -la "${OUTPUT_DIR}/${PROTO_REL%/*}/inference_pb2"*.py
