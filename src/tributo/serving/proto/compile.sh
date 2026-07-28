#!/bin/bash
# 编译 protobuf 生成 Python 代码

set -e

PROTO_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${PROTO_DIR}/generated"

mkdir -p "${OUT_DIR}"

# 使用 grpcio-tools 编译（通过 uv run 使用项目虚拟环境）
uv run python -m grpc_tools.protoc \
    -I "${PROTO_DIR}" \
    --python_out="${OUT_DIR}" \
    --grpc_python_out="${OUT_DIR}" \
    "${PROTO_DIR}/inference.proto"

# 修复导入路径：将绝对导入改为相对导入（兼容 macOS 和 Linux）
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's/^import inference_pb2/from . import inference_pb2/' "${OUT_DIR}/inference_pb2_grpc.py"
    # 修复 __module__ 为完整路径（pickle 序列化需要）
    sed -i '' "s/_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'inference_pb2', _globals)/_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tributo.serving.proto.generated.inference_pb2', _globals)/" "${OUT_DIR}/inference_pb2.py"
else
    sed -i 's/^import inference_pb2/from . import inference_pb2/' "${OUT_DIR}/inference_pb2_grpc.py"
    sed -i "s/_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'inference_pb2', _globals)/_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tributo.serving.proto.generated.inference_pb2', _globals)/" "${OUT_DIR}/inference_pb2.py"
fi

# 生成 __init__.py
touch "${OUT_DIR}/__init__.py"

echo "Generated files in ${OUT_DIR}:"
ls -la "${OUT_DIR}/"
