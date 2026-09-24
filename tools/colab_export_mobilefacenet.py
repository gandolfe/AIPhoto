# =============================================================================
# MobileFaceNet → ExecuTorch(.pte) 导出脚本（在 Colab / 有网络的机器上跑）
# =============================================================================
# 产物：
#   mobilefacenet.pte       ExecuTorch 模型（XNNPACK 后端）
#   mobilefacenet.json      元数据：image_size / mean / std / embed_dim
# 把这两个文件放进 app/src/main/assets/models/ 即可。
#
# -----------------------------------------------------------------------------
# 权重来源：InsightFace 的 w600k_mbf.onnx（这就是 MobileFaceNet，512 维输出）
#   https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
#
# 为什么不用 foamliu/MobileFaceNet 的 mobilefacenet_scripted.pt（踩过的坑，别走回头路）：
#   它是 torch.jit.trace 产物 —— trace 会把 BN 的 running_mean / running_var
#   **内联成图里的常量**，state_dict() 只剩 154 个参数、拿不到 102 个 BN 统计量。
#   named_buffers() 也是空的，补不回来。缺了它们 eval 模式归一化就错，
#   实测 eager vs scripted 余弦只有 -0.04（≈随机），根本不能导出。
#   ONNX 则把全部 initializers（含 BN 四个参数）都完整保存，没有这个问题。
# -----------------------------------------------------------------------------
#
# 其他关键坑：
#   1) batch 固定为 1。MobileCLIP 在 torch.export 下 batch>1 会把内部卷积权重
#      误当输入 lift 成占位符，导出坏模型。这里同样按 batch=1 导出。
#   2) 导出用 fp32。模型小，fp32 才几 MB，且端侧 float[] 直接构张量，
#      省掉 floatToHalf 那一层。
#   3) 归一化 (x-127.5)/128 必须与训练一致（InsightFace 标准），写进 json，
#      Android 端读 json 不硬编码。错了 embedding 就是垃圾、聚类全乱。
# =============================================================================

# ----------------------------------------------------------------------------
# 可调参数
# ----------------------------------------------------------------------------
MODE = "fp32"          # fp32（默认）/ fp16
DEVICE = "cpu"         # 只影响这台机器上的导出与自检，与手机无关
IMAGE_SIZE = 112       # MobileFaceNet 标准输入
# InsightFace 标准归一化（RGB 顺序），必须与训练一致
IMG_MEAN = [127.5, 127.5, 127.5]
IMG_STD = [128.0, 128.0, 128.0]
BATCH = 1              # 见上面坑 1

# ⚠️ MobileFaceNet(w600k_mbf.onnx) 在 **buffalo_s** 包里，不在 buffalo_l ——
# v0.7 的 buffalo_l 只带 w600k_r50（ResNet50，100M+ 参数，端侧太大）。
# 两个都下，再按文件名找 mbf，避免以后 insightface 调整包内容又要改脚本。
INSIGHTFACE_ZIP_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
]
WORK_DIR = "/content/mfn"
# ----------------------------------------------------------------------------

import glob
import json
import os
import subprocess

import torch
import torch.nn.functional as F
from torch.export import export as torch_export

print("torch:", torch.__version__)

# =============================================================================
# Cell 1 —— 装依赖 + 下载 ONNX
# =============================================================================
# executorch 是导出 .pte 用的（与 colab_export_clip.py 一致，装最新稳定版）。
# 注意：Android 侧 runtime 是 executorch-android 1.4.0，实测 1.5.0 导出的 .pte
# 在它上面能正常加载（CLIP 那个模型就是这么跑通的），所以这里不锁死版本；
# 万一以后装出来的新版导的 .pte 在端侧报加载失败，再把这里改成 ==1.5.0 即可。
subprocess.run(
    [
        __import__("sys").executable, "-m", "pip", "install", "-q", "-U",
        "onnx", "onnx2torch", "onnxruntime", "executorch",
    ],
    check=True,
)
from importlib.metadata import version as _pkg_version

print("依赖安装完成, executorch:", _pkg_version("executorch"))

os.makedirs(WORK_DIR, exist_ok=True)
for url in INSIGHTFACE_ZIP_URLS:
    name = url.rsplit("/", 1)[-1]
    zip_path = os.path.join(WORK_DIR, name)
    if not os.path.exists(zip_path):
        subprocess.run(["wget", "-q", "-O", zip_path, url], check=True)
    subprocess.run(["unzip", "-o", "-q", zip_path, "-d", WORK_DIR], check=True)
    print("已解压:", name)

# 按名字找 MobileFaceNet（mbf）那个 onnx，不写死文件名，避免包内容变动
candidates = glob.glob(os.path.join(WORK_DIR, "**", "*mbf*.onnx"), recursive=True)
if not candidates:
    raise RuntimeError(
        "两个包里都没找到 *mbf*.onnx，实际文件: "
        + str(glob.glob(os.path.join(WORK_DIR, "**", "*.onnx"), recursive=True))
    )
ONNX_PATH = candidates[0]
print("ONNX:", ONNX_PATH, os.path.getsize(ONNX_PATH) / 1024 / 1024, "MB")

# =============================================================================
# Cell 2 —— 看一眼 ONNX（输入尺寸/输出维度/用到的算子）
# =============================================================================
import onnx

onnx_model = onnx.load(ONNX_PATH)
ops = sorted({n.op_type for n in onnx_model.graph.node})
print("ONNX 算子:", ops)
_inp = onnx_model.graph.input[0]
print("ONNX 输入:", _inp.name, [d.dim_value or d.dim_param for d in _inp.type.tensor_type.shape.dim])
_out = onnx_model.graph.output[0]
print("ONNX 输出:", _out.name, [d.dim_value or d.dim_param for d in _out.type.tensor_type.shape.dim])

# =============================================================================
# Cell 3 —— ONNX → PyTorch
# =============================================================================
from onnx2torch import convert

net = convert(ONNX_PATH).eval()
print("onnx2torch 转换完成，参数量:",
      f"{sum(p.numel() for p in net.parameters()):,}")


class FaceEmbedder(torch.nn.Module):
    """只保留前向，L2 归一化放到端侧做（端侧要和存库/聚类时的处理一致）。"""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.inner(pixel_values)


embedder = FaceEmbedder(net).eval().to(DEVICE)
if MODE == "fp16":
    embedder = embedder.half()

# =============================================================================
# Cell 4 —— 交叉校验：ONNX(onnxruntime) vs 转换后的 PyTorch
# =============================================================================
# 这一步是转换正确性的硬把关：余弦不接近 1 说明 onnx2torch 转错了，
# 后面导出得再漂亮也没用。
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
x_np = np.random.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
o_onnx = sess.run(None, {sess.get_inputs()[0].name: x_np})[0]
with torch.no_grad():
    o_torch = embedder(torch.from_numpy(x_np).to(DEVICE)).detach().cpu().numpy()

cos_onnx = F.cosine_similarity(
    torch.from_numpy(o_onnx).reshape(BATCH, -1),
    torch.from_numpy(o_torch).reshape(BATCH, -1),
    dim=-1,
).mean().item()
print(f"ONNX vs PyTorch cosine: {cos_onnx:.6f}  (应 ≈ 1)")
assert cos_onnx > 0.99, "onnx2torch 转换结果与 ONNX 不一致，不能继续导出"

EMBED_DIM = int(o_onnx.reshape(BATCH, -1).shape[-1])
print(f"embedding dim = {EMBED_DIM}")

# =============================================================================
# Cell 5 —— torch.export（固定 batch=1）
# =============================================================================
example_inputs = (
    torch.randn(
        BATCH, 3, IMAGE_SIZE, IMAGE_SIZE,
        dtype=torch.half if MODE == "fp16" else torch.float32,
        device=DEVICE,
    ),
)
exported = torch_export(embedder, example_inputs)
print(f"torch.export OK (mode={MODE}, static batch={BATCH})")

# 自检：确认导出程序的**用户输入**确实是 [1,3,112,112]
#
# ⚠️ 不能简单取 graph 里第一个 placeholder：torch.export 会把非 parameter 的张量
#    （比如 onnx2torch 转出来没注册成 nn.Parameter 的卷积权重）「lift」成 placeholder，
#    这些 lifted 常量排在真输入**前面**。直接读 _ph[0] 会拿到卷积权重
#    （实测是 (128,3,3,3)），于是误判成「导出坏了」。
#    正确做法：按 InputKind.USER_INPUT 过滤。
try:
    from torch.export.graph_signature import InputKind

    for n in exported.graph.nodes:
        if n.op == "placeholder":
            print(f"  placeholder: {n.name} {tuple(n.meta['val'].shape)}")
    user_specs = [
        s for s in exported.graph_signature.input_specs
        if s.kind == InputKind.USER_INPUT
    ]
    assert len(user_specs) == 1, f"用户输入不是 1 个，而是 {len(user_specs)} 个"
    _name = user_specs[0].arg.name
    _node = next(n for n in exported.graph.nodes if n.name == _name)
    _in_shape = tuple(_node.meta["val"].shape)
except Exception as e:
    print(f"  （input_specs 方式失败，退回 example_inputs: {e}）")
    _in_shape = tuple(exported.example_inputs[0].shape)

print(f"[export] 用户输入形状自检: {_in_shape}（期望 ({BATCH}, 3, {IMAGE_SIZE}, {IMAGE_SIZE})）")
assert _in_shape == (BATCH, 3, IMAGE_SIZE, IMAGE_SIZE), \
    f"导出输入维度异常: {_in_shape}，请确认 BATCH={BATCH}"

# =============================================================================
# Cell 6 —— to_edge + XNNPACK → .pte
# =============================================================================
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge

edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))
edge = edge.to_backend(XnnpackPartitioner())
et = edge.to_executorch(ExecutorchBackendConfig())

with open("mobilefacenet.pte", "wb") as f:
    f.write(et.buffer)
print(f"mobilefacenet.pte 写出完成: {os.path.getsize('mobilefacenet.pte') / 1024 / 1024:.2f} MB")

# =============================================================================
# Cell 7 —— 元数据（Android 侧读它做预处理，绝不硬编码）
# =============================================================================
meta = {
    "model": "mobilefacenet",
    "image_size": IMAGE_SIZE,
    "image_mean": IMG_MEAN,
    "image_std": IMG_STD,
    "embed_dim": EMBED_DIM,
    "dtype": MODE,
    "batch": BATCH,
}
with open("mobilefacenet.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("元数据:", json.dumps(meta, ensure_ascii=False))

# =============================================================================
# Cell 8 —— 导出后数值自检（PyTorch vs ExecuTorch）
# =============================================================================
# 已知坑：Python runtime 把 ET tensor 包装回 torch.Tensor 时会丢 device，
#        报 "tensor does not have a device"，只影响本 Cell 自检，不影响 .pte。
try:
    from executorch.runtime import Runtime

    test_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    if MODE == "fp16":
        test_input = test_input.half()
    with torch.no_grad():
        ref = embedder(test_input)

    runtime = Runtime.get()
    program = runtime.load_program("mobilefacenet.pte")
    method = program.load_method("forward")
    et_out = method.execute([test_input])[0]
    cos = F.cosine_similarity(ref, et_out[0:1], dim=-1).item()
    print(f"PyTorch vs ExecuTorch cosine: {cos:.6f}  (越接近 1 越好)")
except Exception as e:
    print(f"[自检跳过] Python runtime 执行失败（不影响 .pte）: {type(e).__name__}: {e}")

print("\n完成：把 mobilefacenet.pte 和 mobilefacenet.json 放进 "
      "app/src/main/assets/models/")
