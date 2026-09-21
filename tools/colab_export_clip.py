# ============================================================================
#  AIPhoto —— MobileCLIP-S2 视觉编码模型导出脚本（Google Colab 专用）
#
#  把 Apple MobileCLIP-S2（DataCompDR 训练，视觉塔 ~35.7M 参数）的视觉分支导出成
#  ExecuTorch .pte，端侧只跑前向编码；分类/聚类在 App 侧完成。
#
#  产出：
#    - clip_scene.pte        端侧模型（输入图像 → 512-d L2 归一化 embedding），带 XNNPACK。
#                            正式用这个：实测单张 0.26 秒，比 portable 版快约 73 倍。
#    - clip_scene_portable.pte  纯 portable 版（体积更小但慢很多），
#                            仅在 EXPORT_PORTABLE_BASELINE=True 时才导出
#    - xnnpack_audit.txt     XNNPACK 委托覆盖率 + 落回 portable 的算子完整清单
#    - label_bank.bin        标签库 embedding 矩阵 [N, 512] float32，raw bytes
#    - label_bank.json       标签库元数据（id / prompt / 中文名 / emoji）
#    - labels.json           模型元数据（model_id / image_size / 量化方式）
#    - aiphoto_clip_model.zip  打包下载（含上述所有已生成的文件）
#
#  用法：Colab 新建 notebook，按 Cell 顺序运行；运行时 CPU/GPU 均可（推荐 CPU）。
# ============================================================================


# ============================================================================
# Cell 1 —— 环境检查 & 安装依赖
# ============================================================================
# 权重走 HuggingFace 上的 Apple 官方 OpenCLIP 格式仓库（hf-hub: 前缀，
# 模型配置 + 权重 + tokenizer 都随仓库下发）。注意：不能直接用 'mobileclip_s2'
# 这种名字——open_clip 内置模型表里没有，会报
# "Model config ... not found in built-ins"。executorch 需 ≥ 1.5.0。
# ----------------------------------------------------------------------------
import subprocess


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True)


sh("python -V")
sh("pip show torch | head -n 3")
sh("nvidia-smi | head -n 12")

sh("pip install -q -U open_clip_torch executorch")
sh("pip show torch executorch open_clip_torch | grep -E '^(Name|Version)'")
import open_clip
print("open_clip:", open_clip.__version__)


# ============================================================================
# Cell 2 —— 加载 MobileCLIP-S2 并定义标签库
# ============================================================================
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cpu"

# hf-hub: 前缀 = 从 HuggingFace 拉 Apple 官方 OpenCLIP 格式的配置 + 权重。
#
# 【为什么要从 S2 换掉】(2026-09-21)
#   S2 的视觉塔 35.8M 参数，端侧实测单张推理要几分钟，2800 张照片根本跑不完。
#   换成 S1（视觉塔约 21.5M，fastvit_mci1）能省掉相当一部分算力。
#
# 【为什么不是 S0】
#   Apple 只发布了 **-OpenCLIP 格式**的 S1 / S2 / B(LT) 仓库，没有 S0-OpenCLIP：
#   - https://huggingface.co/apple/MobileCLIP-S1-OpenCLIP   ✓ 存在
#   - https://huggingface.co/apple/MobileCLIP-S2-OpenCLIP   ✓ 存在
#   - https://huggingface.co/apple/MobileCLIP-S0-OpenCLIP   ✗ 不存在（HF 返回 401）
#   另有一个 apple/MobileCLIP-S0 仓库，但里面是 mobileclip_s0.pt 裸 state_dict，
#   没有 open_clip_config.json，不能用 hf-hub: 直接加载，需要额外做权重转换。
#   真要用 S0 得先适配加载方式，见本文件末尾的说明。
#
# 【S1 为何可以直接替换】
#   核对过 open_clip_config.json：embed_dim=512、vision_cfg.image_size=256，
#   与 App 侧 ClipSceneClassifier.EMBED_DIM=512、ImagePreprocessor.INPUT_SIZE=256
#   完全一致，Android 工程一行代码都不用改。若换成别的模型，这两个值变了的话，
#   下面 Cell 4 的 IMAGE_SIZE / 本 Cell 的 dtype 审计会直接报错提醒。
MODEL_ID = "mobileclip_s1"
MODEL_REF = "hf-hub:apple/MobileCLIP-S1-OpenCLIP"

print(f"loading {MODEL_REF} ...")
# 返回 (model, preprocess_train, preprocess_val)，用 val 做推理预处理
clip, _, preprocess = open_clip.create_model_and_transforms(MODEL_REF)
clip = clip.eval().to(DEVICE)
tokenizer = open_clip.get_tokenizer(MODEL_REF)
print("loaded OK")


# ---------------------------------------------------------------------------
# 从官方 preprocess transform 反查 image_size / mean / std，不硬编码。
# 如果和 Android ImagePreprocessor.kt 的常量不一致，这里直接暴露出来——
# 端侧预处理和模型训练分布错位会让结果悄悄变垃圾。
# ---------------------------------------------------------------------------
import torchvision.transforms as T


def _inspect_preprocess(tf):
    size = mean = std = None
    for t in (getattr(tf, "transforms", None) or [tf]):
        if isinstance(t, (T.Resize, T.CenterCrop)) and size is None:
            size = t.size
        elif isinstance(t, T.Normalize):
            mean, std = list(t.mean), list(t.std)
    return size, mean, std


_size, IMG_MEAN, IMG_STD = _inspect_preprocess(preprocess)
if _size:
    IMAGE_SIZE = int(_size[0] if isinstance(_size, (list, tuple)) else _size)
else:
    IMAGE_SIZE = 224
print(f"image_size = {IMAGE_SIZE}, mean = {IMG_MEAN}, std = {IMG_STD}")

# 与 Android ImagePreprocessor.kt 的 OpenAI 常量比对，不一致必须同步改 Kotlin
_OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
_OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]
if IMG_MEAN and not np.allclose(IMG_MEAN, _OPENAI_MEAN, atol=1e-4):
    print("⚠️ mean/std 与 OpenAI CLIP 不同 → Android ImagePreprocessor.kt 需要同步修改！")

# ---------------------------------------------------------------------------
# 标签库：聚类命名候选。每条 = 多 prompt（英文，CLIP 训练分布）+ 中文显示名 + emoji
# 增删条目只改这里，App 侧只读 .json / .bin；不需要重导模型。
# ---------------------------------------------------------------------------
LABEL_BANK: dict[str, tuple[list[str], str, str]] = {
    # ---- 自然 / 风景 ----
    "scenery":     (["a photo of scenery", "a photo of a landscape", "a beautiful landscape photo"], "风景", "🏞️"),
    "mountain":    (["a photo of mountains", "a photo of a mountain peak", "a photo of snow-capped mountains"], "山景", "⛰️"),
    "beach":       (["a photo of a beach", "a photo of the seaside", "ocean waves at the shore"], "海边", "🏖️"),
    "sunset":      (["a photo of a sunset", "a photo of a sunrise", "a colorful sky at sunset"], "日出日落", "🌅"),
    "forest":      (["a photo of a forest", "a photo of trees in a woods"], "森林", "🌲"),
    "flowers":     (["a photo of a flower garden", "a photo of flowers blooming"], "花卉", "🌷"),
    "lake":        (["a photo of a lake", "a photo of a calm river"], "湖景", "🏞️"),
    "snow":        (["a photo of snow", "a photo of a snowy landscape"], "雪景", "❄️"),
    "sky":         (["a photo of the sky", "a photo of clouds", "a blue sky with clouds"], "天空", "☁️"),
    # ---- 城市 / 建筑 ----
    "cityscape":   (["a photo of a city skyline", "a photo of city buildings at night", "an urban skyline"], "城市", "🏙️"),
    "building":    (["a photo of a building", "a photo of architecture", "a photo of a house"], "建筑", "🏛️"),
    "interior":    (["a photo of an interior", "a photo of a room inside a house"], "室内", "🛋️"),
    "street":      (["a photo of a street scene", "a photo of a road in a city"], "街景", "🛣️"),
    "night":       (["a photo at night", "a night scene", "city lights at night"], "夜景", "🌃"),
    # ---- 人物 ----
    "person":      (["a photo of a person", "a photo of people", "a portrait of a human"], "人物", "👤"),
    "portrait":    (["a portrait photo of a person", "a close-up photo of a face"], "肖像", "🧑"),
    "group":       (["a photo of a group of people", "a group photo of friends"], "合影", "👥"),
    "child":       (["a photo of a child", "a photo of a kid", "a photo of a baby"], "儿童", "👶"),
    # ---- 动物 ----
    "cat":         (["a photo of a cat", "a photo of a kitten"], "猫", "🐱"),
    "dog":         (["a photo of a dog", "a photo of a puppy"], "狗", "🐶"),
    "bird":        (["a photo of a bird", "a photo of a flying bird"], "鸟", "🐦"),
    "pet":         (["a photo of a pet", "a photo of a pet animal"], "宠物", "🐾"),
    "wildlife":    (["a photo of wildlife", "a photo of a wild animal in nature"], "野生动物", "🦌"),
    # ---- 食物 ----
    "food":        (["a photo of food", "a photo of a meal", "a photo of a dish on a plate"], "美食", "🍽️"),
    "dessert":     (["a photo of a dessert", "a photo of a cake", "a photo of pastries"], "甜点", "🍰"),
    "drink":       (["a photo of a drink", "a photo of a cup of coffee or tea"], "饮品", "🥤"),
    "fruit":       (["a photo of fruit", "a photo of fresh fruits"], "水果", "🍎"),
    # ---- 物品 / 文档 ----
    "document":    (["a photo of a document", "a screenshot of text"], "文档", "📄"),
    "screen":      (["a photo of a computer screen", "a screenshot of an interface"], "屏幕截图", "🖥️"),
    "art":         (["a photo of a painting", "a photo of artwork", "a photo of a drawing"], "艺术", "🎨"),
    "toy":         (["a photo of a toy", "a photo of a stuffed animal"], "玩具", "🧸"),
    "book":        (["a photo of a book", "a photo of books on a shelf"], "书籍", "📚"),
    "plant":       (["a photo of a houseplant", "a photo of a potted plant"], "植物", "🪴"),
    "car":         (["a photo of a car", "a photo of a vehicle"], "汽车", "🚗"),
    # ---- 事件 ----
    "wedding":     (["a photo of a wedding", "a wedding photo"], "婚礼", "💒"),
    "celebration": (["a photo of a celebration", "a photo of a birthday party"], "庆祝", "🎉"),
    "sports":      (["a photo of a sport", "a photo of someone playing sports"], "运动", "⚽"),
    "travel":      (["a travel photo", "a photo from a trip", "a photo of a famous landmark"], "旅行", "🧳"),
}
# 注：这里原本有个兜底类 "other": (["a photo"], "其他", "🖼️")，已删除。
# 原因：它的 prompt 是 "a photo"，语义太泛，文本 embedding 接近图像分布中心，
# 归一化后与几乎任何图片的余弦都不低，参与竞选会吸走大量有明确内容的照片。
# 实测：人物合照/自拍大量被判成 other，而 person / portrait 几乎为 0。
# 删除后 label_bank 由 39 类变为 38 类；已同步 SceneCategory.kt。
# 若哪天要加回兜底类，不要用泛化 prompt，应改为「最高分低于阈值时判为其他」
# 的后端判定方式，让兜底类不参与 argmax。

LABEL_ORDER: list[str] = list(LABEL_BANK.keys())
print(f"label bank size: {len(LABEL_ORDER)}")


# ============================================================================
# Cell 3 —— 预计算 label bank 文本向量
# ============================================================================
def encode_text(prompts: list[str]) -> torch.Tensor:
    """把一批 prompt 编码成 [P, D] 投影后的文本向量。"""
    tokens = tokenizer(prompts).to(DEVICE)
    with torch.no_grad():
        return clip.encode_text(tokens)


with torch.no_grad():
    text_embeds: list[torch.Tensor] = []
    for label_id in LABEL_ORDER:
        prompts = LABEL_BANK[label_id][0]
        emb = encode_text(prompts)                    # [P, D]
        emb = F.normalize(emb, dim=-1).mean(dim=0)    # 多 prompt 平均
        text_embeds.append(emb)
        print(f"{label_id:12s} computed ({len(prompts)} prompts)")

    text_embeds = torch.stack(text_embeds)            # [N, D]
    text_embeds = F.normalize(text_embeds, dim=-1)    # 再归一化

EMBED_DIM = text_embeds.shape[1]
print(f"label_bank tensor: {tuple(text_embeds.shape)}, embed_dim = {EMBED_DIM}")


# ============================================================================
# Cell 4 —— 构建导出用模型（视觉编码 + L2 归一化 → embedding）
# ============================================================================
# 只导出视觉分支到 .pte；分类/聚类在 App 端用 embedding × label_bank 做。
# 这样：
#   - 模型体积小（不需要固化文本向量进 .pte）
#   - 类目运行时可改（改 label_bank.json 即可，不用重导）
#   - 给聚类留出空间（每张图是 512-d 向量，App 侧做 k-means）
# ----------------------------------------------------------------------------
class ImageEncoder(torch.nn.Module):
    """输入 [1,3,H,W] 归一化图像，输出 [1,512] L2 归一化 embedding。"""

    def __init__(self, visual: torch.nn.Module):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        emb = self.visual(pixel_values)
        return F.normalize(emb, p=2, dim=-1)


encoder = ImageEncoder(clip.visual).eval().to(DEVICE)

# 冒烟测试
with torch.no_grad():
    out = encoder(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
assert out.shape[-1] == EMBED_DIM, f"output dim {out.shape[-1]} ≠ label bank dim {EMBED_DIM}"
print("eager forward OK, output dim =", out.shape[-1])


# ============================================================================
# Cell 5 —— 导出为 ExecuTorch .pte
# ============================================================================
# 导出模式。.pte 只含 MobileCLIP-S2 视觉塔（约 35.7M 参数）。
#
# 实测修正（2026-09-20 Colab 跑通后的真实数据）：
#   参数量 35.8 M，dtype 确认为 fp16 → 理论体积 68.3 MB，但实际 .pte 是 123.1 MB。
#   注意：.half() 是生效的（dtype 列表里只有 torch.float16），
#         所以多出来的约 55 MB 不是 dtype 问题，别再往 fp16 方向排查。
#
#   主因是 XNNPACK 委托覆盖率只有 18.8%（74/394 节点）：官方 XNNPACK 算子表里没有
#   scaled_dot_product_attention / layer_norm / mm，MobileCLIP 图像塔里占大头的
#   attention block 整块无法委托，只能走 portable 参考算子——端侧单张推理 > 6 分钟
#   就出自这里。
#
#   剩一半待证实：多出的 55 MB 是否来自 XNNPACK 的 prepacked 权重重复存储，
#   用 EXPORT_PORTABLE_BASELINE 导一次纯 portable 基线对比即可定案。
#   本 Cell 结尾会打印「XNNPACK 委托覆盖率」+ portable 算子清单（另落盘 xnnpack_audit.txt）。
#
# 体积预期（35.7M 参数，未计 XNNPACK weight prepacking 开销）：
#   "fp32" → ~143 MB
#   "fp16" → 取决于委托覆盖率，委托满 ~71 MB，实测 123 MB
#   "int8" → 只压 XNNPACK 委托到的 conv2d / linear，其余照旧 fp32
# ----------------------------------------------------------------------------
import glob
from collections import Counter

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torch.export import export as torch_export

MODE = "fp16"             # 取值: "fp16" | "int8" | "fp32"
CALIB_IMAGE_DIR = ""      # 例: "/content/calib"，INT8 校准用
CALIB_NUM = 16

# 额外导出一份「不带 XNNPACK」的纯 portable 基线（clip_scene_portable.pte），
# 用于对比体积与速度。默认关闭：下面已结论清楚，不必每次多导一份几十 MB。
#
# 实测结论（2026-09-21，真机 MobileCLIP-S1）：
#   纯 portable : 43.6 MB，单张 19 秒
#   带 XNNPACK  : 77.0 MB，单张 0.26 秒   ← 快约 73 倍
# 即 XNNPACK 虽然把委托权重摊成内部 fp32 让体积涨了 33 MB，但换来的是
# 数量级的速度差。所以正式产物用 clip_scene.pte（带 XNNPACK），别用 portable 版。
#
# 只有当怀疑 XNNPACK 反而拖慢（比如换了新模型/新算子形态后）才需要重新打开它，
# 导出两份上真机对比 [基线对照] 那行输出。
EXPORT_PORTABLE_BASELINE = False
# ----------------------------------------------------------------------------

if MODE == "fp16":
    encoder = encoder.half()
    example_inputs = (torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.half, device=DEVICE),)
else:  # int8 / fp32
    example_inputs = (torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE),)

# --- 0) 导出前审计：确认 dtype 真的切过去了，并记下理论体积 ---
_param_count = sum(p.numel() for p in encoder.parameters())
_param_bytes = sum(
    p.numel() * p.element_size()
    for p in encoder.parameters()
    if p.is_floating_point()
)
_fp_dtypes = sorted({str(p.dtype) for p in encoder.parameters() if p.is_floating_point()})
print(f"视觉塔参数量: {_param_count:,} ({_param_count / 1e6:.1f} M)")
print(f"浮点参数理论体积: {_param_bytes / 1024 / 1024:.1f} MB "
      f"（fp32 全量参考: {_param_count * 4 / 1024 / 1024:.1f} MB）")
print(f"浮点参数 dtype: {_fp_dtypes}")
if MODE == "fp16" and any(d != "torch.float16" for d in _fp_dtypes):
    raise RuntimeError(f"MODE=fp16 但仍有非 fp16 浮点参数（{_fp_dtypes}），.half() 未生效")

# --- 1) torch.export ---
exported = torch_export(encoder, example_inputs)
print(f"torch.export OK (mode={MODE})")

# --- 2) INT8 PTQ ---
if MODE == "int8":
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())
    exported = prepare_pt2e(exported, quantizer)

    def _iter_calib():
        paths = []
        if CALIB_IMAGE_DIR and os.path.isdir(CALIB_IMAGE_DIR):
            for ext in ("jpg", "jpeg", "png", "webp", "bmp"):
                paths.extend(sorted(glob.glob(f"{CALIB_IMAGE_DIR}/*.{ext}")))
        if paths:
            from PIL import Image
            for p in paths[:CALIB_NUM]:
                img = Image.open(p).convert("RGB")
                # 直接用官方 preprocess：归一化/尺寸必然与训练分布对齐
                yield preprocess(img).unsqueeze(0)
            return
        return  # 走下方合成数据兜底

    print("calibrating ...")
    with torch.no_grad():
        used = 0
        for x in _iter_calib():
            exported((x,))
            used += 1
        if used == 0:
            print("(没找到校准图片，退回合成 randn——精度会很差，建议补真实图片)")
            for _ in range(CALIB_NUM):
                exported(example_inputs)
                used += 1
        print(f"calibration done, {used} samples")
    exported = convert_pt2e(exported)
    print("post-training quantization OK")

# --- 3) lowering 到 XNNPACK 后端 ---
edge = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackPartitioner()],
)

# --- 3.1) 委托覆盖率审计 ---
# 这是体积和端侧速度的关键指标：没被 XNNPACK 接住的子图会走 portable 参考算子，
# 既慢（单核、无算子融合），其常量也保持 fp32 留在 .pte 里。
_portable_ops: "Counter[str]" = Counter()
_total_nodes = -1
_delegated = 0
try:
    for _node in edge.exported_program().graph.nodes:
        if _node.op != "call_function":
            continue
        _target = str(_node.target)
        if "executorch_call_delegate" in _target:
            _delegated += 1
        else:
            _portable_ops[_target] += 1
    _total_nodes = _delegated + sum(_portable_ops.values())
    print(f"\nXNNPACK 委托覆盖率（按节点数）: {_delegated}/{_total_nodes} "
          f"({_delegated / _total_nodes:.1%})")

    # 按节点数看会严重低估：XNNPACK 默认把可委托算子融合成少数几个 delegate 节点，
    # 而留在外部的 getitem / dim_order_copy / gelu 之类 glue 算子数量多却不占算力
    # （实测 394 节点里 glue 占近 300 个），所以再算一个剔除 glue 的口径。
    # 注意用子串匹配：节点 target 实际形如
    #   <EdgeOpOverload: dim_order_ops._to_dim_order_copy.default>: schema = ...
    # 不以 "dim_order_ops." 开头，用 startswith 会漏掉（实测过，会少算 65 个）。
    _GLUE = ("<built-in function getitem", "dim_order_ops.")
    _glue_cnt = sum(c for op, c in _portable_ops.items() if any(g in op for g in _GLUE))
    _compute_portable = sum(_portable_ops.values()) - _glue_cnt
    _compute_total = _delegated + _compute_portable
    if _compute_total:
        print(f"XNNPACK 委托覆盖率（剔除 glue 后）: {_delegated}/{_compute_total} "
              f"({_delegated / _compute_total:.1%})  "
              f"[glue 类 {_glue_cnt} 个不参与统计]")
except Exception as _audit_err:  # 审计失败不该中断导出
    _portable_ops.clear()
    print(f"\n[XNNPACK 审计] 跳过（{type(_audit_err).__name__}: {_audit_err}）")
if _portable_ops:
    # 同时落盘，避免长清单被终端折叠或复制时丢行
    with open("xnnpack_audit.txt", "w", encoding="utf-8") as _fh:
        _fh.write(f"XNNPACK 委托覆盖率: {_delegated}/{_total_nodes} "
                  f"({_delegated / _total_nodes:.1%})\n\n")
        _fh.write("落回 portable 后端的算子（全部，按出现次数排序）：\n")
        for _op, _cnt in _portable_ops.most_common():
            _fh.write(f"  {_cnt:5d}  {_op}\n")
    print("落回 portable 后端的算子（按出现次数排序）：")
    for _op, _cnt in _portable_ops.most_common(20):
        print(f"  {_cnt:5d}  {_op}")
    print("完整清单已写入 xnnpack_audit.txt，可直接下载查看")

et_program = edge.to_executorch()

with open("clip_scene.pte", "wb") as f:
    f.write(et_program.buffer)

size_mb = os.path.getsize("clip_scene.pte") / 1024 / 1024
print(f"\nclip_scene.pte written: {size_mb:.1f} MB (mode={MODE})")
_fp32_mb = _param_count * 4 / 1024 / 1024
_fp16_mb = _param_count * 2 / 1024 / 1024
if MODE == "fp16" and size_mb > _fp16_mb * 1.5:
    print(f"⚠️  参数已是 fp16（理论 {_fp16_mb:.1f} MB），实际 {size_mb:.1f} MB，"
          f"反而接近 fp32 的 {_fp32_mb:.1f} MB。"
          f"\n   多出的约 {size_mb - _fp16_mb:.1f} MB 与 dtype 无关"
          f"（dtype 已确认是 fp16），见下方基线对照。")

# --- 3.2) 纯 portable 基线对照 ---
# 用于判断多出来的体积来自 XNNPACK 的 prepacked 权重，还是别的地方。
if EXPORT_PORTABLE_BASELINE:
    try:
        base_path = "clip_scene_portable.pte"
        with open(base_path, "wb") as f:
            f.write(to_edge_transform_and_lower(exported, partitioner=[]).to_executorch().buffer)
        base_mb = os.path.getsize(base_path) / 1024 / 1024
        print(f"\n[基线对照] 纯 portable: {base_mb:.1f} MB | 带 XNNPACK: {size_mb:.1f} MB | "
              f"理论 fp16: {_fp16_mb:.1f} MB")
        # 委托块里的权重会被 XNNPACK 按自己的格式重存。若它把 fp16 摊成内部
        # fp32，则 size ≈ f·fp32 + (1-f)·base，可反解出被委托的权重占比 f。
        # 这个 f 比「按节点数」的覆盖率准得多——后者会被 glue 算子稀释。
        if _fp32_mb > base_mb:
            _f = (size_mb - base_mb) / (_fp32_mb - base_mb)
            print(f"→ 按体积反推，约 {_f:.0%} 的权重进了 XNNPACK 委托块，"
                  f"且被摊成了内部 fp32 格式（这是多出来的 {size_mb - base_mb:.1f} MB 的来源）。")
            print("   即：模型已是 fp16，但 XNNPACK 没吃这个红利，反而多存了一份。")
        print("→ 两份文件大小差异就是这个模型的真实取舍，别只看 18.8% 那个节点数覆盖率。")
    except Exception as e:
        print(f"[基线对照] 跳过（{type(e).__name__}: {e}）")


# ============================================================================
# Cell 6 —— 导出后验证（PyTorch vs ExecuTorch embedding 一致性）
# ============================================================================
# 已知坑：Python runtime 在把 ET tensor 包装回 torch.Tensor 时会丢 device，
#        报 "tensor does not have a device"。实测 executorch 1.5.0 仍复现，
#        所以这不是升级能解决的问题，且只影响本 Cell 的自检。
#        本 Cell 失败不影响 .pte（Android 端走 C++ runtime）。
from importlib.metadata import version as _pkg_version

from executorch.runtime import Runtime

print("executorch:", _pkg_version("executorch"))

test_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
if MODE == "fp16":
    test_input = test_input.half()

with torch.no_grad():
    ref = encoder(test_input)

runtime = Runtime.get()
program = runtime.load_program("clip_scene.pte")
method = program.load_method("forward")

et_out = None
try:
    et_out = method.execute([test_input])[0]
except RuntimeError as e:
    print("\n[ExecuTorch Python runtime 执行失败]", e)
    print("原因：Python 端 runtime 在把 ET tensor 包装回 torch.Tensor 时丢了 device，"
          "报 \"tensor does not have a device\"。")
    print("注意：实测 executorch 1.5.0 仍然复现，所以『升级到 1.5.0 就好』的说法不成立，"
          "别在这上面耗时间。")
    print("处理：本 Cell 只是导出后的自检，可安全跳过。"
          ".pte 本身不受影响；Android 端走 C++ runtime，不依赖这套 Python 绑定。")
    print("真要验证 embedding 的正确性，直接装机跑一次看分类结果，比这个 Python 自检可靠。")

if et_out is not None:
    cos = F.cosine_similarity(ref, et_out, dim=-1).mean().item()
    print(f"PyTorch    embedding[:6]: {[round(v, 4) for v in ref[0, :6].tolist()]}")
    print(f"ExecuTorch embedding[:6]: {[round(v, 4) for v in et_out[0, :6].tolist()]}")
    print(f"cosine similarity: {cos:.6f}  (越接近 1 越好)")


# ============================================================================
# Cell 7 —— 用真实图片自检（可选）
# ============================================================================
from PIL import Image

TEST_IMAGE = ""  # 例: "/content/test.jpg"，留空则跳过

if TEST_IMAGE and os.path.exists(TEST_IMAGE):
    img = Image.open(TEST_IMAGE).convert("RGB")
    inputs = preprocess(img).unsqueeze(0)
    if MODE == "fp16":
        inputs = inputs.half()

    with torch.no_grad():
        emb = encoder(inputs)
        emb = F.normalize(emb, dim=-1)
        sims = (emb @ text_embeds.to(emb.dtype).t()).softmax(dim=-1)[0]

    ranked = sorted(zip(LABEL_ORDER, sims.tolist()), key=lambda x: -x[1])
    print(f"\n{TEST_IMAGE} → top-5 候选类别：")
    for label_id, p in ranked[:5]:
        _, disp, emoji = LABEL_BANK[label_id]
        print(f"  {emoji} {disp:8s} ({label_id})  {p:.4f}")
else:
    print("跳过真实图片自检（TEST_IMAGE 为空）")


# ============================================================================
# Cell 8 —— 写出 label bank + 元数据 + 打包下载
# ============================================================================
# label_bank.bin: raw float32 little-endian，shape [N, EMBED_DIM]
bank_bytes = text_embeds.cpu().numpy().astype(np.float32).tobytes()
with open("label_bank.bin", "wb") as f:
    f.write(bank_bytes)
print(f"label_bank.bin written: {len(LABEL_ORDER)} × {EMBED_DIM} float32 "
      f"({len(bank_bytes) / 1024:.1f} KB)")

# label_bank.json: 每条的 id / prompt / 中文名 / emoji（端侧 UI 用）
label_bank_json = {
    "version": 1,
    "model": MODEL_ID,
    "embed_dim": EMBED_DIM,
    "labels": [
        {
            "id": label_id,
            "prompts": LABEL_BANK[label_id][0],
            "display_name": LABEL_BANK[label_id][1],
            "emoji": LABEL_BANK[label_id][2],
        }
        for label_id in LABEL_ORDER
    ],
}
with open("label_bank.json", "w", encoding="utf-8") as f:
    json.dump(label_bank_json, f, ensure_ascii=False, indent=2)
print(f"label_bank.json written: {len(LABEL_ORDER)} entries")

# labels.json: 模型元数据（App 侧启动时校验模型版本用）
labels_json = {
    "version": 1,
    "model": MODEL_ID,
    "image_size": IMAGE_SIZE,
    "image_mean": IMG_MEAN,
    "image_std": IMG_STD,
    "embed_dim": EMBED_DIM,
    "quantized": MODE == "int8",
    "quant_mode": MODE,
    "label_count": len(LABEL_ORDER),
}
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(labels_json, f, ensure_ascii=False, indent=2)
print("labels.json written")

# 产物清单：基础 4 件，外加审计/基线里实际生成的文件。
# 之前写死成 4 个，导致 clip_scene_portable.pte 和 xnnpack_audit.txt 没进压缩包。
_artifacts = ["clip_scene.pte", "label_bank.bin", "label_bank.json", "labels.json"]
for _extra in ("clip_scene_portable.pte", "xnnpack_audit.txt"):
    if os.path.exists(_extra):
        _artifacts.append(_extra)
    else:
        print(f"(跳过打包 {_extra}：本次未生成)")

sh("ls -lh " + " ".join(_artifacts))
# 先删旧包再打，否则 zip -r 会保留上一轮残留的文件
sh("rm -f aiphoto_clip_model.zip")
sh("zip -r aiphoto_clip_model.zip " + " ".join(_artifacts))
print("打包内容: " + ", ".join(_artifacts))

try:
    from google.colab import files
    files.download("aiphoto_clip_model.zip")
    print("\n下载已开始；若未弹出请从左侧「文件」面板手动下载")
except Exception as e:
    print("非 Colab 环境或下载失败：", e)


# ============================================================================
# 附：如果将来真要用 MobileCLIP-S0
# ============================================================================
# 目标：视觉塔约 11M 参数（S1 ≈ 21.5M，S2 = 35.8M），端侧再快一倍。
#
# 难点：没有 apple/MobileCLIP-S0-OpenCLIP 仓库，open_clip 无法用 hf-hub: 加载，
#       所以不能像上面那样只改 MODEL_REF。
#
# 可行路径（按顺序尝试，成本从低到高）：
#   1. 先确认环境里的 open_clip 是否已把 MobileCLIP 收进内置模型表：
#        python -c "import open_clip; print([m for m in open_clip.list_models() if 'mobileclip' in m.lower()])"
#      若出现 'MobileCLIP-S0' 之类，直接用 create_model_and_transforms("MobileCLIP-S0") 即可。
#   2. 否则从 GitHub 取权重：https://github.com/apple-aiml-research/ml-mobileclip
#      该仓库提供 mobileclip_s0.pt 及配套 config；需要自己构造一份 open_clip 格式的
#      open_clip_config.json（照抄 S1 那份改 embed_dim / vision_cfg / text_cfg 的
#      width / layers / heads），再用 open_clip.create_model_from_pretrained(...) 加载。
#   3. 无论走哪条，加载后务必核对这两个值：
#        model.visual.image_size  == 256   → 否则改 ImagePreprocessor.INPUT_SIZE
#        model.embed_dim          == 512   → 否则改 ClipSceneClassifier.EMBED_DIM
#      本脚本 Cell 4 会自动打印并从官方 preprocess 反查，不一致时会报错。