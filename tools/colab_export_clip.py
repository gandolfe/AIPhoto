# ============================================================================
#  AIPhoto —— MobileCLIP-S1 视觉编码器导出脚本（Google Colab）
#
#  把 Apple MobileCLIP-S1 的视觉塔导出成 ExecuTorch .pte：端侧只做「图像 → embedding」，
#  分类 / 聚类在 App 侧完成（embedding × label_bank）。
#
#  产出：
#    - clip_scene.pte          端侧模型（带 XNNPACK），正式用这个
#    - clip_scene_portable.pte 纯 portable 版，仅在 EXPORT_PORTABLE_BASELINE=True 时导出
#    - xnnpack_audit.txt       XNNPACK 委托覆盖率 + 落回 portable 的算子清单
#    - label_bank.bin          标签库 embedding [N, 512] float32 raw bytes
#    - label_bank.json         标签库元数据（id / prompt / 中文名 / emoji）
#    - labels.json             模型元数据
#    - aiphoto_clip_model.zip  打包下载（含上述已生成的文件）
#
#  用法：Colab 按顺序跑 Cell；CPU / GPU 均可，推荐 CPU。
# ============================================================================


# ============================================================================
# Cell 1 —— 环境检查 & 安装依赖
# ============================================================================
# executorch 需 ≥ 1.5.0。
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
# Cell 2 —— 加载 MobileCLIP-S1 并定义标签库
# ============================================================================
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cpu"

# Apple 官方 OpenCLIP 格式仓库（hf-hub: 前缀下发配置 + 权重 + tokenizer）；
# 不能写 'mobileclip_s1' 这类短名，open_clip 内置表里没有。
# 选 S1：视觉塔约 21.5M 参数，比 S2（35.8M）快得多。
# 换模型要保证 embed_dim == 512、image_size == 256，与 App 侧
# ClipSceneClassifier.EMBED_DIM / ImagePreprocessor.INPUT_SIZE 对齐，否则下面会报错。
MODEL_ID = "mobileclip_s1"
MODEL_REF = "hf-hub:apple/MobileCLIP-S1-OpenCLIP"

print(f"loading {MODEL_REF} ...")
# 返回 (model, preprocess_train, preprocess_val)，推理用 val 那份
clip, _, preprocess = open_clip.create_model_and_transforms(MODEL_REF)
clip = clip.eval().to(DEVICE)
tokenizer = open_clip.get_tokenizer(MODEL_REF)
print("loaded OK")


# 从官方 preprocess 反查 image_size / mean / std，不硬编码；
# 端侧预处理与训练分布错位会让结果悄悄变垃圾。
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

# 标签库：聚类命名候选，每条 = 多 prompt（英文，贴合 CLIP 训练分布）+ 中文显示名 + emoji。
# 增删条目不用重导模型：App 侧只读 .json / .bin。
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
    # cat / dog / bird / pet / wildlife 五类合并而成，prompts 取并集。
    "animal":     (["a photo of a cat", "a photo of a kitten",
                    "a photo of a dog", "a photo of a puppy",
                    "a photo of a bird", "a photo of a flying bird",
                    "a photo of a pet", "a photo of a pet animal",
                    "a photo of wildlife", "a photo of a wild animal in nature"], "动物", "🐾"),
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
# 不要加 "other" 这类泛化兜底：它的文本向量接近图像分布中心，会吸走大量有明确内容的照片。
# 要兜底就用「最高分低于阈值判为其他」的后端判定，别让它参与 argmax。

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
# 只导出视觉分支，文本向量走 label_bank：模型更小、类目可热改，也给 App 侧留出聚类空间。
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
# Cell 5 —— 导出 ExecuTorch .pte
# ============================================================================
# MODE 决定 dtype：fp16 最常用（体积翻倍 vs int8 的取舍），int8 需要校准图。
# 注意 .pte 会比「参数量 × 每参数字节」大：XNNPACK 会按自己的格式重存委托块的权重
# （常摊回 fp32），详见下方 3.1 / 3.2 的覆盖率与基线对照。
# ----------------------------------------------------------------------------
import glob
from collections import Counter

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torch.export import export as torch_export

MODE = "fp16"             # 取值: "fp16" | "int8" | "fp32"
CALIB_IMAGE_DIR = ""      # 例: "/content/calib"，INT8 校准用
CALIB_NUM = 16

# 额外导出纯 portable 基线做体积 / 速度对照，默认关闭。
# 真机实测（S1）：portable 43.6 MB / 单张 19s，XNNPACK 77.0 MB / 单张 0.26s（约快 73 倍）。
# 结论是正式产物用 clip_scene.pte；换了模型形态想重新确认时再打开。
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
# 体积和速度的关键指标：没被 XNNPACK 接住的子图走 portable 参考算子，
# 既慢（单核、无融合），常量也保持 fp32 留在 .pte 里。
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

    # 按节点数会严重低估：getitem / dim_order_copy 这类 glue 算子数量多却不占算力
    # （实测 394 节点里近 300 个是 glue），所以再给一个剔除 glue 的口径。
    # 注意用子串匹配：节点 target 形如
    #   <EdgeOpOverload: dim_order_ops._to_dim_order_copy.default>: schema = ...
    # 不以 "dim_order_ops." 开头，用 startswith 会漏算。
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

# --- 3.2) 纯 portable 基线对照：判断多出来的体积来自 XNNPACK 还是别处 ---
if EXPORT_PORTABLE_BASELINE:
    try:
        base_path = "clip_scene_portable.pte"
        with open(base_path, "wb") as f:
            f.write(to_edge_transform_and_lower(exported, partitioner=[]).to_executorch().buffer)
        base_mb = os.path.getsize(base_path) / 1024 / 1024
        print(f"\n[基线对照] 纯 portable: {base_mb:.1f} MB | 带 XNNPACK: {size_mb:.1f} MB | "
              f"理论 fp16: {_fp16_mb:.1f} MB")
        # 委托块的权重会被 XNNPACK 按自己的格式重存；若它把 fp16 摊成内部 fp32，
        # 则 size ≈ f·fp32 + (1-f)·base，可反解出被委托的权重占比 f
        # （比按节点数的覆盖率准，后者会被 glue 算子稀释）。
        if _fp32_mb > base_mb:
            _f = (size_mb - base_mb) / (_fp32_mb - base_mb)
            print(f"→ 按体积反推，约 {_f:.0%} 的权重进了 XNNPACK 委托块并被摊成内部 "
                  f"fp32（多出来的 {size_mb - base_mb:.1f} MB 来源）：fp16 红利没吃到，反而多存了一份。")
    except Exception as e:
        print(f"[基线对照] 跳过（{type(e).__name__}: {e}）")


# ============================================================================
# Cell 6 —— 导出后验证（PyTorch vs ExecuTorch embedding 一致性）
# ============================================================================
# 已知坑：Python runtime 回包 ET tensor 时会丢 device（"tensor does not have a device"），
# 1.5.0 仍复现。失败只影响本自检，.pte 不受影响（Android 端走 C++ runtime）。
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
    print("已知问题：Python runtime 回包 tensor 时丢 device，升级也治不了，别在这上面耗时间。")
    print("本 Cell 只是自检，可跳过：.pte 不受影响，Android 走 C++ runtime；"
          "正确性以装机实跑的归类结果为准。")

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

# 产物清单：基础 4 件 + 本次实际生成的审计 / 基线文件。
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
# 附：换 MobileCLIP-S0（视觉塔约 11M）
# ============================================================================
# 没有 apple/MobileCLIP-S0-OpenCLIP 仓库，不能只改 MODEL_REF，得先适配加载：
#   1. 先看 open_clip 是否已内置：
#        python -c "import open_clip; print([m for m in open_clip.list_models() if 'mobileclip' in m.lower()])"
#      有就直接 create_model_and_transforms("MobileCLIP-S0")。
#   2. 否则从 https://github.com/apple-aiml-research/ml-mobileclip 取 mobileclip_s0.pt，
#      照 S1 的 open_clip_config.json 改 width / layers / heads，再 create_model_from_pretrained 加载。
#   3. 加载后核对 image_size == 256、embed_dim == 512，变了就同步改 App 侧常量。