# AIPhoto

Android 相册应用，用 **MobileCLIP-S1 + ExecuTorch** 在端侧完成照片场景分类。
全程本地推理，不联网、不上传。

## 它怎么工作

```
照片 → 缩放 256×256 → fp16 张量
     → MobileCLIP-S1 视觉塔 (.pte) → 512 维 L2 归一化 embedding
     → 端侧 与 label_bank 做点积（等于余弦相似度）
     → softmax(logit_scale · sim) → 39 类概率
     → 写入 Room (photo_tags)
```

分类决策在 Kotlin 侧完成，`.pte` 只负责把图像编码成 embedding。
因此**换标签不需要重新导出模型**，只换 `label_bank.bin` / `label_bank.json` 即可。

关键代码：

| 文件 | 职责 |
|---|---|
| `ai/ClipSceneClassifier.kt` | 加载 `.pte`、fp16 编码、forward、embedding × label_bank、softmax |
| `ai/ImagePreprocessor.kt` | 缩放 / 归一化 / CHW 排列 |
| `ai/SceneAnalyzer.kt` | 单线程串行调度（ExecuTorch Module 非并发安全） |
| `data/PhotoTagDao.kt` | 结果读写 |
| `tools/colab_export_clip.py` | 模型导出 |

## 模型产物

放在 `app/src/main/assets/models/`，共 4 个文件：

| 文件 | 来源 | 说明 |
|---|---|---|
| `clip_scene.pte` | Cell 5 | **带 XNNPACK**，正式用这个 |
| `label_bank.bin` | Cell 8 | `[39, 512]` float32 小端，39×512×4 = 79872 字节 |
| `label_bank.json` | Cell 8 | 39 条 id / prompts / 中文名 / emoji |
| `labels.json` | Cell 8 | 元数据，启动时用于校验 |

`build.gradle.kts` 里 `noCompress += "pte"`，保证 `.pte` 以 stored 方式打包，
App 才能用 `openFd()` 拿到真实长度并直接 mmap。

## 导出模型

Colab 新建 notebook，按 `tools/colab_export_clip.py` 的 Cell 顺序运行（推荐 CPU 运行时）。

```
MODEL_ID  = "mobileclip_s1"
MODEL_REF = "hf-hub:apple/MobileCLIP-S1-OpenCLIP"
MODE      = "fp16"        # fp16 | int8 | fp32
EXPORT_PORTABLE_BASELINE = False
```

导出后脚本会打印参数量、dtype、XNNPACK 委托覆盖率、落回 portable 的算子清单
（另落盘 `xnnpack_audit.txt`），并打包下载 `aiphoto_clip_model.zip`。

**换模型时必须核对两个值**，否则端侧会静默出错：

```
model.visual.image_size == 256  → 改 ImagePreprocessor.INPUT_SIZE
model.embed_dim         == 512  → 改 ClipSceneClassifier.EMBED_DIM
```

S1 的 `open_clip_config.json` 实测为 `image_size=256`、`embed_dim=512`，与当前 App 常量一致。

想用更小的 MobileCLIP-S0 需要先解决权重格式问题，见脚本末尾注释。

## 编译运行

AGP 8.7.3 要求 **JDK ≥ 11**。若终端 PATH 上默认是 Java 8，会报
`Dependency requires at least JVM runtime version 11`，需要先指定：

```cmd
set JAVA_HOME=C:\Program Files\Android\Android Studio1\jbr
gradlew.bat assembleDebug
```

安装到真机：

```cmd
adb install -r -g app\build\outputs\apk\debug\app-debug.apk
```

`-g` 会一次性授予全部运行时权限，省去在手机上一一点授权。

## 实测性能（真机 vivo V2528A，MobileCLIP-S1，2804 张照片）

| 版本 | `.pte` 体积 | 单张耗时 | 2800 张总计 |
|---|---|---|---|
| 纯 portable | 43.6 MB | 19 秒 | 约 14.8 小时 |
| **带 XNNPACK** | **77.0 MB** | **0.26 秒** | **约 12 分钟** |

其余环节耗时（供排查参考）：`Module.load` 13 ms、`toClipInput` 3~19 ms、
half 编码 12~45 ms、读图解码 45~49 ms。瓶颈始终在 `module.forward`。

**XNNPACK 的取舍**：它会把委托块的权重摊成内部 fp32，让 `.pte` 涨约 33 MB

## 已知坑

**1. `floatToHalf` 对 `0.0f` 会死循环（已修，别改回去）**

`ClipSceneClassifier.floatToHalf()` 里 `exp == 0` 分支必须写成 if / else 两个分支：

```kotlin
if (mantissa == 0) {
    sign.toShort()          // ±0
} else {
    while ((v and 0x00800000) == 0) { v = v shl 1; e -= 1 }   // v==0 时恒真
    ...
}
```

`if` 只是语句、不会返回。少了 `else`，`mantissa == 0 && exp == 0`（即 `±0.0`）
会继续进入归一化循环，而 `v == 0` 时 `v shl 1` 恒为 0，循环永不退出。
黑色像素归一化后正好是 `0.0f`，**任何一张图都会触发**。
表现为：CPU 100% 空转、进度永远停在 0、三个体积不同的模型表现完全一致。

**2. 已分析过的照片会被跳过**

`TimelineViewModel.startAnalysis()` 里 `pending = all.filter { it.id !in analyzed }`。
换模型后旧结果不兼容（旧模型 5 类，新模型 39 类），需要先用界面上的
**「清空」**按钮清掉历史结果再重新分析，否则只会处理新照片。

**4. 息屏后 vivo 会冻结后台 App**

分析会中途停住。长时间跑批时保持屏幕常亮：

```cmd
adb shell svc power stayon usb
```

## 排查手段

- 计时日志：`ClipSceneClassifier` / `SceneAnalyzer` / `TimelineVM` 三个 tag，
  按 `loadForAnalysis → Module.load → toClipInput → half 编码 → forward` 分段打印耗时
- 直接查库：`adb shell run-as com.aiphoto sqlite3 databases/aiphoto.db "select count(*) from photo_tags;"`
- 已分析文件：`adb shell run-as com.aiphoto ls -l files/models/`
