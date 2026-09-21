package com.aiphoto.ai

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.IOException
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.nio.ByteBuffer

/**
 * MobileCLIP-S2 场景分类器，基于 ExecuTorch 在端侧执行。
 *
 * 架构：
 *   1. 视觉塔 .pte：输入 [1, 3, 256, 256] fp16，归一化图像 → 输出 [1, 512] L2 归一化 embedding
 *   2. label_bank.bin / label_bank.json：39 个类别 id 的 L2 归一化文本向量 + 元数据
 *   3. App 侧：embedding × label_bank → 余弦相似度（点积）→ 乘 logit_scale 后 softmax
 *
 * 使用前提（与 [PLACE_MODEL_HERE] 一致）：
 *   app/src/main/assets/models/
 *     - clip_scene.pte     （Cell 5 产物）
 *     - label_bank.bin     （Cell 8 产物，[N, 512] float32 = 39×512×4 = 79872 字节）
 *     - label_bank.json    （Cell 8 产物，39 条 id / prompts / 中文名 / emoji）
 *     - labels.json        （Cell 8 产物，模型元数据，仅用于校验）
 *
 * 线程安全：ExecuTorch Module 不保证并发安全，调用方需保证串行调用
 * （见 [SceneAnalyzer]，内部用单线程 dispatcher）。
 */
class ClipSceneClassifier private constructor(
    private val module: Module,
    private val labels: List<LabelMeta>,
    private val labelMatrix: FloatArray,
) : AutoCloseable {

    /** 标签元数据 + 行下标（用于索引 labelMatrix）。 */
    data class LabelMeta(
        val index: Int,
        val id: String,
        val displayName: String,
        val emoji: String,
    )

    /** 一次分类的结果。 */
    data class SceneResult(val scores: List<Pair<String, Float>>) {
        val topLabel: String get() = scores.maxByOrNull { it.second }?.first.orEmpty()
        val topScore: Float get() = scores.maxOfOrNull { it.second } ?: 0f
    }

    private val inputShape = longArrayOf(
        1L,
        3L,
        ImagePreprocessor.INPUT_SIZE.toLong(),
        ImagePreprocessor.INPUT_SIZE.toLong(),
    )

    private val inputSize = 1 * 3 * ImagePreprocessor.INPUT_SIZE * ImagePreprocessor.INPUT_SIZE

    /**
     * 对单张图片执行分类。调用方负责保证不在主线程执行。
     *
     * 流程：
     *   image (fp32, [1,3,H,W]) → fp16 half 编码 → tensor → forward → [512] embedding
     *   → 点积 label_bank[N,512] → logits → softmax(·logit_scale) → probs
     */
    fun classify(bitmap: Bitmap): SceneResult {
        // 1. fp32 [1,3,H,W] → ShortBuffer (IEEE half)
        val pixels = ImagePreprocessor.toClipInput(bitmap)
        val halfBuffer = Tensor.allocateHalfBuffer(inputSize)
        for (i in 0 until inputSize) {
            halfBuffer.put(i, floatToHalf(pixels[i]))
        }

        // 2. 构造输入 tensor 并 forward。
        //    这里占掉 classify() 几乎全部耗时（实测 19 秒里的 18.98 秒），
        //    是唯一值得长期保留的计时点；其余环节都是毫秒级，不再逐个打点。
        val inputTensor = Tensor.fromBlob(halfBuffer, inputShape)
        val t0 = SystemClock.elapsedRealtime()
        val outputs = module.forward(EValue.from(inputTensor))
        Log.i(TAG, "module.forward ${SystemClock.elapsedRealtime() - t0} ms")
        if (outputs.isEmpty()) error("模型未返回任何输出")

        // fp16 模型，dataAsFloatArray 由 Tensor_float16 子类实现 fp16→fp32 转换
        val embedding = outputs[0].toTensor().dataAsFloatArray
        require(embedding.size == EMBED_DIM) {
            "embedding dim=${embedding.size} ≠ $EMBED_DIM（labels.json.embed_dim 不一致）"
        }

        // 3. embedding × label_bank：两边都已 L2 归一化，点积即余弦相似度。
        //
        //    不再做 softmax。它只是单调变换（×正数、exp、÷正常数都不改变排序），
        //    取第一名和排序的结果与原始余弦完全一致，纯属多余计算。
        //    而且 softmax 是按「每张图」归一化的，会把另外 38 个标签的表现混进来；
        //    跨照片比较（同类里谁最匹配）时，原始余弦才是可比的量。
        //
        //    副作用：入库分数从概率（0.954）变成余弦（约 0.2~0.4，理论范围 -1~1）。
        //    仅用于分组和排序，不用于展示「置信度 95%」这类百分比文案。
        val n = labels.size
        val scores = ArrayList<Pair<String, Float>>(n)
        for (i in 0 until n) {
            var s = 0f
            val base = i * EMBED_DIM
            for (j in 0 until EMBED_DIM) {
                s += embedding[j] * labelMatrix[base + j]
            }
            scores += labels[i].id to s
        }
        return SceneResult(scores)
    }

    override fun close() {
        runCatching { module.close() }
    }

    companion object {
        private const val TAG = "ClipSceneClassifier"

        const val MODEL_ASSET = "models/clip_scene.pte"
        private const val LABEL_BANK_BIN_ASSET = "models/label_bank.bin"
        private const val LABEL_BANK_JSON_ASSET = "models/label_bank.json"
        private const val METADATA_ASSET = "models/labels.json"

        /** labels.json.embed_dim：MobileCLIP-S2 = 512。 */
        private const val EMBED_DIM = 512

        /** 模型文件是否已经放进 assets。 */
        fun isModelAvailable(context: Context): Boolean = try {
            context.assets.open(MODEL_ASSET).close()
            true
        } catch (e: IOException) {
            false
        }

        /**
         * 加载模型。首次调用会把 .pte 拷到 filesDir，后续直接复用。
         * label_bank.bin/json 很小（~80KB / ~8KB），直接常驻内存。
         *
         * @throws IllegalStateException assets 缺失或标签库维度不一致时。
         */
        fun create(context: Context): ClipSceneClassifier {
            if (!isModelAvailable(context)) {
                error("assets/$MODEL_ASSET 不存在，请先用 tools/colab_export_clip.py 导出模型")
            }

            val t0 = SystemClock.elapsedRealtime()
            val (labels, matrix) = loadLabelBank(context)
            val t1 = SystemClock.elapsedRealtime()
            Log.i(TAG, "loadLabelBank ${t1 - t0} ms，类别=${labels.size}")

            validateMetadata(context, labels.size)

            // ensureModelFile 会把 assets 里的 .pte 拷到 filesDir，体积大，单独计时
            val modelFile = ensureModelFile(context)
            val t2 = SystemClock.elapsedRealtime()
            Log.i(TAG, "ensureModelFile ${t2 - t1} ms，size=${modelFile.length()}")

            Log.i(
                TAG,
                "开始 Module.load ${modelFile.absolutePath} (${modelFile.length() / 1024 / 1024} MB)"
            )
            // MMAP 加载：.pte 大文件按页换入，不一次性占满内存
            val module = Module.load(modelFile.absolutePath, Module.LOAD_MODE_MMAP)
            Log.i(TAG, "Module.load 完成，耗时 ${SystemClock.elapsedRealtime() - t2} ms")
            return ClipSceneClassifier(module, labels, matrix)
        }

        private fun ensureModelFile(context: Context): File {
            val dir = File(context.filesDir, "models").apply { mkdirs() }
            val target = File(dir, "clip_scene.pte")
            val assetSize = assetLength(context, MODEL_ASSET)
            val upToDate = target.exists() &&
                (assetSize <= 0 || target.length() == assetSize)
            if (!upToDate) {
                Log.i(TAG, "拷贝模型到 ${target.absolutePath}")
                context.assets.open(MODEL_ASSET).use { input ->
                    target.outputStream().use { output -> input.copyTo(output, DEFAULT_BUFFER_SIZE) }
                }
            }
            return target
        }

        /** 未压缩 asset 的真实字节数；不可用时返回 -1。 */
        private fun assetLength(context: Context, path: String): Long = try {
            context.assets.openFd(path).use { it.length }
        } catch (e: IOException) {
            -1L
        }

        /**
         * 读 label_bank.json + label_bank.bin，返回 (标签元数据列表, [N, EMBED_DIM] float32 行优先矩阵)。
         */
        private fun loadLabelBank(context: Context): Pair<List<LabelMeta>, FloatArray> {
            val jsonText = context.assets.open(LABEL_BANK_JSON_ASSET)
                .bufferedReader()
                .use { it.readText() }
            val root = JSONObject(jsonText)
            val arr = root.getJSONArray("labels")
            val n = arr.length()
            require(n > 0) { "$LABEL_BANK_JSON_ASSET 为空" }

            // labels.json.embed_dim 与代码里硬编码的 EMBED_DIM 一致才进；不一致直接报错
            val jsonDim = root.optInt("embed_dim", EMBED_DIM)
            require(jsonDim == EMBED_DIM) {
                "label_bank.embed_dim=$jsonDim ≠ 代码里 EMBED_DIM=$EMBED_DIM"
            }

            // 读 [N, EMBED_DIM] float32 小端 raw bytes（Colab 端 np.tofile 写出）
            val expectedBytes = n * EMBED_DIM * 4
            val binBytes = ByteArray(expectedBytes)
            var read = 0
            context.assets.open(LABEL_BANK_BIN_ASSET).use { input ->
                while (read < expectedBytes) {
                    val r = input.read(binBytes, read, expectedBytes - read)
                    if (r < 0) error("$LABEL_BANK_BIN_ASSET 截断：读了 $read/$expectedBytes 字节")
                    read += r
                }
            }
            val buffer = ByteBuffer.wrap(binBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val matrix = FloatArray(n * EMBED_DIM)
            buffer.get(matrix)

            val labels = List(n) { i ->
                val obj = arr.getJSONObject(i)
                LabelMeta(
                    index = i,
                    id = obj.getString("id"),
                    displayName = obj.optString("display_name", obj.getString("id")),
                    emoji = obj.optString("emoji", ""),
                )
            }
            return labels to matrix
        }

        /**
         * 校验 labels.json（Cell 8 的模型元数据）。缺字段不致命，但 image_size 与
         * EMBED_DIM 不一致必须早失败，否则端侧和训练分布错位会静默退化。
         */
        private fun validateMetadata(context: Context, labelCount: Int) {
            val meta = try {
                context.assets.open(METADATA_ASSET)
                    .bufferedReader()
                    .use { JSONObject(it.readText()) }
            } catch (e: Exception) {
                Log.w(TAG, "$METADATA_ASSET 不可用，跳过校验", e)
                return
            }
            val expectedSize = ImagePreprocessor.INPUT_SIZE
            val metaSize = meta.optInt("image_size", expectedSize)
            require(metaSize == expectedSize) {
                "labels.json.image_size=$metaSize ≠ ImagePreprocessor.INPUT_SIZE=$expectedSize，" +
                    "请同步修改 ImagePreprocessor.kt 的 INPUT_SIZE"
            }
            val metaDim = meta.optInt("embed_dim", EMBED_DIM)
            require(metaDim == EMBED_DIM) {
                "labels.json.embed_dim=$metaDim ≠ $EMBED_DIM"
            }
            val metaLabelCount = meta.optInt("label_count", labelCount)
            require(metaLabelCount == labelCount) {
                "labels.json.label_count=$metaLabelCount ≠ label_bank 实际类别数=$labelCount"
            }
        }

        /**
         * IEEE 754 binary32 → binary16 转换。ExecuTorch 1.4 的 fp16 Tensor 必须用
         * ShortBuffer（IEEE half 位模式）传，不能直接传 FloatArray。
         */
        private fun floatToHalf(value: Float): Short {
            val bits = java.lang.Float.floatToIntBits(value)
            val sign = (bits ushr 16) and 0x8000
            val exp = (bits ushr 23) and 0xFF
            val mantissa = bits and 0x007FFFFF

            return when {
                exp == 0xFF -> {
                    // Inf / NaN
                    val payload = if (mantissa != 0) (0x7C00 or 1) else 0x7C00
                    (sign or payload).toShort()
                }
                exp == 0 -> {
                    // ⚠️ 这里必须写成 if/else 的两个分支：
                    // 之前漏了 else，`if (mantissa == 0) sign.toShort()` 只是条语句、
                    // 不会返回，于是 mantissa==0（即 ±0.0）时会继续往下跑归一化循环；
                    // 而 v==0 时 `v shl 1` 恒为 0，循环条件恒真 → 死循环。
                    // 黑色像素归一化后就是 0.0f，图片里必然存在，所以每次必挂。
                    if (mantissa == 0) {
                        sign.toShort()                                        // ±0
                    } else {
                        // Subnormal float → 归一化后再转（罕见，几乎不会遇到）
                        var v = mantissa
                        var e = exp
                        while ((v and 0x00800000) == 0) { v = v shl 1; e -= 1 }
                        v = v and 0x007FFFFF
                        val newExp = e - 127 + 15
                        (sign or (newExp shl 10) or (v ushr 13)).toShort()
                    }
                }
                else -> {
                    val newExp = exp - 127 + 15
                    when {
                        newExp >= 0x1F -> (sign or 0x7C00).toShort()          // overflow → ±Inf
                        newExp <= 0 -> {
                            // Underflow → subnormal half 或 0
                            val m = (mantissa or 0x00800000) ushr (14 - newExp)
                            (sign or m).toShort()
                        }
                        else -> (sign or (newExp shl 10) or (mantissa ushr 13)).toShort()
                    }
                }
            }
        }
    }
}