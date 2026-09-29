package com.aiphoto.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.sqrt

/**
 * MobileFaceNet 人脸特征提取（ExecuTorch）。
 *
 * 与 [ClipSceneClassifier] 的区别：
 *   - 输入是 **fp32**（不是 fp16）。模型只有 ~1M 参数，fp32 也才几 MB，
 *     而 fp32 输入可以直接用 float[] 构造张量，省掉 floatToHalf 那一层转换和出错风险。
 *   - 输入尺寸/归一化参数**不硬编码**，全部从 assets/models/mobilefacenet.json 读。
 *     归一化必须和训练时一致（InsightFace 风格 (x-127.5)/128），否则 embedding 是垃圾、
 *     余弦相似度失去意义、聚类结果全乱。
 *
 * 线程不安全，调用方必须串行（见 [FaceAnalyzer] 的单线程 dispatcher）。
 */
class MobileFaceNet private constructor(
    private val module: Module,
    private val inputSize: Int,
    private val mean: FloatArray,
    private val std: FloatArray,
    val embedDim: Int,
) : AutoCloseable {

    companion object {
        private const val TAG = "MobileFaceNet"
        private const val MODEL_ASSET = "models/mobilefacenet.pte"
        private const val META_ASSET = "models/mobilefacenet.json"

        /** 两个文件（pte + json）都齐了才算可用。 */
        fun isAvailable(context: Context): Boolean = try {
            context.assets.open(META_ASSET).close()
            context.assets.openFd(MODEL_ASSET).close()
            true
        } catch (e: IOException) {
            false
        }

        fun create(context: Context): MobileFaceNet {
            val meta = readMetadata(context)
            val file = ensureModelFile(context)
            Log.i(TAG, "Module.load ${file.absolutePath} (${file.length() / 1024 / 1024} MB)")
            val module = Module.load(file.absolutePath, Module.LOAD_MODE_MMAP)
            return MobileFaceNet(module, meta.size, meta.mean, meta.std, meta.embedDim)
        }

        private class Meta(
            val size: Int,
            val mean: FloatArray,
            val std: FloatArray,
            val embedDim: Int,
        )

        private fun readMetadata(context: Context): Meta {
            val json = context.assets.open(META_ASSET).bufferedReader().use { it.readText() }
            val root = JSONObject(json)
            val size = root.getInt("image_size")
            val meanArr = root.getJSONArray("image_mean")
            val stdArr = root.getJSONArray("image_std")
            require(meanArr.length() == 3 && stdArr.length() == 3) {
                "mobilefacenet.json 的 mean/std 必须是 3 维"
            }
            val embedDim = root.getInt("embed_dim")
            Log.i(TAG, "元数据: size=$size, embed_dim=$embedDim")
            return Meta(
                size = size,
                mean = floatArrayOf(
                    meanArr.getDouble(0).toFloat(),
                    meanArr.getDouble(1).toFloat(),
                    meanArr.getDouble(2).toFloat(),
                ),
                std = floatArrayOf(
                    stdArr.getDouble(0).toFloat(),
                    stdArr.getDouble(1).toFloat(),
                    stdArr.getDouble(2).toFloat(),
                ),
                embedDim = embedDim,
            )
        }

        /**
         * assets → filesDir。.pte 已在 build.gradle 里配了 noCompress，
         * 可以直接 openFd 拿到真实长度，用长度比对决定是否重拷。
         */
        private fun ensureModelFile(context: Context): File {
            val out = File(context.filesDir, "models/mobilefacenet.pte")
            out.parentFile?.mkdirs()
            val expected = context.assets.openFd(MODEL_ASSET).length
            if (out.exists() && out.length() == expected) return out
            context.assets.open(MODEL_ASSET).use { input ->
                FileOutputStream(out).use { output -> input.copyTo(output) }
            }
            return out
        }
    }

    /**
     * 提取人脸 embedding，返回**已 L2 归一化**的向量。
     * 归一化后余弦相似度 = 点积，聚类时直接点乘即可。
     */
    fun embed(source: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(source, inputSize, inputSize, true)
        val pixels = IntArray(inputSize * inputSize)
        scaled.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        if (scaled !== source) scaled.recycle()

        val plane = inputSize * inputSize
        val data = FloatArray(3 * plane)
        for (i in pixels.indices) {
            val c = pixels[i]
            val r = ((c shr 16) and 0xFF).toFloat()
            val g = ((c shr 8) and 0xFF).toFloat()
            val b = (c and 0xFF).toFloat()
            data[i] = (r - mean[0]) / std[0]
            data[plane + i] = (g - mean[1]) / std[1]
            data[2 * plane + i] = (b - mean[2]) / std[2]
        }

        val tensor = Tensor.fromBlob(data, longArrayOf(1L, 3L, inputSize.toLong(), inputSize.toLong()))
        val outputs = module.forward(EValue.from(tensor))
        if (outputs.isEmpty()) error("模型未返回任何输出")

        val embedding = outputs[0].toTensor().dataAsFloatArray
        //做参数的校验，不符合要求会抛出IllegalArgumentException异常
        require(embedding.size == embedDim) {
            "embedding dim=${embedding.size} ≠ mobilefacenet.json 里的 $embedDim"
        }
        //L2向量归一化
        normalize(embedding)
        return embedding
    }

    /**
     * 对向量做L2归一化，使得向量的模长刚好为1
     * 这样比较两个向量的相似度，就只需要做点积运算，而不是复杂的余弦相似度计算
     */
    private fun normalize(v: FloatArray) {
        var sum = 0f
        for (x in v) sum += x * x
        val n = sqrt(sum)
        if (n > 0f) {
            for (i in v.indices) v[i] /= n
        }
    }

    override fun close() {
        module.close()
    }
}
