package com.aiphoto.ai

import android.graphics.Bitmap

/**
 * MobileCLIP-S2 图像预处理。
 *
 * 与导出脚本 `tools/colab_export_clip.py` Cell 2 中反查到的官方 preprocess 对齐：
 *   resize 到 INPUT_SIZE×INPUT_SIZE（当前为 256，见 labels.json.image_size）→
 *   归一化到 [0,1] → mean/std 由 labels.json 给出（mobileclip_s2 为 [0,0,0]/[1,1,1]）→
 *   CHW 排列。
 *
 * 注意：mean/std/输入尺寸不能硬编码——一旦换模型或换导出批次，必须先看
 * labels.json 里的 image_size/image_mean/image_std，再回这里同步。脚本里的检查会
 * 在 mean/std 与默认 OpenAI 不同步时打 ⚠️，但 App 侧没有这种兜底，只能靠人盯。
 *
 * 官方默认是 bicubic + 短边缩放 + 中心裁剪。端侧为省事直接用双线性缩放到正方形，
 * 实测对分类结果影响很小。
 */
object ImagePreprocessor {

    /**
     * MobileCLIP-S2 的视觉塔输入尺寸。改这里之前先确认 labels.json 的 image_size 字段。
     */
    const val INPUT_SIZE = 256

    /**
     * labels.json.image_mean：mobileclip_s2 = [0, 0, 0]（不做均值中心化）。
     * 之前用 OpenAI CLIP 常量是错的，与 mobileclip_s2 不一致，会让结果整体偏移。
     */
    private val MEAN = floatArrayOf(0f, 0f, 0f)

    /**
     * labels.json.image_std：mobileclip_s2 = [1, 1, 1]（不做方差归一化）。
     */
    private val STD = floatArrayOf(1f, 1f, 1f)

    /**
     * 把 Bitmap 转换为 [1, 3, INPUT_SIZE, INPUT_SIZE] 的 NCHW FloatArray。
     */
    fun toClipInput(source: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(source, INPUT_SIZE, INPUT_SIZE, true)

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        if (scaled !== source) {
            scaled.recycle()
        }

        val planeSize = INPUT_SIZE * INPUT_SIZE
        val out = FloatArray(3 * planeSize)

        for (i in pixels.indices) {
            val c = pixels[i]
            val r = ((c shr 16) and 0xFF) / 255f
            val g = ((c shr 8) and 0xFF) / 255f
            val b = (c and 0xFF) / 255f

            out[i] = (r - MEAN[0]) / STD[0]
            out[planeSize + i] = (g - MEAN[1]) / STD[1]
            out[2 * planeSize + i] = (b - MEAN[2]) / STD[2]
        }
        return out
    }
}