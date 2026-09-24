package com.aiphoto.ai

import android.graphics.Bitmap

/**
 * 按人脸框从 Bitmap 里裁出人脸。推理（提特征）和 UI（人物封面）都要用，所以单独抽出来。
 *
 * 框存的是相对比例（0..1），所以这里能直接适配任意分辨率的源图：
 * 分析时用降采样后的缩略图，做封面时可以用更高清的图。
 */
object FaceCropper {

    /**
     * 裁出人脸区域。
     *
     * @param margin 向外扩展的比例（相对人脸框宽/高）。扩边很关键 ——
     *   只裁紧贴的框会丢掉下巴和额头，embedding 质量明显下降。
     * @return 裁好的 Bitmap；框非法时返回 null。
     */
    fun crop(
        source: Bitmap,
        leftPct: Float,
        topPct: Float,
        rightPct: Float,
        bottomPct: Float,
        margin: Float = DEFAULT_MARGIN,
    ): Bitmap? {
        val w = source.width
        val h = source.height
        if (w <= 0 || h <= 0) return null

        val left = (leftPct * w).toInt()
        val top = (topPct * h).toInt()
        val right = (rightPct * w).toInt()
        val bottom = (bottomPct * h).toInt()

        val fw = right - left
        val fh = bottom - top
        if (fw <= 0 || fh <= 0) return null

        val mx = (fw * margin).toInt()
        val my = (fh * margin).toInt()

        val x0 = (left - mx).coerceIn(0, w - 1)
        val y0 = (top - my).coerceIn(0, h - 1)
        val x1 = (right + mx).coerceIn(x0 + 1, w)
        val y1 = (bottom + my).coerceIn(y0 + 1, h)

        return Bitmap.createBitmap(source, x0, y0, x1 - x0, y1 - y0)
    }

    fun crop(source: Bitmap, box: DetectedFaceBox, margin: Float = DEFAULT_MARGIN): Bitmap? =
        crop(source, box.leftPct, box.topPct, box.rightPct, box.bottomPct, margin)

    const val DEFAULT_MARGIN = 0.25f
}
