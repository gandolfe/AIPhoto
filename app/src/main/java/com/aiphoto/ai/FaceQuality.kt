package com.aiphoto.ai

import kotlin.math.abs
import kotlin.math.hypot

/**
 * 依据 MediaPipe 人脸检测输出的 **6 个关键点**评估人脸姿态（正面 / 侧面）。
 *
 * 关键点顺序（MediaPipe Face Detector 固定）：
 *   0 右眼  1 左眼  2 鼻尖  3 嘴中心  4 右耳屏  5 左耳屏
 *
 * 判断正面用的是**几何对称性**，不依赖额外模型：
 *  - [Metrics.eyeSpanRatio] 双眼间距 / 人脸框宽：正脸约 0.4；侧脸两眼投影挤在一起，会明显变小
 *  - [Metrics.noseOffset]   鼻尖相对双眼中点的水平偏移 / 双眼间距：正脸≈0；侧脸显著偏大
 *  - [Metrics.earBalance]   两耳到鼻尖距离的不对称度：正脸≈0；侧脸接近 1（一只耳转到后面）
 *
 * ⚠️ 关键点坐标是**相对整张图**归一化的（不是人脸框），所以换算距离必须传入图片宽高。
 *
 * ⚠️ 阈值要**看日志分布再校准**，不要凭感觉改：运行时会把各指标分布打到 logcat（见 [Stats]）。
 */
object FaceQuality {

    private const val RIGHT_EYE = 0
    private const val LEFT_EYE = 1
    private const val NOSE = 2
    private const val MOUTH = 3
    private const val RIGHT_EAR = 4
    private const val LEFT_EAR = 5

    /** 正面判定阈值。 */
    data class Config(
        /** 双眼间距 / 人脸框宽 的下限。低于它说明脸侧过去了。 */
        val minEyeSpanRatio: Float = 0.28f,
        /** 鼻尖水平偏移 / 双眼间距 的上限。超过它说明脸转开了。 */
        val maxNoseOffset: Float = 0.40f,
        /** 两耳不对称度上限（0=对称，1=一只耳完全转到后面）。null 表示不检查。 */
        val maxEarBalance: Float? = 0.70f,
    )

    data class Metrics(
        val eyeSpanRatio: Float,
        val noseOffset: Float,
        val earBalance: Float?,
    ) {
        fun isFrontal(cfg: Config): Boolean = rejectReason(cfg) == null

        /** 返回拒绝原因（用于日志统计「主要卡在哪个条件」）；通过则返回 null。 */
        fun rejectReason(cfg: Config): String? {
            if (eyeSpanRatio < cfg.minEyeSpanRatio) return "eyeSpan"
            if (noseOffset > cfg.maxNoseOffset) return "noseOffset"
            val eb = earBalance
            val maxEb = cfg.maxEarBalance
            if (eb != null && maxEb != null && eb > maxEb) return "earBalance"
            return null
        }

        override fun toString(): String {
            val eb = earBalance?.let { String.format(java.util.Locale.US, "%.2f", it) } ?: "n/a"
            return String.format(
                java.util.Locale.US,
                "eyeSpan=%.3f noseOffset=%.3f earBal=%s",
                eyeSpanRatio, noseOffset, eb,
            )
        }
    }

    /** 累计分布，用于校准阈值。 */
    class Stats {
        private val eyeSpan = ArrayList<Float>()
        private val noseOff = ArrayList<Float>()
        private val rejects = HashMap<String, Int>()
        var noKeypoints = 0
            private set
        var total = 0
            private set

        fun record(metrics: Metrics?, reason: String?) {
            total++
            if (metrics == null) {
                noKeypoints++
                return
            }
            eyeSpan += metrics.eyeSpanRatio
            noseOff += metrics.noseOffset
            if (reason != null) rejects[reason] = (rejects[reason] ?: 0) + 1
        }

        fun summary(): String {
            fun desc(name: String, v: List<Float>): String {
                if (v.isEmpty()) return "$name: 无样本"
                val s = v.sorted()
                val at =
                    { q: Float -> String.format(java.util.Locale.US, "%.3f", s[((s.size - 1) * q).toInt()]) }
                return "$name: n=${s.size} p10=${at(0.1f)} p50=${at(0.5f)} p90=${at(0.9f)} max=${at(1f)}"
            }
            return buildString {
                append(desc("eyeSpanRatio", eyeSpan)).append('\n')
                append(desc("noseOffset", noseOff)).append('\n')
                append("拒绝原因: ").append(if (rejects.isEmpty()) "无" else rejects.toString())
                append(" | 无关键点: ").append(noKeypoints).append('/').append(total)
            }
        }
    }

    /**
     * 计算姿态指标。
     *
     * @param imgW/imgH 关键点归一化所基于的图片宽高（关键点相对整图，不是人脸框）
     * @return 关键点不足 6 个时返回 null —— 表示**无法判断**，调用方应当放行而不是拒绝，
     *         否则一旦模型不输出关键点会误杀所有人脸。
     */
    fun measure(box: DetectedFaceBox, imgW: Int, imgH: Int): Metrics? {
        val kps = box.keypoints
        if (kps == null || kps.size < 6) return null
        if (imgW <= 0 || imgH <= 0) return null

        val boxW = box.rightPct - box.leftPct
        val boxH = box.bottomPct - box.topPct
        if (boxW <= 0f || boxH <= 0f) return null

        // 关键点 → 像素
        fun px(i: Int): Pair<Float, Float> = kps[i].first * imgW to kps[i].second * imgH

        val (reX, reY) = px(RIGHT_EYE)
        val (leX, leY) = px(LEFT_EYE)
        val (noX, noY) = px(NOSE)

        val interEye = hypot(reX - leX, reY - leY)
        if (interEye <= 0f) return null

        val boxWidthPx = boxW * imgW
        if (boxWidthPx <= 0f) return null

        // 双眼间距占人脸框宽的比例
        val eyeSpanRatio = interEye / boxWidthPx

        // 鼻尖相对双眼中点的水平偏移（yaw 主要体现在水平方向），用双眼间距归一化
        val eyeMidX = (reX + leX) / 2f
        val noseOffset = abs(noX - eyeMidX) / interEye

        return Metrics(eyeSpanRatio, noseOffset, null)
    }
}
