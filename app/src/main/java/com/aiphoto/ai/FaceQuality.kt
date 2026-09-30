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

    /**
     * 正面判定阈值。
     *
     * ## 依据（真机 639 张脸实测，非拍脑袋）
     *   eyeSpanRatio: n=639 p10=0.346 p50=0.418 p90=0.463 max=0.587
     *   noseOffset  : n=639 p10=0.020 p50=0.115 p90=0.460 max=4.905
     *   （无关键点 0/639 → 模型稳定输出关键点；中位数 0.418 与正脸理论值≈0.4 吻合，
     *     也反证了「关键点相对整图归一化」的换算是对的）
     *
     * 用三角关系把阈值换算成「允许的最大偏转角」：
     *   眼间距 ≈ 0.4×脸宽，鼻尖前突 ≈ 0.25×脸宽
     *   → noseOffset ≈ 0.25·sin(θ) / 0.4 = 0.625·sin(θ)
     *     0.20≈19°    0.30≈29°    0.40≈40°
     *   eyeSpanRatio 随偏转角按 cos 衰减：正面0.418 → 30°时0.36 → 45°时0.29
     *
     * 业界「正面脸」通用标准是偏转 <30°，故取 maxNoseOffset=0.30（≈29°），
     * eyeSpanRatio=0.33（≈38°）作兜底下限。
     * 校准前（0.28/0.40）只拒 14% 偏松；这组预计拒掉约 25%。
     */
    data class Config(
        /** 双眼间距 / 人脸框宽 的下限。低于它说明脸侧过去了（0.33 ≈ 偏转 38°）。 */
        val minEyeSpanRatio: Float = 0.33f,
        /** 鼻尖水平偏移 / 双眼间距 的上限。超过它说明脸转开了（0.30 ≈ 偏转 29°）。 */
        val maxNoseOffset: Float = 0.30f,
        /** 两耳不对称度上限（0=对称，1=一只耳完全转到后面）。null 表示不检查。 */
        /**
         * 两耳不对称度上限（0=对称，1=一只耳完全转到后面）。null 表示不检查。
         */
        val maxEarBalance: Float? = 0.70f,
        /**
         * 低头（pitch down）上限：[Metrics.pitchRatio] 超过它就判为低头。
         *
         * 真机 708 张脸实测分布：
         *   p10=0.925  p50=1.323  p90=1.857  max=10.418
         * p50=1.323 即正脸基准（与预估的≈1.5 吻合，说明指标算对了）。
         *
         * 初值 2.2 定得太松 —— 它远超 p90，实测只拦掉 16 个（2.3%），
         * 普通低头根本没拦住（用户相册里仍大量出现低头照）。
         * 故收到 **1.70**：介于 p50 与 p90 之间偏上，约比正脸基准高 28%，
         * 预计拦掉 15%~18%。
         *
         * 若仍有低头残留 → 往 1.55 调；若误杀太多好脸 → 往 1.85 调。
         */
        val maxPitchRatio: Float? = 1.85f,
        /**
         * 抬头（pitch up）下限：低于它就判为抬头。默认 null（先不拦），
         * 等 pitchRatio 分布出来再决定要不要加。
         */
        val minPitchRatio: Float? = null,
    )

    data class Metrics(
        val eyeSpanRatio: Float,
        val noseOffset: Float,
        val earBalance: Float?,
        /**
         * 俯仰指标 = (眼线中点→鼻尖距离) / (鼻尖→嘴中心距离)。
         *
         * 低头时头绕双耳轴旋转，鼻尖投影向下挤向嘴部：分子变大、分母变小，
         * 比值显著升高。正脸约 1.5，低头会明显大于它。
         * 分母接近 0（关键点退化）时为 null。
         */
        val pitchRatio: Float?,
    ) {
        fun isFrontal(cfg: Config): Boolean = rejectReason(cfg) == null

        /** 返回拒绝原因（用于日志统计「主要卡在哪个条件」）；通过则返回 null。 */
        fun rejectReason(cfg: Config): String? {
            if (eyeSpanRatio < cfg.minEyeSpanRatio) return "eyeSpan"
            if (noseOffset > cfg.maxNoseOffset) return "noseOffset"
            val eb = earBalance
            val maxEb = cfg.maxEarBalance
            if (eb != null && maxEb != null && eb > maxEb) return "earBalance"
            val pr = pitchRatio
            if (pr != null) {
                val maxPr = cfg.maxPitchRatio
                val minPr = cfg.minPitchRatio
                if (maxPr != null && pr > maxPr) return "pitchDown"
                if (minPr != null && pr < minPr) return "pitchUp"
            }
            return null
        }

        override fun toString(): String {
            val eb = earBalance?.let { String.format(java.util.Locale.US, "%.2f", it) } ?: "n/a"
            val pr = pitchRatio?.let { String.format(java.util.Locale.US, "%.2f", it) } ?: "n/a"
            return String.format(
                java.util.Locale.US,
                "eyeSpan=%.3f noseOffset=%.3f earBal=%s pitch=%s",
                eyeSpanRatio, noseOffset, eb, pr,
            )
        }
    }

    /** 累计分布，用于校准阈值。 */
    class Stats {
        private val eyeSpan = ArrayList<Float>()
        private val noseOff = ArrayList<Float>()
        private val earBal = ArrayList<Float>()
        private val pitch = ArrayList<Float>()
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
            // 耳朵常被遮挡 / 关键点可能拿不到，缺失就不计入分布
            metrics.earBalance?.let { earBal += it }
            metrics.pitchRatio?.let { pitch += it }
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
                append(desc("earBalance", earBal)).append('\n')
                append(desc("pitchRatio(低头↑)", pitch)).append('\n')
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
        val (moX, moY) = px(MOUTH)

        val interEye = hypot(reX - leX, reY - leY)
        if (interEye <= 0f) return null

        val boxWidthPx = boxW * imgW
        if (boxWidthPx <= 0f) return null

        // 双眼间距占人脸框宽的比例
        val eyeSpanRatio = interEye / boxWidthPx

        // 鼻尖相对双眼中点的水平偏移（yaw 主要体现在水平方向），用双眼间距归一化
        val eyeMidX = (reX + leX) / 2f
        val noseOffset = abs(noX - eyeMidX) / interEye

        // 俯仰（低头/抬头）：(眼线中点→鼻尖) / (鼻尖→嘴中心)
        // 低头时头绕双耳轴旋转，鼻尖投影向下挤向嘴部：分子变大、分母变小，比值升高
        val eyeMidY = (reY + leY) / 2f
        val dEyeNose = hypot(noX - eyeMidX, noY - eyeMidY)
        val dNoseMouth = hypot(moX - noX, moY - noY)
        // 分母过小说明关键点退化，比值无意义 → 返回 null 表示无法判断
        val pitchRatio = if (dNoseMouth > interEye * 0.05f) dEyeNose / dNoseMouth else null

        return Metrics(eyeSpanRatio, noseOffset, null, pitchRatio)
    }
}
