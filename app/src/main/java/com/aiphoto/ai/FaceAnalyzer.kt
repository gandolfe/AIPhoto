package com.aiphoto.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.IOException
import java.util.concurrent.Executors

/**
 * 人脸分析编排：MediaPipe 检测人脸 → **筛出唯一一张清晰正面脸** → 裁出 → MobileFaceNet 提特征。
 *
 * ## 每张照片最多只出一张脸（当前策略）
 * 一张图里即使有多个人，也只保留**质量最好**的那一。理由：多人合照里的小脸、
 * 侧脸 embedding 质量差，是聚类噪声和「一人多簇」的主要来源；先保证每张图
 * 只贡献一张高质量向量，聚类难度会大幅下降。
 *
 * 入选条件（都可调，见伴生对象常量）：
 *   1. 人脸足够大（[MIN_FACE_PX]）
 *   2. 检测置信度足够高（[MIN_EMBED_SCORE]）
 *   3. 姿态为正面（[FaceQuality]：侧脸 / 局部脸会被拒）
 *
 * **只做「这张脸长什么样」的向量化，不做聚类** —— 聚类和落库交给
 * FaceClusterer + ViewModel，这样职责清楚，也方便单独调聚类阈值。
 *
 * 调度方式沿用 [SceneAnalyzer]：单线程 executor + withContext 保证串行。
 * 这里特地**不和 CLIP 共用线程** —— 检测、提特征、场景分类各自都要占内存和 CPU，
 * 分开跑避免互相抢资源。
 */
class FaceAnalyzer(private val appContext: Context) : AutoCloseable {

    /** 一张人脸的检测框 + 特征向量。 */
    data class FaceFeature(
        val box: DetectedFaceBox,
        val embedding: FloatArray,
    )

    // 创建一个单线程线程池，并且命名为"face-inference"
    private val executor = Executors.newSingleThreadExecutor { Thread(it, "face-inference") }

    // 这一步是把线程池包装成协程调度器，支持供协程使用
    private val dispatcher = executor.asCoroutineDispatcher()

    private val loadMutex = Mutex()
    private var detector: FaceDetector? = null
    private var embedder: MobileFaceNet? = null

    /** 正面姿态判定阈值（可调，看 [dumpQualityStats] 的分布再定）。 */
    private val qualityConfig = FaceQuality.Config()

    /** 姿态指标分布，用于校准阈值。 */
    private val qualityStats = FaceQuality.Stats()

    /** 两个模型（tflite 检测器 + pte 特征提取）都齐了才可用。 */
    val isAvailable: Boolean
        get() = MobileFaceNet.isAvailable(appContext) && detectorModelExists()

    private fun detectorModelExists(): Boolean = try {
        appContext.assets.openFd(DETECTOR_ASSET).close()
        true
    } catch (e: IOException) {
        false
    }

    /**
     * 分析一张图，**最多返回一张**人脸的特征（清晰 + 正面 + 质量最好的那张）。
     * 没有合格人脸时返回 null。单张失败只影响这一张，不抛出去打断整批流程。
     */
    suspend fun analyze(bitmap: Bitmap): FaceFeature? = withContext(dispatcher) {
        runCatching {
            val d = ensureDetector() ?: return@runCatching null
            val e = ensureEmbedder() ?: return@runCatching null

            // 这里会把人脸的检测结果返回，可能存在多个人脸
            val boxes = d.detect(bitmap)
            val chosen = pickBestFace(boxes, bitmap) ?: return@runCatching null

            val crop = FaceCropper.crop(bitmap, chosen) ?: return@runCatching null
            try {
                // 做模型推理获取向量
                FaceFeature(chosen, e.embed(crop))
            } finally {
                crop.recycle()
            }
        }.onFailure {
            Log.w(TAG, "人脸分析失败，跳过这张图", it)
        }.getOrNull()
    }

    /**
     * 从检测结果里挑出**唯一一张**合格人脸：足够大 → 置信度够 → 正面姿态，
     * 然后在合格者里取置信度最高的。全不合格返回 null。
     */
    private fun pickBestFace(boxes: List<DetectedFaceBox>, bitmap: Bitmap): DetectedFaceBox? {
        var best: DetectedFaceBox? = null
        var bestScore = -1f

        for (box in boxes) {
            // 1) 太小 / 置信度太低 → embedding 不可靠
            if (!isUsable(box, bitmap)) continue

            // 2) 姿态检查：侧脸、局部脸不要
            val metrics = FaceQuality.measure(box, bitmap.width, bitmap.height)
            if (metrics != null) {
                val reason = metrics.rejectReason(qualityConfig)
                qualityStats.record(metrics, reason)
                if (reason != null) {
                    Log.d(TAG, "姿态不合格($reason): $metrics")
                    continue
                }
            } else {
                // 拿不到人脸五官，直接舍弃
                qualityStats.record(null, null)
                continue
            }

            if (box.score > bestScore) {
                bestScore = box.score
                best = box
            }
        }
        return best
    }

    /** 姿态指标分布，定期打到日志用于校准阈值。 */
    fun dumpQualityStats() {
        Log.i(TAG, "人脸姿态/质量统计:\n${qualityStats.summary()}")
    }

    /**
     * 这张脸值不值得提特征。
     *
     * 人脸太小（放大到 112×112 后是糊的）或检测置信度太低（多半是误检/严重遮挡），
     * 算出来的 embedding 就是噪声 —— 它跟谁都不像，于是自成一家，
     * 变成「一个簇只有 1 张照片」的主要来源。这类噪声不进聚类，比事后清理划算。
     */
    private fun isUsable(box: DetectedFaceBox, bitmap: Bitmap): Boolean {
        val faceW = (box.rightPct - box.leftPct) * bitmap.width
        val faceH = (box.bottomPct - box.topPct) * bitmap.height
        return box.score >= MIN_EMBED_SCORE && minOf(faceW, faceH) >= MIN_FACE_PX
    }

    private suspend fun ensureDetector(): FaceDetector? = loadMutex.withLock {
        detector ?: runCatching { FaceDetector(appContext) }
            .onFailure { Log.w(TAG, "FaceDetector 初始化失败", it) }
            .getOrNull()
            .also { detector = it }
    }

    private suspend fun ensureEmbedder(): MobileFaceNet? = loadMutex.withLock {
        embedder ?: runCatching { MobileFaceNet.create(appContext) }
            .onFailure { Log.w(TAG, "MobileFaceNet 初始化失败", it) }
            .getOrNull()
            .also { embedder = it }
    }

    override fun close() {
        detector?.close()
        detector = null
        embedder?.close()
        embedder = null
        executor.shutdown()
    }

    companion object {
        private const val TAG = "FaceAnalyzer"
        private const val DETECTOR_ASSET = "models/face_detector.tflite"

        /**
         * 人脸最小边长（**分析用缩略图的像素**）。低于此值放大到 112 后太糊。
         * 分析图长边 320px。调高 → 更干净但会漏掉合影里的小脸；调低 → 噪声变多。
         */
        private const val MIN_FACE_PX = 36f

        /**
         * 提特征要求的最低检测置信度。比 [FaceDetector] 的 0.5 更严：
         * 检测只要框准就行，但 embedding 需要人脸足够清晰。
         */
        private const val MIN_EMBED_SCORE = 0.7f
    }
}
