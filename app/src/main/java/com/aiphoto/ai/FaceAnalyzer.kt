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
 * 人脸分析编排：MediaPipe 检测人脸 → 裁出人脸 → MobileFaceNet 提特征。
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
     * 分析一张图，返回图里所有人脸的特征。没有检测到人脸时返回空列表。
     * 单张失败只影响这一张（返回空），不抛出去打断整批流程。
     */
    suspend fun analyze(bitmap: Bitmap): List<FaceFeature> = withContext(dispatcher) {
        runCatching {
            val d = ensureDetector() ?: return@runCatching emptyList<FaceFeature>()
            val e = ensureEmbedder() ?: return@runCatching emptyList<FaceFeature>()

            // 这里会把人脸的检测结果返回，可能存在多个人脸
            val boxes = d.detect(bitmap)
            boxes.mapNotNull { box ->
                val crop = FaceCropper.crop(bitmap, box) ?: return@mapNotNull null
                try {
                    // 做模型推理获取向量
                    FaceFeature(box, e.embed(crop))
                } finally {
                    crop.recycle()
                }
            }
        }.onFailure {
            Log.w(TAG, "人脸分析失败，跳过这张图", it)
        }.getOrDefault(emptyList())
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
    }
}
