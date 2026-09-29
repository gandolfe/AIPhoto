package com.aiphoto.ai

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
// MediaPipe 的 Options 是**嵌套类** FaceDetector.FaceDetectorOptions，不是同名的顶层类
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector as MpFaceDetector

/** 检测到的一张人脸，坐标为相对原图的比例（0..1）。
 * 这里使用比例，可以适配多种分辨率的情况；主要是后面封面的裁剪，读取的可能是原图 */
data class DetectedFaceBox(
    val leftPct: Float,
    val topPct: Float,
    val rightPct: Float,
    val bottomPct: Float,
    /** 检测置信度。 */
    val score: Float,
)

/**
 * MediaPipe 人脸检测（BlazeFace short range）。
 *
 * 只负责**框出人脸位置**，不负责识别是谁。识别靠 [MobileFaceNet] 提特征 + 聚类。
 *
 * 线程不安全（MediaPipe 实例本身就不是线程安全的），由 [FaceAnalyzer] 的单线程
 * dispatcher 保证串行调用。
 */
class FaceDetector(context: Context) : AutoCloseable {

    companion object {
        private const val MODEL_ASSET = "models/face_detector.tflite"
        /**
         * 短距离模型的召回率高但误检也多，0.5 是 MediaPipe 示例里的常用值。
         * 调低→更多框（含误检，会污染聚类）；调高→漏掉小脸/侧脸。
         */
        private const val MIN_CONFIDENCE = 0.75f
    }

    private val detector: MpFaceDetector

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_ASSET)
            .build()
        val options = MpFaceDetector.FaceDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setMinDetectionConfidence(MIN_CONFIDENCE)
            .build()
        detector = MpFaceDetector.createFromOptions(context, options)
    }

    /**
     * 检测一张图里的所有人脸。
     * 返回按置信度从高到低排序（MediaPipe 本身已排序，这里不再重排）。
     *
     *
     * coerceIn 的作用是把一个数值限制（钳制）在指定的区间（最小值 ~ 最大值）之内。
     * 1. 直观理解简单来说，它的效果等价于：
     * 如果数值 小于最小值 ，则返回 最小值；
     * 如果数值 大于最大值 则 返回 最大值；
     * 如果数值 在区间范围内 则 保持原数值 不变。
     */
    fun detect(bitmap: Bitmap): List<DetectedFaceBox> {
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result = detector.detect(mpImage)
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()

        return result.detections().mapNotNull { detection ->
            // boundingBox() 返回的是 RectF（浮点像素坐标，字段 left/top/right/bottom）
            val box = detection.boundingBox()
            val score = detection.categories().firstOrNull()?.score() ?: 0f
            if (score < MIN_CONFIDENCE) return@mapNotNull null
            DetectedFaceBox(
                leftPct = (box.left / w).coerceIn(0f, 1f),
                topPct = (box.top / h).coerceIn(0f, 1f),
                rightPct = (box.right / w).coerceIn(0f, 1f),
                bottomPct = (box.bottom / h).coerceIn(0f, 1f),
                score = score,
            )
        }
    }

    override fun close() {
        detector.close()
    }
}
