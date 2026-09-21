package com.aiphoto.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/**
 * 场景分析入口。内部用单线程 dispatcher 串行执行 ExecuTorch 推理，
 * 既避免 Module 并发访问，也避免多个大模型同时占用内存。
 */
class SceneAnalyzer(private val appContext: Context) : AutoCloseable {

    private val dispatcher = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "clip-inference")
    }.asCoroutineDispatcher()

    private val loadMutex = Mutex()

    @Volatile
    private var classifier: ClipSceneClassifier? = null

    /** 模型是否已就绪（assets 中存在 .pte）。 */
    val isModelAvailable: Boolean
        get() = ClipSceneClassifier.isModelAvailable(appContext)

    /**
     * 对单张图片分类。模型缺失或推理失败时返回 null。
     */
    suspend fun analyze(bitmap: Bitmap): ClipSceneClassifier.SceneResult? = withContext(dispatcher) {
        val engine = ensureClassifier() ?: return@withContext null
        runCatching { engine.classify(bitmap) }
            .onFailure { Log.e(TAG, "推理失败", it) }
            .getOrNull()
    }

    private suspend fun ensureClassifier(): ClipSceneClassifier? = loadMutex.withLock {
        classifier ?: runCatching { ClipSceneClassifier.create(appContext) }
            .onFailure { Log.e(TAG, "模型加载失败", it) }
            .getOrNull()
            ?.also { classifier = it }
    }

    override fun close() {
        runCatching { classifier?.close() }
        classifier = null
        runCatching { dispatcher.close() }
    }

    private companion object {
        const val TAG = "SceneAnalyzer"
    }
}
