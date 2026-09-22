package com.aiphoto.ui.timeline

import android.app.Application
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aiphoto.AiPhotoApp
import com.aiphoto.data.MediaImage
import com.aiphoto.data.PhotoTagEntity
import com.aiphoto.util.ImageLoader
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.Locale

/** 批量分析进度。 */
data class AnalysisState(
    val running: Boolean = false,
    val total: Int = 0,
    val done: Int = 0,
    val failed: Int = 0,
    /** 当前正在做的事，给 UI 显示，避免长时间无变化让人以为卡死。 */
    val phase: String = "",
    val message: String? = null,
)

class TimelineViewModel(application: Application) : AndroidViewModel(application) {

    private val container = (application as AiPhotoApp).container
    private val photoStore = container.photoStore
    private val tagDao = container.photoTagDao
    private val analyzer = container.sceneAnalyzer

    val photos: StateFlow<List<MediaImage>> = photoStore.photos
    val loading: StateFlow<Boolean> = photoStore.loading
    val loadingMore: StateFlow<Boolean> = photoStore.loadingMore
    val fullyLoaded: StateFlow<Boolean> = photoStore.fullyLoaded

    private val _analysis = MutableStateFlow(AnalysisState())
    val analysis: StateFlow<AnalysisState> = _analysis.asStateFlow()

    private val _analyzedIds = MutableStateFlow<Set<Long>>(emptySet())
    val analyzedIds: StateFlow<Set<Long>> = _analyzedIds.asStateFlow()

    private val _modelReady = MutableStateFlow(analyzer.isModelAvailable)
    val modelReady: StateFlow<Boolean> = _modelReady.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            photoStore.refresh()
            _analyzedIds.value = tagDao.analyzedPhotoIds().toHashSet()
            _modelReady.value = analyzer.isModelAvailable
        }
    }

    /** 对尚未分析的图片依次执行场景分类，结果写入数据库。 */
    fun startAnalysis() {
        if (_analysis.value.running) return

        viewModelScope.launch {
            if (!analyzer.isModelAvailable) {
                _analysis.value = AnalysisState(
                    message = "未找到模型，请把 clip_scene.pte 放入 app/src/main/assets/models/",
                )
                return@launch
            }

            // 翻页加载可能还没滚到的图片，保证分析覆盖相册全部图片
            photoStore.ensureAllLoaded()

            val all = photoStore.photos.value
            if (all.isEmpty()) {
                _analysis.value = AnalysisState(message = "没有可分析的图片")
                return@launch
            }

            val analyzed = tagDao.analyzedPhotoIds().toHashSet()
            val pending = all.filter { it.id !in analyzed }
            if (pending.isEmpty()) {
                _analysis.value = AnalysisState(message = "全部图片均已分析")
                return@launch
            }

            _analysis.value = AnalysisState(running = true, total = pending.size)

            val context = getApplication<Application>()
            var done = 0
            var failed = 0

            for ((index, photo) in pending.withIndex()) {
                // 每张开始前就先把状态推给 UI。单张推理可能耗时数分钟，
                _analysis.value = _analysis.value.copy(
                    done = done,
                    failed = failed,
                    phase = if (done == 0) {
                        "加载模型并分析首张…（含首次拷贝+mmap 大模型，耗时明显更长）"
                    } else {
                        "正在分析第 ${index + 1}/${pending.size} 张"
                    },
                )

                val t0 = SystemClock.elapsedRealtime()
                val bitmap = runCatching { ImageLoader.loadForAnalysis(context, photo.uri) }.getOrNull()
                Log.i(
                    TAG,
                    "loadForAnalysis ${SystemClock.elapsedRealtime() - t0} ms, " +
                        "id=${photo.id}, bitmap=${bitmap?.width}x${bitmap?.height}"
                )
                if (bitmap == null) {
                    failed++
                } else {
                    try {
                        val result = analyzer.analyze(bitmap)
                        if (result != null) {
                            tagDao.upsert(
                                PhotoTagEntity(
                                    photoId = photo.id,
                                    topCategory = result.topLabel,
                                    topScore = result.topScore,
                                    scoreDetail = result.scores.joinToString(",") { (label, score) ->
                                        "$label:${String.format(Locale.US, "%.4f", score)}"
                                    },
                                    analyzedAt = System.currentTimeMillis(),
                                ),
                            )
                        } else {
                            failed++
                        }
                    } finally {
                        bitmap.recycle()
                    }
                }

                done++
                // 每张结束都刷新，不再按批节流
                _analysis.value = _analysis.value.copy(done = done, failed = failed)
            }

            _analyzedIds.value = tagDao.analyzedPhotoIds().toHashSet()
            _analysis.value = AnalysisState(
                running = false,
                total = pending.size,
                done = done,
                failed = failed,
                message = if (failed > 0) "分析完成：成功 ${done - failed} 张，失败 $failed 张"
                else "分析完成：共 $done 张",
            )
        }
    }

    fun dismissMessage() {
        _analysis.value = _analysis.value.copy(message = null)
    }

    /** 滚动到底部时由 UI 触发，加载下一页图片。已到底/正在加载时直接忽略。 */
    fun loadMore() {
        if (photoStore.loadingMore.value || photoStore.fullyLoaded.value) return
        viewModelScope.launch { photoStore.loadMore() }
    }

    private companion object {
        const val TAG = "TimelineVM"
    }

    /**
     * 清空全部分析结果。用于换模型后重新跑一遍：旧模型的标签与新 label_bank
     * 不兼容，留着会让 [startAnalysis] 把这些照片当成「已分析」直接跳过
     * （实测：旧 5 类结果 2802 条，导致新 39 类模型只处理了新增的 2 张）。
     *
     * 只删 photo_tags 表，不动 MediaStore 原图。智能相册页走的是 Room Flow 查询，
     * 会自动刷新到空状态；这里另需手动重置 [_analyzedIds]。
     */
    fun clearAnalysis() {
        if (_analysis.value.running) return

        viewModelScope.launch {
            val removed = runCatching { tagDao.analyzedCount() }.getOrDefault(0)
            runCatching { tagDao.clear() }
                .onFailure { e ->
                    _analysis.value = _analysis.value.copy(
                        running = false,
                        message = "清空失败：${e.message ?: e::class.simpleName}",
                    )
                    return@launch
                }
            _analyzedIds.value = emptySet()
            _analysis.value = AnalysisState(
                message = "已清空 $removed 条分析结果，可以重新分析",
            )
        }
    }
}
