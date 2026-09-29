package com.aiphoto.ui.people

import android.app.Application
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aiphoto.AiPhotoApp
import com.aiphoto.ai.FaceClusterer
import com.aiphoto.ai.FaceCropper
import com.aiphoto.data.FaceEntity
import com.aiphoto.data.MediaImage
import com.aiphoto.data.PersonEntity
import com.aiphoto.data.PersonSummary
import com.aiphoto.data.toEmbeddingBytes
import com.aiphoto.util.ImageLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** 人脸分析进度。 */
data class FaceAnalysisState(
    val running: Boolean = false,
    val total: Int = 0,
    val done: Int = 0,
    /** 累计检出的人脸数。 */
    val facesFound: Int = 0,
    /** 累计人物数（簇数）。 */
    val peopleFound: Int = 0,
    val phase: String = "",
    val message: String? = null,
)

class PeopleViewModel(application: Application) : AndroidViewModel(application) {

    private val container = (application as AiPhotoApp).container
    private val faceDao = container.faceDao
    private val photoStore = container.photoStore
    private val faceAnalyzer = container.faceAnalyzer
    private val appContext = application

    /** 人物列表（含每人照片数），由数据库 Flow 驱动 —— 簇是聚类出来的，不能写死。 */
    val people: StateFlow<List<PersonSummary>> = faceDao.observePersonSummaries()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    private val _analysis = MutableStateFlow(FaceAnalysisState())
    val analysis: StateFlow<FaceAnalysisState> = _analysis.asStateFlow()

    /** personId -> 封面（裁出来的人脸小图）。 */
    private val _covers = MutableStateFlow<Map<Long, Bitmap>>(emptyMap())
    val covers: StateFlow<Map<Long, Bitmap>> = _covers.asStateFlow()

    private val _personPhotos = MutableStateFlow<List<MediaImage>>(emptyList())
    val personPhotos: StateFlow<List<MediaImage>> = _personPhotos.asStateFlow()

    private val _selectedPerson = MutableStateFlow<Long?>(null)
    val selectedPerson: StateFlow<Long?> = _selectedPerson.asStateFlow()

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading.asStateFlow()

    /**
     * 人脸检测 + 特征提取 + 聚类。
     *
     * 流程照搬场景分析的骨架（拉全量 → 排除已分析 → 逐张推进进度），
     * 区别是每张图可能产出多张脸，且要顺带维护簇的 centroid。
     */
    fun startFaceAnalysis() {
        if (_analysis.value.running) return
        viewModelScope.launch {
            if (!faceAnalyzer.isAvailable) {
                _analysis.value = FaceAnalysisState(
                    message = "未找到人脸模型：assets/models/ 下需要 face_detector.tflite 与 mobilefacenet.pte + mobilefacenet.json",
                )
                return@launch
            }

            _analysis.value = FaceAnalysisState(running = true, phase = "加载图片列表…")
            photoStore.ensureAllLoaded()

            val analyzed = faceDao.analyzedPhotoIds().toHashSet()
            val pending = photoStore.photos.value.filter { it.id !in analyzed }
            if (pending.isEmpty()) {
                _analysis.value = FaceAnalysisState(message = "所有图片都已完成人脸检测")
                return@launch
            }

            //加载已经识别的人物分类集合
            val clusterer = FaceClusterer()
            clusterer.load(faceDao.allPeople())

            _analysis.value = FaceAnalysisState(running = true, total = pending.size)

            val startedAt = SystemClock.elapsedRealtime()
            var done = 0
            var facesFound = 0

            for (photo in pending) {
                _analysis.value = _analysis.value.copy(
                    phase = if (done == 0) "加载模型并分析首张…" else "检测人脸 ${done + 1}/${pending.size}",
                )

                val bitmap = ImageLoader.loadForAnalysis(appContext, photo.uri)
                if (bitmap == null) {
                    done++
                    _analysis.value = _analysis.value.copy(done = done)
                    continue
                }

                try {
                    val features = faceAnalyzer.analyze(bitmap)
                    if (features.isNotEmpty()) {
                        val entities = ArrayList<FaceEntity>(features.size)
                        for (f in features) {
                            val assignment = clusterer.assign(f.embedding)
                            val personId = if (assignment.isNew) {
                                val id = faceDao.insertPerson(
                                    PersonEntity(
                                        centroid = assignment.centroid.toEmbeddingBytes(),
                                        faceCount = 1,
                                        coverPhotoId = photo.id,
                                    )
                                )
                                clusterer.register(id, f.embedding)
                                id
                            } else {
                                faceDao.updatePerson(
                                    personId = assignment.personId,
                                    centroid = assignment.centroid.toEmbeddingBytes(),
                                    faceCount = assignment.faceCount,
                                    coverPhotoId = photo.id,
                                )
                                assignment.personId
                            }
                            entities += FaceEntity(
                                photoId = photo.id,
                                personId = personId,
                                leftPct = f.box.leftPct,
                                topPct = f.box.topPct,
                                rightPct = f.box.rightPct,
                                bottomPct = f.box.bottomPct,
                                score = f.box.score,
                                embedding = f.embedding.toEmbeddingBytes(),
                            )
                        }
                        faceDao.insertFaces(entities)
                        facesFound += features.size
                    }
                } finally {
                    bitmap.recycle()
                }

                done++
                _analysis.value = _analysis.value.copy(
                    done = done,
                    facesFound = facesFound,
                    peopleFound = clusterer.clusterCount(),
                )
            }

            val seconds = (SystemClock.elapsedRealtime() - startedAt) / 1000
            _analysis.value = _analysis.value.copy(
                running = false,
                phase = "",
                message = "完成：检出 $facesFound 张人脸，归为 ${clusterer.clusterCount()} 个人物，用时 ${seconds}s",
            )
            Log.i(TAG, "人脸分析完成: 图=${pending.size} 脸=$facesFound 人=${clusterer.clusterCount()}")
            loadCovers()
        }
    }

    /** 打开某个人物的相册。 */
    fun loadPerson(personId: Long) {
        viewModelScope.launch {
            _loading.value = true
            _selectedPerson.value = personId
            try {
                if (photoStore.photos.value.isEmpty()) photoStore.refresh()
                if (!photoStore.fullyLoaded.value) photoStore.ensureAllLoaded()
                val ids = faceDao.photoIdsByPerson(personId)
                _personPhotos.value = photoStore.byIds(ids)
            } finally {
                _loading.value = false
            }
        }
    }

    fun clearPerson() {
        _selectedPerson.value = null
        _personPhotos.value = emptyList()
    }

    /**
     * 生成人物封面：取该人物**置信度最高**的那张脸，裁出来做封面。
     * 用最高分而不是第一张，是为了让封面尽量是正脸、清晰的那张。
     */
    fun loadCovers() {
        viewModelScope.launch {
            val current = people.value
            if (current.isEmpty()) return@launch
            // 封面要用 photoStore.byId 拿图片 uri —— 依赖内存索引，
            // 冷启动时列表还没加载，这里先拉全量，否则一个封面都出不来
            if (photoStore.photos.value.isEmpty()) photoStore.ensureAllLoaded()

            val updated = _covers.value.toMutableMap()
            for (person in current) {
                if (updated.containsKey(person.personId)) continue
                val face = faceDao.bestFaceOfPerson(person.personId) ?: continue
                val photo = photoStore.byId(face.photoId) ?: continue
                val bitmap = withContext(Dispatchers.IO) {
                    ImageLoader.loadForAnalysis(appContext, photo.uri, targetSize = 256)
                } ?: continue
                val cover = FaceCropper.crop(
                    bitmap,
                    face.leftPct,
                    face.topPct,
                    face.rightPct,
                    face.bottomPct,
                )
                if (cover !== bitmap) bitmap.recycle()
                if (cover != null) updated[person.personId] = cover
            }
            _covers.value = updated
        }
    }

    /** 清空人脸数据（换模型或重跑时用）。 */
    fun clearFaces() {
        viewModelScope.launch {
            faceDao.clearFaces()
            faceDao.clearPeople()
            _covers.value = emptyMap()
            _analysis.value = FaceAnalysisState(message = "已清空人物数据")
        }
    }

    companion object {
        private const val TAG = "PeopleViewModel"
    }
}
