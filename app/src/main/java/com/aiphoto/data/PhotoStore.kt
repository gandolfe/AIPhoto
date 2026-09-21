package com.aiphoto.data

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * 进程内共享的图片列表缓存，避免时间线与智能相册重复查询 MediaStore。
 */
class PhotoStore(private val repository: MediaStoreRepository) {

    private val _photos = MutableStateFlow<List<MediaImage>>(emptyList())
    val photos: StateFlow<List<MediaImage>> = _photos.asStateFlow()

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading.asStateFlow()

    @Volatile
    private var index: Map<Long, MediaImage> = emptyMap()

    private val _totalCount = MutableStateFlow(0)
    val totalCount: StateFlow<Int> = _totalCount.asStateFlow()

    suspend fun refresh(pageSize: Int = DEFAULT_PAGE_SIZE) {
        _loading.value = true
        try {
            _totalCount.value = repository.countImages()
            val list = repository.loadImages(limit = pageSize)
            _photos.value = list
            index = list.associateBy { it.id }
        } finally {
            _loading.value = false
        }
    }

    fun byId(id: Long): MediaImage? = index[id]

    /** 按给定 id 顺序取出图片；顺序与入参一致，缺失项忽略。 */
    fun byIds(ids: Collection<Long>): List<MediaImage> = ids.mapNotNull { index[it] }

    companion object {
        const val DEFAULT_PAGE_SIZE = 3000
    }
}
