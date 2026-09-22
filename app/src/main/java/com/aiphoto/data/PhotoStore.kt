package com.aiphoto.data

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * 进程内共享的图片列表缓存，避免时间线与智能相册重复查询 MediaStore。
 *
 * 分页策略：
 *   - [refresh] 只加载第一页（[DEFAULT_PAGE_SIZE] 张，按拍摄时间倒序）。
 *   - [loadMore] 加载下一页（offset = 已加载数量），追加到末尾；到底后 [fullyLoaded]=true。
 *   - 内存只存图片元数据（每张几百字节，2 万张约 4~8 MB），不缓存原图位图，
 *     所以即便翻到底、全部 2 万张都在内存里也不会 OOM；缩略图由 Coil 按需解码。
 */
class PhotoStore(private val repository: MediaStoreRepository) {

    private val _photos = MutableStateFlow<List<MediaImage>>(emptyList())
    val photos: StateFlow<List<MediaImage>> = _photos.asStateFlow()

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading.asStateFlow()

    /** 正在加载下一页（滚动触发的翻页）。 */
    private val _loadingMore = MutableStateFlow(false)
    val loadingMore: StateFlow<Boolean> = _loadingMore.asStateFlow()

    /** 所有页都已加载完（已到 MediaStore 末尾）。 */
    private val _fullyLoaded = MutableStateFlow(false)
    val fullyLoaded: StateFlow<Boolean> = _fullyLoaded.asStateFlow()

    private val _totalCount = MutableStateFlow(0)
    val totalCount: StateFlow<Int> = _totalCount.asStateFlow()

    @Volatile
    private var index: Map<Long, MediaImage> = emptyMap()

    // 串行化 loadMore，避免滚动事件高频并发触发重复加载同一页
    private val mutex = Mutex()

    suspend fun refresh(pageSize: Int = DEFAULT_PAGE_SIZE) {
        _loading.value = true
        try {
            _totalCount.value = repository.countImages()
            val first = repository.loadImages(limit = pageSize)
            _photos.value = first
            index = first.associateBy { it.id }
            _fullyLoaded.value = first.size < pageSize
        } finally {
            _loading.value = false
        }
    }

    /**
     * 加载下一页（offset = 当前已加载数量）并追加到列表末尾。
     * 重复调用 / 并发调用都安全：已在加载中或已到底时直接返回。
     */
    suspend fun loadMore(pageSize: Int = DEFAULT_PAGE_SIZE) {
        if (_fullyLoaded.value || _loadingMore.value || _loading.value) return
        mutex.withLock {
            // 拿到锁后双重检查，避免并发下重复加载同一页
            if (_fullyLoaded.value || _loadingMore.value) return
            _loadingMore.value = true
        }
        try {
            val offset = _photos.value.size
            val page = repository.loadImages(limit = pageSize, offset = offset)
            if (page.isEmpty()) {
                _fullyLoaded.value = true
                return
            }
            _photos.value = ArrayList(_photos.value + page)
            index = index + page.associateBy { it.id }
            // 一页没装满，说明已经是最后一页
            _fullyLoaded.value = page.size < pageSize
        } finally {
            _loadingMore.value = false
        }
    }

    /** 顺序翻页直到全部加载完。批量分析前调用，确保覆盖所有图片（含未滚到的页）。 */
    suspend fun ensureAllLoaded(pageSize: Int = DEFAULT_PAGE_SIZE) {
        while (!_fullyLoaded.value) {
            loadMore(pageSize)
        }
    }

    fun byId(id: Long): MediaImage? = index[id]

    /** 按给定 id 顺序取出图片；顺序与入参一致，缺失项忽略。 */
    fun byIds(ids: Collection<Long>): List<MediaImage> = ids.mapNotNull { index[it] }

    companion object {
        const val DEFAULT_PAGE_SIZE = 2000
    }
}
