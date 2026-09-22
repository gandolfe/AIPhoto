package com.aiphoto.ui.album

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aiphoto.AiPhotoApp
import com.aiphoto.data.MediaImage
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class SmartAlbumViewModel(application: Application) : AndroidViewModel(application) {

    private val container = (application as AiPhotoApp).container
    private val tagDao = container.photoTagDao
    private val photoStore = container.photoStore

    /** 类别 id -> 图片数量。 */
    val counts: StateFlow<Map<String, Int>> = tagDao.observeCategoryCounts()
        .map { list -> list.associate { it.category to it.count } }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyMap())

    val totalAnalyzed: StateFlow<Int> = tagDao.observeAnalyzedCount()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), 0)

    private val _categoryPhotos = MutableStateFlow<List<MediaImage>>(emptyList())
    val categoryPhotos: StateFlow<List<MediaImage>> = _categoryPhotos.asStateFlow()

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading.asStateFlow()

    fun loadCategory(categoryId: String) {
        viewModelScope.launch {
            _loading.value = true
            try {
                if (photoStore.photos.value.isEmpty()) {
                    photoStore.refresh()
                }
                // 翻页后未滚到的图片不在内存索引里，这里先把全部图片加载完，
                // 否则 byIds 会漏掉那些还没滚动触发的图片
                if (!photoStore.fullyLoaded.value) {
                    photoStore.ensureAllLoaded()
                }
                val ids = tagDao.photoIdsByCategory(categoryId)
                _categoryPhotos.value = photoStore.byIds(ids)
            } finally {
                _loading.value = false
            }
        }
    }

    fun clearCategory() {
        _categoryPhotos.value = emptyList()
    }
}
