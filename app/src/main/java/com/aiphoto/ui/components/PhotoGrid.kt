package com.aiphoto.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyGridState
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.aiphoto.data.MediaImage

/** 图片网格，用于时间线与分类详情。 */
@Composable
fun PhotoGrid(
    photos: List<MediaImage>,
    modifier: Modifier = Modifier,
    columns: Int = 4,
    // 由调用方持有，便于滚动到底部时触发翻页；不传则内部自己建
    state: LazyGridState = rememberLazyGridState(),
    // 接近列表末尾时回调（用于加载下一页），留空则不触发翻页
    onNearEnd: (() -> Unit)? = null,
) {
    // 滚动接近末尾（距最后 40 个格子内）时触发翻页；loadMore 自身会去重/防到底，
    // 所以这里频繁回调也无副作用。
    LaunchedEffect(state, onNearEnd) {
        if (onNearEnd == null) return@LaunchedEffect
        snapshotFlow { state.layoutInfo }
            .collect { info ->
                val last = info.visibleItemsInfo.lastOrNull()?.index ?: 0
                if (last >= info.totalItemsCount - 40) {
                    onNearEnd()
                }
            }
    }

    LazyVerticalGrid(
        state = state,
        columns = GridCells.Fixed(columns),
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(2.dp),
        horizontalArrangement = Arrangement.spacedBy(2.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        items(photos, key = { it.id }) { photo ->
            PhotoCell(photo)
        }
    }
}

@Composable
fun PhotoCell(photo: MediaImage, modifier: Modifier = Modifier) {
    AsyncImage(
        model = photo.uri,
        contentDescription = photo.displayName,
        contentScale = ContentScale.Crop,
        modifier = modifier
            .aspectRatio(1f)
            .clip(RoundedCornerShape(2.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant),
    )
}
