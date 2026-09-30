package com.aiphoto.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyGridState
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.foundation.shape.RoundedCornerShape
// Dialog / DialogProperties 都在 androidx.compose.ui.window（Compose UI 原生），
// material3 里只有 AlertDialog / BasicAlertDialog，没有裸 Dialog
import androidx.compose.ui.window.Dialog
// DialogProperties 在 androidx.compose.ui.window 下，不在 material3
import androidx.compose.ui.window.DialogProperties
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.aiphoto.data.MediaImage

/** 图片网格，用于时间线与分类详情。点击任意一张会全屏看大图，再点一下退回网格。 */
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
    // 当前全屏查看的图片；非 null 时叠加显示查看器
    var viewing by remember { mutableStateOf<MediaImage?>(null) }

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
            PhotoCell(photo, onClick = { viewing = photo })
        }
    }

    viewing?.let { photo ->
        PhotoViewer(photo = photo, onDismiss = { viewing = null })
    }
}

@Composable
fun PhotoCell(
    photo: MediaImage,
    modifier: Modifier = Modifier,
    // 传了才可点击（默认不可点，避免影响其它调用处）
    onClick: (() -> Unit)? = null,
) {
    val clickable = if (onClick == null) {
        modifier
    } else {
        modifier.clickable(
            // 去掉点击水波纹：网格里点图片不该有涟漪
            interactionSource = remember { MutableInteractionSource() },
            indication = null,
            onClick = onClick,
        )
    }
    AsyncImage(
        model = photo.uri,
        contentDescription = photo.displayName,
        contentScale = ContentScale.Crop,
        modifier = clickable
            .aspectRatio(1f)
            .clip(RoundedCornerShape(2.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant),
    )
}

/**
 * 全屏大图查看：黑底、图片完整显示（Fit），**点任意位置退出**。
 *
 * 用 Dialog 承载而不是普通 Box：Dialog 会盖在整屏之上，
 * 连底部导航栏一起遮住，才算真正的「全屏」；普通 Box 只能在 Scaffold 内容区里，
 * 底部导航栏还会露出来。
 */
@Composable
fun PhotoViewer(
    photo: MediaImage,
    onDismiss: () -> Unit,
) {
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            usePlatformDefaultWidth = false,   // 不要系统默认宽度限制
            decorFitsSystemWindows = false,    // 允许铺满含系统栏的区域
        ),
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null,
                    onClick = onDismiss,
                ),
            contentAlignment = Alignment.Center,
        ) {
            AsyncImage(
                model = photo.uri,
                contentDescription = photo.displayName,
                contentScale = ContentScale.Fit,   // 完整显示，不裁切
                modifier = Modifier.fillMaxSize(),
            )
        }
    }
}
