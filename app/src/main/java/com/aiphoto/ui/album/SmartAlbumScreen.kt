package com.aiphoto.ui.album

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.aiphoto.ai.SceneCategory
import com.aiphoto.data.MediaImage
import com.aiphoto.ui.components.PhotoGrid

@Composable
fun SmartAlbumScreen(
    modifier: Modifier = Modifier,
    viewModel: SmartAlbumViewModel = viewModel(),
) {
    val counts by viewModel.counts.collectAsStateWithLifecycle()
    val total by viewModel.totalAnalyzed.collectAsStateWithLifecycle()
    val categoryPhotos by viewModel.categoryPhotos.collectAsStateWithLifecycle()
    val loading by viewModel.loading.collectAsStateWithLifecycle()

    var selectedId by rememberSaveable { mutableStateOf<String?>(null) }

    val currentId = selectedId
    if (currentId == null) {
        CategoryOverview(
            counts = counts,
            totalAnalyzed = total,
            modifier = modifier,
            onSelect = { id ->
                selectedId = id
                viewModel.loadCategory(id)
            },
        )
    } else {
        CategoryDetail(
            categoryId = currentId,
            photos = categoryPhotos,
            loading = loading,
            modifier = modifier,
            onBack = {
                selectedId = null
                viewModel.clearCategory()
            },
        )
    }
}

@Composable
private fun CategoryOverview(
    counts: Map<String, Int>,
    totalAnalyzed: Int,
    modifier: Modifier,
    onSelect: (String) -> Unit,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 12.dp),
    ) {
        Text(
            text = "智能相册",
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.padding(top = 12.dp),
        )
        Text(
            text = if (totalAnalyzed == 0) {
                "还没有分析结果，请到「时间线」点击 AI 分析"
            } else {
                "已分析 $totalAnalyzed 张图片"
            },
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(top = 4.dp, bottom = 12.dp),
        )

        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(bottom = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            items(SceneCategory.entries.toList(), key = { it.id }) { category ->
                CategoryCard(
                    category = category,
                    count = counts[category.id] ?: 0,
                    onClick = { onSelect(category.id) },
                )
            }
        }
    }
}

@Composable
private fun CategoryCard(
    category: SceneCategory,
    count: Int,
    onClick: () -> Unit,
) {
    Card(
        onClick = onClick,
        modifier = Modifier
            .fillMaxWidth()
            .height(120.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.Center,
        ) {
            Text(text = category.emoji, fontSize = 28.sp)
            Text(
                text = category.displayName,
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(top = 6.dp),
            )
            Text(
                text = "$count 张",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun CategoryDetail(
    categoryId: String,
    photos: List<MediaImage>,
    loading: Boolean,
    modifier: Modifier,
    onBack: () -> Unit,
) {
    val category = SceneCategory.fromId(categoryId)

    Column(modifier = modifier.fillMaxSize()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 4.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "返回")
            }
            Text(
                text = "${category?.emoji.orEmpty()} ${category?.displayName ?: categoryId}",
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.weight(1f),
            )
            Text(
                text = "${photos.size} 张",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(end = 12.dp),
            )
        }

        Box(Modifier.fillMaxSize()) {
            when {
                loading && photos.isEmpty() -> {
                    CircularProgressIndicator(Modifier.align(Alignment.Center))
                }

                photos.isEmpty() -> {
                    Text(
                        text = "该分类下暂无图片",
                        modifier = Modifier.align(Alignment.Center),
                    )
                }

                else -> {
                    PhotoGrid(photos = photos, columns = 3)
                }
            }
        }
    }
}
