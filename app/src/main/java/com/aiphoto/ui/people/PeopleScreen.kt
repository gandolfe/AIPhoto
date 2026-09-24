package com.aiphoto.ui.people

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.aiphoto.data.PersonSummary
import com.aiphoto.ui.components.PhotoGrid

@Composable
fun PeopleScreen(
    modifier: Modifier = Modifier,
    viewModel: PeopleViewModel = viewModel(),
) {
    val people by viewModel.people.collectAsState()
    val covers by viewModel.covers.collectAsState()
    val selectedPerson by viewModel.selectedPerson.collectAsState()

    // 首次进入时补一次封面（分析完也会再调一次）
    LaunchedEffect(Unit) { viewModel.loadCovers() }

    if (selectedPerson != null) {
        PersonDetail(
            personId = selectedPerson!!,
            viewModel = viewModel,
            modifier = modifier,
        )
    } else {
        PeopleOverview(
            people = people,
            covers = covers,
            viewModel = viewModel,
            modifier = modifier,
        )
    }
}

@Composable
private fun PeopleOverview(
    people: List<PersonSummary>,
    covers: Map<Long, android.graphics.Bitmap>,
    viewModel: PeopleViewModel,
    modifier: Modifier = Modifier,
) {
    val analysis by viewModel.analysis.collectAsState()
    var showClearConfirm by remember { mutableStateOf(false) }

    Column(modifier = modifier.fillMaxSize()) {
        // ---- 顶部：人物分类按钮 + 进度 ----
        Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Button(
                    onClick = { viewModel.startFaceAnalysis() },
                    enabled = !analysis.running,
                ) {
                    Text(if (analysis.running) "检测中…" else "人物分类")
                }
                if (people.isNotEmpty()) {
                    TextButton(
                        onClick = { showClearConfirm = true },
                        modifier = Modifier.padding(start = 8.dp),
                    ) {
                        Text("清空")
                    }
                }
            }

            if (analysis.running && analysis.total > 0) {
                LinearProgressIndicator(
                    progress = { analysis.done.toFloat() / analysis.total },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 12.dp),
                )
                Text(
                    text = "${analysis.done} / ${analysis.total}   已检出 ${analysis.facesFound} 张脸、${analysis.peopleFound} 个人物",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(top = 6.dp),
                )
            }

            analysis.phase.takeIf { it.isNotBlank() }?.let {
                Text(
                    text = it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }

            analysis.message?.let {
                Text(
                    text = it,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(top = 6.dp),
                )
            }
        }

        // ---- 人物卡片网格 ----
        if (people.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(32.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = "还没有识别出人物。\n点上方「人物分类」，会检测照片里的人脸并按人聚合。",
                    textAlign = TextAlign.Center,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            LazyVerticalGrid(
                columns = GridCells.Fixed(2),
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                items(people, key = { it.personId }) { person ->
                    PersonCard(
                        person = person,
                        cover = covers[person.personId],
                        onClick = { viewModel.loadPerson(person.personId) },
                    )
                }
            }
        }
    }

    if (showClearConfirm) {
        AlertDialog(
            onDismissRequest = { showClearConfirm = false },
            title = { Text("清空人物数据？") },
            text = { Text("会删除已检出的人脸和人物分组。场景分类结果不受影响。") },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.clearFaces()
                    showClearConfirm = false
                }) { Text("清空") }
            },
            dismissButton = {
                TextButton(onClick = { showClearConfirm = false }) { Text("取消") }
            },
        )
    }
}

@Composable
private fun PersonCard(
    person: PersonSummary,
    cover: android.graphics.Bitmap?,
    onClick: () -> Unit,
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(12.dp),
    ) {
        Column {
            if (cover != null) {
                Image(
                    bitmap = cover.asImageBitmap(),
                    contentDescription = null,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(1f),
                )
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(1f)
                        .background(MaterialTheme.colorScheme.surfaceVariant),
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(
                        Icons.Filled.Person,
                        contentDescription = null,
                        modifier = Modifier.size(48.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            Text(
                text = person.displayName ?: "人物 ${person.personId}",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            )
            Text(
                text = "${person.photoCount} 张",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 12.dp).padding(bottom = 10.dp),
            )
        }
    }
}

@Composable
private fun PersonDetail(
    personId: Long,
    viewModel: PeopleViewModel,
    modifier: Modifier = Modifier,
) {
    val photos by viewModel.personPhotos.collectAsState()
    val cover = viewModel.covers.collectAsState().value[personId]

    Column(modifier = modifier.fillMaxSize()) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
        ) {
            IconButton(onClick = { viewModel.clearPerson() }) {
                Icon(Icons.Filled.ArrowBack, contentDescription = "返回")
            }
            cover?.let {
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = null,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier
                        .size(40.dp)
                        .clip(RoundedCornerShape(20.dp)),
                )
            }
            Text(
                text = "人物 $personId · ${photos.size} 张",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(start = 8.dp),
            )
        }
        PhotoGrid(photos = photos, columns = 4)
    }
}
