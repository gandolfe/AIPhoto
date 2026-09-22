package com.aiphoto.ui.timeline

import android.os.SystemClock
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.aiphoto.ui.components.PhotoGrid
import kotlinx.coroutines.delay

@Composable
fun TimelineScreen(
    modifier: Modifier = Modifier,
    viewModel: TimelineViewModel = viewModel(),
) {
    val photos by viewModel.photos.collectAsStateWithLifecycle()
    val loading by viewModel.loading.collectAsStateWithLifecycle()
    val loadingMore by viewModel.loadingMore.collectAsStateWithLifecycle()
    val fullyLoaded by viewModel.fullyLoaded.collectAsStateWithLifecycle()
    val analysis by viewModel.analysis.collectAsStateWithLifecycle()
    val analyzedIds by viewModel.analyzedIds.collectAsStateWithLifecycle()

    // 网格状态由父级持有，滚动到底部时触发翻页（向上回滚已加载的页是瞬时的）
    val gridState = rememberLazyGridState()

    Column(modifier = modifier.fillMaxSize()) {
        AnalysisHeader(
            analysis = analysis,
            analyzedCount = analyzedIds.size,
            onStart = viewModel::startAnalysis,
            onClear = viewModel::clearAnalysis,
        )

        Box(Modifier.fillMaxSize()) {
            when {
                photos.isEmpty() && loading -> {
                    CircularProgressIndicator(Modifier.align(Alignment.Center))
                }

                photos.isEmpty() -> {
                    Text(
                        text = "未找到图片，请确认已授予相册权限",
                        modifier = Modifier.align(Alignment.Center),
                    )
                }

                else -> {
                    PhotoGrid(
                        photos = photos,
                        columns = 4,
                        state = gridState,
                        onNearEnd = viewModel::loadMore,
                    )
                    // 仅在实际拉取下一页时显示转圈；本地相册不会持续更新内容，
                    // 不显示“加载更多”之类的提示文字，避免误导成信息流式的新内容
                    if (loadingMore) {
                        CircularProgressIndicator(
                            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 16.dp),
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun AnalysisHeader(
    analysis: AnalysisState,
    analyzedCount: Int,
    onStart: () -> Unit,
    onClear: () -> Unit,
) {
    // 清空会一次性删掉全部历史结果且不可恢复，走二次确认
    var showClearConfirm by remember { mutableStateOf(false) }

    // 秒级计时器：单张推理可能几分钟不动进度条，这里让它至少每秒变化一次，
    // 让用户能判断是在算而不是卡死。
    val elapsedState = remember { mutableLongStateOf(0L) }
    LaunchedEffect(key1 = analysis.running) {
        if (!analysis.running) {
            elapsedState.value = 0L
            return@LaunchedEffect
        }
        val startedAt = SystemClock.elapsedRealtime()
        while (true) {
            elapsedState.value = (SystemClock.elapsedRealtime() - startedAt) / 1000L
            delay(1000L)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = "时间线",
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.weight(1f),
            )
            if (analyzedCount > 0) {
                TextButton(
                    onClick = { showClearConfirm = true },
                    enabled = !analysis.running,
                ) {
                    Text("清空")
                }
            }
            Button(onClick = onStart, enabled = !analysis.running) {
                Text(if (analysis.running) "分析中…" else "AI 分析")
            }
        }

        if (showClearConfirm) {
            AlertDialog(
                onDismissRequest = { showClearConfirm = false },
                title = { Text("清空分析结果？") },
                text = {
                    Text(
                        "将删除 $analyzedCount 条已有分析结果（不影响手机里的原始照片）。\n\n" +
                            "之后「AI 分析」会从头分析全部图片，按当前速度可能需要很久。此操作不可撤销。",
                    )
                },
                confirmButton = {
                    TextButton(
                        onClick = {
                            showClearConfirm = false
                            onClear()
                        },
                    ) {
                        Text("清空")
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showClearConfirm = false }) {
                        Text("取消")
                    }
                },
            )
        }

        if (analysis.running) {
            val fraction = if (analysis.total > 0) {
                analysis.done.toFloat() / analysis.total
            } else {
                0f
            }
            LinearProgressIndicator(
                progress = { fraction },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 10.dp),
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    text = "${analysis.done} / ${analysis.total}",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    text = "已用 ${formatDuration(elapsedState.value)}",
                    style = MaterialTheme.typography.bodySmall,
                )
                if (analysis.failed > 0) {
                    Text(
                        text = "失败 ${analysis.failed}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
                val remainSec = estimateRemainingSeconds(
                    elapsedSec = elapsedState.value,
                    done = analysis.done,
                    total = analysis.total,
                )
                if (remainSec != null) {
                    Text(
                        text = "预计还需 ${formatDuration(remainSec)}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
            if (analysis.phase.isNotBlank()) {
                Text(
                    text = analysis.phase,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(top = 2.dp),
                )
            }
        }

        analysis.message?.let { message ->
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(top = 6.dp),
            )
        }
    }
}

/** 把秒数写成「3分25秒」这类人读的形式。 */
private fun formatDuration(seconds: Long): String {
    val minutes = seconds / 60
    val rest = seconds % 60
    return if (minutes > 0) "${minutes}分${rest}秒" else "${rest}秒"
}

/**
 * 用已完成张数的平均耗时估算剩余时间。样本不足（还没跑完一张）时返回 null，
 * 避免给出一个没意义的数字。
 */
private fun estimateRemainingSeconds(elapsedSec: Long, done: Int, total: Int): Long? {
    if (done <= 0 || total <= done) return null
    val remaining = total - done
    return (elapsedSec.toDouble() / done * remaining).toLong()
}
