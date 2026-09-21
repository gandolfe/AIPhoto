package com.aiphoto.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DateRange
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.aiphoto.ui.album.SmartAlbumScreen
import com.aiphoto.ui.timeline.TimelineScreen

@Composable
fun AppRoot() {
    val context = LocalContext.current
    var granted by remember { mutableStateOf(hasMediaPermission(context)) }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
    ) { granted = hasMediaPermission(context) }

    LaunchedEffect(Unit) {
        if (!granted) {
            launcher.launch(Manifest.permission.READ_MEDIA_IMAGES)
        }
    }

    if (!granted) {
        PermissionPrompt(
            onRequest = { launcher.launch(Manifest.permission.READ_MEDIA_IMAGES) },
        )
        return
    }

    MainScaffold()
}

@Composable
private fun MainScaffold() {
    var tab by rememberSaveable { mutableIntStateOf(0) }

    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = tab == 0,
                    onClick = { tab = 0 },
                    icon = { Icon(Icons.Filled.DateRange, contentDescription = null) },
                    label = { Text("时间线") },
                )
                NavigationBarItem(
                    selected = tab == 1,
                    onClick = { tab = 1 },
                    icon = { Icon(Icons.Filled.Star, contentDescription = null) },
                    label = { Text("智能相册") },
                )
            }
        },
    ) { innerPadding ->
        when (tab) {
            0 -> TimelineScreen(modifier = Modifier.padding(innerPadding))
            else -> SmartAlbumScreen(modifier = Modifier.padding(innerPadding))
        }
    }
}

@Composable
private fun PermissionPrompt(onRequest: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "AI 相册需要读取本机图片才能按时间和内容分类。\n所有分析都在本机完成，不会上传任何数据。",
            textAlign = TextAlign.Center,
        )
        Button(
            onClick = onRequest,
            modifier = Modifier.padding(top = 24.dp),
        ) {
            Text("授予访问权限")
        }
    }
}

private fun hasMediaPermission(context: Context): Boolean =
    ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_IMAGES) ==
        PackageManager.PERMISSION_GRANTED
