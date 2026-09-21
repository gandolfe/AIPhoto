package com.aiphoto

import android.content.Context
import com.aiphoto.ai.SceneAnalyzer
import com.aiphoto.data.AppDatabase
import com.aiphoto.data.MediaStoreRepository
import com.aiphoto.data.PhotoStore
import com.aiphoto.data.PhotoTagDao

/**
 * 轻量手写依赖容器（本项目规模下不必引入 Hilt）。
 */
class AppContainer(context: Context) {

    private val appContext = context.applicationContext

    private val mediaStoreRepository = MediaStoreRepository(appContext)

    val photoStore: PhotoStore = PhotoStore(mediaStoreRepository)

    val photoTagDao: PhotoTagDao = AppDatabase.get(appContext).photoTagDao()

    val sceneAnalyzer: SceneAnalyzer = SceneAnalyzer(appContext)
}
