package com.aiphoto.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.util.Size
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max

/**
 * 从 ContentResolver 加载用于 AI 推理的位图。
 *
 * 优先走系统缩略图（快、内存友好），失败再退回完整解码并降采样。
 */
object ImageLoader {

    private const val TAG = "ImageLoader"

    suspend fun loadForAnalysis(
        context: Context,
        uri: Uri,
        targetSize: Int = 320,
    ): Bitmap? = withContext(Dispatchers.IO) {
        loadThumbnail(context, uri, targetSize) ?: decodeScaled(context, uri, targetSize)
    }

    private fun loadThumbnail(context: Context, uri: Uri, targetSize: Int): Bitmap? = try {
        context.contentResolver.loadThumbnail(uri, Size(targetSize, targetSize), null)
    } catch (e: Exception) {
        Log.d(TAG, "缩略图不可用: $uri", e)
        null
    }

    private fun decodeScaled(context: Context, uri: Uri, targetSize: Int): Bitmap? = try {
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.isMutableRequired = false
            val width = info.size.width
            val height = info.size.height
            if (width > 0 && height > 0) {
                val scale = targetSize.toFloat() / max(width, height)
                if (scale < 1f) {
                    decoder.setTargetSize(
                        (width * scale).toInt().coerceAtLeast(1),
                        (height * scale).toInt().coerceAtLeast(1),
                    )
                }
            }
        }
    } catch (e: Exception) {
        Log.e(TAG, "解码失败: $uri", e)
        null
    }

    @Suppress("unused")
    private val sdkInt = Build.VERSION.SDK_INT
}
