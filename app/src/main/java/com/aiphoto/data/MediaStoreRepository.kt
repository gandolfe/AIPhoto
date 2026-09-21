package com.aiphoto.data

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * 通过 MediaStore 读取系统相册图片。
 */
class MediaStoreRepository(private val context: Context) {

    private val collection: Uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI

    private val projection = arrayOf(
        MediaStore.Images.Media._ID,
        MediaStore.Images.Media.DISPLAY_NAME,
        MediaStore.Images.Media.DATE_TAKEN,
        MediaStore.Images.Media.DATE_ADDED,
        MediaStore.Images.Media.WIDTH,
        MediaStore.Images.Media.HEIGHT,
        MediaStore.Images.Media.SIZE,
    )

    /**
     * 按拍摄时间倒序分页读取图片。
     *
     * @param limit 单页数量
     * @param offset 起始偏移
     */
    suspend fun loadImages(limit: Int, offset: Int = 0): List<MediaImage> =
        withContext(Dispatchers.IO) {
            val queryArgs = Bundle().apply {
                putString(
                    ContentResolver.QUERY_ARG_SQL_SORT_ORDER,
                    "${MediaStore.Images.Media.DATE_TAKEN} DESC, ${MediaStore.Images.Media._ID} DESC",
                )
                putInt(ContentResolver.QUERY_ARG_LIMIT, limit)
                putInt(ContentResolver.QUERY_ARG_OFFSET, offset)
            }

            val result = ArrayList<MediaImage>(limit)
            context.contentResolver.query(collection, projection, queryArgs, null)?.use { cursor ->
                val idCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
                val nameCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME)
                val takenCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_TAKEN)
                val addedCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED)
                val widthCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.WIDTH)
                val heightCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.HEIGHT)
                val sizeCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.SIZE)

                while (cursor.moveToNext()) {
                    val id = cursor.getLong(idCol)
                    val taken = cursor.getLong(takenCol).takeIf { it > 0L }
                        ?: (cursor.getLong(addedCol) * 1000L)

                    result += MediaImage(
                        id = id,
                        uri = Uri.withAppendedPath(collection, id.toString()),
                        displayName = cursor.getString(nameCol) ?: "IMG_$id",
                        takenAt = taken,
                        width = cursor.getInt(widthCol),
                        height = cursor.getInt(heightCol),
                        size = cursor.getLong(sizeCol),
                    )
                }
            }
            result
        }

    /** 图片总数，用于展示进度。 */
    suspend fun countImages(): Int = withContext(Dispatchers.IO) {
        context.contentResolver.query(collection, arrayOf(MediaStore.Images.Media._ID), null, null, null)
            ?.use { it.count } ?: 0
    }
}
