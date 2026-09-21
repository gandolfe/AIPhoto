package com.aiphoto.data

import android.net.Uri

/** 从系统媒体库读取到的一张图片。 */
data class MediaImage(
    val id: Long,
    val uri: Uri,
    val displayName: String,
    /** 拍摄时间（毫秒）。缺失时回退为入库时间。 */
    val takenAt: Long,
    val width: Int,
    val height: Int,
    val size: Long,
)
