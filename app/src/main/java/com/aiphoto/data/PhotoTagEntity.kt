package com.aiphoto.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/** 一张图片的场景分类结果。 */
@Entity(tableName = "photo_tags")
data class PhotoTagEntity(
    @PrimaryKey val photoId: Long,
    /** 置信度最高的类别 id，如 "scenery"。 */
    val topCategory: String,
    val topScore: Float,
    /** 完整概率分布，格式 "scenery:0.71,person:0.12,..."。 */
    val scoreDetail: String,
    val analyzedAt: Long,
)
