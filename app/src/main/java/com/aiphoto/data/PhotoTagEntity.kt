package com.aiphoto.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/** 一张图片的场景分类结果。 */
@Entity(tableName = "photo_tags")
data class PhotoTagEntity(
    @PrimaryKey val photoId: Long,
    /** 得分最高的类别 id，如 "scenery"。 */
    val topCategory: String,
    /** 该类别与图片 embedding 的余弦相似度，范围 -1~1，实际多在 0.2~0.4。*/
    val topScore: Float,
    /** 全部类别的余弦相似度，格式 "scenery:0.31,person:0.18,..."。 */
    val scoreDetail: String,
    val analyzedAt: Long,
)
