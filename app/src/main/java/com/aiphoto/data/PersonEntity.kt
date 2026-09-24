package com.aiphoto.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * 一个「人物」簇（同一个人）。
 *
 * 簇是聚类出来的，不是预先定义的标签，所以簇的数量和 id 都运行时才确定 ——
 * 这跟场景分类那套硬编码 enum 完全不同，UI 不能写死列表。
 *
 * centroid 是簇内所有人脸 embedding 的均值（已 L2 归一化），用于增量聚类：
 * 新来一张人脸，跟已有 centroid 比余弦，超过阈值就归到该簇。
 */
@Entity(tableName = "people")
data class PersonEntity(
    @PrimaryKey(autoGenerate = true) val personId: Long = 0,
    /** 簇质心，float32 小端。 */
    val centroid: ByteArray,
    /** 归入该簇的人脸数量。 */
    val faceCount: Int = 0,
    /** 封面用的图片 id（取该簇置信度最高的那张脸所在图片）。 */
    val coverPhotoId: Long = 0,
    /** 预留：用户之后可以给人物起名字。 */
    val displayName: String? = null,
)

/** 人物卡片列表用：id + 封面 + 照片数。 */
data class PersonSummary(
    val personId: Long,
    val coverPhotoId: Long,
    val displayName: String?,
    /** 含该人物的**照片数**（同一张照片里多张同人人脸只算一张）。 */
    val photoCount: Int,
)
