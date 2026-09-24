package com.aiphoto.data

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 一张图片里的**一张**人脸。
 *
 * 与 [PhotoTagEntity] 的区别：一张图只有一个场景标签，但可能有多张人脸，
 * 所以这里主键是自增 id，而不是 photoId。
 *
 * 人脸框存的是**相对比例（0..1）**而不是像素坐标：分析时用的是降采样后的位图，
 * 存比例才能在任意分辨率（缩略图 / 原图）下正确裁切出人脸封面。
 */
@Entity(
    tableName = "face_tags",
    indices = [
        Index(value = ["photoId"]),
        Index(value = ["personId"]),
    ],
)
data class FaceEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    /** MediaStore 图片 id。 */
    val photoId: Long,
    /** 归属的人物簇（对应 people.personId）。 */
    val personId: Long,
    /** 人脸框左上/右下，相对原图的比例。 */
    val leftPct: Float,
    val topPct: Float,
    val rightPct: Float,
    val bottomPct: Float,
    /** MediaPipe 检测置信度，用于挑封面。 */
    val score: Float,
    /** L2 归一化后的 embedding，float32 小端（见 [toEmbeddingBytes]）。 */
    val embedding: ByteArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is FaceEntity) return false
        return id == other.id && photoId == other.photoId && personId == other.personId
    }

    override fun hashCode(): Int = (id * 31 + photoId * 31 + personId).toInt()
}

/** FloatArray → float32 小端字节（存 Room BLOB）。 */
fun FloatArray.toEmbeddingBytes(): ByteArray {
    val buf = ByteBuffer.allocate(size * 4).order(ByteOrder.LITTLE_ENDIAN)
    for (v in this) buf.putFloat(v)
    return buf.array()
}

/** Room BLOB → FloatArray。 */
fun ByteArray.toEmbeddingFloats(): FloatArray {
    val buf = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
    val out = FloatArray(size / 4)
    for (i in out.indices) out[i] = buf.float
    return out
}
