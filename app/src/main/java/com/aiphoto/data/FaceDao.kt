package com.aiphoto.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface FaceDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertFaces(faces: List<FaceEntity>)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPerson(person: PersonEntity): Long

    @Query("UPDATE people SET centroid = :centroid, faceCount = :faceCount, coverPhotoId = :coverPhotoId WHERE personId = :personId")
    suspend fun updatePerson(
        personId: Long,
        centroid: ByteArray,
        faceCount: Int,
        coverPhotoId: Long,
    )

    /** 人物卡片列表（含每人照片数），按照片数从多到少。 */
    @Query(
        """
        SELECT p.personId AS personId, p.coverPhotoId AS coverPhotoId,
               p.displayName AS displayName, COUNT(DISTINCT f.photoId) AS photoCount
        FROM people p LEFT JOIN face_tags f ON f.personId = p.personId
        GROUP BY p.personId
        ORDER BY photoCount DESC, p.personId ASC
        """
    )
    fun observePersonSummaries(): Flow<List<PersonSummary>>

    /** 聚类开始前把已有簇读进内存（增量聚类需要 centroid）。 */
    @Query("SELECT * FROM people ORDER BY personId ASC")
    suspend fun allPeople(): List<PersonEntity>

    @Query("SELECT * FROM people WHERE personId = :personId LIMIT 1")
    suspend fun findPerson(personId: Long): PersonEntity?

    /** 某个人物出现过的所有照片。 */
    @Query("SELECT DISTINCT photoId FROM face_tags WHERE personId = :personId ORDER BY photoId DESC")
    suspend fun photoIdsByPerson(personId: Long): List<Long>

    /** 某个人物置信度最高的那张脸（做封面裁切用）。 */
    @Query("SELECT * FROM face_tags WHERE personId = :personId ORDER BY score DESC LIMIT 1")
    suspend fun bestFaceOfPerson(personId: Long): FaceEntity?

    /** 已跑过人脸检测的图片 id（用于「已分析」去重）。 */
    @Query("SELECT DISTINCT photoId FROM face_tags")
    suspend fun analyzedPhotoIds(): List<Long>

    @Query("SELECT COUNT(*) FROM face_tags")
    suspend fun faceCount(): Int

    @Query("DELETE FROM face_tags")
    suspend fun clearFaces()

    @Query("DELETE FROM people")
    suspend fun clearPeople()
}
