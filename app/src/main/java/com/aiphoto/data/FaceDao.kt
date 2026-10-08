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

    /**
     * 人物卡片列表（含每人照片数），按照片数从多到少。
     *
     * [minPhotos]：只显示照片数达到这个数的人物。聚类出来的小簇（一两张的碎片）
     * 没有展示价值，列表里直接过滤掉，聚类收尾时再从库里删掉。
     */
    @Query(
        """
        SELECT p.personId AS personId, p.coverPhotoId AS coverPhotoId,
               p.displayName AS displayName, COUNT(DISTINCT f.photoId) AS photoCount
        FROM people p LEFT JOIN face_tags f ON f.personId = p.personId
        GROUP BY p.personId
        HAVING COUNT(DISTINCT f.photoId) >= :minPhotos
        ORDER BY photoCount DESC, p.personId ASC
        """
    )
    fun observePersonSummaries(minPhotos: Int): Flow<List<PersonSummary>>

    /** 照片数不足 [minPhotos] 的人物 —— 收尾时把这些碎片簇丢弃。 */
    @Query(
        """
        SELECT p.personId FROM people p
        WHERE (SELECT COUNT(DISTINCT f.photoId) FROM face_tags f WHERE f.personId = p.personId) < :minPhotos
        """
    )
    suspend fun personIdsWithFewerPhotosThan(minPhotos: Int): List<Long>

    /**
     * 把被丢弃人物名下的人脸改成「未归属」（personId = 0）。
     * 保留行本身是为了让这些照片仍算「已分析」，下次不用重复检测；embedding 也留着，
     * 以后想放宽门槛可以直接重聚类。
     */
    @Query("UPDATE face_tags SET personId = 0 WHERE personId IN (:personIds)")
    suspend fun detachFaces(personIds: List<Long>)

    @Query("DELETE FROM people WHERE personId IN (:personIds)")
    suspend fun deletePeople(personIds: List<Long>)

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
