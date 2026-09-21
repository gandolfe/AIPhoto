package com.aiphoto.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

/** 各类别的图片数量。 */
data class CategoryCount(
    val category: String,
    val count: Int,
)

@Dao
interface PhotoTagDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: PhotoTagEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertAll(entities: List<PhotoTagEntity>)

    @Query("SELECT photoId FROM photo_tags")
    suspend fun analyzedPhotoIds(): List<Long>

    @Query("SELECT * FROM photo_tags WHERE photoId = :photoId LIMIT 1")
    suspend fun find(photoId: Long): PhotoTagEntity?

    @Query("SELECT photoId FROM photo_tags WHERE topCategory = :category ORDER BY topScore DESC")
    suspend fun photoIdsByCategory(category: String): List<Long>

    @Query("SELECT COUNT(*) FROM photo_tags")
    fun observeAnalyzedCount(): Flow<Int>

    @Query("SELECT topCategory AS category, COUNT(*) AS count FROM photo_tags GROUP BY topCategory")
    fun observeCategoryCounts(): Flow<List<CategoryCount>>

    @Query("SELECT COUNT(*) FROM photo_tags")
    suspend fun analyzedCount(): Int

    @Query("DELETE FROM photo_tags")
    suspend fun clear()
}
