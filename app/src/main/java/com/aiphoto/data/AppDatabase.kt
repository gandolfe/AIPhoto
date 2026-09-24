package com.aiphoto.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [PhotoTagEntity::class, FaceEntity::class, PersonEntity::class],
    version = 2,
    exportSchema = true,
)
abstract class AppDatabase : RoomDatabase() {

    abstract fun photoTagDao(): PhotoTagDao

    abstract fun faceDao(): FaceDao

    companion object {
        private const val DB_NAME = "aiphoto.db"

        /**
         * v1 → v2：新增人脸表与人物表。
         *
         * ⚠️ 必须写这条迁移：builder 上有 fallbackToDestructiveMigration()，
         * 缺迁移时 Room 会直接**删库重建**，用户已经跑完的几千条场景分析结果会全没了。
         * 这里的 SQL 必须与 Room 生成的 schema 完全一致（列名/类型/NOT NULL/索引名），
         * 否则运行时校验不通过。
         */
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `people` (
                        `personId` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        `centroid` BLOB NOT NULL,
                        `faceCount` INTEGER NOT NULL,
                        `coverPhotoId` INTEGER NOT NULL,
                        `displayName` TEXT
                    )
                    """.trimIndent()
                )
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `face_tags` (
                        `id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        `photoId` INTEGER NOT NULL,
                        `personId` INTEGER NOT NULL,
                        `leftPct` REAL NOT NULL,
                        `topPct` REAL NOT NULL,
                        `rightPct` REAL NOT NULL,
                        `bottomPct` REAL NOT NULL,
                        `score` REAL NOT NULL,
                        `embedding` BLOB NOT NULL
                    )
                    """.trimIndent()
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_face_tags_photoId` ON `face_tags` (`photoId`)"
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_face_tags_personId` ON `face_tags` (`personId`)"
                )
            }
        }

        @Volatile
        private var instance: AppDatabase? = null

        fun get(context: Context): AppDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    DB_NAME,
                )
                    .addMigrations(MIGRATION_1_2)
                    .fallbackToDestructiveMigration()
                    .build()
                    .also { instance = it }
            }
    }
}
