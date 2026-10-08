package com.aiphoto.ai

import com.aiphoto.data.PersonEntity
import com.aiphoto.data.toEmbeddingFloats
import kotlin.math.sqrt

/**
 * 人脸 embedding 的**增量聚类**（在线聚类）。
 *
 * 为什么不用「最后一次性聚类」：2750 张图可能产出几千张脸，两两比较是 O(n²)
 * （几千² × 512 维 ≈ 几十亿次运算），端侧会卡到不可接受。增量聚类只跟已有簇的
 * centroid 比，是 O(n·k)（k = 人数，通常几十），快几个数量级。
 *
 * 代价是结果依赖处理顺序、且 centroid 会被逐步拉偏，但对相册去重聚合这种场景够用。
 *
 * ⚠️ [threshold] 是这个功能最主要的调节点，见 [DEFAULT_THRESHOLD] 的注释。
 */
class FaceClusterer(private val threshold: Float = DEFAULT_THRESHOLD) {

    /** 一次聚类的结果。 */
    data class Assignment(
        /** 归属的人物 id；[isNew] 为 true 时这个值无意义（还没落库）。 */
        val personId: Long,
        val isNew: Boolean,
        /** 更新后的簇质心（已归一化），调用方写回 people.centroid。 */
        val centroid: FloatArray,
        /** 更新后的簇内人脸数。 */
        val faceCount: Int,
        /** 与最相似簇的相似度，便于调阈值时观察。 */
        val bestScore: Float,
    )

    private class Cluster(
        val personId: Long,
        var centroid: FloatArray,
        var count: Int,
    )

    private val clusters = ArrayList<Cluster>()

    /** 开始前把库里已有的簇读进内存。 */
    fun load(people: List<PersonEntity>) {
        clusters.clear()
        for (p in people) {
            clusters += Cluster(
                personId = p.personId,
                centroid = p.centroid.toEmbeddingFloats(),
                count = p.faceCount.coerceAtLeast(1),
            )
        }
    }

    /**
     * 给一张新人脸分配人物。
     * 命中已有簇时**就地更新**该簇的 centroid（调用方随后写回数据库）。
     */
    fun assign(embedding: FloatArray): Assignment {
        var best: Cluster? = null
        var bestScore = -1f
        for (c in clusters) {
            // embedding 都已 L2 归一化 → 点积即余弦
            val s = dot(embedding, c.centroid)
            if (s > bestScore) {
                bestScore = s
                best = c
            }
        }

        val hit = best
        if (hit != null && bestScore >= threshold) {
            val n = hit.count + 1
            // 只要命中簇，就会用当前照片更新中心向量
            val updated = FloatArray(embedding.size) { i ->
                (hit.centroid[i] * hit.count + embedding[i]) / n
            }
            normalize(updated)
            hit.centroid = updated
            hit.count = n
            return Assignment(hit.personId, false, updated.copyOf(), n, bestScore)
        }

        return Assignment(NEW_PERSON_ID, true, embedding.copyOf(), 1, bestScore)
    }

    /** 调用方把新簇插库拿到 personId 后，登记进内存。 */
    fun register(personId: Long, embedding: FloatArray) {
        clusters += Cluster(personId, embedding.copyOf(), 1)
    }

    /** 数据库里删掉某些人物后同步内存（否则计数和后续分配会用上已删除的簇）。 */
    fun remove(personIds: Collection<Long>) {
        if (personIds.isEmpty()) return
        val drop = personIds.toHashSet()
        clusters.removeAll { it.personId in drop }
    }

    fun clusterCount(): Int = clusters.size

    /** 人脸数达到 [minCount] 的簇数量：进度里展示这个，免得把待丢弃的碎片也算进「人物数」。 */
    fun clusterCountAtLeast(minCount: Int): Int = clusters.count { it.count >= minCount }

    companion object {
        /**
         * 余弦相似度阈值：高于它认为是同一个人。
         *
         * 这是整个人物功能**最该调的参数**，不同训练方式的模型分布不一样：
         *   - 调低 → 更容易合并，风险是不同人被并成一个人
         *   - 调高 → 更容易拆分，风险是同一个人被拆成好几个「人物」
         * 经验范围 0.4 ~ 0.7。先用 0.5 跑一遍看结果，再按上面两条微调。
         * 余弦相似度0.5f的阈值不能动，低于0.5就会出现同一个人物相册混了多个人
         */
        const val DEFAULT_THRESHOLD = 0.5f

        const val NEW_PERSON_ID = -1L

        /**
         * 一个「人物」至少要出现在这几张**照片**里才会被保留。
         *
         * 聚类总会掉出一堆只有 1~2 张的碎片簇（侧脸、糊图、路人），没有做成相册的价值，
         * 数量还特别多，所以收尾统一丢弃，只留够份的。想更严格就调大。
         */
        const val MIN_PHOTOS_PER_PERSON = 8

        /**
         * 这里是做点积运算
         * 把两个向量做点积，返回的分数越接近1.0,就说明这两张图片越是一个人
         */
        private fun dot(a: FloatArray, b: FloatArray): Float {
            var s = 0f
            val n = minOf(a.size, b.size)
            for (i in 0 until n) s += a[i] * b[i]
            return s
        }

        // 向量做L2归一化
        private fun normalize(v: FloatArray) {
            var sum = 0f
            for (x in v) sum += x * x
            val n = sqrt(sum)
            if (n > 0f) {
                for (i in v.indices) v[i] /= n
            }
        }
    }
}
