package com.aiphoto.ai

/**
 * 智能分类的类别定义。
 *
 * 顺序必须与 [tools/colab_export_clip.py] 中 `LABEL_BANK` 的声明顺序一致：
 * 模型端不会输出概率向量，而是输出 512-d 视觉 embedding；App 侧用
 * embedding × label_bank.bin 做余弦相似度，取最高者。所以这里的 id 顺序要
 * 跟 label_bank.json / label_bank.bin 的行顺序对齐，否则 top-1 选错类别。
 *
 * 运行时不依赖这份硬编码的顺序作为分类依据——真实顺序由 label_bank.json 决定，
 * 这里主要给 UI（智能相册 / 类别卡片）展示用。读不到 label_bank.json 时才回退到
 * [DEFAULT_ORDER] 当兜底。
 */
enum class SceneCategory(
    val id: String,
    val displayName: String,
    val emoji: String,
) {
    // ---- 自然 / 风景 ----
    SCENERY("scenery", "风景", "🏞️"),
    MOUNTAIN("mountain", "山景", "⛰️"),
    BEACH("beach", "海边", "🏖️"),
    SUNSET("sunset", "日出日落", "🌅"),
    FOREST("forest", "森林", "🌲"),
    FLOWERS("flowers", "花卉", "🌷"),
    LAKE("lake", "湖景", "🏞️"),
    SNOW("snow", "雪景", "❄️"),
    SKY("sky", "天空", "☁️"),
    // ---- 城市 / 建筑 ----
    CITYSCAPE("cityscape", "城市", "🏙️"),
    BUILDING("building", "建筑", "🏛️"),
    INTERIOR("interior", "室内", "🛋️"),
    STREET("street", "街景", "🛣️"),
    NIGHT("night", "夜景", "🌃"),
    // ---- 人物 ----
    PERSON("person", "人物", "👤"),
    PORTRAIT("portrait", "肖像", "🧑"),
    GROUP("group", "合影", "👥"),
    CHILD("child", "儿童", "👶"),
    // ---- 动物 ----
    CAT("cat", "猫", "🐱"),
    DOG("dog", "狗", "🐶"),
    BIRD("bird", "鸟", "🐦"),
    PET("pet", "宠物", "🐾"),
    WILDLIFE("wildlife", "野生动物", "🦌"),
    // ---- 食物 ----
    FOOD("food", "美食", "🍽️"),
    DESSERT("dessert", "甜点", "🍰"),
    DRINK("drink", "饮品", "🥤"),
    FRUIT("fruit", "水果", "🍎"),
    // ---- 物品 / 文档 ----
    DOCUMENT("document", "文档", "📄"),
    SCREEN("screen", "屏幕截图", "🖥️"),
    ART("art", "艺术", "🎨"),
    TOY("toy", "玩具", "🧸"),
    BOOK("book", "书籍", "📚"),
    PLANT("plant", "植物", "🪴"),
    CAR("car", "汽车", "🚗"),
    // ---- 事件 ----
    WEDDING("wedding", "婚礼", "💒"),
    CELEBRATION("celebration", "庆祝", "🎉"),
    SPORTS("sports", "运动", "⚽"),
    TRAVEL("travel", "旅行", "🧳"),
    ;
    // 注：原本有个兜底类 OTHER("other", "其他")，已移除。
    // 它的 prompt 是 "a photo"，语义太泛，embedding 接近图像分布中心，
    // 参与竞选会吸走大量有明确内容的照片（实测人物合照/自拍几乎全被它抢走）。
    // 现在没有兜底类：每张照片都会归入 38 个具体类别中得分最高的一个。

    companion object {
        private val byId = entries.associateBy { it.id }

        fun fromId(id: String): SceneCategory? = byId[id]

        /** 默认类别 ID 顺序，与导出脚本 LABEL_ORDER 保持一致。 */
        val DEFAULT_ORDER: List<String> = entries.map { it.id }
    }
}