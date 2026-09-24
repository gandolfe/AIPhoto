# AIPhoto

Android 端侧 AI 相册：按**场景**分类照片、按**人物**聚合人脸。
所有推理都在手机本地完成，不联网、不上传。

## 技术栈

| 领域    | 选型 |
|-------|---|
| UI    | Jetpack Compose + Material3，单 Activity + 底部导航 |
| 异步    | Kotlin Coroutines / Flow |
| 数据    | Room（SQLite）+ MediaStore |
| 依赖注入  | 手写 `AppContainer`（项目规模下不引入 Hilt） |
| 图片加载  | Coil |
| 端侧推理  | **ExecuTorch**（`executorch-android`，XNNPACK 后端，加载 `.pte`） |
| 人脸框检测 | **MediaPipe Tasks Vision**（`tasks-vision`，加载 `.tflite`） |
| 模型导出  | PyTorch `torch.export` → ExecuTorch（脚本在 `tools/`，Colab 跑） |

## 整体架构

```
ui/         界面层：Compose 画面 + ViewModel（持有进度/结果状态）
   ↓
ai/         推理层：模型加载、预处理、forward、聚类
   ↓
data/       数据层：MediaStore 读图、Room 落库、内存图片索引
```

三个 tab 对应三块功能：**时间线**（照片流 + 场景分析入口）、**智能相册**（分类浏览）、**人物**（人脸聚合）。

### 分层职责

| 包 | 职责 |
|---|---|
| `ui/` | 页面与 ViewModel；批量分析的进度推进、已分析去重 |
| `ai/` | 模型封装与推理编排；**不含 UI、不直接写库** |
| `data/` | `MediaImage` / `PhotoStore`（内存索引、分页）/ Room 实体与 DAO |
| `AppContainer` | 全局单例，持有 Store、DAO、Analyzer |

### 两个贯穿全局的设计

1. **推理不并发**：`ExecuTorch Module` 非线程安全，每个链路各用一个**单线程 dispatcher** 串行执行（场景用 `clip-inference`，人脸用 `face-inference`，互不抢资源）。
2. **`.pte` 只出 embedding，决策在 Kotlin 侧**：模型负责把图编成向量，分类/聚类/排序都在 App 里算。所以**换标签或调聚类阈值不用重新导出模型**。

## 两条推理链路

### 1. 场景分类

```
图片 → 缩放 256×256 → fp16 张量 → MobileCLIP 视觉塔(.pte)
     → 512 维 embedding → 与 label_bank 点积(即余弦) → softmax → 34 类
     → 写入 Room (photo_tags)
```

### 2. 人物聚合

```
图片 → MediaPipe 检出人脸框 → 按框裁出人脸(扩边 25%)
     → MobileFaceNet(.pte) → 512 维 embedding
     → 与已有簇质心比余弦，超阈值归簇、否则新建簇（增量聚类 O(n·k)）
     → 写入 Room (face_tags / people)
```

聚类用增量方式而非「最后两两比较」——几千张脸两两比是 O(n²)，端侧不可接受。

## 模型文件

放 `app/src/main/assets/models/`（缺文件时对应功能会提示「未找到模型」而非崩溃）：

| 文件 | 用途 | 来源 |
|---|---|---|
| `clip_scene.pte` | 场景编码 | `tools/colab_export_clip.py` |
| `labels.json` / `label_bank.json` / `label_bank.bin` | 类别元数据与标签向量 | 同上 |
| `face_detector.tflite` | MediaPipe 人脸检测 | BlazeFace short range |
| `mobilefacenet.pte` + `mobilefacenet.json` | 人脸特征提取与元数据 | `tools/colab_export_mobilefacenet.py` |

`build.gradle.kts` 里对 `pte` / `tflite` / `task` 配了 `noCompress`，保证模型未被压缩（MediaPipe 要求；`.pte` 也依赖它才能 `openFd` + mmap）。

## 构建运行

AGP 8.7.3 需 JDK ≥ 11：

```cmd
set JAVA_HOME=C:\Program Files\Android\Android Studio1\jbr
gradlew.bat assembleDebug
adb install -r -g app\build\outputs\apk\debug\app-debug.apk
```

## 几个要点

- **归一化参数不硬编码**：`image_size` / `mean` / `std` / `embed_dim` 都从模型的 `.json` 元数据读（`labels.json`、`mobilefacenet.json`）。换模型只换文件即可。
- **数据库有 Migration**：`AppDatabase` 是 v2 且提供了 `Migration(1,2)`。升级版本时必须补迁移，否则 Room 的 `fallbackToDestructiveMigration()` 会**删库重建**、已分析结果全丢。
- **已分析过的照片会跳过**：换模型后旧结果不兼容，需先用界面上的「清空」清掉再跑。
- **息屏会被冻结**：vivo 等机型息屏后批量分析会中断，长时间跑批先 `adb shell svc power stayon usb`。
- **人物聚类阈值**：`FaceClusterer.DEFAULT_THRESHOLD`（默认 0.5，范围 0.4~0.7）。同人被拆成多个就调低，不同人被并成一个就调高。

## 排查手段

- 计时日志：`ClipSceneClassifier` / `SceneAnalyzer` / `TimelineVM` / `FaceAnalyzer` / `MobileFaceNet` 各 tag 分段打印耗时
- 直接查库：`adb shell run-as com.aiphoto sqlite3 databases/aiphoto.db "select count(*) from face_tags;"`
- 已落盘模型：`adb shell run-as com.aiphoto ls -l files/models/`
