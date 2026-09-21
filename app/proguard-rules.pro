# ExecuTorch 通过 JNI 反射调用，需保留
-keep class org.pytorch.executorch.** { *; }
-keepclassmembers class org.pytorch.executorch.** { *; }

# Room 生成的实现
-keep class * extends androidx.room.RoomDatabase { *; }
