package com.aiphoto

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.aiphoto.ui.AppRoot
import com.aiphoto.ui.theme.AiPhotoTheme

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent {
            AiPhotoTheme {
                AppRoot()
            }
        }
    }
}
