package com.aiphoto.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val GreenPrimary = Color(0xFF14654A)
private val GreenSecondary = Color(0xFF4C8C6F)
private val Amber = Color(0xFFF2A93B)

private val LightColors = lightColorScheme(
    primary = GreenPrimary,
    secondary = GreenSecondary,
    tertiary = Amber,
    background = Color(0xFFF7F8F7),
    surface = Color(0xFFFFFFFF),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF7BD0AA),
    secondary = Color(0xFF9CD4BB),
    tertiary = Amber,
    background = Color(0xFF101413),
    surface = Color(0xFF191E1C),
)

@Composable
fun AiPhotoTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        content = content,
    )
}
