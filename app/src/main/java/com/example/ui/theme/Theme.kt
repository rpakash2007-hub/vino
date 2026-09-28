package com.example.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val NephroScanColorScheme = darkColorScheme(
    primary = CyanPrimary,
    onPrimary = NavyDark,
    primaryContainer = NavySurfaceVariant,
    onPrimaryContainer = CyanPrimary,
    secondary = CyanSecondary,
    onSecondary = NavyDark,
    background = NavyDark,
    onBackground = TextWhite,
    surface = NavySurface,
    onSurface = TextWhite,
    surfaceVariant = NavySurfaceVariant,
    onSurfaceVariant = TextMuted,
    error = MedicalRed,
    onError = Color.White
)

@Composable
fun NephroScanTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = NephroScanColorScheme,
        typography = Typography,
        content = content
    )
}
