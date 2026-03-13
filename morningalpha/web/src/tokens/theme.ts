export const colors = {
  bg: '#0a0e1a',
  surface: '#0f1523',
  surface2: '#151d2e',
  border: '#1e2d45',
  textPrimary: '#e2e8f0',
  textSecondary: '#64748b',
  textCode: '#34d399',
  accentGreen: '#10b981',
  accentRed: '#ef4444',
  accentBlue: '#3b82f6',
  accentPurple: '#8b5cf6',
} as const

export const lightColors = {
  surface: '#ffffff',
  surface2: '#f8fafc',
  border: '#e2e8f0',
  textPrimary: '#0f172a',
  textSecondary: '#64748b',
} as const

export const fonts = {
  mono: "'JetBrains Mono', monospace",
  ui: "'Inter', system-ui, sans-serif",
} as const

// Shared Plotly layout base — apply via spread: { ...plotlyBase, title: '...' }
export const plotlyBase = {
  paper_bgcolor: colors.surface,
  plot_bgcolor: colors.surface2,
  font: { family: fonts.mono, color: colors.textPrimary, size: 12 },
  xaxis: {
    gridcolor: colors.border,
    linecolor: colors.border,
    tickfont: { color: colors.textSecondary },
    zeroline: false,
  },
  yaxis: {
    gridcolor: colors.border,
    linecolor: colors.border,
    tickfont: { color: colors.textSecondary },
    zeroline: false,
  },
  legend: { bgcolor: 'transparent', font: { color: colors.textSecondary } },
  margin: { t: 40, r: 20, b: 40, l: 60 },
} as const

export const plotlyBaseLight = {
  paper_bgcolor: lightColors.surface,
  plot_bgcolor: lightColors.surface2,
  font: { family: fonts.mono, color: lightColors.textPrimary, size: 12 },
  xaxis: {
    gridcolor: lightColors.border,
    linecolor: lightColors.border,
    tickfont: { color: lightColors.textSecondary },
    zeroline: false,
  },
  yaxis: {
    gridcolor: lightColors.border,
    linecolor: lightColors.border,
    tickfont: { color: lightColors.textSecondary },
    zeroline: false,
  },
  legend: { bgcolor: 'transparent', font: { color: lightColors.textSecondary } },
  margin: { t: 40, r: 20, b: 40, l: 60 },
} as const
