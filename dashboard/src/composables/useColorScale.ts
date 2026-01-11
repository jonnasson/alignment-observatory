/**
 * Color Scale Composable
 *
 * Provides color mapping functions for heatmap visualizations.
 */

import { computed, type Ref } from 'vue'

export type ColorScaleName = 'viridis' | 'inferno' | 'coolwarm' | 'blues' | 'rdbu'

// Color scale definitions (RGB values)
const COLOR_SCALES: Record<ColorScaleName, number[][]> = {
  viridis: [
    [68, 1, 84],
    [72, 40, 120],
    [62, 73, 137],
    [49, 104, 142],
    [38, 130, 142],
    [31, 158, 137],
    [53, 183, 121],
    [110, 206, 88],
    [181, 222, 43],
    [253, 231, 37],
  ],
  inferno: [
    [0, 0, 4],
    [27, 12, 65],
    [74, 12, 107],
    [120, 28, 109],
    [165, 44, 96],
    [207, 68, 70],
    [237, 105, 37],
    [251, 155, 6],
    [247, 209, 61],
    [252, 255, 164],
  ],
  coolwarm: [
    [59, 76, 192],
    [102, 136, 238],
    [136, 187, 255],
    [176, 208, 255],
    [216, 218, 235],
    [247, 217, 196],
    [244, 165, 130],
    [214, 96, 77],
    [178, 24, 43],
  ],
  blues: [
    [247, 251, 255],
    [222, 235, 247],
    [198, 219, 239],
    [158, 202, 225],
    [107, 174, 214],
    [66, 146, 198],
    [33, 113, 181],
    [8, 81, 156],
    [8, 48, 107],
  ],
  rdbu: [
    [103, 0, 31],
    [178, 24, 43],
    [214, 96, 77],
    [244, 165, 130],
    [253, 219, 199],
    [247, 247, 247],
    [209, 229, 240],
    [146, 197, 222],
    [67, 147, 195],
    [33, 102, 172],
    [5, 48, 97],
  ],
}

/**
 * Interpolate between two colors
 */
function interpolateColor(color1: number[], color2: number[], t: number): number[] {
  const r1 = color1[0] ?? 0
  const g1 = color1[1] ?? 0
  const b1 = color1[2] ?? 0
  const r2 = color2[0] ?? 0
  const g2 = color2[1] ?? 0
  const b2 = color2[2] ?? 0
  return [
    Math.round(r1 + (r2 - r1) * t),
    Math.round(g1 + (g2 - g1) * t),
    Math.round(b1 + (b2 - b1) * t),
  ]
}

/**
 * Get color from scale at normalized position [0, 1]
 */
function getColorAtPosition(scale: number[][], position: number): number[] {
  const clampedPos = Math.max(0, Math.min(1, position))
  const scaledPos = clampedPos * (scale.length - 1)
  const idx = Math.floor(scaledPos)
  const t = scaledPos - idx

  if (idx >= scale.length - 1) {
    return scale[scale.length - 1] ?? [0, 0, 0]
  }

  const color1 = scale[idx] ?? [0, 0, 0]
  const color2 = scale[idx + 1] ?? [0, 0, 0]
  return interpolateColor(color1, color2, t)
}

/**
 * Convert RGB to hex string
 */
function rgbToHex(rgb: number[]): string {
  return `#${rgb.map((c) => c.toString(16).padStart(2, '0')).join('')}`
}

/**
 * Convert RGB to CSS rgb() string
 */
function rgbToCss(rgb: number[]): string {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
}

export interface ColorScaleOptions {
  scaleName?: ColorScaleName
  min?: number
  max?: number
  reverse?: boolean
}

export function useColorScale(options: Ref<ColorScaleOptions> | ColorScaleOptions = {}) {
  const opts = computed(() => {
    const o = 'value' in options ? options.value : options
    return {
      scaleName: o.scaleName ?? 'viridis',
      min: o.min ?? 0,
      max: o.max ?? 1,
      reverse: o.reverse ?? false,
    }
  })

  const scale = computed(() => {
    const colors = COLOR_SCALES[opts.value.scaleName]
    return opts.value.reverse ? [...colors].reverse() : colors
  })

  /**
   * Get color for a value (normalized between min and max)
   */
  function getColor(value: number, format: 'hex' | 'rgb' | 'array' = 'hex'): string | number[] {
    const { min, max } = opts.value
    const normalized = max === min ? 0.5 : (value - min) / (max - min)
    const rgb = getColorAtPosition(scale.value, normalized)

    switch (format) {
      case 'hex':
        return rgbToHex(rgb)
      case 'rgb':
        return rgbToCss(rgb)
      case 'array':
        return rgb
      default:
        return rgbToHex(rgb)
    }
  }

  /**
   * Get CSS gradient for the color scale
   */
  function getGradient(direction: 'horizontal' | 'vertical' = 'horizontal'): string {
    const stops = scale.value.map((color, i) => {
      const percent = (i / (scale.value.length - 1)) * 100
      return `${rgbToCss(color)} ${percent}%`
    })

    const angle = direction === 'horizontal' ? '90deg' : '180deg'
    return `linear-gradient(${angle}, ${stops.join(', ')})`
  }

  /**
   * Generate color stops for SVG gradient
   */
  function getGradientStops(): Array<{ offset: string; color: string }> {
    return scale.value.map((color, i) => ({
      offset: `${(i / (scale.value.length - 1)) * 100}%`,
      color: rgbToHex(color),
    }))
  }

  /**
   * Map an array of values to colors
   */
  function mapColors(values: number[], format: 'hex' | 'rgb' = 'hex'): string[] {
    return values.map((v) => getColor(v, format) as string)
  }

  /**
   * Get color for a 2D position (useful for attention matrices)
   */
  function getColorAt2D(data: number[][], row: number, col: number, format: 'hex' | 'rgb' = 'hex'): string {
    const rowData = data[row]
    if (row < 0 || row >= data.length || !rowData || col < 0 || col >= rowData.length) {
      return format === 'hex' ? '#000000' : 'rgb(0, 0, 0)'
    }
    const value = rowData[col]
    return getColor(value ?? 0, format) as string
  }

  return {
    getColor,
    getGradient,
    getGradientStops,
    mapColors,
    getColorAt2D,
    scale,
    scaleName: computed(() => opts.value.scaleName),
    min: computed(() => opts.value.min),
    max: computed(() => opts.value.max),
  }
}
