import { defineConfig, presetWind4, presetTypography } from 'unocss'

export default defineConfig({
  presets: [presetWind4(), presetTypography()],
  theme: {
    colors: {
      paper: '#F7F5EE',
      'paper-deep': '#EFEBDD',
      ink: '#1A2B1E',
      moss: '#2E5339',
      leaf: '#4A7C59',
      fern: '#8FAE94',
      vein: '#D8D2BF',
      specimen: '#B5541E',
    },
    font: {
      serif: '"Fraunces Variable", Fraunces, Georgia, serif',
      sans: '"Inter Variable", Inter, system-ui, sans-serif',
      mono: '"IBM Plex Mono", ui-monospace, monospace',
    },
  },
  shortcuts: {
    'hairline': 'border border-vein',
    'rule-double': 'border-0 border-b-4 border-b-double border-vein',
    'latin': 'font-serif italic text-moss',
    'plate': 'bg-paper hairline',
    'meta-label': 'font-mono text-[0.65rem] tracking-[0.18em] uppercase text-fern',
  },
})
