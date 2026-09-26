<script setup lang="ts">
/**
 * Carl Linnaeus (Roslin, 1775 — public domain) rendered as a halftone
 * magnetic dot matrix. Particles sit on a grid whose radius is driven by
 * image luminance; the pointer repels them and a spring brings them home.
 * Honors prefers-reduced-motion by painting the static halftone once.
 */
const canvas = ref<HTMLCanvasElement | null>(null)
const wrap = ref<HTMLElement | null>(null)
const caption = 'Carolus Linnæus · Roslin, 1775'

interface Dot { hx: number; hy: number; x: number; y: number; vx: number; vy: number; r: number; lum: number }

onMounted(() => {
  const cv = canvas.value!
  const ctx = cv.getContext('2d')!
  const img = new Image()
  img.src = useRuntimeConfig().app.baseURL + 'linnaeus-portrait.jpg'
  img.onload = () => {
    // render at a fixed logical size, dpr-aware
    const W = wrap.value!.clientWidth
    const H = Math.round(W * (img.height / img.width))
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    cv.width = W * dpr; cv.height = H * dpr
    cv.style.width = `${W}px`; cv.style.height = `${H}px`
    ctx.scale(dpr, dpr)

    // sample luminance at grid points
    const gap = Math.max(5, Math.round(W / 64))
    const off = document.createElement('canvas')
    off.width = W; off.height = H
    const octx = off.getContext('2d')!
    octx.drawImage(img, 0, 0, W, H)
    const data = octx.getImageData(0, 0, W, H).data

    const dots: Dot[] = []
    for (let y = gap / 2; y < H; y += gap) {
      for (let x = gap / 2; x < W; x += gap) {
        const i = ((y | 0) * W + (x | 0)) * 4
        let lum = (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 255
        lum = Math.pow(lum, 1.25) // deepen mid-tones so the figure reads
        if (lum > 0.88) continue // skip near-paper background
        const r = 0.4 + (1 - lum) * (gap * 0.62)
        dots.push({ hx: x, hy: y, x, y, vx: 0, vy: 0, r, lum })
      }
    }

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const ink = [26, 43, 30], moss = [46, 83, 57], leaf = [74, 124, 89]
    const draw = () => {
      ctx.clearRect(0, 0, W, H)
      for (const d of dots) {
        const disp = Math.hypot(d.x - d.hx, d.y - d.hy)
        const t = Math.min(disp / 24, 1) // displaced dots bloom toward leaf green
        const c = d.lum < 0.45 ? ink : moss
        const cr = Math.round(c[0] + (leaf[0] - c[0]) * t)
        const cg = Math.round(c[1] + (leaf[1] - c[1]) * t)
        const cb = Math.round(c[2] + (leaf[2] - c[2]) * t)
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`
        ctx.beginPath()
        ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    if (reduced) { draw(); return }

    const mouse = { x: -9999, y: -9999 }
    const R = Math.max(70, W * 0.22)
    const onMove = (e: PointerEvent) => {
      const b = cv.getBoundingClientRect()
      mouse.x = e.clientX - b.left; mouse.y = e.clientY - b.top
    }
    const onLeave = () => { mouse.x = -9999; mouse.y = -9999 }
    cv.addEventListener('pointermove', onMove)
    cv.addEventListener('pointerleave', onLeave)

    let raf = 0, running = false
    const tick = () => {
      for (const d of dots) {
        const dx = d.x - mouse.x, dy = d.y - mouse.y
        const dist = Math.hypot(dx, dy)
        if (dist < R && dist > 0.01) {
          const f = (1 - dist / R) * 1.35
          d.vx += (dx / dist) * f
          d.vy += (dy / dist) * f
        }
        d.vx = (d.vx + (d.hx - d.x) * 0.045) * 0.88
        d.vy = (d.vy + (d.hy - d.y) * 0.045) * 0.88
        d.x += d.vx; d.y += d.vy
      }
      draw()
      if (running) raf = requestAnimationFrame(tick)
    }
    // animate only while on screen
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !running) { running = true; raf = requestAnimationFrame(tick) }
      else if (!e.isIntersecting && running) { running = false; cancelAnimationFrame(raf) }
    })
    io.observe(cv)
    onBeforeUnmount(() => { cancelAnimationFrame(raf); io.disconnect(); cv.removeEventListener('pointermove', onMove); cv.removeEventListener('pointerleave', onLeave) })
  }
})
</script>

<template>
  <figure ref="wrap" class="plate relative">
    <div class="px-5 pt-4 pb-3 flex items-baseline justify-between border-b border-vein">
      <span class="meta-label">Tab. 0</span>
      <span class="meta-label">the taxonomist</span>
    </div>
    <div class="p-4 bg-paper">
      <canvas ref="canvas" class="block w-full cursor-crosshair" :aria-label="caption" role="img" />
    </div>
    <figcaption class="px-5 pb-4 pt-3 border-t border-vein flex items-baseline justify-between">
      <span class="latin text-sm">{{ caption }}</span>
      <span class="meta-label">dot matrix</span>
    </figcaption>
  </figure>
</template>
