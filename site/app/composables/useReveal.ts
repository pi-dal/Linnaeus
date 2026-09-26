/**
 * Scroll-reveal: elements marked `data-reveal` rise & fade in once.
 * Mount once in app.vue; respects prefers-reduced-motion.
 */
export function useReveal() {
  onMounted(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      document.querySelectorAll('[data-reveal]').forEach((el) => el.classList.add('revealed'))
      return
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add('revealed')
            io.unobserve(e.target)
          }
        }
      },
      { threshold: 0.12, rootMargin: '0px 0px -40px 0px' },
    )
    document.querySelectorAll('[data-reveal]').forEach((el, i) => {
      ;(el as HTMLElement).style.transitionDelay = `${Math.min(i % 6, 5) * 60}ms`
      io.observe(el)
    })
    onBeforeUnmount(() => io.disconnect())
  })
}
