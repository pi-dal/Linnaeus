export default defineNuxtConfig({
  ssr: true,
  nitro: {
    preset: 'static',
    prerender: { crawlLinks: true, failOnError: false },
  },
  modules: ['@unocss/nuxt', '@nuxt/content'],
  content: {
    build: { markdown: { toc: { depth: 3 } } },
  },
  app: {
    head: {
      title: 'Linnaeus — a taxonomist for decisions',
      meta: [
        { name: 'description', content: 'A 2B decision model that classifies states into distributions — no generation, calibrated, on-device.' },
      ],
      htmlAttrs: { lang: 'en' },
      link: [{ rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
    },
  },
  css: ['~/assets/css/main.css'],
  compatibilityDate: '2025-01-01',
})
