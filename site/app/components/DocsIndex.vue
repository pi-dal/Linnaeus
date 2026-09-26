<script setup lang="ts">
const { data: pages } = await useAsyncData('docs-index', () =>
  queryCollection('content').all()
)
const pretty = (p) => {
  const stem = (p.path?.split('/').filter(Boolean).pop() ?? 'index')
  return stem.split('-').map(w => w[0].toUpperCase() + w.slice(1)).join(' ')
}
const docs = computed(() =>
  (pages.value ?? []).filter(p => p.path?.startsWith('/docs/') && !p.path.includes('figures') && !p.path.endsWith('readme'))
    .map(p => ({ ...p, title: (p.title && p.title !== p.path && p.title !== 'Untitled') ? p.title : pretty(p) }))
    .sort((a, b) => a.title.localeCompare(b.title))
)
</script>

<template>
  <div class="hairline divide-y divide-vein">
    <NuxtLink v-for="p in docs" :key="p.path" :to="p.path"
              class="flex items-baseline justify-between px-5 py-4 hover:bg-paper-deep/70 transition-colors group">
      <span class="font-serif text-lg text-moss group-hover:text-leaf transition-colors">
        {{ p.title }}
      </span>
      <span class="meta-label">folium</span>
    </NuxtLink>
  </div>
</template>
