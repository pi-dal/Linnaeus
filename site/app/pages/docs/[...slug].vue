<script setup lang="ts">
const route = useRoute()
const { data: page } = await useAsyncData(`doc-${route.path}`, () =>
  queryCollection('content').path(route.path).first()
)
const prettyTitle = computed(() =>
  (route.path.split('/').filter(Boolean).pop() ?? 'index')
    .split('-').map(w => w[0].toUpperCase() + w.slice(1)).join(' ')
)
useHead(() => ({ title: page.value?.title ? `${page.value.title} — Linnaeus` : 'Linnaeus docs' }))
</script>

<template>
  <main class="mx-auto max-w-3xl px-6 py-14">
    <template v-if="page">
      <p class="meta-label mb-3">Documentum</p>
      <h1 class="font-serif text-3xl sm:text-4xl text-moss mb-2">{{ page.title || prettyTitle }}</h1>
      <div class="h-px w-16 bg-leaf mb-10" />
      <article class="prose prose-neutral max-w-none
                      prose-headings:font-serif prose-headings:text-moss prose-headings:font-normal
                      prose-a:text-leaf prose-a:no-underline hover:prose-a:text-moss
                      prose-code:text-moss prose-code:bg-paper-deep prose-code:px-1 prose-code:py-0.5
                      prose-pre:bg-paper-deep prose-pre:border prose-pre:border-vein
                      prose-table:border-vein prose-th:text-moss prose-td:border-vein
                      prose-strong:text-moss prose-hr:border-vein">
        <ContentRenderer :value="page" />
      </article>
    </template>
    <template v-else>
      <p class="meta-label mb-3">Documentum</p>
      <h1 class="font-serif text-4xl text-moss mb-8">Docs</h1>
      <DocsIndex />
    </template>
  </main>
</template>
