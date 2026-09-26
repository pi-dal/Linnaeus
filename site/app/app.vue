<template>
  <div class="min-h-screen bg-paper text-ink font-sans">
    <header class="border-b border-vein">
      <div class="mx-auto max-w-5xl px-6 py-4 flex items-baseline justify-between">
        <NuxtLink to="/" class="flex items-baseline gap-3 group">
          <LeafMark class="w-5 h-5 text-leaf transition-transform duration-500 group-hover:rotate-12" />
          <span class="font-serif text-xl tracking-[0.08em] text-moss">LINNAEUS</span>
          <span class="latin text-sm hidden sm:inline">decisio</span>
        </NuxtLink>
        <nav class="flex gap-6 text-sm">
          <NuxtLink to="/docs" class="text-fern hover:text-moss transition-colors">Docs</NuxtLink>
          <a href="https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B" class="text-fern hover:text-moss transition-colors">Hugging Face</a>
          <a href="https://github.com/pi-dal/Linnaeus" class="text-fern hover:text-moss transition-colors">GitHub</a>
        </nav>
      </div>
    </header>
    <NuxtPage />
    <footer class="border-t border-vein mt-24">
      <div class="mx-auto max-w-5xl px-6 py-10 flex flex-col sm:flex-row justify-between gap-4">
        <p class="text-sm text-fern">
          Named for Carl Linnaeus — a taxonomist for decisions.<br>
          Forked from <span class="font-mono text-xs">dohnuts</span> (PsiACE), Apache-2.0.
        </p>
        <p class="meta-label self-end">Syst. Nat. · MMXXVI</p>
      </div>
    </footer>
  </div>
</template>
