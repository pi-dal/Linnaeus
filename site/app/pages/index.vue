<script setup lang="ts">
const plates = [
  { n: 'I', value: '80.78%', label: 'held-out macro accuracy', sub: '26 groups · 180k held-out examples' },
  { n: 'II', value: '73.16%', label: 'JevBench v1.2.2', sub: '231 public tasks · top of the 2B class' },
  { n: 'III', value: '70.56%', label: 'on-device MLX', sub: '8-bit · iPhone & Mac, text + image' },
  { n: 'IV', value: '33MB', label: 'adapter size', sub: 'rank-8 LoRA + scalar decision head' },
]

const species = [
  { name: 'decisio', variety: 'typus', size: '33 MB', note: 'LoRA adapter on Qwen3.5-2B — the reference artifact', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B' },
  { name: 'decisio', variety: 'concreta', size: '4.3 GB', note: 'merged full weights; torch MPS & conversion source', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B-merged' },
  { name: 'decisio', variety: 'octo', size: '1.9 GB', note: 'MLX 8-bit text — Mac & iPhone quality pick', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B-MLX-8bit' },
  { name: 'decisio', variety: 'quatuor', size: '1.0 GB', note: 'MLX 4-bit text — smallest footprint', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B-MLX-4bit' },
  { name: 'decisio', variety: 'octo visionis', size: '2.5 GB', note: 'MLX VLM 8-bit — text + image decisions on device', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B-MLX-VLM-8bit' },
  { name: 'decisio', variety: 'quatuor visionis', size: '1.6 GB', note: 'MLX VLM 4-bit — multimodal, size pick', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B-MLX-VLM-4bit' },
]
</script>

<template>
  <main>
    <!-- ═══ Hero ═══ -->
    <section class="mx-auto max-w-5xl px-6 pt-20 pb-16">
      <p class="meta-label mb-6">Systema Naturae · ed. MMXXVI</p>
      <h1 class="font-serif text-moss text-[clamp(3rem,9vw,6.5rem)] leading-[0.95] tracking-tight">
        LINNAEUS
      </h1>
      <p class="latin text-2xl sm:text-3xl mt-3">Linnaeus decisio</p>
      <div class="mt-8 max-w-xl">
        <div class="h-px w-24 bg-leaf mb-6" />
        <p class="text-lg leading-relaxed text-ink/90">
          A 2-billion-parameter model that <em class="font-serif not-italic text-moss">decides</em>
          rather than generates. Give it a state and candidate answers —
          it returns calibrated probabilities over them. No prose, no drift,
          small enough for a phone.
        </p>
      </div>
      <div class="mt-10 flex gap-4 text-sm">
        <a href="https://github.com/pi-dal/Linnaeus"
           class="px-4 py-2 bg-moss text-paper hover:bg-leaf transition-colors">Repository</a>
        <NuxtLink to="/docs"
           class="px-4 py-2 hairline text-moss hover:bg-paper-deep transition-colors">Read the docs</NuxtLink>
      </div>
    </section>

    <!-- ═══ Plates ═══ -->
    <section class="border-y border-vein bg-paper-deep/60">
      <div class="mx-auto max-w-5xl px-6 py-12 grid grid-cols-2 lg:grid-cols-4 gap-px bg-vein/40">
        <div v-for="p in plates" :key="p.n" class="plate px-6 py-6 group">
          <p class="meta-label mb-3">Tab. {{ p.n }}</p>
          <p class="font-mono text-3xl text-moss font-medium">{{ p.value }}</p>
          <p class="mt-2 text-sm text-ink/85">{{ p.label }}</p>
          <p class="mt-1 text-xs text-fern">{{ p.sub }}</p>
        </div>
      </div>
    </section>

    <!-- ═══ Mechanism as taxonomic key ═══ -->
    <section class="mx-auto max-w-5xl px-6 py-20">
      <p class="meta-label mb-2">Classis</p>
      <h2 class="font-serif text-3xl sm:text-4xl text-moss mb-2">The mechanism</h2>
      <div class="h-px w-16 bg-leaf mb-10" />
      <div class="grid md:grid-cols-2 gap-12 items-start">
        <div class="space-y-5 text-ink/90 leading-relaxed">
          <p>
            Each request carries a <strong class="text-moss font-medium">state</strong> and
            independent <strong class="text-moss font-medium">questions</strong>. Every
            candidate option is closed by a reserved marker; a shared scalar head scores
            the hidden state at each marker position, and a temperature-calibrated
            softmax yields the distribution — one forward, many questions.
          </p>
          <p class="text-sm text-fern">
            Three question genera: <span class="latin">choice</span> (select among candidates),
            <span class="latin">noul</span> (probability a statement holds),
            <span class="latin">score</span> (expected ordinal level).
          </p>
        </div>
        <pre class="plate px-6 py-6 font-mono text-[0.8rem] leading-relaxed overflow-x-auto text-moss">predict(
  state    = { user_intent: "play jazz" },
  questions = {
    app: choice ─┬─ spotify   ▮ 0.999
                 ├─ maps      ▮ 0.000
                 └─ settings  ▮ 0.001
    urgent: noul ── true ▮ 0.93
    level:  score ── E[·] = 1.15
})</pre>
      </div>
    </section>

    <!-- ═══ Genus table ═══ -->
    <section class="mx-auto max-w-5xl px-6 pb-20">
      <p class="meta-label mb-2">Genus</p>
      <h2 class="font-serif text-3xl sm:text-4xl text-moss mb-2">The specimens</h2>
      <div class="h-px w-16 bg-leaf mb-10" />
      <div class="hairline">
        <div v-for="(s, i) in species" :key="i"
             class="grid sm:grid-cols-[1fr_6rem_1.6fr] gap-2 sm:gap-6 px-6 py-4 items-baseline border-b border-vein last:border-b-0 hover:bg-paper-deep/70 transition-colors">
          <a :href="s.href" class="font-serif italic text-lg text-moss hover:text-leaf transition-colors">
            Linnaeus {{ s.name }} <span class="not-italic text-fern text-sm">var.</span> {{ s.variety }}
          </a>
          <span class="font-mono text-sm text-ink/80">{{ s.size }}</span>
          <span class="text-sm text-ink/75">{{ s.note }}</span>
        </div>
      </div>
      <p class="mt-4 text-xs text-fern">All specimens at huggingface.co/pi-dal — Apache-2.0.</p>
    </section>

    <!-- ═══ Honesty strip ═══ -->
    <section class="border-t border-vein">
      <div class="mx-auto max-w-5xl px-6 py-12 text-sm text-fern leading-relaxed max-w-none">
        <p class="meta-label mb-3">Nota bene</p>
        <p>
          Measured, not claimed: multi-step numeric decisions (JevBench
          <span class="font-mono text-xs">temporal_numeric</span>, 2/15) are the known weak
          cell; hard-tier borderline tasks lose ~3pp off-device. Every figure on this page
          traces to a replayable artifact in the repository.
        </p>
      </div>
    </section>
  </main>
</template>
