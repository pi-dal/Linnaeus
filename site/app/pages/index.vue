<script setup lang="ts">
const plates = [
  { n: 'I', value: '80.78%', label: 'held-out macro accuracy', sub: '26 groups · 180k examples · upstream 78.21%' },
  { n: 'II', value: '73.16%', label: 'JevBench v1.2.2', sub: '231 public tasks · top of the ~2B class' },
  { n: 'III', value: '70.56%', label: 'on-device (MLX 8-bit)', sub: 'same tasks, fully local · +image via VLM' },
  { n: 'IV', value: '33MB', label: 'adapter size', sub: 'rank-8 LoRA + scalar decision head' },
]

// Held-out suite, measured per-group accuracy vs upstream Dohnuts-0.1.0-0.8B
const heldOut = [
  { group: 'scienceqa', ours: '92.7', upstream: '84.6' },
  { group: 'esci_us', ours: '57.6', upstream: '51.0' },
  { group: 'sharc', ours: '73.2', upstream: '66.8' },
  { group: 'clevr_count', ours: '89.7', upstream: '83.5' },
  { group: 'aokvqa', ours: '83.7', upstream: '78.1' },
  { group: 'vqav2_yesno', ours: '85.9', upstream: '80.8' },
  { group: 'xnli_zh', ours: '78.2', upstream: '75.6' },
  { group: 'screenqa_choice †', ours: '22.1', upstream: '22.4' },
  { group: 'macro — all 26 groups', ours: '80.78', upstream: '78.21', strong: true },
]

const jevbench = [
  { system: 'GPT-5.6 Luna', acc: '97.4%', cls: 'API' },
  { system: 'Jev 1.13.0 (TypeSafe AI)', acc: '86.6%', cls: 'API' },
  { system: 'SemIf (Qwen3.5-4B)', acc: '80.9%', cls: 'local 4B' },
  { system: 'Linnaeus-0.1.0-2B', acc: '73.2%', cls: 'local 2B', ours: true },
  { system: 'Qwen3.8 27B (Chutes TEE)', acc: '69.0%', cls: 'cloud 27B' },
  { system: 'upstream Dohnuts-0.1.0-0.8B', acc: '65.8%', cls: 'local 0.8B' },
  { system: 'DeBERTa-v3-large classifiers', acc: '52.4%', cls: 'local <1B' },
]

const deviceLadder = [
  { path: 'CUDA + fla/Triton kernels', acc: '73.16%', note: 'training-time reference' },
  { path: 'torch MPS (reference kernels)', acc: '71.00%', note: 'Mac, merged-hf' },
  { path: 'MLX 8-bit', acc: '70.56%', note: '1.9 GB · Mac + iPhone' },
  { path: 'MLX 4-bit', acc: '67.53%', note: '1.0 GB · smallest' },
]

const species = [
  { variety: 'typus', size: '33 MB', note: 'LoRA adapter on Qwen3.5-2B — the recorded artifact', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B' },
  { variety: 'concreta', size: '4.3 GB', note: 'merged full weights — Mac torch path & conversion source', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B-merged' },
  { variety: 'octo', size: '1.9 GB', note: 'MLX 8-bit, text — Mac & iPhone quality pick', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B-MLX-8bit' },
  { variety: 'quatuor', size: '1.0 GB', note: 'MLX 4-bit, text — smallest footprint', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B-MLX-4bit' },
  { variety: 'octo visionis', size: '2.5 GB', note: 'MLX VLM 8-bit — adds image decisions, vision tower bf16', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B-MLX-VLM-8bit' },
  { variety: 'quatuor visionis', size: '1.6 GB', note: 'MLX VLM 4-bit — multimodal, size pick', href: 'https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B-MLX-VLM-4bit' },
]

const notes = [
  {
    k: 'I',
    title: 'Decisions, not generations',
    body: 'Product interfaces need distributions, not prose. Each candidate gets a reserved marker; one forward scores all of them through a shared scalar head — then per-type temperature calibration turns logits into honest probabilities.',
  },
  {
    k: 'II',
    title: 'A hybrid backbone, kept frozen',
    body: 'Qwen3.5-2B interleaves delta-rule linear attention with full attention (3:1) — long states are cheap to carry. We freeze it entirely and train rank-8 LoRA + the head, upstream’s recipe unchanged, so the comparison is the backbone’s and nothing else’s.',
  },
  {
    k: 'III',
    title: 'The score-row trick',
    body: 'For on-device export the scalar head is appended as an extra lm_head row. Any stock LM runtime — MLX, llama.cpp, plain torch — then yields the decision score as logits[marker, score_row]. No custom head code ships to any runtime.',
  },
  {
    k: 'IV',
    title: 'Measured humility',
    body: '4-bit quantization was not the main cost: MLX at bf16 already trails CUDA by ~3pp, all on hard-tier borderline tasks — the fla chunked delta-rule kernel vs reference implementations. We publish the ladder, not the highlight.',
  },
]

const hortus = [
  { name: 'Dohnuts · PsiACE', role: 'upstream architecture, recipe, and evaluation protocol — this work is a rebase onto a stronger backbone', href: 'https://github.com/PsiACE/dohnuts' },
  { name: 'JevBench', role: 'the decision benchmark our numbers answer to', href: 'https://github.com/fstandhartinger/jevbench' },
  { name: 'Laya · Convai', role: 'application-suite protocol for the 11-group comparisons', href: '#' },
  { name: 'Qwen team', role: 'Qwen3.5-2B base model', href: 'https://huggingface.co/Qwen/Qwen3.5-2B' },
  { name: 'MLX / mlx-vlm · Apple', role: 'the Apple Silicon runtime the VLM builds run on', href: 'https://github.com/ml-explore/mlx-lm' },
  { name: 'OpenBayes', role: 'the RTX 4090 that trained and evaluated everything here', href: 'https://openbayes.com' },
]
</script>

<template>
  <main>
    <!-- ═══ Hero ═══ -->
    <section class="mx-auto max-w-5xl px-6 pt-20 pb-16">
      <p class="meta-label mb-6">Systema Naturae · ed. MMXXVI</p>
      <h1 class="font-serif text-moss text-[clamp(3rem,9vw,6.5rem)] leading-[0.95] tracking-tight">LINNAEUS</h1>
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
        <div v-for="p in plates" :key="p.n" class="plate px-6 py-6">
          <p class="meta-label mb-3">Tab. {{ p.n }}</p>
          <p class="font-mono text-3xl text-moss font-medium">{{ p.value }}</p>
          <p class="mt-2 text-sm text-ink/85">{{ p.label }}</p>
          <p class="mt-1 text-xs text-fern">{{ p.sub }}</p>
        </div>
      </div>
    </section>

    <!-- ═══ Data ═══ -->
    <section class="mx-auto max-w-5xl px-6 py-20">
      <p class="meta-label mb-2">Observatio</p>
      <h2 class="font-serif text-3xl sm:text-4xl text-moss mb-2">The measurements</h2>
      <div class="h-px w-16 bg-leaf mb-10" />

      <div class="grid lg:grid-cols-2 gap-10">
        <div>
          <p class="meta-label mb-3">Held-out suite · 26 groups</p>
          <table class="w-full text-sm hairline">
            <thead>
              <tr class="border-b border-vein text-left">
                <th class="px-4 py-2 font-normal text-fern">group</th>
                <th class="px-4 py-2 font-normal text-moss text-right">linnaeus</th>
                <th class="px-4 py-2 font-normal text-fern text-right">upstream</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in heldOut" :key="r.group"
                  class="border-b border-vein/60 last:border-b-0"
                  :class="r.strong ? 'bg-paper-deep/70' : ''">
                <td class="px-4 py-2.5" :class="r.strong ? 'text-moss font-medium' : 'text-ink/80'">{{ r.group }}</td>
                <td class="px-4 py-2.5 font-mono text-right text-moss">{{ r.ours }}%</td>
                <td class="px-4 py-2.5 font-mono text-right text-fern">{{ r.upstream }}%</td>
              </tr>
            </tbody>
          </table>
          <p class="mt-3 text-xs text-fern">20 wins, 6 ties, 0 losses across all 26 groups vs upstream 0.8B —
          shown: the largest gains plus the one cell where we trail. † screenqa_choice is a
          task-format artifact upstream shares (both at ~22%), not a regression.</p>
        </div>

        <div class="space-y-10">
          <div>
            <p class="meta-label mb-3">JevBench v1.2.2 · 231 public tasks</p>
            <table class="w-full text-sm hairline">
              <tbody>
                <tr v-for="r in jevbench" :key="r.system"
                    class="border-b border-vein/60 last:border-b-0"
                    :class="r.ours ? 'bg-paper-deep/80' : ''">
                  <td class="px-4 py-2.5" :class="r.ours ? 'text-moss font-medium' : 'text-ink/80'">{{ r.system }}</td>
                  <td class="px-4 py-2.5 text-xs text-fern">{{ r.cls }}</td>
                  <td class="px-4 py-2.5 font-mono text-right" :class="r.ours ? 'text-specimen font-medium' : 'text-moss'">{{ r.acc }}</td>
                </tr>
              </tbody>
            </table>
            <p class="mt-3 text-xs text-fern">Head-to-head on identical tasks: ±0 vs Gemma-4-E2B LoRA,
            −18 net vs the 4B classifiers, +24 to +48 vs every sub-1B model.</p>
          </div>

          <div>
            <p class="meta-label mb-3">The on-device ladder · same 231 tasks</p>
            <table class="w-full text-sm hairline">
              <tbody>
                <tr v-for="r in deviceLadder" :key="r.path" class="border-b border-vein/60 last:border-b-0">
                  <td class="px-4 py-2.5 text-ink/80">{{ r.path }}</td>
                  <td class="px-4 py-2.5 text-xs text-fern">{{ r.note }}</td>
                  <td class="px-4 py-2.5 font-mono text-right text-moss">{{ r.acc }}</td>
                </tr>
              </tbody>
            </table>
            <p class="mt-3 text-xs text-fern">The gap is kernel fidelity — fla's chunked delta-rule vs
            reference implementations — not quantization. Easy + standard tiers match 100%.</p>
          </div>
        </div>
      </div>
    </section>

    <!-- ═══ Mechanism ═══ -->
    <section class="border-t border-vein">
      <div class="mx-auto max-w-5xl px-6 py-20">
        <p class="meta-label mb-2">Classis</p>
        <h2 class="font-serif text-3xl sm:text-4xl text-moss mb-2">The mechanism</h2>
        <div class="h-px w-16 bg-leaf mb-10" />
        <div class="grid md:grid-cols-2 gap-12 items-start">
          <div class="space-y-5 text-ink/90 leading-relaxed">
            <p>
              Each request carries a <strong class="text-moss font-medium">state</strong> and
              independent <strong class="text-moss font-medium">questions</strong>. Every
              candidate ends with a reserved marker; the head scores the hidden state at each
              marker, and a temperature-calibrated softmax yields the distribution.
            </p>
            <p class="text-sm text-fern">
              Three question genera: <span class="latin">choice</span>,
              <span class="latin">noul</span>, <span class="latin">score</span>.
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
      </div>
    </section>

    <!-- ═══ Design notes ═══ -->
    <section class="border-t border-vein bg-paper-deep/50">
      <div class="mx-auto max-w-5xl px-6 py-20">
        <p class="meta-label mb-2">Ratio</p>
        <h2 class="font-serif text-3xl sm:text-4xl text-moss mb-2">Why it is built this way</h2>
        <div class="h-px w-16 bg-leaf mb-10" />
        <div class="grid sm:grid-cols-2 gap-x-12 gap-y-10">
          <div v-for="n in notes" :key="n.k" class="flex gap-5">
            <span class="font-serif italic text-2xl text-fern shrink-0 w-8">{{ n.k }}.</span>
            <div>
              <h3 class="font-serif text-lg text-moss mb-1.5">{{ n.title }}</h3>
              <p class="text-sm leading-relaxed text-ink/80">{{ n.body }}</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ═══ Genus table ═══ -->
    <section class="mx-auto max-w-5xl px-6 py-20">
      <p class="meta-label mb-2">Genus</p>
      <h2 class="font-serif text-3xl sm:text-4xl text-moss mb-2">The specimens</h2>
      <div class="h-px w-16 bg-leaf mb-10" />
      <div class="hairline">
        <div v-for="(s, i) in species" :key="i"
             class="grid sm:grid-cols-[1fr_6rem_1.6fr] gap-2 sm:gap-6 px-6 py-4 items-baseline border-b border-vein last:border-b-0 hover:bg-paper-deep/70 transition-colors">
          <a :href="s.href" class="font-serif italic text-lg text-moss hover:text-leaf transition-colors">
            Linnaeus decisio <span class="not-italic text-fern text-sm">var.</span> {{ s.variety }}
          </a>
          <span class="font-mono text-sm text-ink/80">{{ s.size }}</span>
          <span class="text-sm text-ink/75">{{ s.note }}</span>
        </div>
      </div>
      <p class="mt-4 text-xs text-fern">All specimens at huggingface.co/pi-dal — Apache-2.0.</p>
    </section>

    <!-- ═══ Hortus — acknowledgements ═══ -->
    <section class="border-t border-vein">
      <div class="mx-auto max-w-5xl px-6 py-20">
        <p class="meta-label mb-2">Hortus</p>
        <h2 class="font-serif text-3xl sm:text-4xl text-moss mb-2">The garden it grew in</h2>
        <div class="h-px w-16 bg-leaf mb-10" />
        <div class="hairline divide-y divide-vein">
          <a v-for="h in hortus" :key="h.name" :href="h.href"
             class="grid sm:grid-cols-[14rem_1fr] gap-1 sm:gap-8 px-6 py-4 items-baseline hover:bg-paper-deep/70 transition-colors group">
            <span class="font-serif text-moss group-hover:text-leaf transition-colors">{{ h.name }}</span>
            <span class="text-sm text-ink/70">{{ h.role }}</span>
          </a>
        </div>
      </div>
    </section>

    <!-- ═══ Nota bene ═══ -->
    <section class="border-t border-vein">
      <div class="mx-auto max-w-5xl px-6 py-12 text-sm text-fern leading-relaxed">
        <p class="meta-label mb-3">Nota bene</p>
        <p>
          Measured, not claimed: multi-step numeric decisions (JevBench
          <span class="font-mono text-xs">temporal_numeric</span>, 2/15) are the known weak
          cell. Every figure on this page traces to a replayable artifact in the repository.
        </p>
      </div>
    </section>
  </main>
</template>
