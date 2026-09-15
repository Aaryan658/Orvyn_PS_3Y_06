"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const FIELDS: { name: string; label: string; unit: string; default: number }[] = [
  { name: "ph", label: "pH", unit: "", default: 7.0 },
  { name: "Hardness", label: "Hardness", unit: "mg/L", default: 197 },
  { name: "Solids", label: "Solids (TDS)", unit: "ppm", default: 20927 },
  { name: "Chloramines", label: "Chloramines", unit: "ppm", default: 7.1 },
  { name: "Sulfate", label: "Sulfate", unit: "mg/L", default: 333 },
  { name: "Conductivity", label: "Conductivity", unit: "µS/cm", default: 421 },
  { name: "Organic_carbon", label: "Organic Carbon", unit: "ppm", default: 14.2 },
  { name: "Trihalomethanes", label: "Trihalomethanes", unit: "µg/L", default: 66.4 },
  { name: "Turbidity", label: "Turbidity", unit: "NTU", default: 3.96 },
];

const BENCHMARKS = [
  { model: "Random Forest (baseline)", accuracy: 0.674, auc: 0.661 },
  { model: "HistGradientBoosting (tuned)", accuracy: 0.669, auc: 0.666 },
  { model: "Stacking + SMOTE (RF+HGB+XGB)", accuracy: 0.659, auc: 0.673 },
  { model: "TabPFN (shipped model)", accuracy: 0.688, auc: 0.723, current: true },
];

type Metrics = {
  potability_accuracy: number;
  potability_f1: number;
  potability_roc_auc: number;
  wqi_r2: number;
  wqi_category_accuracy: number;
  dataset_rows: number;
  feature_importance: Record<string, number>;
};

type Prediction = {
  prediction: string;
  confidence: number;
  risk_band: string;
  who_flags: string[];
  wqi_computed: number;
  wqi_computed_category: string;
  wqi_predicted: number;
  wqi_predicted_category: string;
};

type BatchResult = {
  rows: { row: number; wqi: number; risk_category: string; potability: string; confidence: number }[];
  total: number;
  category_counts: Record<string, number>;
  parameter_stats: Record<string, { min: number; max: number; mean: number }>;
};

function bandStyle(label: string) {
  const map: Record<string, string> = {
    Excellent: "bg-emerald-500/20 text-emerald-300 border-emerald-400/40",
    Good: "bg-emerald-500/20 text-emerald-300 border-emerald-400/40",
    Fair: "bg-amber-500/20 text-amber-300 border-amber-400/40",
    Poor: "bg-orange-500/20 text-orange-300 border-orange-400/40",
    "Very Poor": "bg-red-500/20 text-red-300 border-red-400/40",
    Potable: "bg-emerald-500/20 text-emerald-300 border-emerald-400/40",
    "Not Potable": "bg-red-500/20 text-red-300 border-red-400/40",
    Safe: "bg-emerald-500/20 text-emerald-300 border-emerald-400/40",
    "Moderate Risk": "bg-amber-500/20 text-amber-300 border-amber-400/40",
    "High Risk": "bg-orange-500/20 text-orange-300 border-orange-400/40",
    Unsafe: "bg-red-500/20 text-red-300 border-red-400/40",
  };
  return map[label] ?? "bg-white/10 text-white border-white/20";
}

const RISK_ORDER = ["Safe", "Moderate Risk", "High Risk", "Unsafe"];
const RISK_BAR_COLOR: Record<string, string> = {
  Safe: "bg-emerald-400",
  "Moderate Risk": "bg-amber-400",
  "High Risk": "bg-orange-400",
  Unsafe: "bg-red-400",
};

function DistributionChart({ counts }: { counts: Record<string, number> }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  return (
    <div className="space-y-3">
      {RISK_ORDER.map((label) => {
        const count = counts[label] ?? 0;
        const pct = (count / total) * 100;
        return (
          <div key={label}>
            <div className="flex justify-between text-xs text-cyan-200/70">
              <span>{label}</span>
              <span>{count} ({pct.toFixed(0)}%)</span>
            </div>
            <div className="mt-1 h-3 w-full overflow-hidden rounded-full bg-white/10">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.8 }}
                className={`h-full ${RISK_BAR_COLOR[label]}`}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TrendChart({ points }: { points: number[] }) {
  const width = 600;
  const height = 160;
  const pad = 12;
  const minV = Math.min(...points);
  const maxV = Math.max(...points);
  const range = maxV - minV || 1;
  const stepX = points.length > 1 ? (width - pad * 2) / (points.length - 1) : 0;
  const toY = (v: number) => height - pad - ((v - minV) / range) * (height - pad * 2);
  const path = points.map((v, i) => `${i === 0 ? "M" : "L"}${(pad + i * stepX).toFixed(1)},${toY(v).toFixed(1)}`).join(" ");
  const bandLines = [173.4, 192.7, 212.4].filter((b) => b >= minV && b <= maxV);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
      {bandLines.map((b) => (
        <line key={b} x1={pad} x2={width - pad} y1={toY(b)} y2={toY(b)} stroke="white" strokeOpacity={0.15} strokeDasharray="4 4" />
      ))}
      <motion.path
        d={path}
        fill="none"
        stroke="#22d3ee"
        strokeWidth={2}
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.2 }}
      />
    </svg>
  );
}

function ParameterStats({ stats }: { stats: Record<string, { min: number; max: number; mean: number }> }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {Object.entries(stats).map(([name, s]) => (
        <div key={name} className="glass rounded-xl p-3">
          <div className="text-xs text-cyan-200/70">{name}</div>
          <div className="mt-1 text-lg font-semibold text-white">{s.mean.toFixed(1)}</div>
          <div className="text-[10px] text-cyan-200/50">range {s.min.toFixed(1)}&ndash;{s.max.toFixed(1)}</div>
        </div>
      ))}
    </div>
  );
}

function Badge({ label }: { label: string }) {
  return (
    <span className={`inline-block rounded-full border px-3 py-1 text-sm font-medium ${bandStyle(label)}`}>
      {label}
    </span>
  );
}

function WaveDivider() {
  const wave =
    "M0,32L48,37.3C96,43,192,53,288,58.7C384,64,480,64,576,58.7C672,53,768,43,864,37.3C960,32,1056,32,1152,37.3C1248,43,1344,53,1392,58.7L1440,64L1440,120L0,120Z";
  return (
    <div className="relative h-24 w-full overflow-hidden">
      <div className="wave-track wave-slow absolute inset-0 flex opacity-40">
        <svg viewBox="0 0 1440 120" className="h-full w-1/2 shrink-0 fill-cyan-400">
          <path d={wave} />
        </svg>
        <svg viewBox="0 0 1440 120" className="h-full w-1/2 shrink-0 fill-cyan-400">
          <path d={wave} />
        </svg>
      </div>
      <div className="wave-track absolute inset-0 flex">
        <svg viewBox="0 0 1440 120" className="h-full w-1/2 shrink-0 fill-cyan-300">
          <path d={wave} />
        </svg>
        <svg viewBox="0 0 1440 120" className="h-full w-1/2 shrink-0 fill-cyan-300">
          <path d={wave} />
        </svg>
      </div>
    </div>
  );
}

function Reveal({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.5, delay }}
    >
      {children}
    </motion.div>
  );
}

export default function Home() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [values, setValues] = useState<Record<string, number>>(
    Object.fromEntries(FIELDS.map((f) => [f.name, f.default]))
  );
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [batch, setBatch] = useState<BatchResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/metrics`)
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => setError("Backend not reachable -- start it with: uvicorn api:app --port 8000"));
  }, []);

  async function predict() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch {
      setError("Backend not reachable -- start it with: uvicorn api:app --port 8000");
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(file: File) {
    setUploading(true);
    setUploadError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/predict_batch`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      setBatch(await res.json());
    } catch {
      setUploadError("Upload failed -- check the CSV has the 9 required columns.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <main className="min-h-full bg-gradient-to-b from-[#04121f] via-[#062338] to-[#04121f] text-cyan-50">
      {/* Hero */}
      <section className="relative flex flex-col items-center justify-center gap-6 px-6 pt-24 pb-8 text-center">
        <motion.span
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-4 py-1 text-xs tracking-wide text-cyan-300 uppercase"
        >
          SDG 6 &middot; Clean Water &amp; Sanitation
        </motion.span>
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="max-w-3xl text-5xl font-bold tracking-tight text-white float"
        >
          💧 AquaSense
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="max-w-xl text-cyan-200/80"
        >
          A machine-learning system that scores drinking-water risk from raw lab readings --
          honestly benchmarked, not oversold.
        </motion.p>
        <motion.a
          href="#demo"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.97 }}
          className="mt-2 rounded-full bg-cyan-400 px-6 py-3 font-semibold text-[#04121f] shadow-lg shadow-cyan-500/20"
        >
          Try the live demo &darr;
        </motion.a>
      </section>
      <WaveDivider />

      {/* About */}
      <section className="mx-auto max-w-4xl px-6 py-16">
        <Reveal>
          <h2 className="text-2xl font-bold text-white">What this does</h2>
          <p className="mt-3 text-cyan-200/80">
            Enter 9 physicochemical readings from a water sample. AquaSense returns a potability
            risk prediction, a WHO/EPA guideline cross-check, and a computed Water Quality Index --
            three independent signals instead of one black-box number.
          </p>
        </Reveal>
        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          {[
            { icon: "🧠", title: "ML Prediction", body: "TabPFN, tuned via a held-out accuracy-optimal threshold." },
            { icon: "📋", title: "Rule Cross-Check", body: "Flags parameters outside WHO/EPA guideline ranges." },
            { icon: "📊", title: "Water Quality Index", body: "Deterministic score + an ML model that reconstructs it." },
          ].map((c, i) => (
            <Reveal key={c.title} delay={i * 0.1}>
              <div className="glass h-full rounded-2xl p-5">
                <div className="text-3xl">{c.icon}</div>
                <h3 className="mt-2 font-semibold text-white">{c.title}</h3>
                <p className="mt-1 text-sm text-cyan-200/70">{c.body}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* Metrics */}
      <section className="mx-auto max-w-5xl px-6 py-16">
        <Reveal>
          <h2 className="text-2xl font-bold text-white">Performance</h2>
          <p className="mt-2 text-sm text-cyan-200/70">
            Two tasks, two honest numbers -- see Methodology below for why they differ.
          </p>
        </Reveal>
        <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
          {metrics
            ? [
                { label: "Potability Accuracy", value: `${(metrics.potability_accuracy * 100).toFixed(1)}%` },
                { label: "Potability ROC-AUC", value: metrics.potability_roc_auc.toFixed(3) },
                { label: "WQI R²", value: metrics.wqi_r2.toFixed(3) },
                { label: "WQI Category Acc.", value: `${(metrics.wqi_category_accuracy * 100).toFixed(1)}%` },
              ].map((m, i) => (
                <Reveal key={m.label} delay={i * 0.08}>
                  <div className="glass rounded-2xl p-4 text-center">
                    <div className="text-3xl font-bold text-cyan-300">{m.value}</div>
                    <div className="mt-1 text-xs text-cyan-200/70">{m.label}</div>
                  </div>
                </Reveal>
              ))
            : (
                <p className="col-span-4 text-sm text-cyan-200/60">
                  {error ?? "Loading live metrics from the model..."}
                </p>
              )}
        </div>
      </section>

      {/* Methodology */}
      <section className="mx-auto max-w-4xl px-6 py-16">
        <Reveal>
          <h2 className="text-2xl font-bold text-white">Methodology</h2>
          <p className="mt-3 text-cyan-200/80">
            Raw feature&ndash;target correlation for Potability is &asymp; 0 (checked directly on the
            3,276-row Kaggle dataset) -- so we benchmarked five approaches honestly instead of
            chasing an inflated number:
          </p>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="glass mt-6 overflow-x-auto rounded-2xl">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-white/10 text-cyan-300/80">
                  <th className="px-4 py-3 font-medium">Model</th>
                  <th className="px-4 py-3 font-medium">Accuracy</th>
                  <th className="px-4 py-3 font-medium">ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {BENCHMARKS.map((b) => (
                  <tr key={b.model} className={`border-b border-white/5 ${b.current ? "bg-cyan-400/10" : ""}`}>
                    <td className="px-4 py-3">{b.model}{b.current && <span className="ml-2 text-xs text-cyan-300">✓ shipped</span>}</td>
                    <td className="px-4 py-3">{(b.accuracy * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3">{b.auc.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Reveal>
        <Reveal delay={0.15}>
          <p className="mt-4 text-sm text-cyan-200/70">
            WQI, unlike Potability, is a formula computed <em>from</em> these same 9 features --
            so a model can legitimately reconstruct it with much higher accuracy. That&apos;s also
            why most published &ldquo;95%+&rdquo; water-quality claims are about WQI, not raw potability.
          </p>
        </Reveal>
      </section>

      {/* Live demo */}
      <section id="demo" className="mx-auto max-w-4xl px-6 py-16">
        <Reveal>
          <h2 className="text-2xl font-bold text-white">Live demo</h2>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="glass mt-6 rounded-2xl p-6">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              {FIELDS.map((f) => (
                <label key={f.name} className="text-xs text-cyan-200/70">
                  {f.label} {f.unit && <span className="opacity-60">({f.unit})</span>}
                  <input
                    type="number"
                    value={values[f.name]}
                    onChange={(e) => setValues((v) => ({ ...v, [f.name]: parseFloat(e.target.value) || 0 }))}
                    className="mt-1 w-full rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
                  />
                </label>
              ))}
            </div>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={predict}
              disabled={loading}
              className="mt-6 w-full rounded-xl bg-cyan-400 py-3 font-semibold text-[#04121f] disabled:opacity-60"
            >
              {loading ? "Analyzing sample..." : "Predict"}
            </motion.button>
            {error && <p className="mt-3 text-sm text-red-300">{error}</p>}
          </div>
        </Reveal>

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-6 grid gap-4"
            >
              <div className="glass rounded-2xl p-6">
                <h3 className="font-semibold text-white">ML Prediction</h3>
                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <Badge label={result.prediction} />
                  <Badge label={result.risk_band} />
                  <span className="text-sm text-cyan-200/70">{(result.confidence * 100).toFixed(0)}% confidence</span>
                </div>
                <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-white/10">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.confidence * 100}%` }}
                    transition={{ duration: 0.8 }}
                    className="h-full bg-cyan-400"
                  />
                </div>
              </div>

              <div className="glass rounded-2xl p-6">
                <h3 className="font-semibold text-white">WHO/EPA Cross-Check</h3>
                <p className="mt-2 text-sm">
                  {result.who_flags.length ? (
                    <span className="text-red-300">Outside typical range: {result.who_flags.join(", ")}</span>
                  ) : (
                    <span className="text-emerald-300">All parameters within typical safe ranges</span>
                  )}
                </p>
              </div>

              <div className="glass rounded-2xl p-6">
                <h3 className="font-semibold text-white">Water Quality Index</h3>
                <div className="mt-3 grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-cyan-200/60">Computed</div>
                    <div className="text-2xl font-bold text-white">{result.wqi_computed.toFixed(1)}</div>
                    <Badge label={result.wqi_computed_category} />
                  </div>
                  <div>
                    <div className="text-xs text-cyan-200/60">ML-predicted</div>
                    <div className="text-2xl font-bold text-white">{result.wqi_predicted.toFixed(1)}</div>
                    <Badge label={result.wqi_predicted_category} />
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      {/* Dashboard */}
      <section id="dashboard" className="mx-auto max-w-5xl px-6 py-16">
        <Reveal>
          <h2 className="text-2xl font-bold text-white">Monitoring dashboard</h2>
          <p className="mt-2 text-sm text-cyan-200/70">
            Upload a CSV of samples ({FIELDS.map((f) => f.name).join(", ")}) for a batch risk
            breakdown.{" "}
            <a href="/sample.csv" download className="text-cyan-300 underline">
              Try sample data
            </a>
          </p>
        </Reveal>
        <Reveal delay={0.1}>
          <label className="glass mt-6 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-cyan-400/30 p-10 text-center transition hover:border-cyan-400/60">
            <span className="text-3xl">📁</span>
            <span className="text-sm text-cyan-200/80">{uploading ? "Processing..." : "Click to upload a CSV"}</span>
            <input
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
            />
          </label>
          {uploadError && <p className="mt-2 text-sm text-red-300">{uploadError}</p>}
        </Reveal>

        <AnimatePresence>
          {batch && (
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mt-6 grid gap-6">
              <div className="glass rounded-2xl p-6">
                <h3 className="font-semibold text-white">Risk distribution &middot; {batch.total} samples</h3>
                <div className="mt-4">
                  <DistributionChart counts={batch.category_counts} />
                </div>
              </div>
              <div className="glass rounded-2xl p-6">
                <h3 className="font-semibold text-white">WQI trend across samples</h3>
                <p className="text-xs text-cyan-200/50">By upload row order -- this dataset has no timestamps</p>
                <div className="mt-4">
                  <TrendChart points={batch.rows.map((r) => r.wqi)} />
                </div>
              </div>
              <div className="glass rounded-2xl p-6">
                <h3 className="font-semibold text-white">Parameter breakdown</h3>
                <div className="mt-4">
                  <ParameterStats stats={batch.parameter_stats} />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      <footer className="border-t border-white/10 px-6 py-8 text-center text-xs text-cyan-200/50">
        Tech Utsav 2026 &middot; PS-3Y-06 &middot; Dataset: Kaggle Water Potability ({metrics?.dataset_rows ?? "3,276"} samples)
      </footer>
    </main>
  );
}
