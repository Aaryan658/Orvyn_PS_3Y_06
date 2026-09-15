"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const FIELDS: { name: string; label: string; unit: string; default: number }[] = [
  { name: "ph", label: "pH", unit: "—", default: 7.0 },
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
  active_model?: string;
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
  active_model?: string;
};

type BatchResult = {
  rows: { row: number; wqi: number; risk_category: string; potability: string; confidence: number }[];
  total: number;
  category_counts: Record<string, number>;
  parameter_stats: Record<string, { min: number; max: number; mean: number }>;
};

const RISK_ORDER = ["Safe", "Moderate Risk", "High Risk", "Unsafe"];
const RISK_THEME: Record<string, { bar: string; text: string }> = {
  Safe: { bar: "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.5)]", text: "text-emerald-300" },
  "Moderate Risk": { bar: "bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.5)]", text: "text-amber-300" },
  "High Risk": { bar: "bg-orange-400 shadow-[0_0_10px_rgba(251,146,60,0.5)]", text: "text-orange-300" },
  Unsafe: { bar: "bg-rose-400 shadow-[0_0_10px_rgba(244,63,94,0.5)]", text: "text-rose-300" },
};

function DistributionChart({ counts, total }: { counts: Record<string, number>; total: number }) {
  const safeTotal = total || Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  return (
    <div className="space-y-6">
      {RISK_ORDER.map((label) => {
        const count = counts[label] ?? 0;
        const pct = (count / safeTotal) * 100;
        const theme = RISK_THEME[label] ?? { bar: "bg-white/70", text: "text-white/70" };
        return (
          <div key={label} className="grid grid-cols-[160px_1fr_100px] gap-6 items-center">
            <span className={`text-base md:text-lg italic ${theme.text}`}>{label}</span>
            <div className="h-[1px] w-full bg-white/10 relative">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 1.2, ease: "easeOut" }}
                className={`absolute top-[-1px] left-0 h-[3px] rounded-full ${theme.bar}`}
              />
            </div>
            <span className="text-right text-base text-indigo-200/80 font-sans">
              {count} <span className="text-xs text-indigo-300/40">({pct.toFixed(1)}%)</span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

function TrendChart({ points }: { points: number[] }) {
  const width = 800;
  const height = 140;
  const pad = 10;
  const minV = Math.min(...points);
  const maxV = Math.max(...points);
  const range = maxV - minV || 1;
  const stepX = points.length > 1 ? (width - pad * 2) / (points.length - 1) : 0;
  const toY = (v: number) => height - pad - ((v - minV) / range) * (height - pad * 2);
  const path = points.map((v, i) => `${i === 0 ? "M" : "L"}${(pad + i * stepX).toFixed(1)},${toY(v).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto opacity-85 overflow-visible">
      <motion.path
        d={path}
        fill="none"
        stroke="#9e96f8"
        strokeWidth={1.8}
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.5, ease: "easeInOut" }}
      />
    </svg>
  );
}

function Reveal({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 1, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}

function SectionDivider() {
  return (
    <div className="w-full flex justify-center py-24 opacity-30">
      <div className="w-12 border-t border-white/50"></div>
    </div>
  );
}

export default function Home() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [metricsError, setMetricsError] = useState<string | null>(null);

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
    setMetricsLoading(true);
    fetch(`${API}/metrics`)
      .then((r) => {
        if (!r.ok) throw new Error("Unreachable");
        return r.json();
      })
      .then((data) => {
        setMetrics(data);
        setMetricsLoading(false);
      })
      .catch(() => {
        setMetricsError("Backend not reachable -- start it with: uvicorn api:app --port 8000");
        setMetricsLoading(false);
      });
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
    <main className="relative z-10 min-h-full text-foreground pb-32">
      {/* Subtle Glowing Background for Depth */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[80vw] h-[80vh] bg-indigo-500/10 blur-[120px] rounded-full pointer-events-none opacity-50 mix-blend-screen" />
      
      {/* Hero */}
      <section className="relative flex flex-col items-center justify-center gap-8 px-6 min-h-[100vh] text-center overflow-x-clip pt-20 pb-28">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.5, delay: 0.2 }}
          className="relative z-10 inline-flex items-center gap-2.5 px-4 py-1.5 rounded-full border border-indigo-400/25 bg-indigo-500/10 text-indigo-300 text-sm tracking-[0.2em] uppercase mb-2 backdrop-blur-sm font-sans"
        >
          <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.9)]" />
          SDG 6 &middot; Clean Water &amp; Sanitation
        </motion.div>

        <div className="relative my-2 py-2">
          {/* Authentic Fluid Water Wave Basin directly under & around AquaSense */}
          <div className="absolute inset-x-[-15%] -bottom-6 top-6 pointer-events-none z-0 overflow-hidden flex items-end justify-center">
            <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-cyan-500/20 via-sky-500/10 to-transparent blur-3xl mix-blend-screen opacity-90" />
            
            <svg
              viewBox="0 0 1440 240"
              preserveAspectRatio="none"
              className="w-full h-36 md:h-44 overflow-visible opacity-85"
            >
              <defs>
                <linearGradient id="wave-deep" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.28" />
                  <stop offset="50%" stopColor="#0284c7" stopOpacity="0.16" />
                  <stop offset="100%" stopColor="#0369a1" stopOpacity="0.0" />
                </linearGradient>

                <linearGradient id="wave-mid" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#67e8f9" stopOpacity="0.4" />
                  <stop offset="60%" stopColor="#06b6d4" stopOpacity="0.18" />
                  <stop offset="100%" stopColor="#0891b2" stopOpacity="0.0" />
                </linearGradient>

                <linearGradient id="wave-crest" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#a5f3fc" stopOpacity="0.2" />
                  <stop offset="30%" stopColor="#38bdf8" stopOpacity="0.8" />
                  <stop offset="70%" stopColor="#67e8f9" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#a5f3fc" stopOpacity="0.2" />
                </linearGradient>

                <filter id="water-glow" x="-10%" y="-10%" width="120%" height="120%">
                  <feGaussianBlur stdDeviation="3" result="glow" />
                  <feMerge>
                    <feMergeNode in="glow" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>

              {/* Layer 1: Deep Slow Background Wave */}
              <motion.path
                d="M 0 130 C 240 100, 480 160, 720 130 C 960 100, 1200 160, 1440 130 L 1440 240 L 0 240 Z"
                fill="url(#wave-deep)"
                animate={{
                  d: [
                    "M 0 130 C 240 100, 480 160, 720 130 C 960 100, 1200 160, 1440 130 L 1440 240 L 0 240 Z",
                    "M 0 145 C 240 165, 480 105, 720 145 C 960 165, 1200 105, 1440 145 L 1440 240 L 0 240 Z",
                    "M 0 130 C 240 100, 480 160, 720 130 C 960 100, 1200 160, 1440 130 L 1440 240 L 0 240 Z"
                  ]
                }}
                transition={{ duration: 9, repeat: Infinity, ease: "easeInOut" }}
              />

              {/* Layer 2: Mid Dynamic Fluid Wave */}
              <motion.path
                d="M 0 150 C 200 175, 420 115, 680 155 C 940 190, 1180 120, 1440 150 L 1440 240 L 0 240 Z"
                fill="url(#wave-mid)"
                animate={{
                  d: [
                    "M 0 150 C 200 175, 420 115, 680 155 C 940 190, 1180 120, 1440 150 L 1440 240 L 0 240 Z",
                    "M 0 135 C 220 105, 460 170, 720 135 C 980 100, 1220 175, 1440 135 L 1440 240 L 0 240 Z",
                    "M 0 150 C 200 175, 420 115, 680 155 C 940 190, 1180 120, 1440 150 L 1440 240 L 0 240 Z"
                  ]
                }}
                transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
              />

              {/* Layer 3: Surface Wave Crest Line */}
              <motion.path
                d="M 0 148 C 220 115, 440 175, 720 140 C 980 105, 1220 170, 1440 142"
                fill="none"
                stroke="url(#wave-crest)"
                strokeWidth="2"
                filter="url(#water-glow)"
                animate={{
                  d: [
                    "M 0 148 C 220 115, 440 175, 720 140 C 980 105, 1220 170, 1440 142",
                    "M 0 138 C 240 170, 480 115, 740 150 C 980 180, 1220 115, 1440 138",
                    "M 0 148 C 220 115, 440 175, 720 140 C 980 105, 1220 170, 1440 142"
                  ]
                }}
                transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
              />

              {/* Layer 4: Subtle secondary ripple thread */}
              <motion.path
                d="M 0 162 C 260 180, 520 138, 760 165 C 1020 190, 1240 140, 1440 160"
                fill="none"
                stroke="#67e8f9"
                strokeWidth="1.2"
                strokeDasharray="4 8"
                opacity="0.4"
                animate={{
                  d: [
                    "M 0 162 C 260 180, 520 138, 760 165 C 1020 190, 1240 140, 1440 160",
                    "M 0 152 C 240 132, 500 175, 760 148 C 1000 125, 1260 172, 1440 154",
                    "M 0 162 C 260 180, 520 138, 760 165 C 1020 190, 1240 140, 1440 160"
                  ]
                }}
                transition={{ duration: 8.5, repeat: Infinity, ease: "easeInOut" }}
              />
            </svg>
          </div>

          <motion.h1
            initial={{ opacity: 0, filter: "blur(10px)" }}
            animate={{ opacity: 1, filter: "blur(0px)" }}
            transition={{ duration: 1.2, ease: "easeOut" }}
            className="relative z-10 max-w-5xl text-7xl md:text-[9.5rem] tracking-normal font-normal leading-none italic bg-gradient-to-b from-[#e0f7fa] via-[#9e96f8] to-[#8378f2] bg-clip-text text-transparent drop-shadow-[0_6px_28px_rgba(56,189,248,0.3)]"
          >
            AquaSense
          </motion.h1>
        </div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.8 }}
          className="max-w-3xl text-2xl md:text-3xl lg:text-4xl text-indigo-100/80 leading-relaxed italic"
        >
          A machine-learning system that scores drinking-water risk from raw lab readings &mdash; honestly benchmarked, not oversold.
        </motion.p>

        <div className="relative mt-24 md:mt-32 flex items-center justify-center pb-12">
          <motion.a
            href="#demo"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 1.2 }}
            className="flex items-center justify-center gap-4 text-lg md:text-xl tracking-[0.25em] uppercase border-b border-indigo-300/30 pb-2 text-indigo-100/90 hover:text-[#d4d1ff] hover:border-indigo-300/80 transition-all duration-500 group relative z-10"
          >
            <span>Try the live demo &rarr;</span>
            <span className="relative inline-flex items-center">
              {/* Dotted Flow Animation: Plump, organic, continuous cursive ribbon loop */}
              <span className="absolute left-[calc(100%+14px)] top-1/2 -translate-y-[150px] w-[1600px] h-[520px] pointer-events-none z-0 hidden md:block overflow-visible text-left">
                <svg viewBox="0 0 1600 520" className="w-full h-full overflow-visible">
                  <defs>
                    <filter id="arrow-dot-glow" x="-20%" y="-20%" width="140%" height="140%">
                      <feGaussianBlur stdDeviation="2.2" result="blur" />
                      <feMerge>
                        <feMergeNode in="blur" />
                        <feMergeNode in="SourceGraphic" />
                      </feMerge>
                    </filter>
                    <mask id="arrow-path-reveal">
                      <motion.path
                        d="M 0 150 L 90 150 C 130 150, 175 145, 205 120 C 235 95, 245 60, 230 25 C 215 -10, 175 -10, 145 15 C 115 40, 100 95, 90 150 C 80 205, 90 260, 125 300 C 175 355, 270 395, 420 420 C 650 455, 1050 475, 1600 480"
                        fill="none"
                        stroke="white"
                        strokeWidth="38"
                        strokeLinecap="round"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: 2.4, ease: [0.16, 1, 0.3, 1], delay: 0.8 }}
                      />
                    </mask>
                  </defs>

                  <g mask="url(#arrow-path-reveal)">
                    <motion.path
                      d="M 0 150 L 90 150 C 130 150, 175 145, 205 120 C 235 95, 245 60, 230 25 C 215 -10, 175 -10, 145 15 C 115 40, 100 95, 90 150 C 80 205, 90 260, 125 300 C 175 355, 270 395, 420 420 C 650 455, 1050 475, 1600 480"
                      fill="none"
                      stroke="rgba(194, 190, 255, 0.45)"
                      strokeWidth="5"
                      strokeDasharray="0 17"
                      strokeLinecap="round"
                      filter="url(#arrow-dot-glow)"
                      animate={{ strokeDashoffset: [0, -34] }}
                      transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
                    />
                    <motion.path
                      d="M 0 150 L 90 150 C 130 150, 175 145, 205 120 C 235 95, 245 60, 230 25 C 215 -10, 175 -10, 145 15 C 115 40, 100 95, 90 150 C 80 205, 90 260, 125 300 C 175 355, 270 395, 420 420 C 650 455, 1050 475, 1600 480"
                      fill="none"
                      stroke="#f3f1ff"
                      strokeWidth="3.2"
                      strokeDasharray="0 17"
                      strokeLinecap="round"
                      animate={{ strokeDashoffset: [0, -34] }}
                      transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
                    />
                  </g>
                </svg>
              </span>
            </span>
          </motion.a>
        </div>
      </section>

      <SectionDivider />

      {/* "What this does" section */}
      <section className="mx-auto max-w-5xl px-6">
        <Reveal>
          <div className="text-center mb-16">
            <span className="text-xs uppercase tracking-[0.3em] text-indigo-400/60 font-medium block mb-3 font-sans">System Architecture</span>
            <h2 className="text-4xl md:text-5xl text-[#edeaff] font-normal tracking-wide">What this does</h2>
            <p className="text-xl md:text-2xl leading-relaxed text-indigo-200/80 max-w-3xl mx-auto mt-6">
              Enter 9 physicochemical readings from a water sample. AquaSense returns a potability risk prediction, a WHO/EPA guideline cross-check, and a computed Water Quality Index &mdash; three independent signals instead of one black-box number.
            </p>
          </div>
        </Reveal>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Signal 01: ML Prediction */}
          <Reveal delay={0.1}>
            <div className="relative group rounded-2xl border border-indigo-400/20 bg-gradient-to-b from-[#19153a]/90 via-[#120e2c]/95 to-[#0b081c] p-8 md:p-9 backdrop-blur-xl shadow-[0_16px_40px_rgba(0,0,0,0.4)] hover:border-indigo-300/40 transition-all duration-500 h-full flex flex-col justify-between overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 rounded-full blur-2xl pointer-events-none group-hover:bg-indigo-500/20 transition-all duration-700" />
              
              <div>
                <div className="flex items-center justify-between mb-8">
                  <span className="text-[11px] font-sans uppercase tracking-[0.25em] text-indigo-300/50 font-medium">
                    SIGNAL &bull; 01
                  </span>
                  <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-400/20 flex items-center justify-center group-hover:scale-110 group-hover:border-indigo-400/40 transition-all duration-300 shadow-[0_0_15px_rgba(158,150,248,0.15)]">
                    <svg className="w-5 h-5 text-indigo-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
                    </svg>
                  </div>
                </div>

                <h3 className="text-2xl font-serif font-normal text-[#edeaff] mb-3 group-hover:text-white transition-colors">
                  ML Prediction
                </h3>
                
                <p className="text-base md:text-lg text-indigo-200/75 leading-relaxed italic">
                  TabPFN, tuned via a held-out accuracy-optimal threshold.
                </p>
              </div>

              <div className="mt-8 pt-4 border-t border-indigo-400/10 flex items-center justify-between text-xs font-sans text-indigo-300/40">
                <span>Inference Engine</span>
                <span className="text-indigo-300/70 font-medium">Prior-Data Fitted</span>
              </div>
            </div>
          </Reveal>

          {/* Signal 02: Rule Cross-Check */}
          <Reveal delay={0.2}>
            <div className="relative group rounded-2xl border border-indigo-400/20 bg-gradient-to-b from-[#19153a]/90 via-[#120e2c]/95 to-[#0b081c] p-8 md:p-9 backdrop-blur-xl shadow-[0_16px_40px_rgba(0,0,0,0.4)] hover:border-indigo-300/40 transition-all duration-500 h-full flex flex-col justify-between overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/10 rounded-full blur-2xl pointer-events-none group-hover:bg-amber-500/15 transition-all duration-700" />
              
              <div>
                <div className="flex items-center justify-between mb-8">
                  <span className="text-[11px] font-sans uppercase tracking-[0.25em] text-indigo-300/50 font-medium">
                    SIGNAL &bull; 02
                  </span>
                  <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-400/20 flex items-center justify-center group-hover:scale-110 group-hover:border-amber-400/40 transition-all duration-300 shadow-[0_0_15px_rgba(251,191,36,0.15)]">
                    <svg className="w-5 h-5 text-amber-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
                    </svg>
                  </div>
                </div>

                <h3 className="text-2xl font-serif font-normal text-[#edeaff] mb-3 group-hover:text-white transition-colors">
                  Rule Cross-Check
                </h3>
                
                <p className="text-base md:text-lg text-indigo-200/75 leading-relaxed italic">
                  Flags parameters outside WHO/EPA guideline ranges.
                </p>
              </div>

              <div className="mt-8 pt-4 border-t border-indigo-400/10 flex items-center justify-between text-xs font-sans text-indigo-300/40">
                <span>Determinism</span>
                <span className="text-amber-300/80 font-medium">Zero-Tolerant Boundary</span>
              </div>
            </div>
          </Reveal>

          {/* Signal 03: Water Quality Index */}
          <Reveal delay={0.3}>
            <div className="relative group rounded-2xl border border-indigo-400/20 bg-gradient-to-b from-[#19153a]/90 via-[#120e2c]/95 to-[#0b081c] p-8 md:p-9 backdrop-blur-xl shadow-[0_16px_40px_rgba(0,0,0,0.4)] hover:border-indigo-300/40 transition-all duration-500 h-full flex flex-col justify-between overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-full blur-2xl pointer-events-none group-hover:bg-cyan-500/20 transition-all duration-700" />
              
              <div>
                <div className="flex items-center justify-between mb-8">
                  <span className="text-[11px] font-sans uppercase tracking-[0.25em] text-indigo-300/50 font-medium">
                    SIGNAL &bull; 03
                  </span>
                  <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-400/20 flex items-center justify-center group-hover:scale-110 group-hover:border-cyan-400/40 transition-all duration-300 shadow-[0_0_15px_rgba(34,211,238,0.15)]">
                    <svg className="w-5 h-5 text-cyan-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
                    </svg>
                  </div>
                </div>

                <h3 className="text-2xl font-serif font-normal text-[#edeaff] mb-3 group-hover:text-white transition-colors">
                  Water Quality Index
                </h3>
                
                <p className="text-base md:text-lg text-indigo-200/75 leading-relaxed italic">
                  Deterministic score + an ML model that reconstructs it.
                </p>
              </div>

              <div className="mt-8 pt-4 border-t border-indigo-400/10 flex items-center justify-between text-xs font-sans text-indigo-300/40">
                <span>Weighted Aggregation</span>
                <span className="text-cyan-300/80 font-medium">Arithmetic WQI</span>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      <SectionDivider />

      {/* Performance section */}
      <section className="mx-auto max-w-5xl px-6">
        <Reveal>
          <div className="text-center mb-16">
            <span className="text-xs uppercase tracking-[0.3em] text-indigo-400/60 font-medium block mb-3 font-sans">Telemetry</span>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-4">
              <h2 className="text-4xl md:text-5xl text-[#edeaff] font-normal tracking-wide">Performance</h2>
              <span className="px-3.5 py-1 rounded-full text-xs font-sans tracking-wider uppercase bg-indigo-500/15 text-indigo-300 border border-indigo-400/30">
                serving: {metrics?.active_model ?? "TabPFN"}
              </span>
            </div>
            <p className="text-xl md:text-2xl text-indigo-200/80 max-w-2xl mx-auto italic">
              Two tasks, two honest numbers &mdash; see Methodology below for why they differ.
            </p>
            {metricsLoading && (
              <p className="mt-4 text-sm text-indigo-300/60 italic font-sans animate-pulse">
                Loading live metrics from the model...
              </p>
            )}
            {metricsError && (
              <p className="mt-4 text-sm text-rose-300/90 italic font-sans">
                {metricsError}
              </p>
            )}
          </div>
        </Reveal>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <Reveal delay={0.1}>
            <div className="rounded-2xl border border-indigo-500/20 bg-gradient-to-b from-[#181436]/80 to-[#100c24]/90 p-6 backdrop-blur-md shadow-[0_12px_36px_rgba(0,0,0,0.35)] text-center">
              <span className="text-xs tracking-[0.2em] text-indigo-300/60 uppercase font-sans font-medium block mb-3">
                Potability Accuracy
              </span>
              <p className="text-5xl text-[#d4d1ff] font-light mb-2 drop-shadow-[0_0_20px_rgba(180,175,255,0.35)]">
                {metrics ? (metrics.potability_accuracy * 100).toFixed(1) + "%" : "68.8%"}
              </p>
              <p className="text-sm text-indigo-300/50 italic font-sans">Accuracy on test split</p>
            </div>
          </Reveal>

          <Reveal delay={0.2}>
            <div className="rounded-2xl border border-indigo-500/20 bg-gradient-to-b from-[#181436]/80 to-[#100c24]/90 p-6 backdrop-blur-md shadow-[0_12px_36px_rgba(0,0,0,0.35)] text-center">
              <span className="text-xs tracking-[0.2em] text-indigo-300/60 uppercase font-sans font-medium block mb-3">
                Potability ROC-AUC
              </span>
              <p className="text-5xl text-[#c2beff] font-light mb-2 drop-shadow-[0_0_20px_rgba(194,190,255,0.35)]">
                {metrics ? metrics.potability_roc_auc.toFixed(3) : "0.723"}
              </p>
              <p className="text-sm text-indigo-300/50 italic font-sans">Discrimination ability</p>
            </div>
          </Reveal>

          <Reveal delay={0.3}>
            <div className="rounded-2xl border border-emerald-500/20 bg-gradient-to-b from-[#12232a]/60 to-[#0e1620]/80 p-6 backdrop-blur-md shadow-[0_12px_36px_rgba(0,0,0,0.35)] text-center">
              <span className="text-xs tracking-[0.2em] text-emerald-300/60 uppercase font-sans font-medium block mb-3">
                WQI R&sup2;
              </span>
              <p className="text-5xl text-[#9ef5db] font-light mb-2 drop-shadow-[0_0_20px_rgba(158,245,219,0.35)]">
                {metrics ? metrics.wqi_r2.toFixed(3) : "0.985"}
              </p>
              <p className="text-sm text-emerald-300/50 italic font-sans">Variance explained</p>
            </div>
          </Reveal>

          <Reveal delay={0.4}>
            <div className="rounded-2xl border border-emerald-500/20 bg-gradient-to-b from-[#12232a]/60 to-[#0e1620]/80 p-6 backdrop-blur-md shadow-[0_12px_36px_rgba(0,0,0,0.35)] text-center">
              <span className="text-xs tracking-[0.2em] text-emerald-300/60 uppercase font-sans font-medium block mb-3">
                WQI Category Acc.
              </span>
              <p className="text-5xl text-[#86efac] font-light mb-2 drop-shadow-[0_0_20px_rgba(134,239,172,0.35)]">
                {metrics ? (metrics.wqi_category_accuracy * 100).toFixed(1) + "%" : "93.3%"}
              </p>
              <p className="text-sm text-emerald-300/50 italic font-sans">5-tier category match</p>
            </div>
          </Reveal>
        </div>
      </section>

      <SectionDivider />

      {/* Methodology section */}
      <section className="mx-auto max-w-4xl px-6">
        <Reveal>
          <div className="text-center mb-12">
            <span className="text-xs uppercase tracking-[0.3em] text-indigo-400/60 font-medium block mb-3 font-sans">Scientific Rigor</span>
            <h2 className="text-4xl md:text-5xl text-[#edeaff] font-normal tracking-wide">Methodology</h2>
            <p className="text-xl md:text-2xl leading-relaxed text-indigo-200/80 max-w-3xl mx-auto mt-6">
              Raw feature&ndash;target correlation for Potability is &asymp; 0 (checked directly on the 3,276-row Kaggle dataset) &mdash; so we benchmarked five approaches honestly instead of chasing an inflated number:
            </p>
          </div>
        </Reveal>

        <Reveal delay={0.2}>
          <div className="rounded-2xl border border-indigo-400/15 bg-gradient-to-b from-[#14122c]/70 to-[#0e0b1d]/85 p-6 md:p-8 backdrop-blur-md shadow-[0_8px_30px_rgba(0,0,0,0.3)]">
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-indigo-400/20 font-sans text-xs uppercase tracking-widest text-indigo-300/60">
                    <th className="pb-4 font-medium">Model</th>
                    <th className="pb-4 font-medium text-right">Accuracy</th>
                    <th className="pb-4 font-medium text-right">ROC-AUC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-indigo-400/10">
                  {BENCHMARKS.map((b) => (
                    <tr
                      key={b.model}
                      className={b.current ? "bg-indigo-500/15" : "hover:bg-white/[0.02]"}
                    >
                      <td className="py-4 font-serif text-lg md:text-xl text-[#edeaff] flex items-center gap-3">
                        {b.model}
                        {b.current && (
                          <span className="text-xs font-sans tracking-wider uppercase px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                            &check; shipped
                          </span>
                        )}
                      </td>
                      <td className={`py-4 font-serif text-lg md:text-xl text-right ${b.current ? "text-emerald-300 font-medium" : "text-indigo-200/70"}`}>
                        {(b.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className={`py-4 font-sans text-sm md:text-base text-right ${b.current ? "text-emerald-300 font-medium" : "text-indigo-300/50"}`}>
                        {b.auc.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="mt-8 text-base md:text-lg text-indigo-200/70 leading-relaxed italic border-t border-indigo-400/15 pt-6">
              WQI, unlike Potability, is a formula computed <em>from</em> these same 9 features &mdash; so a model can legitimately reconstruct it with much higher accuracy. That&apos;s also why most published &ldquo;95%+&rdquo; water-quality claims are about WQI, not raw potability.
            </p>
          </div>
        </Reveal>

        {/* Related work subsection */}
        <Reveal delay={0.3}>
          <div className="mt-12 rounded-2xl border border-indigo-400/15 bg-gradient-to-b from-[#121028]/70 to-[#0a0818]/85 p-8 md:p-10 backdrop-blur-md">
            <span className="text-xs uppercase tracking-[0.3em] text-indigo-400/60 font-medium block mb-3 font-sans">Literature Comparison</span>
            <h3 className="text-2xl md:text-3xl font-serif text-[#edeaff] mb-4">Related work</h3>
            <p className="text-lg md:text-xl leading-relaxed text-indigo-200/80">
              Abdullah et al. (IJACSA, 2023) report 95.08% accuracy with Random Forest on a different Kaggle set (8,000 rows, 21 contaminant-threshold features, an <code className="text-indigo-300 bg-indigo-500/20 px-1.5 py-0.5 rounded font-mono text-sm">is_safe</code> label largely derivable from three features alone). It&apos;s a different, easier problem, not a higher bar we missed &mdash; leakage handling isn&apos;t specified in their preprocessing, only mean imputation is used, and results come from a single train/test split. This project splits before imputing/scaling, uses median imputation, cross-checks five modeling approaches, and ships a deployed 5-band risk output instead of a binary label and an offline benchmark.
            </p>
          </div>
        </Reveal>
      </section>

      <SectionDivider />

      {/* Live demo section */}
      <section id="demo" className="mx-auto max-w-4xl px-6">
        <Reveal>
          <div className="text-center mb-12">
            <span className="text-xs uppercase tracking-[0.3em] text-indigo-400/60 font-medium block mb-3 font-sans">Interactive Diagnostics</span>
            <h2 className="text-4xl md:text-5xl text-[#edeaff] font-normal tracking-wide">Live demo</h2>
            <p className="text-xl md:text-2xl text-indigo-200/80 max-w-xl mx-auto mt-4 italic">
              Enter 9 physicochemical laboratory readings to evaluate drinking safety and standard compliance.
            </p>
          </div>
        </Reveal>

        <Reveal delay={0.2}>
          <div className="rounded-3xl border border-indigo-400/25 bg-gradient-to-b from-[#171438]/90 via-[#120f2b]/95 to-[#0b081c] p-8 md:p-12 shadow-[0_20px_50px_rgba(0,0,0,0.5)] backdrop-blur-xl relative overflow-hidden">
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-5">
              {FIELDS.map((f) => (
                <div
                  key={f.name}
                  className="rounded-xl border border-indigo-400/15 bg-white/[0.02] hover:bg-white/[0.04] p-4.5 transition-all duration-300 focus-within:border-indigo-400/60 focus-within:bg-indigo-500/5 group"
                >
                  <div className="flex items-baseline justify-between mb-2">
                    <label className="text-xs tracking-wider text-indigo-200/70 uppercase font-sans font-medium">
                      {f.label}
                    </label>
                    {f.unit && (
                      <span className="text-xs font-sans text-indigo-300/45 italic">
                        {f.unit}
                      </span>
                    )}
                  </div>
                  <input
                    type="number"
                    value={values[f.name]}
                    onChange={(e) => setValues((v) => ({ ...v, [f.name]: parseFloat(e.target.value) || 0 }))}
                    className="w-full bg-transparent text-2xl font-serif text-[#f3f1ff] outline-none group-focus-within:text-white transition-colors"
                  />
                </div>
              ))}
            </div>
            
            <div className="mt-12 pt-8 border-t border-indigo-400/15 flex flex-col sm:flex-row items-center justify-between gap-6">
              <span className="text-xs tracking-widest text-indigo-300/50 uppercase font-sans">
                Standard: WHO &amp; EPA Drinking Guidelines
              </span>
              <button
                onClick={predict}
                disabled={loading}
                className="w-full sm:w-auto text-sm tracking-[0.25em] uppercase font-sans font-medium px-10 py-3.5 rounded-full bg-gradient-to-r from-[#8378f2] to-[#9e96f8] text-[#0a071d] shadow-[0_0_25px_rgba(158,150,248,0.4)] hover:shadow-[0_0_35px_rgba(158,150,248,0.65)] hover:scale-[1.02] active:scale-[0.98] transition-all duration-300 disabled:opacity-50"
              >
                {loading ? "Analyzing sample..." : "Predict"}
              </button>
            </div>
            {error && <p className="mt-4 text-center text-sm italic text-rose-300/90 font-sans">{error}</p>}
          </div>
        </Reveal>

        {/* Result panel: Three cards */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-10 grid grid-cols-1 md:grid-cols-3 gap-6"
            >
              {/* Card 1: ML Prediction */}
              <div className="rounded-2xl border border-indigo-400/25 bg-gradient-to-b from-[#161233]/90 to-[#0e0b21]/95 p-6 backdrop-blur-xl shadow-[0_15px_40px_rgba(0,0,0,0.4)] flex flex-col justify-between">
                <div>
                  <span className="text-xs tracking-[0.2em] text-indigo-300/60 uppercase block mb-4 font-sans font-medium">
                    ML Prediction
                  </span>
                  <div className="flex flex-wrap items-center gap-2 mb-4">
                    <span className={`px-3 py-1 rounded-full text-xs font-sans font-medium uppercase tracking-wider ${
                      result.prediction === "Potable"
                        ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                        : "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                    }`}>
                      {result.prediction}
                    </span>
                    <span className="px-3 py-1 rounded-full text-xs font-sans font-medium uppercase tracking-wider bg-indigo-500/20 text-indigo-300 border border-indigo-400/30">
                      {result.risk_band}
                    </span>
                  </div>
                  <p className="text-2xl font-serif text-[#edeaff] mb-3">
                    {(result.confidence * 100).toFixed(0)}% confidence
                  </p>
                </div>
                {/* Progress bar */}
                <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden mt-4">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.confidence * 100}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    className={`h-full ${result.prediction === "Potable" ? "bg-emerald-400" : "bg-rose-400"}`}
                  />
                </div>
              </div>

              {/* Card 2: WHO/EPA Cross-Check */}
              <div className="rounded-2xl border border-indigo-400/25 bg-gradient-to-b from-[#161233]/90 to-[#0e0b21]/95 p-6 backdrop-blur-xl shadow-[0_15px_40px_rgba(0,0,0,0.4)] flex flex-col justify-between">
                <div>
                  <span className="text-xs tracking-[0.2em] text-indigo-300/60 uppercase block mb-4 font-sans font-medium">
                    WHO/EPA Cross-Check
                  </span>
                  {result.who_flags.length ? (
                    <div className="space-y-2">
                      <p className="text-sm font-sans font-medium uppercase tracking-wider text-rose-300">
                        Outside typical range:
                      </p>
                      <p className="text-base text-rose-200/90 italic">
                        {result.who_flags.join(", ")}
                      </p>
                    </div>
                  ) : (
                    <p className="text-lg font-serif italic text-emerald-300">
                      All parameters within typical safe ranges
                    </p>
                  )}
                </div>
                <span className="text-xs text-indigo-300/40 uppercase font-sans tracking-widest mt-6">
                  Threshold validation
                </span>
              </div>

              {/* Card 3: Water Quality Index */}
              <div className="rounded-2xl border border-indigo-400/25 bg-gradient-to-b from-[#161233]/90 to-[#0e0b21]/95 p-6 backdrop-blur-xl shadow-[0_15px_40px_rgba(0,0,0,0.4)] flex flex-col justify-between">
                <div>
                  <span className="text-xs tracking-[0.2em] text-indigo-300/60 uppercase block mb-4 font-sans font-medium">
                    Water Quality Index
                  </span>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="border-r border-indigo-400/15 pr-3">
                      <span className="text-xs font-sans text-indigo-300/50 uppercase block mb-1">Computed</span>
                      <p className="text-2xl font-serif text-[#edeaff] mb-2">{result.wqi_computed.toFixed(1)}</p>
                      <span className="inline-block px-2 py-0.5 rounded text-[11px] font-sans uppercase tracking-wider bg-indigo-500/20 text-indigo-200 border border-indigo-400/20">
                        {result.wqi_computed_category}
                      </span>
                    </div>
                    <div className="pl-1">
                      <span className="text-xs font-sans text-indigo-300/50 uppercase block mb-1">ML-predicted</span>
                      <p className="text-2xl font-serif text-[#9ef5db] mb-2">{result.wqi_predicted.toFixed(1)}</p>
                      <span className="inline-block px-2 py-0.5 rounded text-[11px] font-sans uppercase tracking-wider bg-emerald-500/20 text-emerald-200 border border-emerald-400/20">
                        {result.wqi_predicted_category}
                      </span>
                    </div>
                  </div>
                </div>
                <span className="text-xs text-indigo-300/40 uppercase font-sans tracking-widest mt-6">
                  Reconstructed index
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      <SectionDivider />

      {/* Monitoring dashboard section */}
      <section className="mx-auto max-w-5xl px-6">
        <Reveal>
          <div className="text-center mb-12">
            <span className="text-xs uppercase tracking-[0.3em] text-indigo-400/60 font-medium block mb-3 font-sans">Batch Analytics</span>
            <h2 className="text-4xl md:text-5xl text-[#edeaff] font-normal tracking-wide">Monitoring dashboard</h2>
            <p className="text-xl md:text-2xl text-indigo-200/80 max-w-3xl mx-auto mt-4">
              Upload a CSV of samples (ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity) for a batch risk breakdown.
            </p>
            <div className="mt-4">
              <a
                href="/sample.csv"
                download="sample.csv"
                className="inline-flex items-center gap-1.5 text-base text-[#d4d1ff] hover:text-white border-b border-[#d4d1ff]/40 hover:border-white pb-0.5 transition-colors font-sans"
              >
                <span>Try sample data</span>
                <span>&darr;</span>
              </a>
            </div>
          </div>
        </Reveal>

        <Reveal delay={0.2}>
          <div className="rounded-3xl border border-indigo-400/20 bg-gradient-to-b from-[#121028]/80 to-[#0c091b]/90 p-8 md:p-12 backdrop-blur-xl shadow-[0_15px_40px_rgba(0,0,0,0.3)] text-center">
            <label className="cursor-pointer inline-flex flex-col items-center justify-center w-full border-2 border-dashed border-indigo-400/30 rounded-2xl p-12 hover:border-indigo-300/70 hover:bg-indigo-500/5 transition-all duration-300 group">
              <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">
                📁
              </div>
              <span className="text-lg tracking-[0.2em] uppercase font-sans font-medium text-indigo-100 group-hover:text-white mb-2">
                {uploading ? "Processing..." : "Click to upload a CSV"}
              </span>
              <span className="text-sm italic text-indigo-300/50 font-sans">
                Drag &amp; drop or browse local files
              </span>
              <input
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
              />
            </label>
            {uploadError && <p className="mt-6 text-sm italic text-rose-300/90 font-sans">{uploadError}</p>}
          </div>
        </Reveal>

        {/* Results: Three cards */}
        <AnimatePresence>
          {batch && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-16 space-y-10">
              {/* Card 1: Risk distribution */}
              <div className="rounded-2xl border border-indigo-400/20 bg-[#120f2b]/80 p-8 backdrop-blur-xl">
                <h3 className="text-base tracking-widest text-indigo-200 uppercase mb-8 font-sans font-medium text-center">
                  Risk distribution &middot; {batch.total} samples
                </h3>
                <DistributionChart counts={batch.category_counts} total={batch.total} />
              </div>
              
              {/* Card 2: WQI trend across samples */}
              <div className="rounded-2xl border border-indigo-400/20 bg-[#120f2b]/80 p-8 backdrop-blur-xl">
                <h3 className="text-base tracking-widest text-indigo-200 uppercase mb-2 font-sans font-medium text-center">
                  WQI trend across samples
                </h3>
                <p className="text-xs text-indigo-300/50 text-center uppercase tracking-wider mb-8 font-sans">
                  By upload row order &mdash; this dataset has no timestamps
                </p>
                <TrendChart points={batch.rows.map((r) => r.wqi)} />
              </div>

              {/* Card 3: Parameter breakdown */}
              <div className="rounded-2xl border border-indigo-400/20 bg-[#120f2b]/80 p-8 backdrop-blur-xl">
                <h3 className="text-base tracking-widest text-indigo-200 uppercase mb-8 font-sans font-medium text-center">
                  Parameter breakdown
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
                  {FIELDS.map((f) => {
                    const stats = batch.parameter_stats[f.name];
                    if (!stats) return null;
                    return (
                      <div
                        key={f.name}
                        className="rounded-xl border border-indigo-400/15 bg-white/[0.02] p-4"
                      >
                        <div className="flex justify-between items-baseline mb-2">
                          <span className="text-xs uppercase font-sans tracking-wider text-indigo-200/80 font-medium">
                            {f.label}
                          </span>
                          {f.unit !== "—" && (
                            <span className="text-[11px] font-sans italic text-indigo-300/50">
                              {f.unit}
                            </span>
                          )}
                        </div>
                        <p className="text-2xl font-serif text-[#edeaff] mb-1">
                          {stats.mean.toFixed(1)} <span className="text-xs font-sans text-indigo-300/50 uppercase tracking-widest">mean</span>
                        </p>
                        <p className="text-xs font-sans text-indigo-300/60">
                          Range: {stats.min.toFixed(1)} &ndash; {stats.max.toFixed(1)}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      {/* Footer */}
      <footer className="mt-32 border-t border-indigo-400/10 px-6 py-12 text-center text-sm md:text-base tracking-widest uppercase text-indigo-300/50 font-sans">
        Tech Utsav 2026 &middot; PS-3Y-06 &middot; Dataset: Kaggle Water Potability ({metrics?.dataset_rows ?? "3,276"} samples)
      </footer>
    </main>
  );
}