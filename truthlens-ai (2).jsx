import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  Shield, LayoutDashboard, FolderPlus, Microscope, Bot, Eye,
  FileText, History, BarChart3, Settings, Search, Bell, Sun, Moon, Upload,
  Image as ImageIcon, Video, ChevronRight, ChevronDown, Play, Pause,
  CheckCircle2, AlertTriangle, XCircle, Clock, Cpu, Activity, Zap, Pin,
  Download, Share2, GitCompare, Keyboard, Layers, Fingerprint, Brain,
  Gauge as GaugeIcon, Radio, ScanFace, FileSearch,
  CircleDot, Sparkles, Lock, Users, Database, TrendingUp, TrendingDown,
  ArrowUpRight, Plus, Flame, Network,
  ShieldAlert, UserCheck, RefreshCw, ExternalLink, Globe, Send, ThumbsDown
} from "lucide-react";
import {
  ResponsiveContainer, AreaChart, Area, LineChart, Line, BarChart, Bar,
  PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from "recharts";

/* ============================================================
   DESIGN TOKENS — TruthLens AI enterprise dark theme
   ============================================================ */
const T = {
  dark: {
    bg: "#0B1220", card: "#151E2D", card2: "#1B2638", border: "#233046",
    primary: "#4F8EF7", accent: "#00C9A7", danger: "#FF5A5F", warning: "#FFC857",
    text: "#F5F7FA", sub: "#A3B3C9", success: "#22C55E", dim: "#5B6B85",
  },
  light: {
    bg: "#F4F6FA", card: "#FFFFFF", card2: "#EEF2F8", border: "#DDE4EE",
    primary: "#2F6FE4", accent: "#00A98C", danger: "#E5484D", warning: "#D9A514",
    text: "#101827", sub: "#5A6B85", success: "#16A34A", dim: "#8A99B3",
  },
};

const font = { fontFamily: "'Inter', system-ui, -apple-system, sans-serif" };

/* ============================================================
   MOCK DATA
   ============================================================ */
const CASES = [
  { id: "TL-2026-0342", name: "Fabricated Boardroom Photo — Wire Fraud", dept: "Financial Crimes", investigator: "A. Sharma", priority: "Critical", type: "Composite Image", status: "In Progress", verdict: null, confidence: null, created: "Jul 8, 2026", evidence: 4, pinned: true },
  { id: "TL-2026-0341", name: "Fabricated Press Conference Footage", dept: "Media Integrity", investigator: "R. Verma", priority: "High", type: "Deepfake Image", status: "Completed", verdict: "FAKE", confidence: 94, created: "Jul 7, 2026", evidence: 12, pinned: true },
  { id: "TL-2026-0339", name: "KYC Onboarding Face Swap Attempt", dept: "Banking Fraud", investigator: "S. Iyer", priority: "High", type: "Deepfake Image", status: "Completed", verdict: "FAKE", confidence: 97, created: "Jul 6, 2026", evidence: 3, pinned: false },
  { id: "TL-2026-0336", name: "Ransom Note Photo Verification", dept: "Law Enforcement", investigator: "K. Nair", priority: "Critical", type: "Image Forensics", status: "Completed", verdict: "AUTHENTIC", confidence: 88, created: "Jul 5, 2026", evidence: 2, pinned: false },
  { id: "TL-2026-0333", name: "Insurance Claim Photo Tampering", dept: "Insurance", investigator: "A. Sharma", priority: "Medium", type: "Image Forensics", status: "Pending Review", verdict: "SUSPICIOUS", confidence: 71, created: "Jul 3, 2026", evidence: 8, pinned: false },
  { id: "TL-2026-0329", name: "Election Poster Forgery Attribution", dept: "Gov Affairs", investigator: "M. Das", priority: "High", type: "Synthetic Media", status: "Completed", verdict: "FAKE", confidence: 91, created: "Jul 1, 2026", evidence: 4, pinned: false },
];

const EVIDENCE = [
  { id: "EV-01", name: "boardroom_still_4k.png", kind: "image", size: "8.2 MB", hash: "a3f2…9c1d", verdict: "FAKE", confidence: 96 },
  { id: "EV-02", name: "press_photo_variant.png", kind: "image", size: "5.6 MB", hash: "77be…04aa", verdict: "FAKE", confidence: 92 },
  { id: "EV-03", name: "id_badge_scan.jpg", kind: "image", size: "1.4 MB", hash: "c9d0…e2f7", verdict: "AUTHENTIC", confidence: 89 },
  { id: "EV-04", name: "email_header_capture.png", kind: "image", size: "0.6 MB", hash: "5a41…77d2", verdict: "AUTHENTIC", confidence: 82 },
];

const AGENTS = [
  { key: "plan", name: "Investigation Planner", icon: Network, color: "#4F8EF7", role: "Understands the case & orchestrates the investigation", adapter: "A", model: "Planning Model", store: "past-case task plans",
    purpose: "Assesses the uploaded media and decides which specialist agents to run — simple cases take a short path, complex ones fan out.",
    inputs: ["Uploaded image + metadata", "Case context"], outputs: ["Dispatch plan", "Pre-detected issues"],
    steps: ["Assess media type & complexity", "Query similar past cases", "Detect obvious issues (e.g. lighting)", "Route to needed agents only"], deps: [], ms: 118, conf: 92 },
  { key: "forensic", name: "Forensic Analysis", icon: Fingerprint, color: "#00C9A7", role: "Detects low-level manipulation artifacts", adapter: "B", model: "Forensic Analysis Model", store: "generator artifact fingerprints",
    purpose: "Finds pixel- and frequency-level traces of manipulation and attributes the generator family using a mixture-of-experts head.",
    inputs: ["Image pixels", "Cached FFT/PRNU/ELA features"], outputs: ["Artifact findings", "Generator attribution", "Heatmap / ELA images"],
    steps: ["Run MoE expert heads", "FFT/DCT frequency analysis", "PRNU sensor-noise check", "ELA compression analysis"], deps: ["Investigation Planner"], ms: 842, conf: 96 },
  { key: "semantic", name: "Semantic & Context", icon: ScanFace, color: "#63D2FF", role: "Checks scene & content consistency", adapter: "B", model: "Semantic Analysis Model", store: "scene-consistency embeddings",
    purpose: "Reasons about the picture's meaning — lighting direction, reflections, object relationships — to catch physically impossible composites.",
    inputs: ["Image", "CLIP/VLM embeddings"], outputs: ["Semantic conflicts", "Consistency score"],
    steps: ["Estimate lighting direction", "Check reflections & specular highlights", "CLIP scene-consistency", "Object relationship check"], deps: ["Investigation Planner"], ms: 610, conf: 88 },
  { key: "retrieval", name: "Retrieval & Comparison", icon: FileSearch, color: "#FFC857", role: "Matches against external & known sources", adapter: "A", model: "Retrieval Intelligence Model", store: "reverse-image + generator registry",
    purpose: "Searches trusted sources and prior cases to find the origin image or related deepfakes.",
    inputs: ["Image embedding", "Generator fingerprint"], outputs: ["External matches", "Related cases"],
    steps: ["Reverse-image search", "Query known-fake DB", "Match generator toolchain", "Link prior cases"], deps: ["Forensic Analysis", "Semantic & Context"], ms: 940, conf: 90 },
  { key: "fusion", name: "Evidence Fusion & Debate", icon: Users, color: "#FF8A5F", role: "Consolidates evidence & resolves disagreements", adapter: "C", model: "Evidence Fusion Model", store: "past conflict resolutions",
    purpose: "Merges every agent's findings, weights them by reliability, and runs a debate round to reconcile any conflicts before a verdict is formed.",
    inputs: ["All agent outputs"], outputs: ["Unified evidence", "Resolved conflicts", "Consensus"],
    steps: ["Deduplicate findings", "Weight by agent reliability", "Run debate on conflicts", "Counterfactual checks"], deps: ["Forensic Analysis", "Semantic & Context", "Retrieval & Comparison"], ms: 720, conf: 94 },
  { key: "decision", name: "Decision & Confidence", icon: GaugeIcon, color: "#FF5A5F", role: "Produces the calibrated verdict", adapter: "C", model: "Decision Support Model", store: "calibration set",
    purpose: "Turns consolidated evidence into a final Real/Fake prediction with a calibrated confidence, risk level and threat score.",
    inputs: ["Unified evidence"], outputs: ["Verdict", "Calibrated confidence", "Risk level"],
    steps: ["Aggregate weighted evidence", "Calibrate confidence (ECE)", "Assign risk & threat score", "Flag low-confidence for review"], deps: ["Evidence Fusion & Debate"], ms: 260, conf: 96 },
  { key: "report", name: "Report Generation", icon: FileText, color: "#22C55E", role: "Writes the explainable forensic report", adapter: "B", model: "Report Generation Model", store: "report templates",
    purpose: "Assembles findings, visuals and reasoning into a signed, human-readable forensic report with recommendations.",
    inputs: ["Verdict + all evidence"], outputs: ["Signed forensic report", "Recommendations"],
    steps: ["Compose executive summary", "Embed visual exhibits", "Generate reasoning & recommendations", "Sign & seal"], deps: ["Decision & Confidence"], ms: 2100, conf: 95 },
];

const AGENT_LOGS = {
  plan: [
    { t: "thinking", text: "Case contains 4 image exhibits. Querying task-plan vector store for similar past cases (ChromaDB, k=5)." },
    { t: "tool", text: "Tool: vector_store.query · Adapter A (orchestration) hot-swapped in 42 ms", ms: 118 },
    { t: "evidence", text: "Nearest precedent: TL-2026-0329 (doctored proof-image fraud, cosine 0.91). Dispatch plan: Forensic ∥ Semantic → Retrieval → Fusion → Decision → Report." },
    { t: "evidence", text: "Issue pre-detected from thumbnail scan: lighting mismatch on right side of face vs background reflection." },
    { t: "output", text: "Dynamic dispatch issued — Forensic and Semantic in parallel, Retrieval conditional on artifact hits.", conf: 92 },
  ],
  forensic: [
    { t: "thinking", text: "Running mixture-of-experts heads on shared EfficientNet features: GAN expert, diffusion expert, compression expert." },
    { t: "tool", text: "Tool: moe_consensus_head · precomputed FFT/DCT + PRNU + ELA features from cache (.npy)", ms: 842 },
    { t: "evidence", text: "Expert votes — diffusion 0.71, GAN 0.19, compression 0.06, inconclusive 0.04. Consensus head verdict: diffusion-family synthesis." },
    { t: "evidence", text: "PRNU residual: sensor-noise mismatch inside face region (corr 0.09 vs 0.84 background). ELA: uniform high error in face — post-compression insertion." },
    { t: "output", text: "SYNTHETIC, generator family = diffusion. This generator was in a held-out LOGO split at validation — score remains reliable (LOGO ROC-AUC 0.981).", conf: 96 },
  ],
  semantic: [
    { t: "thinking", text: "CLIP/VLM consistency pass: object-scene plausibility, lighting direction, specular reflections, text-in-image." },
    { t: "tool", text: "Tool: lighting_estimator + clip_consistency", ms: 610 },
    { t: "evidence", text: "Key light on face: 40° left. Key light on background windows: 15° right. Reflection in glass panel does not contain the subject." },
    { t: "output", text: "2 semantic conflicts confirmed. Scene is physically inconsistent with a single capture.", conf: 88 },
  ],
  retrieval: [
    { t: "thinking", text: "Adapter A hot-swap. Reverse-search against generator registry + known real/fake reference sets." },
    { t: "tool", text: "Tool: registry_match · vector store: reverse-image embeddings", ms: 940 },
    { t: "evidence", text: "Background matches a stock boardroom photo (registry hit, sim 0.94) — subject inserted. Latent decoder fingerprint matches the generator toolchain seen in TL-2026-0329." },
    { t: "output", text: "External corroboration secured: source background identified, generator toolchain linked to prior case.", conf: 90 },
  ],
  fusion: [
    { t: "thinking", text: "Collecting all evidence. Debate round 1: Forensic (fake) vs Semantic (fake) vs Retrieval (fake) — checking for conflicts before consensus." },
    { t: "tool", text: "Tool: debate_orchestrator · Adapter C (argumentation)", ms: 720 },
    { t: "evidence", text: "Conflict raised: Semantic flagged lighting, but Forensic's compression expert scored low (0.06) — could imply clean single capture. Resolved: ELA insertion evidence outweighs; compression expert low score expected for high-quality diffusion output." },
    { t: "evidence", text: "Precedent store: 3 similar disagreements previously resolved toward 'composite' with 100% outcome accuracy." },
    { t: "output", text: "Consensus reached 4/4 weighted by reliability. Timeline: fabricated image emailed 6 min before the approval deadline — social-engineering pattern TTP-114.", conf: 94 },
  ],
  decision: [
    { t: "thinking", text: "Calibrating raw consensus against outcome-history calibration set (ECE 0.021, July window)." },
    { t: "tool", text: "Tool: confidence_calibrator", ms: 260 },
    { t: "evidence", text: "Calibrated: FAKE 96% [94.1, 97.8]. Composite threat 9.1/10 — financial approval workflow targeted." },
    { t: "output", text: "FINAL VERDICT: FAKE · HIGH RISK. Recommendation: block wire transfer, escalate to legal hold, human review of EV-04 (below auto-close threshold).", conf: 96 },
  ],
  report: [
    { t: "thinking", text: "Adapter B synthesis: mapping structured classifier outputs to generation-method-aware language per forensic glossary." },
    { t: "tool", text: "Tool: report_synthesize · ISO/IEC 27042 structure, 8 sections + annexes", ms: 2100 },
    { t: "evidence", text: "Embedded exhibits: Grad-CAM, ELA, MoE consensus chart, FFT spectrum, correlation graph, debate transcript." },
    { t: "output", text: "Report TL-2026-0342-R1 generated & signed. Explanations are generator-specific ('diffusion upsampling artifacts'), not generic.", conf: 95 },
  ],
};

const TIMELINE = [
  { label: "Evidence uploaded (5 files)", time: "10:52:04", icon: Upload },
  { label: "Metadata extracted & hashes sealed", time: "10:52:11", icon: Database },
  { label: "Face located — 1 subject, 94% quality", time: "10:52:19", icon: ScanFace },
  { label: "Deepfake ensemble executed", time: "10:52:41", icon: Cpu },
  { label: "Fusion & Debate round started", time: "10:52:58", icon: Brain },
  { label: "Fusion debate — consensus 5/5", time: "10:53:22", icon: Users },
  { label: "Forensic report generated", time: "10:53:40", icon: FileText },
];

const THREAT_TREND = [
  { d: "Jul 1", fake: 14, real: 22, susp: 5 }, { d: "Jul 2", fake: 18, real: 19, susp: 7 },
  { d: "Jul 3", fake: 11, real: 25, susp: 4 }, { d: "Jul 4", fake: 22, real: 17, susp: 9 },
  { d: "Jul 5", fake: 27, real: 20, susp: 6 }, { d: "Jul 6", fake: 19, real: 24, susp: 8 },
  { d: "Jul 7", fake: 31, real: 18, susp: 11 }, { d: "Jul 8", fake: 24, real: 21, susp: 7 },
];

const EMOTION = [
  { axis: "Neutral", v: 82 }, { axis: "Urgency", v: 64 }, { axis: "Stress", v: 31 },
  { axis: "Confidence", v: 74 }, { axis: "Fear", v: 12 }, { axis: "Anger", v: 8 },
];

const ATTACK_MIX = [
  { name: "Face swap", value: 34, c: "#4F8EF7" }, { name: "Diffusion synthesis", value: 28, c: "#B084F5" },
  { name: "GAN generation", value: 21, c: "#00C9A7" }, { name: "Splice / composite", value: 10, c: "#FFC857" },
  { name: "Other", value: 7, c: "#5B6B85" },
];

/* Deterministic pseudo-random for stable visuals */
const prng = (i, j = 0) => {
  const x = Math.sin(i * 127.1 + j * 311.7) * 43758.5453;
  return x - Math.floor(x);
};

/* ============================================================
   SHARED PRIMITIVES
   ============================================================ */
const Card = ({ c, children, style, pad = 20, onClick, hover }) => {
  const [h, setH] = useState(false);
  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}
      style={{
        background: `linear-gradient(180deg, rgba(255,255,255,.028), rgba(255,255,255,0) 46%), ${c.card}`,
        border: `1px solid ${h && hover ? c.primary + "77" : c.border}`,
        borderRadius: 16, padding: pad, transition: "all .22s cubic-bezier(.4,0,.2,1)",
        boxShadow: h && hover
          ? `0 16px 44px rgba(0,0,0,.5), 0 0 0 1px ${c.primary}22, inset 0 1px 0 rgba(255,255,255,.04)`
          : "0 1px 2px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.03)",
        transform: h && hover ? "translateY(-3px)" : "none",
        cursor: onClick ? "pointer" : "default", ...style,
      }}>
      {children}
    </div>
  );
};

const Badge = ({ c, tone = "sub", children, dot }) => {
  const map = {
    danger: [c.danger, c.danger + "1A"], success: [c.success, c.success + "1A"],
    warning: [c.warning, c.warning + "1A"], primary: [c.primary, c.primary + "1A"],
    accent: [c.accent, c.accent + "1A"], sub: [c.sub, c.sub + "1A"],
  };
  const [fg, bg] = map[tone];
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 6, background: bg, color: fg,
      fontSize: 11, fontWeight: 600, padding: "3px 9px", borderRadius: 999, letterSpacing: .3,
      whiteSpace: "nowrap",
    }}>
      {dot && <span style={{ width: 6, height: 6, borderRadius: 99, background: fg }} />}
      {children}
    </span>
  );
};

const verdictTone = (v) => v === "FAKE" ? "danger" : v === "AUTHENTIC" ? "success" : "warning";
const prioTone = (p) => p === "Critical" ? "danger" : p === "High" ? "warning" : "primary";

const SectionTitle = ({ c, icon: Icon, title, right }) => (
  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      {Icon && <Icon size={16} color={c.sub} />}
      <span style={{ fontSize: 13, fontWeight: 700, color: c.text, letterSpacing: .2 }}>{title}</span>
    </div>
    {right}
  </div>
);

const Ring = ({ c, value, size = 96, stroke = 9, color, label }) => {
  const r = (size - stroke) / 2, circ = 2 * Math.PI * r;
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} stroke={c.border} strokeWidth={stroke} fill="none" />
        <circle cx={size / 2} cy={size / 2} r={r} stroke={color} strokeWidth={stroke} fill="none"
          strokeDasharray={circ} strokeDashoffset={circ * (1 - value / 100)}
          strokeLinecap="round" transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: size / 4.5, fontWeight: 800, color: c.text }}>{value}%</span>
        {label && <span style={{ fontSize: 10, color: c.sub }}>{label}</span>}
      </div>
    </div>
  );
};

const ThreatGauge = ({ c, value }) => {
  // semicircular gauge 0-10
  const pct = value / 10;
  const angle = -90 + pct * 180;
  const color = value >= 7 ? c.danger : value >= 4 ? c.warning : c.success;
  return (
    <div style={{ position: "relative", width: 180, height: 104, margin: "0 auto" }}>
      <svg width="180" height="104" viewBox="0 0 180 104">
        <path d="M 14 96 A 76 76 0 0 1 166 96" fill="none" stroke={c.border} strokeWidth="14" strokeLinecap="round" />
        <path d="M 14 96 A 76 76 0 0 1 166 96" fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
          strokeDasharray={239} strokeDashoffset={239 * (1 - pct)} style={{ transition: "stroke-dashoffset 1.4s ease" }} />
        <g transform={`rotate(${angle} 90 96)`} style={{ transition: "transform 1.4s ease" }}>
          <line x1="90" y1="96" x2="90" y2="34" stroke={c.text} strokeWidth="3" strokeLinecap="round" />
          <circle cx="90" cy="96" r="6" fill={c.text} />
        </g>
      </svg>
      <div style={{ position: "absolute", bottom: -8, width: "100%", textAlign: "center" }}>
        <span style={{ fontSize: 26, fontWeight: 800, color }}>{value.toFixed(1)}</span>
        <span style={{ fontSize: 12, color: c.sub }}> / 10 threat</span>
      </div>
    </div>
  );
};

const Skeleton = ({ c, h = 14, w = "100%", style }) => (
  <div style={{ height: h, width: w, borderRadius: 6, background: `linear-gradient(90deg, ${c.card2} 25%, ${c.border} 50%, ${c.card2} 75%)`, backgroundSize: "200% 100%", animation: "tlshimmer 1.4s infinite", ...style }} />
);

const Toast = ({ c, toast }) => !toast ? null : (
  <div style={{
    position: "fixed", bottom: 24, right: 24, zIndex: 100, background: c.card,
    border: `1px solid ${c.border}`, borderLeft: `3px solid ${c.accent}`, color: c.text,
    padding: "12px 18px", borderRadius: 10, fontSize: 13, boxShadow: "0 10px 40px rgba(0,0,0,.5)",
    display: "flex", alignItems: "center", gap: 10, animation: "tlslideup .25s ease",
  }}>
    <CheckCircle2 size={16} color={c.accent} /> {toast}
  </div>
);

/* ============================================================
   SIMULATED VISUAL FORENSICS (SVG)
   ============================================================ */
const FaceExhibit = ({ c, overlay }) => (
  <svg viewBox="0 0 300 220" style={{ width: "100%", borderRadius: 10, background: "#0A0F1A", display: "block" }}>
    {/* abstract subject */}
    <rect x="0" y="0" width="300" height="220" fill="#101826" />
    <rect x="0" y="150" width="300" height="70" fill="#0D1420" />
    <ellipse cx="150" cy="98" rx="46" ry="56" fill="#2A3A52" />
    <ellipse cx="150" cy="188" rx="72" ry="40" fill="#22304a" />
    <ellipse cx="132" cy="88" rx="7" ry="4.5" fill="#0B1220" />
    <ellipse cx="168" cy="88" rx="7" ry="4.5" fill="#0B1220" />
    <path d="M136 122 Q150 130 164 122" stroke="#0B1220" strokeWidth="3" fill="none" strokeLinecap="round" />
    <path d="M150 96 L146 110 L154 110 Z" fill="#1E2C42" />
    {overlay === "heat" && (
      <>
        <defs>
          <radialGradient id="hot1"><stop offset="0%" stopColor="#FF5A5F" stopOpacity=".85" /><stop offset="100%" stopColor="#FF5A5F" stopOpacity="0" /></radialGradient>
          <radialGradient id="hot2"><stop offset="0%" stopColor="#FFC857" stopOpacity=".7" /><stop offset="100%" stopColor="#FFC857" stopOpacity="0" /></radialGradient>
        </defs>
        <ellipse cx="150" cy="140" rx="52" ry="26" fill="url(#hot1)" />
        <ellipse cx="118" cy="92" rx="26" ry="20" fill="url(#hot2)" />
        <ellipse cx="184" cy="96" rx="20" ry="16" fill="url(#hot2)" />
        <rect x="102" y="40" width="96" height="122" fill="none" stroke="#00C9A7" strokeWidth="1.5" strokeDasharray="5 4" rx="6" />
        <text x="106" y="34" fill="#00C9A7" fontSize="9" fontFamily="Inter">FACE 0.94</text>
        {[
          [150, 140, "Blend boundary — jawline, 14px feather · SHAP 34%"],
          [118, 92, "Specular highlight mismatch (left eye) · light source conflict"],
          [184, 96, "Reflection inconsistency (right) · absent in glass panel"],
        ].map(([x, y, tip], i) => (
          <g key={i} style={{ cursor: "pointer" }}>
            <title>{tip}</title>
            <circle cx={x} cy={y} r="9" fill="none" stroke="#F5F7FA" strokeWidth="1.5" opacity=".9">
              <animate attributeName="r" values="7;11;7" dur="1.8s" repeatCount="indefinite" />
              <animate attributeName="opacity" values=".9;.35;.9" dur="1.8s" repeatCount="indefinite" />
            </circle>
            <circle cx={x} cy={y} r="14" fill="transparent" />
          </g>
        ))}
      </>
    )}
    {overlay === "attn" && [...Array(60)].map((_, i) => {
      const x = 20 + (i % 10) * 27, y = 20 + Math.floor(i / 10) * 32;
      const d = Math.hypot(x - 150, y - 118) / 150;
      const o = Math.max(0, .75 - d) * (0.5 + prng(i) * 0.6);
      return <rect key={i} x={x} y={y} width="24" height="28" fill="#4F8EF7" opacity={o.toFixed(2)} />;
    })}
    {overlay === "ela" && [...Array(120)].map((_, i) => {
      const x = (i % 15) * 20, y = Math.floor(i / 15) * 28;
      const inFace = x > 100 && x < 200 && y > 40 && y < 165;
      const v = prng(i, 7) * (inFace ? 0.9 : 0.18);
      return <rect key={i} x={x} y={y} width="20" height="28" fill={v > 0.5 ? "#FF5A5F" : "#4F8EF7"} opacity={(v * 0.7).toFixed(2)} />;
    })}
  </svg>
);

const CorrelationGraph = ({ c }) => {
  const nodes = [
    { x: 250, y: 60, label: "Case 0342", color: c.primary, r: 26 },
    { x: 90, y: 150, label: "EV-01 image", color: c.danger, r: 18 },
    { x: 200, y: 190, label: "EV-02 variant", color: "#B084F5", r: 18 },
    { x: 330, y: 180, label: "TTP-114", color: c.warning, r: 16 },
    { x: 430, y: 110, label: "Case 0329", color: c.sub, r: 16 },
    { x: 400, y: 220, label: "Generator FP", color: c.accent, r: 14 },
  ];
  const edges = [[0, 1], [0, 2], [0, 3], [3, 4], [2, 5], [4, 5]];
  return (
    <svg viewBox="0 0 520 260" style={{ width: "100%", display: "block" }}>
      {edges.map(([a, b], i) => (
        <line key={i} x1={nodes[a].x} y1={nodes[a].y} x2={nodes[b].x} y2={nodes[b].y}
          stroke={c.border} strokeWidth="1.5" strokeDasharray={i > 2 ? "5 4" : "none"} />
      ))}
      {nodes.map((n, i) => (
        <g key={i}>
          <circle cx={n.x} cy={n.y} r={n.r} fill={n.color + "22"} stroke={n.color} strokeWidth="2" />
          <text x={n.x} y={n.y + n.r + 14} textAnchor="middle" fill={c.sub} fontSize="10" fontFamily="Inter">{n.label}</text>
        </g>
      ))}
    </svg>
  );
};

/* ============================================================
   AGENT STREAMING HOOK
   ============================================================ */
const useAgentStream = (running) => {
  const [progress, setProgress] = useState({}); // key -> count of visible entries
  useEffect(() => {
    if (!running) return;
    setProgress({});
    let step = 0;
    const order = [];
    // interleave agents in pipeline order
    const seq = ["plan", "forensic", "semantic", "retrieval", "fusion", "decision", "report"];
    seq.forEach((k, ai) => AGENT_LOGS[k].forEach((_, li) => order.push([k, li, ai])));
    order.sort((a, b) => (a[2] * 2.4 + a[1]) - (b[2] * 2.4 + b[1]));
    const iv = setInterval(() => {
      if (step >= order.length) { clearInterval(iv); return; }
      const [k] = order[step];
      setProgress(p => ({ ...p, [k]: (p[k] || 0) + 1 }));
      step++;
    }, 700);
    return () => clearInterval(iv);
  }, [running]);
  return progress;
};

const AgentEntry = ({ c, e }) => {
  const map = {
    thinking: { icon: Sparkles, color: c.sub, label: "Thinking" },
    tool: { icon: Zap, color: c.primary, label: "Tool" },
    evidence: { icon: Fingerprint, color: c.accent, label: "Evidence" },
    output: { icon: CheckCircle2, color: c.success, label: "Output" },
  };
  const m = map[e.t];
  const I = m.icon;
  return (
    <div style={{ display: "flex", gap: 10, padding: "8px 0", animation: "tlfadein .35s ease" }}>
      <I size={14} color={m.color} style={{ marginTop: 2, flexShrink: 0 }} />
      <div style={{ flex: 1 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 10, fontWeight: 700, color: m.color, letterSpacing: .8, textTransform: "uppercase" }}>{m.label}</span>
          {e.ms && <span style={{ fontSize: 10, color: c.dim }}>{e.ms} ms</span>}
          {e.conf && <Badge c={c} tone={e.conf > 90 ? "success" : "warning"}>{e.conf}% confidence</Badge>}
        </div>
        <p style={{ margin: "3px 0 0", fontSize: 12.5, lineHeight: 1.55, color: e.t === "thinking" ? c.sub : c.text, fontStyle: e.t === "thinking" ? "italic" : "normal" }}>{e.text}</p>
      </div>
    </div>
  );
};

/* Live LoRA adapter hot-swap indicator */
const ADAPTERS = { A: ["Adapter A", "orchestration", "#4F8EF7"], B: ["Adapter B", "forensic synthesis", "#00C9A7"], C: ["Adapter C", "argumentation", "#FF8A5F"] };
const AdapterSwapIndicator = ({ c }) => {
  const [i, setI] = useState(0);
  useEffect(() => { const iv = setInterval(() => setI(x => (x + 1) % 3), 2200); return () => clearInterval(iv); }, []);
  const k = ["A", "B", "C"][i];
  const [name, role, col] = ADAPTERS[k];
  return (
    <div key={k} style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6, animation: "tlfadein .4s" }}>
      <span style={{ width: 7, height: 7, borderRadius: 99, background: col, animation: "tlpulse 1.1s infinite" }} />
      <span style={{ fontSize: 10, color: c.sub }}>hot-swap: <b style={{ color: col }}>{name}</b> · {role}</span>
    </div>
  );
};

const AdapterChip = ({ a, label }) => {
  const [name, , col] = ADAPTERS[a];
  return <span title={`Specialized ${name} (adapter ${a}: ${ADAPTERS[a][1]})`} style={{ fontSize: 9, fontWeight: 800, color: col, background: col + "1C", border: `1px solid ${col}55`, borderRadius: 5, padding: "1px 6px", letterSpacing: .3, whiteSpace: "nowrap" }}>{label || "Model " + a}</span>;
};

/* ============================================================
   RBAC — roles, permissions, users, audit (mirrors backend)
   ============================================================ */
const RBAC_MODULES = ["dashboard", "investigation", "sentinel", "complaint_management", "report_generation", "user_management", "role_management", "audit_logs", "settings"];
const RBAC_PERMS = ["view", "create", "edit", "delete", "export", "download", "approve", "configure"];
const ROLE_DEFS = {
  super_admin: { label: "Super Admin", all: true },
  administrator: { label: "Administrator", all: true },
  investigation_manager: { label: "Investigation Manager", grant: { dashboard: ["view"], investigation: ["view","create","edit","approve","export"], sentinel: ["view","create","approve","export"], complaint_management: ["view","approve","download"], report_generation: ["view","download","export"], audit_logs: ["view"], settings: ["view"] } },
  investigator: { label: "Investigator", grant: { dashboard: ["view"], investigation: ["view","create","edit","export"], sentinel: ["view","create","export"], report_generation: ["view","download"], complaint_management: ["view"] } },
  analyst: { label: "Analyst", grant: { dashboard: ["view"], investigation: ["view","export"], sentinel: ["view","export"], report_generation: ["view","download"] } },
  human_reviewer: { label: "Human Reviewer", grant: { dashboard: ["view"], investigation: ["view","approve"], sentinel: ["view","approve"], report_generation: ["view"], complaint_management: ["view","approve"] } },
  auditor: { label: "Auditor", grant: { dashboard: ["view"], audit_logs: ["view","export","download"], investigation: ["view"], sentinel: ["view"] } },
  read_only: { label: "Read Only", grant: { dashboard: ["view"], investigation: ["view"], sentinel: ["view"] } },
};
const roleCan = (role, mod, perm) => {
  const d = ROLE_DEFS[role]; if (!d) return false;
  if (d.all) return true;
  return (d.grant[mod] || []).includes(perm);
};
const SEED_USERS = [
  { username: "admin", name: "A. Sharma", role: "super_admin", enabled: true, locked: false, last: "just now" },
  { username: "r.verma", name: "R. Verma", role: "investigator", enabled: true, locked: false, last: "2h ago" },
  { username: "s.iyer", name: "S. Iyer", role: "analyst", enabled: true, locked: false, last: "1d ago" },
  { username: "k.nair", name: "K. Nair", role: "human_reviewer", enabled: true, locked: false, last: "3h ago" },
  { username: "m.das", name: "M. Das", role: "auditor", enabled: false, locked: false, last: "5d ago" },
];

/* ============================================================
   NAVIGATION SHELL
   ============================================================ */
const NAV = [
  { mod: "dashboard", id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { mod: "investigation", id: "new", label: "New Investigation", icon: FolderPlus },
  { mod: "investigation", id: "workspace", label: "Investigation Workspace", icon: Microscope },
  { mod: "investigation", id: "console", label: "AI Agent Console", icon: Bot },
  { mod: "investigation", id: "explain", label: "Explainability", icon: Eye },
  { mod: "report_generation", id: "report", label: "Investigation Report", icon: FileText },
  { mod: "sentinel", id: "sentinel", label: "Sentinel Threat Intel", icon: ShieldAlert, isNew: true },
  { mod: "investigation", id: "history", label: "Case History", icon: History },
  { mod: "dashboard", id: "analytics", label: "Analytics", icon: BarChart3 },
  { id: "settings", label: "Settings", icon: Settings, mod: "settings" },
  { id: "users", label: "User Management", icon: Users, mod: "user_management", admin: true },
  { id: "roles", label: "Role Management", icon: Lock, mod: "role_management", admin: true },
  { id: "audit", label: "Audit Logs", icon: Database, mod: "audit_logs", admin: true },
];

const Sidebar = ({ c, page, setPage, collapsed, setCollapsed, currentUser }) => {
  const visible = NAV.filter(n => !n.mod || roleCan(currentUser.role, n.mod, "view"));
  return (
  <aside style={{
    width: collapsed ? 64 : 236, background: c.card, borderRight: `1px solid ${c.border}`,
    display: "flex", flexDirection: "column", transition: "width .25s ease", flexShrink: 0,
    height: "100vh", position: "sticky", top: 0,
  }}>
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "18px 16px", borderBottom: `1px solid ${c.border}` }}>
      <div style={{ width: 32, height: 32, borderRadius: 9, background: `linear-gradient(135deg, ${c.primary}, ${c.accent})`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, boxShadow: `0 4px 18px ${c.primary}66` }}>
        <Shield size={17} color="#fff" />
      </div>
      {!collapsed && (
        <div>
          <div style={{ fontSize: 14, fontWeight: 800, color: c.text, letterSpacing: -.2 }}>TruthLens AI</div>
          <div style={{ fontSize: 10, color: c.sub }}>Multimedia Forensics</div>
        </div>
      )}
    </div>
    <nav style={{ flex: 1, padding: "10px 8px", overflowY: "auto" }}>
      {visible.map((n, idx) => {
        const active = page === n.id;
        const showDivider = n.admin && (idx === 0 || !visible[idx-1].admin);
        return (
          <React.Fragment key={n.id}>
          {showDivider && !collapsed && <div style={{ fontSize: 9.5, fontWeight: 800, color: c.dim, letterSpacing: 1, padding: "14px 12px 6px", textTransform: "uppercase" }}>Administration</div>}
          {showDivider && collapsed && <div style={{ height: 1, background: c.border, margin: "8px 6px" }} />}
          <button onClick={() => setPage(n.id)} title={n.label} style={{
            display: "flex", alignItems: "center", gap: 11, width: "100%", padding: collapsed ? "10px" : "9px 12px",
            justifyContent: collapsed ? "center" : "flex-start",
            background: active ? `linear-gradient(90deg, ${c.primary}2E, ${c.primary}0A)` : "transparent", border: "none",
            boxShadow: active ? `inset 3px 0 0 ${c.primary}` : "none",
            borderRadius: 9, cursor: "pointer", marginBottom: 2, transition: "background .15s",
            color: active ? c.primary : c.sub, ...font,
          }}
            onMouseEnter={e => !active && (e.currentTarget.style.background = c.card2)}
            onMouseLeave={e => !active && (e.currentTarget.style.background = "transparent")}>
            <n.icon size={17} />
            {!collapsed && <span style={{ fontSize: 12.5, fontWeight: active ? 700 : 500 }}>{n.label}</span>}
            {!collapsed && n.isNew && <span style={{ marginLeft: "auto", fontSize: 8.5, fontWeight: 800, color: c.accent, background: c.accent + "1F", border: `1px solid ${c.accent}55`, borderRadius: 5, padding: "1px 5px", letterSpacing: .6 }}>EXT</span>}
            {!collapsed && active && !n.isNew && <CircleDot size={8} style={{ marginLeft: "auto" }} />}
          </button>
          </React.Fragment>
        );
      })}
    </nav>
    <div style={{ padding: 12, borderTop: `1px solid ${c.border}` }}>
      {!collapsed && (
        <Card c={{ ...c, card: c.card2 }} pad={12} style={{ marginBottom: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <Cpu size={13} color={c.accent} />
            <span style={{ fontSize: 11, fontWeight: 700, color: c.text }}>SLM Backbone</span>
            <Badge c={c} tone="success" dot>Live</Badge>
          </div>
          <div style={{ fontSize: 10.5, color: c.text, fontWeight: 600 }}>Qwen-2.5-7B · vLLM</div>
          <AdapterSwapIndicator c={c} />
          <div style={{ height: 5, borderRadius: 99, background: c.border, marginTop: 7 }}>
            <div style={{ width: "62%", height: "100%", borderRadius: 99, background: c.accent }} />
          </div>
          <div style={{ fontSize: 10, color: c.sub, marginTop: 5 }}>62% VRAM · single GPU · 3 LoRA adapters</div>
        </Card>
      )}
      <button onClick={() => setCollapsed(!collapsed)} style={{ width: "100%", background: "transparent", border: `1px solid ${c.border}`, borderRadius: 8, padding: 7, cursor: "pointer", color: c.sub }}>
        <ChevronRight size={14} style={{ transform: collapsed ? "none" : "rotate(180deg)", transition: "transform .2s" }} />
      </button>
    </div>
  </aside>
  );
};

const Topbar = ({ c, dark, setDark, setPage, notify, currentUser, onLogout }) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [q, setQ] = useState("");
  return (
    <header style={{
      height: 58, display: "flex", alignItems: "center", gap: 14, padding: "0 24px",
      borderBottom: `1px solid ${c.border}`, background: c.bg + "D9", position: "sticky", top: 0, zIndex: 50,
      backdropFilter: "blur(14px) saturate(140%)", WebkitBackdropFilter: "blur(14px) saturate(140%)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, background: c.card, border: `1px solid ${c.border}`, borderRadius: 9, padding: "7px 12px", width: 340 }}>
        <Search size={14} color={c.dim} />
        <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search cases, evidence, hashes…  (⌘K)"
          style={{ background: "transparent", border: "none", outline: "none", color: c.text, fontSize: 12.5, width: "100%", ...font }} />
      </div>
      <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
        <Badge c={c} tone="success" dot>All models healthy</Badge>
        <button onClick={() => notify("3 critical alerts — 2 pending reviews assigned to you")} style={{ background: c.card, border: `1px solid ${c.border}`, borderRadius: 9, padding: 8, cursor: "pointer", position: "relative", color: c.sub }}>
          <Bell size={15} />
          <span style={{ position: "absolute", top: 5, right: 5, width: 7, height: 7, borderRadius: 99, background: c.danger }} />
        </button>
        <button onClick={() => setDark(!dark)} style={{ background: c.card, border: `1px solid ${c.border}`, borderRadius: 9, padding: 8, cursor: "pointer", color: c.sub }}>
          {dark ? <Sun size={15} /> : <Moon size={15} />}
        </button>
        <button onClick={() => setPage("new")} style={{
          display: "flex", alignItems: "center", gap: 7, background: `linear-gradient(135deg, ${c.primary}, #3B7BE0)`, color: "#fff",
          border: "none", borderRadius: 9, padding: "8px 15px", fontSize: 12.5, fontWeight: 700, cursor: "pointer", boxShadow: `0 4px 16px ${c.primary}4D`, ...font,
        }}>
          <Plus size={14} /> New Investigation
        </button>
        <div style={{ position: "relative" }}>
          <button onClick={() => setMenuOpen(o => !o)} style={{ display: "flex", alignItems: "center", gap: 8, background: c.card, border: `1px solid ${c.border}`, borderRadius: 9, padding: "5px 8px 5px 5px", cursor: "pointer", ...font }}>
            <div style={{ width: 28, height: 28, borderRadius: 99, background: `linear-gradient(135deg, ${c.primary}, ${c.accent})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800, color: "#fff" }}>{currentUser.name.split(" ").map(x => x[0]).join("")}</div>
            <div style={{ textAlign: "left" }}><div style={{ fontSize: 11.5, fontWeight: 700, color: c.text, lineHeight: 1.1 }}>{currentUser.name}</div><div style={{ fontSize: 9.5, color: c.sub }}>{ROLE_DEFS[currentUser.role].label}</div></div>
            <ChevronDown size={13} color={c.dim} />
          </button>
          {menuOpen && <>
            <div onClick={() => setMenuOpen(false)} style={{ position: "fixed", inset: 0, zIndex: 40 }} />
            <div style={{ position: "absolute", top: 42, right: 0, width: 200, background: c.card, border: `1px solid ${c.border}`, borderRadius: 11, boxShadow: "0 12px 40px rgba(0,0,0,.5)", zIndex: 41, overflow: "hidden", animation: "tlfadein .15s" }}>
              <div style={{ padding: "12px 14px", borderBottom: `1px solid ${c.border}` }}><div style={{ fontSize: 12.5, fontWeight: 700, color: c.text }}>{currentUser.name}</div><div style={{ fontSize: 10.5, color: c.sub, fontFamily: "monospace" }}>{currentUser.username}</div></div>
              <button onClick={() => { setPage("settings"); setMenuOpen(false); }} style={{ display: "flex", gap: 9, alignItems: "center", width: "100%", padding: "10px 14px", background: "none", border: "none", cursor: "pointer", color: c.text, fontSize: 12.5, ...font }} onMouseEnter={e => e.currentTarget.style.background = c.card2} onMouseLeave={e => e.currentTarget.style.background = "none"}><Settings size={14} color={c.sub} /> Settings</button>
              <button onClick={onLogout} style={{ display: "flex", gap: 9, alignItems: "center", width: "100%", padding: "10px 14px", background: "none", border: "none", cursor: "pointer", color: c.danger, fontSize: 12.5, fontWeight: 600, ...font }} onMouseEnter={e => e.currentTarget.style.background = c.danger + "12"} onMouseLeave={e => e.currentTarget.style.background = "none"}><Lock size={14} /> Sign out</button>
            </div>
          </>}
        </div>
      </div>
    </header>
  );
};

/* ============================================================
   PAGE: DASHBOARD
   ============================================================ */
const StatCard = ({ c, icon: Icon, label, value, delta, up, tone = "primary" }) => (
  <Card c={c} hover>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
      <div>
        <div style={{ fontSize: 11.5, color: c.sub, fontWeight: 600, marginBottom: 6 }}>{label}</div>
        <div style={{ fontSize: 30, fontWeight: 800, color: c.text, letterSpacing: -.8, fontFamily: "'Space Grotesk', Inter, sans-serif", fontVariantNumeric: "tabular-nums" }}>{value}</div>
        <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 6, fontSize: 11.5, color: up ? c.success : c.danger }}>
          {up ? <TrendingUp size={13} /> : <TrendingDown size={13} />} {delta}
          <span style={{ color: c.dim }}>vs last week</span>
        </div>
      </div>
      <div style={{ background: c[tone] + "1A", padding: 10, borderRadius: 11 }}>
        <Icon size={19} color={c[tone]} />
      </div>
    </div>
  </Card>
);

const Dashboard = ({ c, setPage, notify }) => (
  <div style={{ display: "grid", gap: 18 }}>
    <div>
      <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text, letterSpacing: -.4 }}>Forensic Operations Center</h1>
      <p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub }}>Wednesday, July 8, 2026 · 14 investigations today · SOC-2 audit mode active</p>
    </div>

    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: 14 }}>
      <StatCard c={c} icon={Flame} label="Critical alerts" value="3" delta="+2" up={false} tone="danger" />
      <StatCard c={c} icon={Microscope} label="Today's investigations" value="14" delta="+18%" up tone="primary" />
      <StatCard c={c} icon={CheckCircle2} label="Detection success rate" value="97.2%" delta="+0.4%" up tone="accent" />
      <StatCard c={c} icon={Clock} label="Pending reviews" value="5" delta="-3" up tone="warning" />
    </div>

    <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 18 }}>
      <Card c={c}>
        <SectionTitle c={c} icon={Activity} title="Threat detections — last 8 days"
          right={<Badge c={c} tone="danger" dot>Fake volume up 29%</Badge>} />
        <ResponsiveContainer width="100%" height={230}>
          <AreaChart data={THREAT_TREND}>
            <defs>
              <linearGradient id="gf" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={c.danger} stopOpacity=".35" /><stop offset="100%" stopColor={c.danger} stopOpacity="0" /></linearGradient>
              <linearGradient id="gr" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={c.accent} stopOpacity=".3" /><stop offset="100%" stopColor={c.accent} stopOpacity="0" /></linearGradient>
            </defs>
            <CartesianGrid stroke={c.border} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="d" stroke={c.dim} fontSize={11} tickLine={false} axisLine={false} />
            <YAxis stroke={c.dim} fontSize={11} tickLine={false} axisLine={false} width={28} />
            <Tooltip contentStyle={{ background: c.card2, border: `1px solid ${c.border}`, borderRadius: 10, fontSize: 12 }} labelStyle={{ color: c.text }} />
            <Area type="monotone" dataKey="fake" name="Confirmed fake" stroke={c.danger} fill="url(#gf)" strokeWidth={2} />
            <Area type="monotone" dataKey="real" name="Authentic" stroke={c.accent} fill="url(#gr)" strokeWidth={2} />
            <Area type="monotone" dataKey="susp" name="Suspicious" stroke={c.warning} fill="none" strokeWidth={2} strokeDasharray="5 4" />
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      <div style={{ display: "grid", gap: 14, alignContent: "start" }}>
        <Card c={c}>
          <SectionTitle c={c} icon={Cpu} title="System health" />
          {[["Detection ensemble", 99, c.success], ["XAI pipeline (Grad-CAM/SHAP)", 97, c.success], ["Agent orchestrator", 100, c.success], ["GPU utilization", 62, c.primary], ["Queue depth", 18, c.warning]].map(([l, v, col]) => (
            <div key={l} style={{ marginBottom: 11 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, marginBottom: 4 }}>
                <span style={{ color: c.sub }}>{l}</span><span style={{ color: c.text, fontWeight: 700 }}>{v}%</span>
              </div>
              <div style={{ height: 5, borderRadius: 99, background: c.border }}>
                <div style={{ width: `${v}%`, height: "100%", borderRadius: 99, background: col, transition: "width 1s ease" }} />
              </div>
            </div>
          ))}
        </Card>
        <Card c={c}>
          <SectionTitle c={c} icon={Zap} title="Quick actions" />
          {[["Create new investigation", FolderPlus, () => setPage("new")], ["Open agent console", Bot, () => setPage("console")], ["Review pending cases", Clock, () => setPage("history")], ["Export analytics", Download, () => notify("Analytics export queued — you'll be notified when ready")]].map(([l, I, fn]) => (
            <button key={l} onClick={fn} style={{ display: "flex", alignItems: "center", gap: 10, width: "100%", padding: "9px 10px", background: "transparent", border: "none", borderRadius: 8, cursor: "pointer", color: c.text, fontSize: 12.5, ...font, textAlign: "left" }}
              onMouseEnter={e => e.currentTarget.style.background = c.card2} onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
              <I size={15} color={c.primary} /> {l} <ArrowUpRight size={13} color={c.dim} style={{ marginLeft: "auto" }} />
            </button>
          ))}
        </Card>
      </div>
    </div>

    <Card c={c}>
      <SectionTitle c={c} icon={History} title="Recent cases"
        right={<button onClick={() => setPage("history")} style={{ background: "none", border: "none", color: c.primary, fontSize: 12, fontWeight: 600, cursor: "pointer", ...font }}>View all →</button>} />
      <CaseTable c={c} rows={CASES.slice(0, 4)} setPage={setPage} />
    </Card>
  </div>
);

const CaseTable = ({ c, rows, setPage }) => (
  <div style={{ overflowX: "auto" }}>
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
      <thead>
        <tr>{["Case", "Type", "Priority", "Investigator", "Verdict", "Status", ""].map(h => (
          <th key={h} style={{ textAlign: "left", padding: "8px 10px", color: c.dim, fontSize: 10.5, fontWeight: 700, letterSpacing: .8, textTransform: "uppercase", borderBottom: `1px solid ${c.border}` }}>{h}</th>
        ))}</tr>
      </thead>
      <tbody>
        {rows.map(r => (
          <tr key={r.id} onClick={() => setPage("workspace")} style={{ cursor: "pointer" }}
            onMouseEnter={e => e.currentTarget.style.background = c.card2} onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
            <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}>
              <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                {r.pinned && <Pin size={12} color={c.warning} />}
                <div>
                  <div style={{ fontWeight: 700, color: c.text }}>{r.name}</div>
                  <div style={{ fontSize: 10.5, color: c.dim, fontFamily: "monospace" }}>{r.id} · {r.evidence} exhibits</div>
                </div>
              </div>
            </td>
            <td style={{ padding: "11px 10px", color: c.sub, borderBottom: `1px solid ${c.border}` }}>{r.type}</td>
            <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}><Badge c={c} tone={prioTone(r.priority)}>{r.priority}</Badge></td>
            <td style={{ padding: "11px 10px", color: c.sub, borderBottom: `1px solid ${c.border}` }}>{r.investigator}</td>
            <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}>
              {r.verdict ? <Badge c={c} tone={verdictTone(r.verdict)}>{r.verdict} {r.confidence}%</Badge> : <Badge c={c} tone="primary" dot>Analyzing</Badge>}
            </td>
            <td style={{ padding: "11px 10px", color: c.sub, borderBottom: `1px solid ${c.border}` }}>{r.status}</td>
            <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}><ChevronRight size={14} color={c.dim} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

/* ============================================================
   PAGE: NEW INVESTIGATION
   ============================================================ */
const Field = ({ c, label, children }) => (
  <label style={{ display: "block" }}>
    <span style={{ fontSize: 11.5, fontWeight: 700, color: c.sub, display: "block", marginBottom: 6 }}>{label}</span>
    {children}
  </label>
);

const inputStyle = (c) => ({
  width: "100%", background: c.card2, border: `1px solid ${c.border}`, borderRadius: 9,
  padding: "10px 12px", color: c.text, fontSize: 13, outline: "none", boxSizing: "border-box", ...font,
});

const NewInvestigation = ({ c, setPage, notify }) => {
  const [files, setFiles] = useState([]);
  const [drag, setDrag] = useState(false);
  const [priority, setPriority] = useState("Critical");
  const addMock = (kind) => setFiles(f => [...f, {
    name: `exhibit_${f.length + 1}.png`,
    kind, size: (0.4 + prng(f.length) * 8).toFixed(1) + " MB", progress: 0,
  }]);
  useEffect(() => {
    if (!files.some(f => f.progress < 100)) return;
    const iv = setInterval(() => setFiles(fs => fs.map(f => f.progress < 100 ? { ...f, progress: Math.min(100, f.progress + 8 + prng(f.progress) * 14) } : f)), 120);
    return () => clearInterval(iv);
  }, [files.length]);

  return (
    <div style={{ maxWidth: 920, display: "grid", gap: 18 }}>
      <div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>New Investigation</h1>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub }}>All evidence is hashed (SHA-256) and sealed to an immutable chain of custody on upload.</p>
      </div>
      <Card c={c}>
        <SectionTitle c={c} icon={FolderPlus} title="Case details" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <Field c={c} label="Case name"><input style={inputStyle(c)} defaultValue="Fabricated Boardroom Photo — Wire Fraud" /></Field>
          <Field c={c} label="Investigator"><input style={inputStyle(c)} defaultValue="A. Sharma (Badge #4417)" /></Field>
          <Field c={c} label="Department">
            <select style={inputStyle(c)}><option>Financial Crimes</option><option>Law Enforcement</option><option>Media Integrity</option><option>Banking Fraud</option></select>
          </Field>
          <Field c={c} label="Incident type">
            <select style={inputStyle(c)}><option>Deepfake Image</option><option>Composite / Face Swap</option><option>Fully Synthetic Media</option><option>Coordinated Campaign</option></select>
          </Field>
          <Field c={c} label="Case priority">
            <div style={{ display: "flex", gap: 8 }}>
              {["Critical", "High", "Medium", "Low"].map(p => (
                <button key={p} onClick={() => setPriority(p)} style={{
                  flex: 1, padding: "9px 0", borderRadius: 8, cursor: "pointer", fontSize: 12, fontWeight: 700, ...font,
                  background: priority === p ? (p === "Critical" ? c.danger : p === "High" ? c.warning : c.primary) + "22" : c.card2,
                  color: priority === p ? (p === "Critical" ? c.danger : p === "High" ? c.warning : c.primary) : c.sub,
                  border: `1px solid ${priority === p ? (p === "Critical" ? c.danger : p === "High" ? c.warning : c.primary) : c.border}`,
                }}>{p}</button>
              ))}
            </div>
          </Field>
          <Field c={c} label="Linked ticket (optional)"><input style={inputStyle(c)} placeholder="e.g. FRAUD-2291" /></Field>
        </div>
        <div style={{ marginTop: 14 }}>
          <Field c={c} label="Description">
            <textarea rows={3} style={{ ...inputStyle(c), resize: "vertical" }} defaultValue="Treasury received an email authorizing a ₹4.2 Cr wire transfer, attaching a boardroom photo as 'proof of meeting' plus a press-photo variant. Verify authenticity of all image exhibits." />
          </Field>
        </div>
      </Card>

      <Card c={c}>
        <SectionTitle c={c} icon={Upload} title="Evidence upload" right={<Badge c={c} tone="accent">Images · Video (soon)</Badge>} />
        <div
          onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
          onDrop={e => { e.preventDefault(); setDrag(false); addMock("image"); }}
          style={{
            border: `2px dashed ${drag ? c.primary : c.border}`, borderRadius: 12, padding: "36px 20px",
            textAlign: "center", background: drag ? c.primary + "0D" : c.card2, transition: "all .2s",
          }}>
          <Upload size={26} color={drag ? c.primary : c.dim} style={{ margin: "0 auto 10px", display: "block" }} />
          <div style={{ fontSize: 13.5, fontWeight: 700, color: c.text }}>Drag & drop evidence, or add exhibits below</div>
          <div style={{ fontSize: 11.5, color: c.sub, marginTop: 4 }}>PNG · JPG · WEBP · TIFF — up to 2 GB per exhibit</div>
          <div style={{ display: "flex", gap: 10, justifyContent: "center", marginTop: 16 }}>
            <button onClick={() => addMock("image")} style={{ display: "flex", gap: 6, alignItems: "center", background: c.card, border: `1px solid ${c.border}`, borderRadius: 8, padding: "8px 14px", color: c.text, fontSize: 12, fontWeight: 600, cursor: "pointer", ...font }}><ImageIcon size={14} color={c.primary} /> Add image</button>            <button disabled style={{ display: "flex", gap: 6, alignItems: "center", background: c.card, border: `1px dashed ${c.border}`, borderRadius: 8, padding: "8px 14px", color: c.dim, fontSize: 12, fontWeight: 600, cursor: "not-allowed", ...font }}><Video size={14} /> Video — Q4 2026</button>
          </div>
        </div>
        {files.length > 0 && (
          <div style={{ marginTop: 14, display: "grid", gap: 8 }}>
            {files.map((f, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, background: c.card2, borderRadius: 9, padding: "10px 14px", animation: "tlfadein .3s" }}>
                <ImageIcon size={16} color={c.primary} />
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}>
                    <span style={{ color: c.text, fontWeight: 600 }}>{f.name}</span>
                    <span style={{ color: c.sub }}>{f.progress >= 100 ? "Hashed & sealed ✓" : Math.round(f.progress) + "%"}</span>
                  </div>
                  <div style={{ height: 4, borderRadius: 99, background: c.border, marginTop: 5 }}>
                    <div style={{ width: `${f.progress}%`, height: "100%", borderRadius: 99, background: f.progress >= 100 ? c.success : c.primary, transition: "width .15s" }} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
        <button onClick={() => notify("Draft saved to case queue")} style={{ background: "transparent", border: `1px solid ${c.border}`, color: c.sub, borderRadius: 9, padding: "10px 18px", fontSize: 13, fontWeight: 600, cursor: "pointer", ...font }}>Save draft</button>
        <button onClick={() => { notify("Investigation TL-2026-0342 opened — agents dispatched"); setPage("workspace"); }} style={{ display: "flex", alignItems: "center", gap: 8, background: `linear-gradient(135deg, ${c.primary}, #3B7BE0)`, border: "none", color: "#fff", borderRadius: 9, padding: "10px 20px", fontSize: 13, fontWeight: 700, cursor: "pointer", boxShadow: `0 4px 16px ${c.primary}4D`, ...font }}>
          <Play size={14} /> Launch investigation
        </button>
      </div>
    </div>
  );
};


/* Six anomaly-class thumbnails (matches EADIS reference report, section B) */
const ANOM_THUMBS = [
  ["Lighting mismatch", "light"], ["Reflection inconsistency", "reflect"], ["Blending artifacts", "blend"],
  ["Boundary anomaly", "bound"], ["Noise pattern deviation", "noise"], ["Frequency anomaly", "freq"],
];
const AnomThumb = ({ c, kind, label }) => (
  <div style={{ textAlign: "center" }}>
    <svg viewBox="0 0 90 60" style={{ width: "100%", borderRadius: 8, background: "#0A0F1A", display: "block", border: `1px solid ${c.border}` }}>
      <ellipse cx="45" cy="30" rx="16" ry="20" fill="#2A3A52" />
      {kind === "light" && <><path d="M45 10 A16 20 0 0 1 45 50 Z" fill="#FFC857" opacity=".45" /><line x1="45" y1="6" x2="45" y2="54" stroke="#FFC857" strokeDasharray="3 3" strokeWidth="1" /></>}
      {kind === "reflect" && <><rect x="64" y="12" width="20" height="36" fill="#16223A" rx="2" /><ellipse cx="74" cy="30" rx="6" ry="9" fill="#2A3A52" opacity=".25" /><text x="74" y="33" textAnchor="middle" fill="#FF5A5F" fontSize="10">✕</text></>}
      {kind === "blend" && <path d="M31 38 Q45 48 59 38" stroke="#FF5A5F" strokeWidth="2" fill="none" strokeDasharray="4 3" />}
      {kind === "bound" && <rect x="27" y="8" width="36" height="44" fill="none" stroke="#00C9A7" strokeWidth="1.5" strokeDasharray="4 3" rx="3" />}
      {kind === "noise" && [...Array(40)].map((_, i) => <circle key={i} cx={8 + prng(i) * 74} cy={6 + prng(i, 3) * 48} r=".9" fill={prng(i, 7) > .5 ? "#FF5A5F" : "#4F8EF7"} opacity=".8" />)}
      {kind === "freq" && [...Array(14)].map((_, i) => { const h = 6 + prng(i, 9) * 22 + (i === 9 ? 22 : 0); return <rect key={i} x={7 + i * 6} y={54 - h} width="4" height={h} fill={i === 9 ? "#FF5A5F" : "#4F8EF7"} opacity=".85" />; })}
    </svg>
    <div style={{ fontSize: 9.5, color: c.sub, marginTop: 5, lineHeight: 1.3 }}>{label}</div>
  </div>
);

/* Evidence summary counts (EADIS reference report, section C) */
const EVIDENCE_COUNTS = [
  ["Artifact anomalies", 18, "danger"], ["Face inconsistencies", 12, "danger"], ["Lighting issues", 9, "warning"],
  ["Compression traces", 11, "warning"], ["Semantic conflicts", 7, "warning"], ["External matches", 4, "accent"], ["Metadata issues", 5, "primary"],
];
const EvidenceCounts = ({ c, compact }) => (
  <div style={{ display: "grid", gridTemplateColumns: compact ? "repeat(7, 1fr)" : "repeat(auto-fit, minmax(120px, 1fr))", gap: 10 }}>
    {EVIDENCE_COUNTS.map(([l, n, tone]) => (
      <div key={l} style={{ background: c.card2, borderRadius: 10, padding: "12px 8px", textAlign: "center", borderTop: `2px solid ${c[tone]}` }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: c[tone], fontFamily: "'Space Grotesk', Inter, sans-serif" }}>{n}</div>
        <div style={{ fontSize: 10, color: c.sub, marginTop: 3, lineHeight: 1.3 }}>{l}</div>
      </div>
    ))}
  </div>
);

/* Interactive robustness stress lab — brownie point 5, live */
const STRESS_MODES = [
  ["none", "Original", 96.0], ["jpeg", "JPEG q70", 94.8], ["noise", "Gaussian noise", 93.6], ["blur", "Blur 1.5px", 92.1], ["resize", "Downscale 0.5×", 93.9],
];
const StressLab = ({ c }) => {
  const [mode, setMode] = useState("none");
  const score = STRESS_MODES.find(m => m[0] === mode)[2];
  const filt = mode === "blur" ? "blur(1.4px)" : mode === "jpeg" ? "contrast(1.08) saturate(.9)" : mode === "resize" ? "blur(.7px) contrast(1.04)" : "none";
  return (
    <Card c={c}>
      <SectionTitle c={c} icon={Shield} title="Robustness stress lab — attack the evidence"
        right={<Badge c={c} tone={score > 90 ? "success" : "warning"}>verdict unchanged</Badge>} />
      <div style={{ position: "relative", filter: filt, transition: "filter .3s" }}>
        <FaceExhibit c={c} overlay="heat" />
        {mode === "noise" && (
          <svg viewBox="0 0 300 220" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
            {[...Array(260)].map((_, i) => <rect key={i} x={prng(i) * 300} y={prng(i, 3) * 220} width="1.6" height="1.6" fill={prng(i, 7) > .5 ? "#fff" : "#000"} opacity=".28" />)}
          </svg>
        )}
        {mode === "jpeg" && (
          <svg viewBox="0 0 300 220" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", opacity: .18 }}>
            {[...Array(15 * 11)].map((_, i) => <rect key={i} x={(i % 15) * 20} y={Math.floor(i / 15) * 20} width="20" height="20" fill="none" stroke="#000" strokeWidth=".6" />)}
          </svg>
        )}
      </div>
      <div style={{ display: "flex", gap: 7, marginTop: 12, flexWrap: "wrap" }}>
        {STRESS_MODES.map(([k, l]) => (
          <button key={k} onClick={() => setMode(k)} style={{
            padding: "7px 12px", borderRadius: 8, fontSize: 11.5, fontWeight: 600, cursor: "pointer", ...font,
            background: mode === k ? c.primary + "1F" : "transparent", color: mode === k ? c.primary : c.sub,
            border: `1px solid ${mode === k ? c.primary + "66" : c.border}`,
          }}>{l}</button>
        ))}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 14, background: c.card2, borderRadius: 10, padding: "12px 16px" }}>
        <span style={{ fontSize: 26, fontWeight: 800, color: score > 90 ? c.danger : c.warning, fontFamily: "'Space Grotesk', Inter, sans-serif" }}>{score.toFixed(1)}%</span>
        <div style={{ fontSize: 11.5, color: c.sub, lineHeight: 1.5 }}>
          fake probability under this manipulation · Δ {(96 - score).toFixed(1)} pts from original.
          The MoE ensemble was trained with augmentation sweeps, so unseen recompression or noise cannot wash out the generator fingerprint.
        </div>
      </div>
    </Card>
  );
};

/* Failure analysis & self-reflection — brownie point 1 */
const FailureAnalysis = ({ c }) => (
  <Card c={c}>
    <SectionTitle c={c} icon={AlertTriangle} title="Failure analysis & self-reflection" right={<Badge c={c} tone="warning">1 exhibit flagged for human review</Badge>} />
    <div style={{ background: c.card2, borderRadius: 10, padding: 12, marginBottom: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <ImageIcon size={14} color={c.primary} />
        <span style={{ fontSize: 12.5, fontWeight: 700, color: c.text }}>EV-04 · email_header_capture.png — 82%</span>
        <Badge c={c} tone="warning">below 90% auto-close threshold</Badge>
      </div>
      <div style={{ fontSize: 11.5, color: c.sub, lineHeight: 1.6 }}>
        Why uncertain: screenshot-of-a-screenshot provenance destroys sensor noise (PRNU unusable) and the crop removes EXIF. The system <b style={{ color: c.text }}>knows what it cannot know</b> — routed to analyst queue rather than forced into a verdict.
      </div>
    </div>
    <div style={{ fontSize: 11.5, color: c.sub, lineHeight: 1.7 }}>
      <b style={{ color: c.text }}>Stated limitations:</b> heavy re-editing after generation can mask decoder fingerprints (mitigated by semantic + retrieval agents); registry coverage is bounded by known generators (unseen families fall back to LOGO-validated artifact heads); uncertainty is split and reported — model (epistemic) ±1.4%, data quality (aleatoric) ±0.8%.
    </div>
  </Card>
);

/* Live pipeline DAG — nodes light up as agents complete, edges carry flowing dashes */
const AgentDetailPanel = ({ c, agent, onClose }) => {
  if (!agent) return null;
  const Row = ({ label, children }) => (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 10, fontWeight: 800, color: c.dim, letterSpacing: 1, marginBottom: 6 }}>{label.toUpperCase()}</div>
      <div style={{ fontSize: 12.5, color: c.text, lineHeight: 1.6 }}>{children}</div>
    </div>
  );
  return (
    <>
      <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.5)", zIndex: 60, animation: "tlfadein .2s" }} />
      <div style={{ position: "fixed", top: 0, right: 0, bottom: 0, width: 420, maxWidth: "92vw", background: c.card, borderLeft: `1px solid ${c.border}`, zIndex: 61, boxShadow: "-20px 0 60px rgba(0,0,0,.5)", overflowY: "auto", animation: "tlslidein .28s cubic-bezier(.4,0,.2,1)" }}>
        <div style={{ position: "sticky", top: 0, background: c.card, borderBottom: `1px solid ${c.border}`, padding: "18px 20px", display: "flex", alignItems: "center", gap: 12, zIndex: 1 }}>
          <div style={{ width: 40, height: 40, borderRadius: 11, background: agent.color + "1F", display: "flex", alignItems: "center", justifyContent: "center" }}><agent.icon size={20} color={agent.color} /></div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 15, fontWeight: 800, color: c.text }}>{agent.name}</div>
            <div style={{ fontSize: 11, color: c.sub }}>{agent.role}</div>
          </div>
          <button onClick={onClose} style={{ background: c.card2, border: `1px solid ${c.border}`, borderRadius: 8, width: 30, height: 30, cursor: "pointer", color: c.sub, fontSize: 16, lineHeight: 1 }}>×</button>
        </div>
        <div style={{ padding: 20 }}>
          <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
            <AdapterChip a={agent.adapter} label={agent.model} />
            <Badge c={c} tone="sub">{agent.ms} ms avg</Badge>
            <Badge c={c} tone={agent.conf > 90 ? "success" : "warning"}>{agent.conf}% confidence</Badge>
          </div>
          <Row label="Purpose">{agent.purpose}</Row>
          <Row label="AI model used">{agent.model} <span style={{ color: c.dim }}>· specialized on a shared open-source backbone, hot-swapped at runtime (adapter {agent.adapter}).</span></Row>
          <Row label="Inputs">{agent.inputs.map((x, i) => <span key={i} style={{ display: "inline-block", background: c.card2, borderRadius: 6, padding: "3px 8px", margin: "0 5px 5px 0", fontSize: 11 }}>{x}</span>)}</Row>
          <Row label="Outputs">{agent.outputs.map((x, i) => <span key={i} style={{ display: "inline-block", background: agent.color + "18", color: agent.color, borderRadius: 6, padding: "3px 8px", margin: "0 5px 5px 0", fontSize: 11, fontWeight: 600 }}>{x}</span>)}</Row>
          <Row label="Internal workflow / processing steps">
            <div style={{ display: "grid", gap: 7 }}>
              {agent.steps.map((st, i) => (
                <div key={i} style={{ display: "flex", gap: 9, alignItems: "flex-start" }}>
                  <span style={{ width: 18, height: 18, borderRadius: 99, background: agent.color + "22", color: agent.color, fontSize: 10, fontWeight: 800, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 1 }}>{i + 1}</span>
                  <span style={{ fontSize: 12 }}>{st}</span>
                </div>
              ))}
            </div>
          </Row>
          <Row label="Dependencies">{agent.deps.length ? agent.deps.join(" · ") : "None — runs first"}</Row>
        </div>
      </div>
    </>
  );
};

const PipelineDAG = ({ c, progress }) => {
  const [sel, setSel] = useState(null);
  const pos = {
    plan: [64, 88], forensic: [235, 54], semantic: [235, 122],
    retrieval: [420, 88], fusion: [580, 88], decision: [725, 88], report: [862, 88],
  };
  const edges = [
    ["plan", "forensic"], ["plan", "semantic"],
    ["forensic", "retrieval"], ["semantic", "retrieval"],
    ["retrieval", "fusion"], ["fusion", "decision"], ["decision", "report"],
  ];
  const state = (k) => {
    const total = AGENT_LOGS[k].length, n = Math.min(progress[k] || 0, total);
    return n >= total ? "done" : n > 0 ? "active" : "idle";
  };
  const legend = [["Agent", c.text, "circle"], ["Specialist model", c.accent, "text"], ["Data flow", c.accent, "line"], ["Running", c.primary, "dot"], ["Completed", c.success, "dot"], ["Waiting", c.border, "dot"]];
  return (
    <Card c={c} pad={18}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4, flexWrap: "wrap", gap: 8 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 800, color: c.text, display: "flex", gap: 8, alignItems: "center", fontFamily: "'Space Grotesk', Inter, sans-serif" }}><Network size={16} color={c.primary} /> AI Investigation Pipeline</div>
          <div style={{ fontSize: 11, color: c.sub, marginTop: 2 }}>Seven specialist models on one efficient backbone. Click any agent to see how it works.</div>
        </div>
        <div style={{ display: "flex", gap: 14, fontSize: 10.5, color: c.sub, flexWrap: "wrap" }}>
          {legend.map(([l, col, kind]) => (
            <span key={l} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              {kind === "line" ? <span style={{ width: 14, height: 0, borderTop: `2px dashed ${col}` }} />
                : kind === "text" ? <span style={{ fontSize: 8, fontWeight: 800, color: col }}>Aa</span>
                : <span style={{ width: 8, height: 8, borderRadius: kind === "circle" ? 99 : 2, background: kind === "circle" ? "transparent" : col, border: kind === "circle" ? `2px solid ${col}` : "none" }} />}
              {l}
            </span>
          ))}
        </div>
      </div>
      <svg viewBox="0 0 920 176" style={{ width: "100%", display: "block" }}>
        {edges.map(([a, b], i) => {
          const [x1, y1] = pos[a], [x2, y2] = pos[b];
          const live = state(a) === "done" && state(b) !== "idle";
          const doneEdge = state(a) === "done";
          return (
            <g key={i}>
              <path d={`M ${x1 + 34} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2 - 34} ${y2}`}
                fill="none" stroke={doneEdge ? c.accent : c.border} strokeWidth={live ? 2 : 1.5}
                strokeDasharray="6 6" style={doneEdge ? { animation: "tldash 1s linear infinite" } : {}} opacity={doneEdge ? .9 : .5} />
              {live && <circle r="3" fill={c.accent}><animateMotion dur="1.4s" repeatCount="indefinite" path={`M ${x1 + 34} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2 - 34} ${y2}`} /></circle>}
            </g>
          );
        })}
        {AGENTS.map(a => {
          const [x, y] = pos[a.key], st = state(a.key);
          const ring = st === "done" ? c.success : st === "active" ? a.color : c.border;
          return (
            <g key={a.key} style={{ cursor: "pointer" }} onClick={() => setSel(a)}>
              <title>{a.name} — {a.role}. Click for details.</title>
              {st === "active" && <circle cx={x} cy={y} r={26} fill={a.color} opacity=".14"><animate attributeName="r" values="22;30;22" dur="1.6s" repeatCount="indefinite" /></circle>}
              <circle cx={x} cy={y} r={21} fill={c.card2} stroke={ring} strokeWidth="2.5" style={{ transition: "stroke .4s" }} />
              <foreignObject x={x - 10} y={y - 10} width="20" height="20"><a.icon size={20} color={st === "idle" ? c.dim : a.color} /></foreignObject>
              <text x={x} y={y + 37} textAnchor="middle" fill={st === "idle" ? c.dim : c.text} fontSize="9" fontWeight="700" fontFamily="Inter">{a.name.split(" ")[0]}</text>
              <text x={x} y={y + 47} textAnchor="middle" fill={c.sub} fontSize="7.5" fontFamily="Inter">{a.name.split(" ").slice(1).join(" ").replace("& ", "&\u200a")}</text>
              <text x={x} y={y - 30} textAnchor="middle" fill={ADAPTERS[a.adapter][2]} fontSize="7.5" fontWeight="700" fontFamily="Inter">{a.model.replace(" Model", "")}</text>
            </g>
          );
        })}
      </svg>
      <AgentDetailPanel c={c} agent={sel} onClose={() => setSel(null)} />
    </Card>
  );
};

/* ============================================================
   PAGE: INVESTIGATION WORKSPACE
   ============================================================ */
const Workspace = ({ c, setPage }) => {
  const [running, setRunning] = useState(true);
  const [selected, setSelected] = useState(EVIDENCE[0]);
  const [tab, setTab] = useState("heatmap");
  const [tlStep, setTlStep] = useState(0);
  const progress = useAgentStream(running);
  useEffect(() => {
    const iv = setInterval(() => setTlStep(s => Math.min(TIMELINE.length, s + 1)), 1400);
    return () => clearInterval(iv);
  }, []);
  const tabs = [["heatmap", "Heatmap"], ["attn", "Attention map"], ["meta", "Metadata"], ["compress", "Compression / ELA"]];

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <h1 style={{ margin: 0, fontSize: 20, fontWeight: 800, color: c.text }}>Fabricated Boardroom Photo — Wire Fraud</h1>
            <Badge c={c} tone="danger">Critical</Badge>
            <Badge c={c} tone="primary" dot>Live investigation</Badge>
          </div>
          <div style={{ fontSize: 12, color: c.sub, marginTop: 3, fontFamily: "monospace" }}>TL-2026-0342 · opened 10:52 IST · investigator A. Sharma</div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => setRunning(r => !r)} style={{ display: "flex", gap: 6, alignItems: "center", background: c.card, border: `1px solid ${c.border}`, borderRadius: 8, padding: "8px 14px", color: c.text, fontSize: 12, fontWeight: 600, cursor: "pointer", ...font }}>
            {running ? <Pause size={13} /> : <Play size={13} />} {running ? "Pause agents" : "Resume"}
          </button>
          <button onClick={() => setPage("report")} style={{ display: "flex", gap: 6, alignItems: "center", background: c.accent, border: "none", borderRadius: 8, padding: "8px 14px", color: "#06251F", fontSize: 12, fontWeight: 700, cursor: "pointer", ...font }}>
            <FileText size={13} /> View report
          </button>
        </div>
      </div>

      <PipelineDAG c={c} progress={progress} />

      <div style={{ display: "grid", gridTemplateColumns: "270px 1fr 330px", gap: 14, alignItems: "start" }}>
        {/* LEFT: evidence + timeline */}
        <div style={{ display: "grid", gap: 14 }}>
          <Card c={c} pad={14}>
            <SectionTitle c={c} icon={Layers} title="Evidence (4)" />
            {EVIDENCE.map(ev => (
              <div key={ev.id} onClick={() => { setSelected(ev); setTab("heatmap"); }}
                style={{
                  padding: "9px 10px", borderRadius: 9, cursor: "pointer", marginBottom: 4,
                  background: selected.id === ev.id ? c.primary + "14" : "transparent",
                  border: `1px solid ${selected.id === ev.id ? c.primary + "55" : "transparent"}`,
                }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <ImageIcon size={14} color={c.primary} />
                  <span style={{ fontSize: 12, fontWeight: 600, color: c.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>{ev.name}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 5, alignItems: "center" }}>
                  <span style={{ fontSize: 10, color: c.dim, fontFamily: "monospace" }}>{ev.hash}</span>
                  <Badge c={c} tone={verdictTone(ev.verdict)}>{ev.verdict}</Badge>
                </div>
              </div>
            ))}
          </Card>
          <Card c={c} pad={14}>
            <SectionTitle c={c} icon={Clock} title="Investigation timeline" />
            {TIMELINE.map((t, i) => {
              const done = i < tlStep, active = i === tlStep;
              return (
                <div key={i} style={{ display: "flex", gap: 10, opacity: done || active ? 1 : .35, transition: "opacity .4s" }}>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                    <div style={{
                      width: 22, height: 22, borderRadius: 99, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                      background: done ? c.accent + "22" : active ? c.primary + "22" : c.card2,
                      border: `1.5px solid ${done ? c.accent : active ? c.primary : c.border}`,
                      animation: active ? "tlpulse 1.4s infinite" : "none",
                    }}>
                      <t.icon size={11} color={done ? c.accent : active ? c.primary : c.dim} />
                    </div>
                    {i < TIMELINE.length - 1 && <div style={{ width: 2, flex: 1, minHeight: 16, background: done ? c.accent + "55" : c.border }} />}
                  </div>
                  <div style={{ paddingBottom: 14 }}>
                    <div style={{ fontSize: 11.5, fontWeight: 600, color: done || active ? c.text : c.sub }}>{t.label}</div>
                    <div style={{ fontSize: 10, color: c.dim, fontFamily: "monospace" }}>{done ? t.time : active ? "in progress…" : "queued"}</div>
                  </div>
                </div>
              );
            })}
          </Card>
        </div>

        {/* CENTER: explainability */}
        <Card c={c}>
          <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
            {tabs.map(([k, l]) => (
              <button key={k} onClick={() => setTab(k)} style={{
                padding: "7px 13px", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer", ...font,
                background: tab === k ? c.primary + "1F" : "transparent", color: tab === k ? c.primary : c.sub,
                border: `1px solid ${tab === k ? c.primary + "55" : c.border}`,
              }}>{l}</button>
            ))}
            <span style={{ marginLeft: "auto", alignSelf: "center" }}>
              <Badge c={c} tone={verdictTone(selected.verdict)}>{selected.verdict} · {selected.confidence}%</Badge>
            </span>
          </div>

          {tab === "heatmap" && <>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div><div style={{ fontSize: 11, color: c.sub, marginBottom: 6, fontWeight: 600 }}>ORIGINAL</div><FaceExhibit c={c} /></div>
              <div><div style={{ fontSize: 11, color: c.sub, marginBottom: 6, fontWeight: 600 }}>GRAD-CAM MANIPULATION HEATMAP</div><FaceExhibit c={c} overlay="heat" /></div>
            </div>
            <div style={{ marginTop: 12, padding: 12, background: c.card2, borderRadius: 10, fontSize: 12.5, color: c.sub, lineHeight: 1.6 }}>
              <span style={{ color: c.danger, fontWeight: 700 }}>High suspicion:</span> jawline blend boundary and mouth region dominate model attention. <span style={{ color: c.warning, fontWeight: 700 }}>Moderate:</span> asymmetric specular highlights across eyes suggest inconsistent light source — classic composite artifact.
            </div>
          </>}
          {tab === "attn" && <>
            <FaceExhibit c={c} overlay="attn" />
            <div style={{ marginTop: 12, fontSize: 12.5, color: c.sub, lineHeight: 1.6 }}>ViT patch attention (layer 11, head 7). Attention mass concentrates on facial region at 4.2× background rate — the ensemble's decision is driven by face patches, not context bias.</div>
          </>}
          {tab === "meta" && (
            <div style={{ display: "grid", gap: 0 }}>
              {[["Dimensions", "3840 × 2160"], ["Camera Make/Model", "absent ⚠"], ["Software", "PIL 10.1 ⚠"], ["Created", "2026-07-08 10:48:12 (postdates email) ⚠"], ["Color profile", "sRGB (stripped ICC)"], ["Quantization tables", "non-standard, double-compressed ⚠"], ["SHA-256", "a3f2bb17…9c1d"]].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "10px 4px", borderBottom: `1px solid ${c.border}`, fontSize: 12.5 }}>
                  <span style={{ color: c.sub }}>{k}</span>
                  <span style={{ color: v.includes("⚠") ? c.warning : c.text, fontWeight: 600, fontFamily: "monospace", fontSize: 11.5 }}>{v}</span>
                </div>
              ))}
            </div>
          )}
          {tab === "compress" && <>
            <div style={{ fontSize: 11, color: c.sub, marginBottom: 6, fontWeight: 600 }}>ERROR LEVEL ANALYSIS (ELA)</div>
            <FaceExhibit c={c} overlay="ela" />
            <div style={{ marginTop: 12, fontSize: 12.5, color: c.sub, lineHeight: 1.6 }}>Face region shows uniform high error levels against a low-error background — the face was inserted after the original compression pass. Double-JPEG quantization detected (q₁≈92, q₂≈78).</div>
          </>}
        </Card>

        {/* RIGHT: live agents */}
        <Card c={c} pad={14} style={{ maxHeight: "78vh", overflowY: "auto" }}>
          <SectionTitle c={c} icon={Bot} title="Live AI agents" right={<Badge c={c} tone={running ? "accent" : "sub"} dot>{running ? "Streaming" : "Paused"}</Badge>} />
          {AGENTS.map(a => {
            const total = AGENT_LOGS[a.key].length;
            const shown = Math.min(progress[a.key] || 0, total);
            const done = shown >= total;
            const last = shown > 0 ? AGENT_LOGS[a.key][shown - 1] : null;
            return (
              <div key={a.key} style={{ marginBottom: 12, background: c.card2, borderRadius: 10, padding: 12, borderLeft: `3px solid ${a.color}` }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <a.icon size={14} color={a.color} />
                  <span style={{ fontSize: 12, fontWeight: 700, color: c.text, flex: 1 }}>{a.name}</span>
                  <AdapterChip a={a.adapter} label={a.model} />
                  {done ? <CheckCircle2 size={14} color={c.success} /> : shown > 0 ? <span style={{ width: 8, height: 8, borderRadius: 99, background: a.color, animation: "tlpulse 1.2s infinite" }} /> : <Clock size={13} color={c.dim} />}
                </div>
                <div style={{ height: 4, borderRadius: 99, background: c.border, margin: "8px 0" }}>
                  <div style={{ width: `${(shown / total) * 100}%`, height: "100%", borderRadius: 99, background: a.color, transition: "width .5s ease" }} />
                </div>
                {last
                  ? <p style={{ margin: 0, fontSize: 11.5, color: c.sub, lineHeight: 1.5, animation: "tlfadein .3s" }}>{last.text}</p>
                  : <Skeleton c={c} h={11} w="80%" />}
              </div>
            );
          })}
        </Card>
      </div>
    </div>
  );
};

/* ============================================================
   PAGE: AGENT CONSOLE
   ============================================================ */
const AgentConsole = ({ c }) => {
  const [active, setActive] = useState("plan");
  const [visible, setVisible] = useState(1);
  useEffect(() => {
    setVisible(1);
    const iv = setInterval(() => setVisible(v => v + 1), 650);
    return () => clearInterval(iv);
  }, [active]);
  const agent = AGENTS.find(a => a.key === active);
  const logs = AGENT_LOGS[active].slice(0, visible);
  const totalMs = AGENT_LOGS[active].reduce((s, e) => s + (e.ms || 0), 0);

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>AI Agent Console</h1>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub }}>Full reasoning transcripts — every conclusion is auditable down to the tool call.</p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "250px 1fr", gap: 14, alignItems: "start" }}>
        <Card c={c} pad={10}>
          {AGENTS.map(a => (
            <button key={a.key} onClick={() => setActive(a.key)} style={{
              display: "flex", alignItems: "center", gap: 10, width: "100%", padding: "11px 12px",
              background: active === a.key ? a.color + "14" : "transparent",
              border: `1px solid ${active === a.key ? a.color + "44" : "transparent"}`,
              borderRadius: 9, cursor: "pointer", marginBottom: 4, textAlign: "left", ...font,
            }}>
              <a.icon size={16} color={a.color} />
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12.5, fontWeight: 700, color: c.text }}>{a.name}</div>
                <div style={{ fontSize: 10.5, color: c.dim }}>{a.role}</div>
              </div>
              <ChevronRight size={13} color={c.dim} />
            </button>
          ))}
        </Card>
        <Card c={c}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, paddingBottom: 12, borderBottom: `1px solid ${c.border}`, marginBottom: 8 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: agent.color + "1F", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <agent.icon size={18} color={agent.color} />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 14, fontWeight: 800, color: c.text }}>{agent.name}</div>
              <div style={{ fontSize: 11, color: c.sub }}>{agent.role} · case TL-2026-0342</div>
            </div>
            <Badge c={c} tone="sub">Σ tool time {totalMs} ms</Badge>
            <AdapterChip a={agent.adapter} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 7, padding: "8px 10px", background: c.card2, borderRadius: 8, marginBottom: 8 }}>
            <Database size={12} color={c.primary} />
            <span style={{ fontSize: 11, color: c.sub }}>Vector store (ChromaDB): <b style={{ color: c.text }}>{agent.store}</b></span>
          </div>
          <div style={{ minHeight: 320 }}>
            {logs.map((e, i) => <AgentEntry key={i} c={c} e={e} />)}
            {visible < AGENT_LOGS[active].length && (
              <div style={{ display: "flex", gap: 6, padding: "10px 0 0 24px" }}>
                {[0, 1, 2].map(i => <span key={i} style={{ width: 6, height: 6, borderRadius: 99, background: c.dim, animation: `tlbounce 1.2s ${i * .18}s infinite` }} />)}
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};

/* ============================================================
   PAGE: EXPLAINABILITY DASHBOARD
   ============================================================ */
const Explainability = ({ c }) => {
  const contrib = [
    { name: "Blend boundary", v: 34, c: "#FF5A5F" }, { name: "Frequency anomaly", v: 26, c: "#FFC857" },
    { name: "Metadata conflicts", v: 18, c: "#4F8EF7" }, { name: "Lighting mismatch", v: 13, c: "#B084F5" },
    { name: "Texture entropy", v: 9, c: "#00C9A7" },
  ];
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>Explainability Dashboard</h1>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub }}>Why the model decided FAKE — decomposed into human-verifiable evidence.</p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>
        <Card c={c} style={{ textAlign: "center" }}>
          <SectionTitle c={c} icon={GaugeIcon} title="Composite verdict" />
          <div style={{ display: "flex", justifyContent: "center" }}><Ring c={c} value={96} color={c.danger} label="fake probability" size={120} /></div>
          <div style={{ marginTop: 12 }}><Badge c={c} tone="danger">FAKE — high confidence</Badge></div>
          <div style={{ fontSize: 11, color: c.sub, marginTop: 8 }}>95% CI: [94.1, 97.8] · calibration ECE 0.021</div>
        </Card>
        <Card c={c} style={{ textAlign: "center" }}>
          <SectionTitle c={c} icon={AlertTriangle} title="Threat level" />
          <ThreatGauge c={c} value={9.1} />
          <div style={{ fontSize: 11.5, color: c.sub, marginTop: 18, lineHeight: 1.6 }}>Coordinated multi-modal impersonation targeting a financial approval workflow.</div>
        </Card>
        <Card c={c}>
          <SectionTitle c={c} icon={Layers} title="Decision contribution" />
          {contrib.map(x => (
            <div key={x.name} style={{ marginBottom: 11 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, marginBottom: 4 }}>
                <span style={{ color: c.sub }}>{x.name}</span><span style={{ color: c.text, fontWeight: 700 }}>{x.v}%</span>
              </div>
              <div style={{ height: 6, borderRadius: 99, background: c.border }}>
                <div style={{ width: `${x.v * 2.6}%`, height: "100%", borderRadius: 99, background: x.c, transition: "width 1s" }} />
              </div>
            </div>
          ))}
          <div style={{ fontSize: 10.5, color: c.dim, marginTop: 6 }}>SHAP values normalized over ensemble output</div>
        </Card>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 14 }}>
        <Card c={c}>
          <SectionTitle c={c} icon={ScanFace} title="Visual evidence — EV-01" right={<Badge c={c} tone="danger">6 anomaly classes</Badge>} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div><div style={{ fontSize: 11, color: c.sub, marginBottom: 6, fontWeight: 600 }}>GRAD-CAM</div><FaceExhibit c={c} overlay="heat" /></div>
            <div><div style={{ fontSize: 11, color: c.sub, marginBottom: 6, fontWeight: 600 }}>ELA</div><FaceExhibit c={c} overlay="ela" /></div>
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
            {["Blend boundary", "Reflection inconsistency", "Frequency anomaly", "Noise deviation", "Double JPEG", "Lighting mismatch"].map(t => <Badge key={t} c={c} tone="warning">{t}</Badge>)}
          </div>
        </Card>
        <Card c={c}>
          <SectionTitle c={c} icon={Network} title="Evidence correlation graph" />
          <CorrelationGraph c={c} />
          <div style={{ fontSize: 11.5, color: c.sub, lineHeight: 1.6 }}>Generator fingerprint (latent decoder signature) links this case to TL-2026-0329 — probable common toolchain. Dashed edges = inferred links pending analyst confirmation.</div>
        </Card>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card c={c}>
          <SectionTitle c={c} icon={Layers} title="Mixture-of-experts consensus — generator attribution"
            right={<Badge c={c} tone="danger">Diffusion-family</Badge>} />
          {[["Diffusion expert", 71, "#B084F5"], ["GAN expert", 19, "#4F8EF7"], ["Compression expert", 6, "#FFC857"], ["Inconclusive", 4, "#5B6B85"]].map(([l, v, col]) => (
            <div key={l} style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, marginBottom: 4 }}>
                <span style={{ color: c.sub }}>{l}</span><span style={{ color: c.text, fontWeight: 700 }}>{v}%</span>
              </div>
              <div style={{ height: 8, borderRadius: 99, background: c.border }}>
                <div style={{ width: `${v}%`, height: "100%", borderRadius: 99, background: col, transition: "width 1.2s cubic-bezier(.4,0,.2,1)" }} />
              </div>
            </div>
          ))}
          <div style={{ fontSize: 11.5, color: c.sub, lineHeight: 1.6, background: c.card2, borderRadius: 9, padding: 10 }}>
            Per-generator expert heads over a shared EfficientNet extractor; the consensus head weights them into an attribution, not a single opaque score. This generator family was <b style={{ color: c.accent }}>held out in LOGO validation</b> (ROC-AUC 0.981 on unseen generators) — the score generalizes.
          </div>
        </Card>
        <Card c={c}>
          <SectionTitle c={c} icon={Users} title="Fusion & Debate — agent votes" right={<Badge c={c} tone="success">Consensus 4/4</Badge>} />
          {[
            ["Forensic Analysis", "FAKE", "MoE consensus 0.96, diffusion family", true, "#00C9A7"],
            ["Semantic & Context", "FAKE", "lighting + reflection conflicts", true, "#63D2FF"],
            ["Retrieval & Comparison", "FAKE", "background = registry stock photo (0.94)", true, "#FFC857"],
            ["Decision & Confidence", "FAKE", "calibrated 96%, threat 9.1", true, "#FF5A5F"],
          ].map(([n, v, why, ok, col]) => (
            <div key={n} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 2px", borderBottom: `1px solid ${c.border}` }}>
              <span style={{ width: 8, height: 8, borderRadius: 99, background: col, flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: c.text }}>{n}</span>
                <div style={{ fontSize: 10.5, color: c.dim }}>{why}</div>
              </div>
              <Badge c={c} tone="danger">{v}</Badge>
              {ok ? <CheckCircle2 size={14} color={c.success} /> : <XCircle size={14} color={c.danger} />}
            </div>
          ))}
          <div style={{ fontSize: 11.5, color: c.sub, marginTop: 10, lineHeight: 1.55 }}>
            <b style={{ color: c.warning }}>Conflict resolved:</b> compression expert's low score vs Semantic's lighting flag — precedent store showed 3 similar disagreements, all resolved toward "composite" with correct outcome.
          </div>
        </Card>
      </div>
      <Card c={c}>
        <SectionTitle c={c} icon={Layers} title="Forensic evidence summary — anomaly counts across all exhibits" right={<Badge c={c} tone="danger">66 total findings</Badge>} />
        <EvidenceCounts c={c} compact />
      </Card>
      <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 14, alignItems: "start" }}>
        <StressLab c={c} />
        <FailureAnalysis c={c} />
      </div>
      <Card c={c}>
        <SectionTitle c={c} icon={Eye} title="Counterfactual challenges (debate round 2)" right={<Badge c={c} tone="success">All hypotheses survived</Badge>} />
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
          {[
            ["Could compression alone explain frequency peaks?", "No — peaks persist across q90/q70/q50 recompression sweep.", "rejected"],
            ["Could the lighting mismatch be a flash artifact?", "No — flash would raise specular energy on the background glass too; it is absent.", "rejected"],
            ["Is the stock-photo registry match reliable?", "Yes — pixel-aligned at 0.94 similarity across 3 independent crops.", "confirmed"],
          ].map(([q, a, s]) => (
            <div key={q} style={{ background: c.card2, borderRadius: 10, padding: 14 }}>
              <div style={{ fontSize: 12.5, fontWeight: 700, color: c.text, marginBottom: 6 }}>{q}</div>
              <div style={{ fontSize: 12, color: c.sub, lineHeight: 1.55, marginBottom: 8 }}>{a}</div>
              <Badge c={c} tone={s === "rejected" ? "accent" : "success"}>{s === "rejected" ? "Alternative rejected" : "Assumption confirmed"}</Badge>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};


/* Interactive decision-threshold lab: drag τ, watch FAR/FRR/verdict react */
const ThresholdLab = ({ c }) => {
  const [tau, setTau] = useState(0.5);
  const sample = 0.96; // this exhibit's fake probability
  const far = Math.exp(-6 * tau) * 100;          // false accept (fakes passing)
  const frr = Math.exp(-6 * (1 - tau)) * 100;    // false reject (genuine flagged)
  const eerTau = 0.5;
  const flagged = sample >= tau;
  const curve = [...Array(41)].map((_, i) => {
    const t = i / 40;
    return { t: t.toFixed(2), FAR: +(Math.exp(-6 * t) * 100).toFixed(2), FRR: +(Math.exp(-6 * (1 - t)) * 100).toFixed(2) };
  });
  return (
    <Card c={c}>
      <SectionTitle c={c} icon={GaugeIcon} title="Decision threshold lab — drag τ"
        right={<Badge c={c} tone={flagged ? "danger" : "success"}>{flagged ? "EV-01: FLAGGED" : "EV-01: PASSES ⚠"}</Badge>} />
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={curve}>
          <CartesianGrid stroke={c.border} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="t" stroke={c.dim} fontSize={10} tickLine={false} axisLine={false} interval={9} />
          <YAxis stroke={c.dim} fontSize={10} tickLine={false} axisLine={false} width={30} unit="%" />
          <Tooltip contentStyle={{ background: c.card2, border: `1px solid ${c.border}`, borderRadius: 10, fontSize: 12 }} />
          <Line type="monotone" dataKey="FAR" name="False accept" stroke={c.danger} dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="FRR" name="False reject" stroke={c.primary} dot={false} strokeWidth={2} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </LineChart>
      </ResponsiveContainer>
      <input type="range" min="0.05" max="0.95" step="0.01" value={tau} onChange={e => setTau(+e.target.value)}
        style={{ width: "100%", accentColor: c.accent, cursor: "grab", margin: "6px 0 10px" }} aria-label="Fake decision threshold" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, textAlign: "center" }}>
        {[["τ threshold", tau.toFixed(2), c.text], ["FAR", far.toFixed(1) + "%", c.danger], ["FRR", frr.toFixed(1) + "%", c.primary], ["EER point", "τ=" + eerTau + " · 1.7%", c.accent]].map(([k, v, col]) => (
          <div key={k} style={{ background: c.card2, borderRadius: 9, padding: "10px 4px" }}>
            <div style={{ fontSize: 15, fontWeight: 800, color: col }}>{v}</div>
            <div style={{ fontSize: 10, color: c.sub, marginTop: 2 }}>{k}</div>
          </div>
        ))}
      </div>
      <div style={{ fontSize: 11, color: c.sub, marginTop: 10, lineHeight: 1.55 }}>
        Judges' metric made tangible: EER is where the two curves cross. Push τ above {sample} and this fabricated image would slip through — the demo shows exactly why threshold choice is a security decision.
      </div>
    </Card>
  );
};

/* ============================================================
   PAGE: INVESTIGATION REPORT
   ============================================================ */
const Report = ({ c, notify }) => {
  const Sec = ({ n, title, children }) => (
    <div style={{ marginBottom: 26 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, borderBottom: `2px solid ${c.border}`, paddingBottom: 8, marginBottom: 12 }}>
        <span style={{ fontSize: 11, fontWeight: 800, color: c.primary, fontFamily: "monospace" }}>{n}</span>
        <span style={{ fontSize: 15, fontWeight: 800, color: c.text }}>{title}</span>
      </div>
      <div style={{ fontSize: 13, color: c.sub, lineHeight: 1.75 }}>{children}</div>
    </div>
  );
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 10 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>Forensic Investigation Report</h1>
          <p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub, fontFamily: "monospace" }}>TL-2026-0342-R1 · generated 10:53:40 IST · report agent v2.3</p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {[["Download PDF", Download], ["Share (signed link)", Share2], ["Compare with TL-0329", GitCompare]].map(([l, I]) => (
            <button key={l} onClick={() => notify(l + " — done")} style={{ display: "flex", gap: 6, alignItems: "center", background: c.card, border: `1px solid ${c.border}`, borderRadius: 8, padding: "8px 14px", color: c.text, fontSize: 12, fontWeight: 600, cursor: "pointer", ...font }}>
              <I size={13} color={c.primary} /> {l}
            </button>
          ))}
        </div>
      </div>

      <Card c={c} pad={36} style={{ maxWidth: 880 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingBottom: 18, borderBottom: `3px solid ${c.primary}`, marginBottom: 24 }}>
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <div style={{ width: 42, height: 42, borderRadius: 11, background: `linear-gradient(135deg, ${c.primary}, ${c.accent})`, display: "flex", alignItems: "center", justifyContent: "center" }}><Shield size={22} color="#fff" /></div>
            <div>
              <div style={{ fontSize: 17, fontWeight: 800, color: c.text }}>TruthLens AI — Forensic Report</div>
              <div style={{ fontSize: 11, color: c.sub }}>ISO/IEC 27042-aligned · Confidential · Chain of custody sealed</div>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 30, fontWeight: 900, color: c.danger, letterSpacing: -1 }}>FAKE</div>
            <div style={{ fontSize: 12, color: c.sub }}>composite confidence <b style={{ color: c.text }}>96%</b></div>
          </div>
        </div>

        {/* A. Final verdict — matches EADIS reference layout */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 26 }}>
          <div style={{ border: `1px solid ${c.border}`, borderRadius: 12, padding: 16, textAlign: "center" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: c.dim, letterSpacing: 1, marginBottom: 10 }}>A. FINAL VERDICT</div>
            <div style={{ display: "inline-block", transform: "rotate(-7deg)", border: `3px solid ${c.danger}`, color: c.danger, borderRadius: 8, padding: "4px 22px", fontSize: 30, fontWeight: 900, letterSpacing: 3, fontFamily: "'Space Grotesk', Inter, sans-serif", boxShadow: `0 0 24px ${c.danger}33` }}>FAKE</div>
          </div>
          <div style={{ border: `1px solid ${c.border}`, borderRadius: 12, padding: 16, textAlign: "center" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: c.dim, letterSpacing: 1, marginBottom: 8 }}>CONFIDENCE SCORE</div>
            <div style={{ fontSize: 34, fontWeight: 900, color: c.text, fontFamily: "'Space Grotesk', Inter, sans-serif" }}>96%</div>
            <div style={{ fontSize: 10.5, color: c.sub }}>(high confidence · CI [94.1, 97.8])</div>
          </div>
          <div style={{ border: `1px solid ${c.border}`, borderRadius: 12, padding: 16, textAlign: "center" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: c.dim, letterSpacing: 1, marginBottom: 8 }}>RISK LEVEL</div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, marginTop: 6 }}>
              <AlertTriangle size={22} color={c.warning} />
              <span style={{ fontSize: 26, fontWeight: 900, color: c.danger, fontFamily: "'Space Grotesk', Inter, sans-serif" }}>HIGH</span>
            </div>
            <div style={{ fontSize: 10.5, color: c.sub, marginTop: 4 }}>threat 9.1 / 10 · active distribution</div>
          </div>
        </div>

        {/* B. Evidence visualization */}
        <div style={{ marginBottom: 26 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: c.dim, letterSpacing: 1, marginBottom: 10 }}>B. EVIDENCE VISUALIZATION</div>
          <div style={{ display: "grid", gridTemplateColumns: "1.1fr 2fr", gap: 14, alignItems: "start" }}>
            <div style={{ display: "flex", gap: 8 }}>
              <div style={{ flex: 1 }}><FaceExhibit c={c} overlay="heat" /></div>
              <div style={{ width: 14, borderRadius: 7, background: `linear-gradient(180deg, ${c.danger}, ${c.warning}, ${c.primary}33)`, position: "relative" }}>
                <span style={{ position: "absolute", top: 2, left: 18, fontSize: 8.5, color: c.sub, whiteSpace: "nowrap" }}>high</span>
                <span style={{ position: "absolute", bottom: 2, left: 18, fontSize: 8.5, color: c.sub, whiteSpace: "nowrap" }}>low</span>
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
              {ANOM_THUMBS.map(([label, kind]) => <AnomThumb key={kind} c={c} kind={kind} label={label} />)}
            </div>
          </div>
        </div>

        {/* C. Evidence summary */}
        <div style={{ marginBottom: 26 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: c.dim, letterSpacing: 1, marginBottom: 10 }}>C. EVIDENCE SUMMARY</div>
          <EvidenceCounts c={c} />
        </div>

        <Sec n="01" title="Executive summary">
          Both primary exhibits are synthetic. The boardroom photo (EV-01) is a diffusion-generated composite with an inserted face (ensemble score 96%), and the press-photo variant (EV-02) shares the same generator fingerprint (92%). Registry retrieval identified the source background as a stock photo, and metadata places creation 6 minutes before the approval deadline — a targeted social-engineering attack on the treasury wire-approval workflow. <b style={{ color: c.danger }}>Immediate recommendation: block the pending ₹4.2 Cr transfer and initiate legal hold.</b>
        </Sec>
        <Sec n="02" title="Case information">
          Case TL-2026-0342 · Financial Crimes · Investigator A. Sharma (#4417) · Opened Jul 8 2026, 10:52 IST · Priority Critical · 5 exhibits, all SHA-256 sealed at ingestion.
        </Sec>
        <Sec n="03" title="Media summary">
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
            <thead><tr>{["Exhibit", "Type", "Verdict", "Confidence"].map(h => <th key={h} style={{ textAlign: "left", padding: "7px 8px", borderBottom: `1px solid ${c.border}`, color: c.dim, fontSize: 10.5, textTransform: "uppercase", letterSpacing: .6 }}>{h}</th>)}</tr></thead>
            <tbody>{EVIDENCE.map(e => (
              <tr key={e.id}>
                <td style={{ padding: "8px", borderBottom: `1px solid ${c.border}`, color: c.text, fontWeight: 600 }}>{e.id} · {e.name}</td>
                <td style={{ padding: "8px", borderBottom: `1px solid ${c.border}` }}>{e.kind}</td>
                <td style={{ padding: "8px", borderBottom: `1px solid ${c.border}` }}><Badge c={c} tone={verdictTone(e.verdict)}>{e.verdict}</Badge></td>
                <td style={{ padding: "8px", borderBottom: `1px solid ${c.border}`, color: c.text }}>{e.confidence}%</td>
              </tr>
            ))}</tbody>
          </table>
        </Sec>
        <Sec n="04" title="Image findings (EV-01)">
          Grad-CAM localizes manipulation to jawline and mouth; ELA shows uniform high error inside the face region against a low-error background (post-compression insertion). FFT reveals periodic peaks at 0.31/0.47 cycles-per-pixel consistent with diffusion decoder upsampling. Metadata: camera fields absent, Software=PIL 10.1, creation timestamp postdates the delivery email.
        </Sec>
        <Sec n="05" title="Provenance & retrieval findings">
          Registry reverse-search matched the background to a licensed stock boardroom photo (similarity 0.94, three independent crops) — the subject was inserted post-hoc. The latent decoder fingerprint shared by EV-01 and EV-02 matches the generator toolchain in case TL-2026-0329, indicating a common operator. EV-03 carries a valid C2PA manifest; EV-04 headers are consistent with genuine transport.
        </Sec>
        <Sec n="06" title="Agent decisions & consensus">
          Seven agents on a single SLM backbone (Qwen-2.5-7B, hot-swapped LoRA adapters A/B/C) reached weighted consensus 4/4 in the Fusion & Debate round; one inter-agent conflict (compression score vs lighting flag) was resolved via the precedent store. Counterfactual challenges in debate round 2 overturned nothing; calibrated interval: 96±2%. Full transcripts are preserved in the Agent Console and appended as Annex B.
        </Sec>
        <Sec n="07" title="Explainability & confidence">
          Decision decomposition (SHAP): blend boundary 34%, frequency anomaly 26%, metadata conflicts 18%, lighting mismatch 13%, texture entropy 9%. Calibration ECE 0.021 on the July validation window; the reported 96% is a calibrated probability, not a raw logit.
        </Sec>
        <Sec n="08" title="Recommendations">
          1) Block pending wire transfer and freeze the approval session. 2) Escalate to legal hold; preserve mail server logs 10:30–11:10 IST. 3) Add sender-domain verification to the approval workflow. 4) Cross-reference generator fingerprint with TL-2026-0329 (probable same operator). 5) Human review of EV-04 (82% confidence, below auto-close threshold).
        </Sec>
        <div style={{ display: "flex", justifyContent: "space-between", paddingTop: 16, borderTop: `1px solid ${c.border}`, fontSize: 10.5, color: c.dim, fontFamily: "monospace" }}>
          <span>Report hash: 9e12ffb3…77aa · signed by orchestrator key TL-PROD-03</span>
          <span>Page 1 of 12</span>
        </div>
      </Card>
    </div>
  );
};

/* ============================================================
   PAGE: CASE HISTORY
   ============================================================ */
const CaseHistory = ({ c, setPage }) => {
  const [q, setQ] = useState("");
  const [f, setF] = useState("All");
  const rows = CASES.filter(r =>
    (f === "All" || (f === "Pinned" ? r.pinned : r.status === f)) &&
    (r.name.toLowerCase().includes(q.toLowerCase()) || r.id.toLowerCase().includes(q.toLowerCase()))
  );
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>Case History</h1>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub }}>{CASES.length} cases · retention policy 7 years · WORM storage</p>
      </div>
      <Card c={c}>
        <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, background: c.card2, border: `1px solid ${c.border}`, borderRadius: 9, padding: "8px 12px", flex: 1, minWidth: 220 }}>
            <Search size={14} color={c.dim} />
            <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search by case name or ID…" style={{ background: "transparent", border: "none", outline: "none", color: c.text, fontSize: 12.5, width: "100%", ...font }} />
          </div>
          {["All", "Pinned", "In Progress", "Completed", "Pending Review"].map(x => (
            <button key={x} onClick={() => setF(x)} style={{
              padding: "8px 14px", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer", ...font,
              background: f === x ? c.primary + "1F" : "transparent", color: f === x ? c.primary : c.sub,
              border: `1px solid ${f === x ? c.primary + "55" : c.border}`,
            }}>{x === "Pinned" ? "📌 Pinned" : x}</button>
          ))}
        </div>
        <CaseTable c={c} rows={rows} setPage={setPage} />
        {rows.length === 0 && <div style={{ padding: 30, textAlign: "center", color: c.sub, fontSize: 13 }}>No cases match. Clear the search or create a new investigation.</div>}
      </Card>
    </div>
  );
};

/* ============================================================
   PAGE: ANALYTICS
   ============================================================ */
const Analytics = ({ c }) => (
  <div style={{ display: "grid", gap: 14 }}>
    <div>
      <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>Analytics</h1>
      <p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub }}>Fleet-wide detection intelligence · July 2026</p>
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: 14 }}>
      <StatCard c={c} icon={Microscope} label="Cases this month" value="212" delta="+31%" up tone="primary" />
      <StatCard c={c} icon={Flame} label="Confirmed fakes" value="87" delta="+44%" up={false} tone="danger" />
      <StatCard c={c} icon={Clock} label="Median time-to-verdict" value="48s" delta="-12s" up tone="accent" />
      <StatCard c={c} icon={Users} label="Human overrides" value="1.8%" delta="-0.6%" up tone="warning" />
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 14 }}>
      <Card c={c}>
        <SectionTitle c={c} icon={BarChart3} title="Verdicts by day" />
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={THREAT_TREND}>
            <CartesianGrid stroke={c.border} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="d" stroke={c.dim} fontSize={11} tickLine={false} axisLine={false} />
            <YAxis stroke={c.dim} fontSize={11} tickLine={false} axisLine={false} width={28} />
            <Tooltip contentStyle={{ background: c.card2, border: `1px solid ${c.border}`, borderRadius: 10, fontSize: 12 }} cursor={{ fill: c.border + "44" }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="fake" name="Fake" stackId="a" fill={c.danger} radius={[0, 0, 0, 0]} />
            <Bar dataKey="susp" name="Suspicious" stackId="a" fill={c.warning} />
            <Bar dataKey="real" name="Authentic" stackId="a" fill={c.accent} radius={[5, 5, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
      <Card c={c}>
        <SectionTitle c={c} icon={Layers} title="Attack technique mix" />
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={ATTACK_MIX} dataKey="value" innerRadius={52} outerRadius={80} paddingAngle={3} strokeWidth={0}>
              {ATTACK_MIX.map((e, i) => <Cell key={i} fill={e.c} />)}
            </Pie>
            <Tooltip contentStyle={{ background: c.card2, border: `1px solid ${c.border}`, borderRadius: 10, fontSize: 12 }} />
          </PieChart>
        </ResponsiveContainer>
        <div style={{ display: "grid", gap: 6 }}>
          {ATTACK_MIX.map(a => (
            <div key={a.name} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
              <span style={{ width: 9, height: 9, borderRadius: 3, background: a.c }} />
              <span style={{ color: c.sub, flex: 1 }}>{a.name}</span>
              <span style={{ color: c.text, fontWeight: 700 }}>{a.value}%</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
    <Card c={c}>
      <SectionTitle c={c} icon={TrendingUp} title="Model performance — image deepfake classifier" right={<Badge c={c} tone="accent">Judging-sheet metrics</Badge>} />
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
          <thead><tr>{["Classifier", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "EER"].map(h => (
            <th key={h} style={{ textAlign: "left", padding: "8px 10px", color: c.dim, fontSize: 10.5, fontWeight: 700, letterSpacing: .8, textTransform: "uppercase", borderBottom: `1px solid ${c.border}` }}>{h}</th>
          ))}</tr></thead>
          <tbody>
            {[
              ["Image deepfake (MoE consensus)", "98.1%", "97.4%", "96.8%", "97.1%", "0.994", "1.4%"],
            ].map(r => (
              <tr key={r[0]}>{r.map((v, i) => (
                <td key={i} style={{ padding: "10px", borderBottom: `1px solid ${c.border}`, color: i === 0 ? c.text : c.sub, fontWeight: i === 0 ? 700 : 500 }}>{v}</td>
              ))}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
    <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 14 }}>
      <Card c={c}>
        <SectionTitle c={c} icon={Shield} title="LOGO validation — leave-one-generator-out" right={<Badge c={c} tone="primary">Robustness on unknown attacks</Badge>} />
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
            <thead><tr>{["Held-out generator", "Accuracy", "F1", "ROC-AUC"].map(h => (
              <th key={h} style={{ textAlign: "left", padding: "8px 10px", color: c.dim, fontSize: 10.5, fontWeight: 700, letterSpacing: .8, textTransform: "uppercase", borderBottom: `1px solid ${c.border}` }}>{h}</th>
            ))}</tr></thead>
            <tbody>
              {[["Diffusion family", "95.2%", "94.7%", "0.981"], ["GAN family", "96.4%", "95.9%", "0.987"], ["Face-swap tools", "93.8%", "93.1%", "0.972"], ["Compression/edit", "97.0%", "96.6%", "0.990"]].map(r => (
                <tr key={r[0]}>{r.map((v, i) => (
                  <td key={i} style={{ padding: "9px 10px", borderBottom: `1px solid ${c.border}`, color: i === 0 ? c.text : c.sub, fontWeight: i === 0 ? 600 : 500 }}>{v}</td>
                ))}</tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ fontSize: 11, color: c.sub, marginTop: 10 }}>Trained on the other 3 families, tested purely on the held-out one — per-generator numbers, not an averaged headline.</div>
      </Card>
      <ThresholdLab c={c} />
    </div>
  </div>
);

/* ============================================================
   PAGE: SETTINGS
   ============================================================ */
const Toggle = ({ c, on, set }) => (
  <button onClick={() => set(!on)} style={{ width: 40, height: 22, borderRadius: 99, border: "none", cursor: "pointer", background: on ? c.accent : c.border, position: "relative", transition: "background .2s", flexShrink: 0 }}>
    <span style={{ position: "absolute", top: 3, left: on ? 21 : 3, width: 16, height: 16, borderRadius: 99, background: "#fff", transition: "left .2s" }} />
  </button>
);

const SettingsPage = ({ c, dark, setDark }) => {
  const [s, setS] = useState({ stream: true, critic: true, autoblock: false, redact: true, notif: true, keys: true });
  const set = k => v => setS(p => ({ ...p, [k]: v }));
  const Row = ({ k, title, sub, custom }) => (
    <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px 2px", borderBottom: `1px solid ${c.border}` }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: c.text }}>{title}</div>
        <div style={{ fontSize: 11.5, color: c.sub, marginTop: 2 }}>{sub}</div>
      </div>
      {custom || <Toggle c={c} on={s[k]} set={set(k)} />}
    </div>
  );
  return (
    <div style={{ maxWidth: 720, display: "grid", gap: 14 }}>
      <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>Settings</h1>
      <Card c={c}>
        <SectionTitle c={c} icon={Bot} title="Agent behavior" />
        <Row k="stream" title="Stream agent reasoning live" sub="Show token-level agent progress in the workspace" />
        <Row k="critic" title="Mandatory critic pass" sub="No verdict finalizes without adversarial counterfactual review" />
        <Row k="autoblock" title="Auto-escalate critical verdicts" sub="Automatically open a legal-hold ticket when threat ≥ 9.0" />
      </Card>
      <Card c={c}>
        <SectionTitle c={c} icon={Lock} title="Security & privacy" />
        <Row k="redact" title="PII redaction in transcripts" sub="Mask names, account numbers and phone numbers in agent logs" />
        <Row title="Report signing key" sub="TL-PROD-03 · Ed25519 · rotates every 90 days" custom={<Badge c={c} tone="success">Active</Badge>} />
        <Row title="Data residency" sub="All evidence processed in-region (ap-south-1) · WORM storage" custom={<Badge c={c} tone="primary">India</Badge>} />
      </Card>
      <Card c={c}>
        <SectionTitle c={c} icon={Settings} title="Interface" />
        <Row title="Theme" sub="Dark is recommended for evidence review" custom={
          <div style={{ display: "flex", gap: 6 }}>
            {[["Dark", true], ["Light", false]].map(([l, v]) => (
              <button key={l} onClick={() => setDark(v)} style={{ padding: "6px 14px", borderRadius: 8, fontSize: 12, fontWeight: 700, cursor: "pointer", ...font, background: dark === v ? c.primary + "1F" : "transparent", color: dark === v ? c.primary : c.sub, border: `1px solid ${dark === v ? c.primary : c.border}` }}>{l}</button>
            ))}
          </div>} />
        <Row k="notif" title="Desktop notifications" sub="Critical alerts and completed investigations" />
        <Row k="keys" title="Keyboard shortcuts" sub="⌘K search · N new case · G+D dashboard · G+W workspace" />
      </Card>
    </div>
  );
};

/* ============================================================
   PAGE: SENTINEL THREAT INTEL (plug-and-play extension)
   Consumes InvestigationCompletedEvent → ThreatIntelligenceResult
   ============================================================ */
const SENTINEL_NODES = [
  { id: "threat_intel", name: "Threat Intelligence", icon: ShieldAlert, desc: "Entity reputation vs threat feeds (VirusTotal, AbuseIPDB adapters)" },
  { id: "internet_intel", name: "Internet Intelligence", icon: Globe, desc: "Reverse-image + keyword crawl using inherited pHash / CLIP keys", degradable: true },
  { id: "correlation", name: "Correlation", icon: Layers, desc: "Generator fingerprint vs main-app ChromaDB case store" },
  { id: "kgraph", name: "Knowledge Graph", icon: Network, desc: "Link cases, URLs, accounts, infrastructure" },
  { id: "score", name: "Threat Score", icon: GaugeIcon, desc: "confidence×5 + live hosts + campaign + feed hits" },
  { id: "hitl", name: "Human Review (HITL)", icon: UserCheck, desc: "Analyst reviews full package — nothing files without approval" },
  { id: "cybercrime", name: "Cybercrime Report", icon: FileText, desc: "NCRP-ready signed complaint with per-URL evidence basis", degradable: true },
  { id: "monitor", name: "Continuous Monitoring", icon: Radio, desc: "pHash enrolled in 24h re-crawl watch + takedown tracking" },
];

/* Search keys Sentinel inherits from the main pipeline — the knowledge reuse */
const SEARCH_KEYS = [
  { key: "Perceptual hash (pHash)", val: "f0e1d2c3…9687 · 64-bit", from: "Forensic Agent — exhibit preprocessing", color: "#4F8EF7" },
  { key: "CLIP embedding", val: "emb_0342_ev01 · 512-d", from: "Semantic Agent — ChromaDB scene embeddings", color: "#63D2FF" },
  { key: "Generator fingerprint", val: "GEN-FP-0342 · 128-d latent signature", from: "Forensic Agent — ChromaDB artifact fingerprints", color: "#00C9A7" },
  { key: "Entity keywords", val: "boardroom · CFO · pay-fastsettle.example", from: "Planner Agent — entity extraction", color: "#FFC857" },
];

/* Crawl matches WITH evidence basis — how each was found and judged */
const CRAWL_MATCHES = [
  { platform: "Video platform", handle: "@quick_finance_tips", url: "videosite.example/watch?v=x91", status: "live", views: "48.2K", region: "IN", firstSeen: "Jul 6",
    ev: { method: "Reverse image", provider: "SearchProvider: google_images", query: "pHash f0e1d2c3… + CLIP embedding", phash: 3, cos: 0.97, fp: true, cls: "exact copy" } },
  { platform: "Social network", handle: "@ceo.updates.official", url: "social.example/p/88231", status: "live", views: "12.7K", region: "IN", firstSeen: "Jul 6",
    ev: { method: "Reverse image", provider: "SearchProvider: google_images", query: "pHash f0e1d2c3… + CLIP embedding", phash: 5, cos: 0.95, fp: true, cls: "exact copy" } },
  { platform: "Messaging channel", handle: "t.me/insider_market_x", url: "t.me/insider_market_x/2231", status: "live", views: "8.9K", region: "AE", firstSeen: "Jul 7",
    ev: { method: "Platform crawl", provider: "CrawlerAdapter: telegram", query: "watchlist channels · image-hash sweep", phash: 6, cos: 0.94, fp: true, cls: "exact copy" } },
  { platform: "Forum mirror", handle: "board/vid/49221", url: "forum.example/board/vid/49221", status: "removed", views: "—", region: "US", firstSeen: "Jul 5",
    ev: { method: "Keyword search", provider: "SearchProvider: google_images", query: "boardroom CFO pay-fastsettle", phash: 14, cos: 0.91, fp: null, cls: "variant (cropped)" } },
];

const EvidenceBasis = ({ c, ev }) => (
  <div style={{ background: c.card, borderRadius: 9, padding: "10px 12px", marginTop: 8, border: `1px solid ${c.border}`, animation: "tlfadein .25s" }}>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginBottom: 8 }}>
      {[["pHash distance", ev.phash + " / 64 bits", ev.phash <= 8 ? c.success : c.warning],
        ["CLIP cosine", ev.cos.toFixed(2), ev.cos >= .9 ? c.success : c.warning],
        ["Generator fingerprint", ev.fp === true ? "match ✓" : ev.fp === false ? "no match" : "n/a (removed)", ev.fp ? c.success : c.dim]].map(([k, v, col]) => (
        <div key={k} style={{ background: c.card2, borderRadius: 8, padding: "8px 6px", textAlign: "center" }}>
          <div style={{ fontSize: 13, fontWeight: 800, color: col, fontFamily: "'Space Grotesk', Inter, sans-serif" }}>{v}</div>
          <div style={{ fontSize: 9, color: c.sub, marginTop: 2 }}>{k}</div>
        </div>
      ))}
    </div>
    <div style={{ fontSize: 10.5, color: c.sub, lineHeight: 1.6 }}>
      <b style={{ color: c.text }}>Found via:</b> {ev.method} · <span style={{ fontFamily: "monospace" }}>{ev.provider}</span><br />
      <b style={{ color: c.text }}>Query:</b> <span style={{ fontFamily: "monospace" }}>{ev.query}</span><br />
      <b style={{ color: c.text }}>Judgment:</b> <Badge c={c} tone={ev.cls.startsWith("exact") ? "danger" : "warning"}>{ev.cls}</Badge>
      <span style={{ marginLeft: 6 }}>of the exhibit already ruled FAKE by the 7-agent pipeline — Sentinel proves distribution, it never re-detects.</span>
    </div>
  </div>
);

const SentinelInvestigation = ({ c, notify }) => {
  const [state, setState] = useState("idle");
  const [node, setNode] = useState(-1);
  const [failCrawler, setFailCrawler] = useState(false);
  const [logs, setLogs] = useState([]);
  const [score, setScore] = useState(0);
  const [decision, setDecision] = useState(null);
  const [ackNo, setAckNo] = useState(null);
  const [expanded, setExpanded] = useState(null);
  const [pkgOpen, setPkgOpen] = useState(false);

  const addLog = (agent, message, ok = true) =>
    setLogs(l => [...l, { t: new Date().toLocaleTimeString(), agent, message, ok }]);

  const run = () => {
    setState("running"); setNode(0); setLogs([]); setScore(0); setDecision(null); setAckNo(null); setExpanded(null); setPkgOpen(false);
    addLog("EventBus", "InvestigationCompletedEvent — TL-2026-0342 (FAKE, 96%). Search keys inherited from main pipeline: pHash, CLIP embedding, generator fingerprint, entities.");
  };

  useEffect(() => {
    if (state !== "running") return;
    if (node >= 6) { setState("awaiting_hitl"); addLog("HITL", "Threat ≥ 7.0 — full package preview built and routed to analyst queue. Pipeline paused at gate."); return; }
    const iv = setTimeout(() => {
      const n = SENTINEL_NODES[node];
      if (n.id === "internet_intel" && failCrawler) {
        addLog(n.name, "CrawlerAdapter timeout (simulated). Degrading gracefully — continuing with ThreatFeedProvider data only.", false);
      } else {
        const msgs = {
          threat_intel: "pay-fastsettle.example flagged by 3 providers (VirusTotal, AbuseIPDB) — payment-fraud infrastructure.",
          internet_intel: "Reverse-image (pHash + CLIP) + keyword + platform crawl → 4 matches, 3 live. Evidence basis recorded per match: pHash Δ, CLIP cosine, fingerprint.",
          correlation: "Generator fingerprint queried against the SAME ChromaDB case store the Retrieval agent uses → TL-2026-0329 at distance 0.11 → campaign CAMP-2026-017.",
          kgraph: "+5 nodes, +5 edges. Shared payment domain links both cases and 3 distribution URLs.",
          score: "9.1/10 = confidence×5 (4.8) + 3 live hosts (3.0) + campaign (2.0) + feed hits (1.0), capped.",
        };
        addLog(n.name, msgs[n.id] || "Completed.");
        if (n.id === "score") setScore(9.1);
      }
      setNode(x => x + 1);
    }, 1100);
    return () => clearTimeout(iv);
  }, [state, node, failCrawler]);

  const approve = () => {
    setDecision("approved"); setNode(6);
    addLog("HITL", "Analyst A. Sharma APPROVED filing — digital signature captured over the reviewed package.");
    setTimeout(() => {
      addLog("Cybercrime Report", "Package sealed: complaint narrative, 4 URL observations with evidence basis, forensic PDF, custody manifest, takedown list.");
      setNode(7);
      setTimeout(() => {
        const ack = "NCRP-2026-" + Math.floor(100000 + Math.random() * 899999);
        setAckNo(ack);
        addLog("CyberCrimeProvider: ncrp", `Complaint filed — acknowledgment ${ack}.`);
        addLog("Continuous Monitoring", "pHash enrolled in 24h re-crawl watch. Takedown requests dispatched to 3 live hosts.");
        setState("filed"); setNode(8);
        notify("Cybercrime complaint filed — " + ack);
      }, 1400);
    }, 1200);
  };
  const reject = () => {
    setDecision("rejected");
    addLog("HITL", "Analyst REJECTED auto-filing — returned to manual queue. Nothing submitted externally.", false);
    setState("filed"); setNode(8);
  };

  const nodeState = (i) => i < node ? "done" : i === node && (state === "running" || state === "awaiting_hitl") ? "active" : "idle";
  const crawlDone = node > 1 && !failCrawler;

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 10 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>Sentinel Threat Intelligence</h1>
            <Badge c={c} tone="accent">Plug-and-play extension</Badge>
          </div>
          <p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub }}>Inherits the verdict and learned artifacts from the 7-agent pipeline, hunts the fake across the internet, and — with analyst approval — files the cybercrime complaint.</p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <label style={{ display: "flex", gap: 7, alignItems: "center", fontSize: 11.5, color: c.sub, cursor: "pointer" }}>
            <Toggle c={c} on={failCrawler} set={setFailCrawler} /> Simulate crawler failure
          </label>
          <button onClick={run} disabled={state === "running" || state === "awaiting_hitl"} style={{ display: "flex", gap: 7, alignItems: "center", background: state === "idle" || state === "filed" ? `linear-gradient(135deg, ${c.primary}, #3B7BE0)` : c.card, border: "none", borderRadius: 9, padding: "9px 16px", color: state === "idle" || state === "filed" ? "#fff" : c.dim, fontSize: 12.5, fontWeight: 700, cursor: state === "running" ? "wait" : "pointer", boxShadow: state === "idle" || state === "filed" ? `0 4px 16px ${c.primary}4D` : "none", ...font }}>
            {state === "running" || state === "awaiting_hitl" ? <RefreshCw size={14} style={{ animation: "tlspin 1s linear infinite" }} /> : <Zap size={14} />}
            {state === "idle" ? "Trigger InvestigationCompletedEvent" : state === "filed" ? "Re-run extension" : "Running…"}
          </button>
        </div>
      </div>

      <Card c={c} pad={14} style={{ borderLeft: `3px solid ${c.accent}` }}>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", fontSize: 12, color: c.sub }}>
          <Badge c={c} tone="accent" dot>Zero core changes</Badge>
          <span>Core pipeline calls <code style={{ background: c.card2, padding: "2px 7px", borderRadius: 6, color: c.accent, fontSize: 11.5 }}>await sentinel.analyze(investigation_result)</code> → returns <code style={{ background: c.card2, padding: "2px 7px", borderRadius: 6, color: c.primary, fontSize: 11.5 }}>ThreatIntelligenceResult</code>. Detection & report agents untouched.</span>
        </div>
      </Card>

      {/* Inherited search keys — the knowledge-reuse card */}
      {node >= 0 && (
        <Card c={c}>
          <SectionTitle c={c} icon={Database} title="Search keys — inherited from the main investigation (no re-analysis)"
            right={<Badge c={c} tone="primary">Reused training & memory</Badge>} />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
            {SEARCH_KEYS.map(k => (
              <div key={k.key} style={{ background: c.card2, borderRadius: 10, padding: 12, borderLeft: `3px solid ${k.color}`, animation: "tlfadein .4s" }}>
                <div style={{ fontSize: 11.5, fontWeight: 700, color: c.text }}>{k.key}</div>
                <div style={{ fontSize: 11, color: k.color, fontFamily: "monospace", margin: "4px 0" }}>{k.val}</div>
                <div style={{ fontSize: 9.5, color: c.dim }}>↳ {k.from}</div>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card c={c}>
        <SectionTitle c={c} icon={Network} title="Sentinel LangGraph — 8 agents" right={
          state === "awaiting_hitl" ? <Badge c={c} tone="warning" dot>Paused at HITL gate</Badge>
            : state === "filed" ? <Badge c={c} tone={decision === "approved" ? "success" : "warning"}>{decision === "approved" ? "Complaint filed" : "Returned to manual queue"}</Badge>
              : <Badge c={c} tone={state === "running" ? "primary" : "sub"} dot={state === "running"}>{state === "running" ? "Streaming" : "Idle — awaiting event"}</Badge>} />
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
          {SENTINEL_NODES.map((n, i) => {
            const st = nodeState(i);
            const degraded = n.id === "internet_intel" && failCrawler && i < node;
            const col = degraded ? c.warning : st === "done" ? c.success : st === "active" ? c.primary : c.dim;
            return (
              <div key={n.id} style={{ background: c.card2, borderRadius: 10, padding: 12, border: `1px solid ${st === "active" ? c.primary + "66" : c.border}`, opacity: st === "idle" ? .55 : 1, transition: "all .4s", position: "relative" }}>
                {st === "active" && <span style={{ position: "absolute", top: 10, right: 10, width: 8, height: 8, borderRadius: 99, background: c.primary, animation: "tlpulse 1.2s infinite" }} />}
                {degraded && <span style={{ position: "absolute", top: 8, right: 8 }}><AlertTriangle size={13} color={c.warning} /></span>}
                {st === "done" && !degraded && <span style={{ position: "absolute", top: 8, right: 8 }}><CheckCircle2 size={13} color={c.success} /></span>}
                <n.icon size={16} color={col} />
                <div style={{ fontSize: 11.5, fontWeight: 700, color: c.text, marginTop: 7 }}>{i + 1}. {n.name}</div>
                <div style={{ fontSize: 10, color: c.sub, marginTop: 3, lineHeight: 1.45 }}>{degraded ? "Adapter failed — workflow continued (graceful degradation)" : n.desc}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1.35fr 1fr", gap: 14, alignItems: "start" }}>
        <Card c={c}>
          <SectionTitle c={c} icon={Globe} title="Internet intelligence — click a match to see its evidence basis"
            right={<Badge c={c} tone={crawlDone ? "danger" : "sub"}>{node > 1 ? (failCrawler ? "Degraded — feed data only" : "3 live · 1 removed") : "Awaiting crawl"}</Badge>} />
          {crawlDone ? CRAWL_MATCHES.map((m, i) => (
            <div key={i} style={{ borderBottom: `1px solid ${c.border}`, animation: "tlfadein .4s", animationDelay: `${i * .1}s`, animationFillMode: "backwards" }}>
              <div onClick={() => setExpanded(expanded === i ? null : i)} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 4px", cursor: "pointer" }}>
                <ChevronRight size={13} color={c.dim} style={{ transform: expanded === i ? "rotate(90deg)" : "none", transition: "transform .2s", flexShrink: 0 }} />
                <ExternalLink size={14} color={m.status === "live" ? c.danger : c.dim} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12.5, fontWeight: 700, color: c.text }}>{m.platform} · <span style={{ fontFamily: "monospace", fontWeight: 500 }}>{m.handle}</span></div>
                  <div style={{ fontSize: 10.5, color: c.dim, fontFamily: "monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.url} · first seen {m.firstSeen} · {m.views} views · {m.region}</div>
                </div>
                <Badge c={c} tone={m.ev.cls.startsWith("exact") ? "danger" : "warning"}>{m.ev.cls}</Badge>
                <Badge c={c} tone={m.status === "live" ? "danger" : "sub"} dot={m.status === "live"}>{m.status}</Badge>
              </div>
              {expanded === i && <div style={{ paddingBottom: 12, paddingLeft: 25 }}><EvidenceBasis c={c} ev={m.ev} /></div>}
            </div>
          )) : (
            <div style={{ padding: "26px 0", textAlign: "center" }}>
              {failCrawler && node > 1
                ? <div style={{ fontSize: 12.5, color: c.warning, lineHeight: 1.6 }}><AlertTriangle size={18} style={{ display: "block", margin: "0 auto 8px" }} />CrawlerAdapter failed — Sentinel continued with ThreatFeedProvider data only.<br /><span style={{ color: c.sub, fontSize: 11.5 }}>Spec rule honored: never break the main TruthLens workflow.</span></div>
                : <><Skeleton c={c} h={12} w="70%" style={{ margin: "0 auto 8px" }} /><Skeleton c={c} h={12} w="55%" style={{ margin: "0 auto 8px" }} /><Skeleton c={c} h={12} w="62%" style={{ margin: "0 auto" }} /></>}
            </div>
          )}
          {node > 2 && (
            <div style={{ marginTop: 12, padding: 11, background: c.card2, borderRadius: 9, fontSize: 12, color: c.sub, lineHeight: 1.6, animation: "tlfadein .4s" }}>
              <b style={{ color: c.warning }}>Campaign CAMP-2026-017:</b> generator fingerprint hit TL-2026-0329 at distance 0.11 in the main app's ChromaDB case store — same operator toolchain, shared payment domain.
            </div>
          )}
        </Card>

        <div style={{ display: "grid", gap: 14 }}>
          <Card c={c} style={{ textAlign: "center" }}>
            <SectionTitle c={c} icon={GaugeIcon} title="Composite threat score" />
            <ThreatGauge c={c} value={score} />
            <div style={{ fontSize: 11, color: c.sub, marginTop: 18 }}>{score > 0 ? "confidence×5 + live hosts + campaign + feed hits · ≥ 7.0 → human review" : "Computed after correlation completes"}</div>
          </Card>
          <Card c={c} style={{ border: state === "awaiting_hitl" ? `1.5px solid ${c.warning}` : undefined }}>
            <SectionTitle c={c} icon={UserCheck} title="Human-in-the-loop gate" right={state === "awaiting_hitl" ? <Badge c={c} tone="warning" dot>Action required</Badge> : decision ? <Badge c={c} tone={decision === "approved" ? "success" : "sub"}>{decision}</Badge> : <Badge c={c} tone="sub">Waiting</Badge>} />
            {state === "awaiting_hitl" ? (
              <>
                <p style={{ fontSize: 12.5, color: c.sub, lineHeight: 1.6, margin: "0 0 10px" }}>Sentinel proposes filing for <b style={{ color: c.text }}>TL-2026-0342</b> (threat 9.1, 3 live hosts). Review exactly what will be submitted:</p>
                <button onClick={() => setPkgOpen(o => !o)} style={{ display: "flex", alignItems: "center", gap: 7, width: "100%", background: c.card2, border: `1px solid ${c.border}`, borderRadius: 9, padding: "9px 12px", color: c.text, fontSize: 12, fontWeight: 700, cursor: "pointer", marginBottom: 10, ...font }}>
                  <FileText size={13} color={c.primary} /> {pkgOpen ? "Hide" : "Review"} cybercrime package (4 observations)
                  <ChevronRight size={13} color={c.dim} style={{ marginLeft: "auto", transform: pkgOpen ? "rotate(90deg)" : "none", transition: "transform .2s" }} />
                </button>
                {pkgOpen && (
                  <div style={{ background: c.card2, borderRadius: 10, padding: 12, marginBottom: 12, fontSize: 11, color: c.sub, lineHeight: 1.6, maxHeight: 240, overflowY: "auto", animation: "tlfadein .25s" }}>
                    <div style={{ fontWeight: 700, color: c.text, marginBottom: 4 }}>Complaint: Online financial fraud — synthetic media · ref TL-2026-0342</div>
                    <div style={{ marginBottom: 8 }}>Original verdict attached: FAKE 96% (calibrated, signed report TL-2026-0342-R1).</div>
                    <div style={{ fontWeight: 700, color: c.text, marginBottom: 4 }}>Observations to be submitted:</div>
                    {CRAWL_MATCHES.map((m, i) => (
                      <div key={i} style={{ padding: "6px 0", borderTop: `1px solid ${c.border}`, fontFamily: "monospace", fontSize: 10.5 }}>
                        {i + 1}. {m.url} — {m.ev.cls} (pHash Δ{m.ev.phash}, CLIP {m.ev.cos}{m.ev.fp ? ", fingerprint ✓" : ""}) · {m.status} · found via {m.ev.method.toLowerCase()}
                      </div>
                    ))}
                    <div style={{ fontWeight: 700, color: c.text, margin: "8px 0 4px" }}>Attachments:</div>
                    forensic_report.pdf (signed) · evidence_hashes.json · knowledge_graph.json · takedown_url_list.txt
                  </div>
                )}
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={approve} style={{ flex: 1, display: "flex", gap: 7, alignItems: "center", justifyContent: "center", background: c.success, border: "none", borderRadius: 9, padding: "10px 0", color: "#062712", fontSize: 12.5, fontWeight: 800, cursor: "pointer", ...font }}><Send size={14} /> Approve & file</button>
                  <button onClick={reject} style={{ flex: 1, display: "flex", gap: 7, alignItems: "center", justifyContent: "center", background: "transparent", border: `1px solid ${c.border}`, borderRadius: 9, padding: "10px 0", color: c.sub, fontSize: 12.5, fontWeight: 700, cursor: "pointer", ...font }}><ThumbsDown size={14} /> Reject</button>
                </div>
              </>
            ) : (
              <div style={{ fontSize: 12, color: c.sub, lineHeight: 1.6 }}>
                {ackNo
                  ? <><CheckCircle2 size={16} color={c.success} style={{ marginBottom: 6 }} /><div><b style={{ color: c.success }}>Filed.</b> Acknowledgment <b style={{ color: c.text, fontFamily: "monospace" }}>{ackNo}</b> — the reviewed package (4 URL observations with evidence basis + signed forensic report) was submitted. pHash now under 24h re-crawl watch.</div></>
                  : decision === "rejected" ? "Auto-filing rejected — case returned to the manual queue. No external submission was made."
                    : "Activates when threat ≥ 7.0. The analyst reviews the full package — every URL, how it was found, and the evidence basis — before anything leaves the system."}
              </div>
            )}
          </Card>
        </div>
      </div>

      <Card c={c}>
        <SectionTitle c={c} icon={Database} title="Sentinel audit log" right={<Badge c={c} tone="sub">{logs.length} entries · append-only</Badge>} />
        <div style={{ maxHeight: 210, overflowY: "auto", fontFamily: "monospace" }}>
          {logs.length === 0 && <div style={{ fontSize: 12, color: c.dim, padding: "14px 0" }}>Trigger the event to start — every adapter call, failure and human decision lands here.</div>}
          {logs.map((l, i) => (
            <div key={i} style={{ display: "flex", gap: 10, padding: "6px 0", borderBottom: `1px solid ${c.border}`, fontSize: 11.5, animation: "tlfadein .3s" }}>
              <span style={{ color: c.dim, flexShrink: 0 }}>{l.t}</span>
              <span style={{ color: l.ok ? c.accent : c.warning, flexShrink: 0, fontWeight: 700, minWidth: 150 }}>{l.agent}</span>
              <span style={{ color: c.sub }}>{l.message}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

/* ============================================================
   SENTINEL — enterprise SOC workspace (tabbed)
   ============================================================ */
const SENTINEL_TREND = [
  { d: "Mon", found: 4, taken: 2 }, { d: "Tue", found: 7, taken: 3 }, { d: "Wed", found: 5, taken: 5 },
  { d: "Thu", found: 11, taken: 6 }, { d: "Fri", found: 8, taken: 7 }, { d: "Sat", found: 6, taken: 4 }, { d: "Sun", found: 9, taken: 8 },
];
const GEO_ROWS = [["India", 24, "#4F8EF7"], ["UAE", 11, "#00C9A7"], ["United States", 8, "#FFC857"], ["Singapore", 5, "#B084F5"], ["Others", 6, "#5B6B85"]];

const SentinelOverview = ({ c }) => (
  <div style={{ display: "grid", gap: 14 }}>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 14 }}>
      <StatCard c={c} icon={Globe} label="Distribution points found" value="54" delta="+18" up={false} tone="danger" />
      <StatCard c={c} icon={Shield} label="Takedowns issued" value="35" delta="+9" up tone="accent" />
      <StatCard c={c} icon={FileText} label="Complaints filed" value="7" delta="+2" up tone="primary" />
      <StatCard c={c} icon={Radio} label="Active monitors" value="12" delta="+3" up tone="warning" />
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 14 }}>
      <Card c={c}>
        <SectionTitle c={c} icon={Activity} title="Distribution vs takedowns — this week" right={<Badge c={c} tone="accent">65% takedown rate</Badge>} />
        <ResponsiveContainer width="100%" height={230}>
          <AreaChart data={SENTINEL_TREND}>
            <defs><linearGradient id="sf" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={c.danger} stopOpacity=".35" /><stop offset="100%" stopColor={c.danger} stopOpacity="0" /></linearGradient>
              <linearGradient id="st" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={c.accent} stopOpacity=".3" /><stop offset="100%" stopColor={c.accent} stopOpacity="0" /></linearGradient></defs>
            <CartesianGrid stroke={c.border} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="d" stroke={c.dim} fontSize={11} tickLine={false} axisLine={false} />
            <YAxis stroke={c.dim} fontSize={11} tickLine={false} axisLine={false} width={26} />
            <Tooltip contentStyle={{ background: c.card2, border: `1px solid ${c.border}`, borderRadius: 10, fontSize: 12 }} />
            <Area type="monotone" dataKey="found" name="New distributions" stroke={c.danger} fill="url(#sf)" strokeWidth={2} />
            <Area type="monotone" dataKey="taken" name="Taken down" stroke={c.accent} fill="url(#st)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </Card>
      <Card c={c}>
        <SectionTitle c={c} icon={Globe} title="Geographic distribution" />
        {GEO_ROWS.map(([country, n, col]) => (
          <div key={country} style={{ marginBottom: 13 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, marginBottom: 4 }}><span style={{ color: c.sub }}>{country}</span><span style={{ color: c.text, fontWeight: 700 }}>{n}</span></div>
            <div style={{ height: 7, borderRadius: 99, background: c.border }}><div style={{ width: `${n * 4}%`, height: "100%", borderRadius: 99, background: col, transition: "width 1s" }} /></div>
          </div>
        ))}
      </Card>
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
      <Card c={c}>
        <SectionTitle c={c} icon={Clock} title="Recent activity" />
        {[["New copy detected on messaging channel", "2m ago", c.danger], ["Takedown confirmed — social network", "22m ago", c.success], ["Complaint CMP-707F acknowledged by portal", "1h ago", c.primary], ["Monitor enrolled for TL-2026-0341", "3h ago", c.warning]].map(([t, time, col], i) => (
          <div key={i} style={{ display: "flex", gap: 10, padding: "9px 0", borderBottom: `1px solid ${c.border}`, alignItems: "center" }}>
            <span style={{ width: 7, height: 7, borderRadius: 99, background: col, flexShrink: 0 }} />
            <span style={{ fontSize: 12, color: c.text, flex: 1 }}>{t}</span><span style={{ fontSize: 10.5, color: c.dim }}>{time}</span>
          </div>
        ))}
      </Card>
      <Card c={c}>
        <SectionTitle c={c} icon={Layers} title="Investigation queue" />
        {[["TL-2026-0342", "Awaiting HITL approval", "warning"], ["TL-2026-0341", "Monitoring active", "accent"], ["TL-2026-0339", "Complaint filed", "primary"], ["TL-2026-0333", "Correlation running", "sub"]].map(([id, st, tone], i) => (
          <div key={i} style={{ display: "flex", gap: 10, padding: "9px 0", borderBottom: `1px solid ${c.border}`, alignItems: "center" }}>
            <span style={{ fontSize: 12, fontFamily: "monospace", color: c.text, fontWeight: 600 }}>{id}</span>
            <Badge c={c} tone={tone}>{st}</Badge>
          </div>
        ))}
      </Card>
    </div>
  </div>
);

const WEBSITE_ROWS = [
  { name: "QuickFinance Tips", url: "videosite.example/watch?v=x91", domain: "videosite.example", rep: "Low", cat: "Video sharing", country: "IN", firstSeen: "Jul 6", lastSeen: "Jul 8", cred: "Unverified", risk: 92, threat: "HIGH", status: "live", sim: 98 },
  { name: "CEO Updates Official", url: "social.example/p/88231", domain: "social.example", rep: "Medium", cat: "Social network", country: "IN", firstSeen: "Jul 6", lastSeen: "Jul 8", cred: "Unverified", risk: 88, threat: "HIGH", status: "live", sim: 96 },
  { name: "Insider Market X", url: "t.me/insider_market_x", domain: "t.me", rep: "Low", cat: "Messaging", country: "AE", firstSeen: "Jul 7", lastSeen: "Jul 8", cred: "Anonymous", risk: 90, threat: "HIGH", status: "live", sim: 94 },
  { name: "Forum Mirror 49221", url: "forum.example/board/vid/49221", domain: "forum.example", rep: "Low", cat: "Forum", country: "US", firstSeen: "Jul 5", lastSeen: "Jul 6", cred: "Unverified", risk: 61, threat: "MEDIUM", status: "removed", sim: 91 },
];
const SentinelWebsites = ({ c }) => {
  const [sort, setSort] = useState("risk");
  const rows = [...WEBSITE_ROWS].sort((a, b) => sort === "sim" ? b.sim - a.sim : sort === "date" ? a.firstSeen.localeCompare(b.firstSeen) : b.risk - a.risk);
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <Card c={c}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, flexWrap: "wrap", gap: 8 }}>
          <SectionTitle c={c} icon={Globe} title="Discovered websites — full investigation detail" right={null} />
          <div style={{ display: "flex", gap: 6 }}>{[["risk", "Risk"], ["sim", "Similarity"], ["date", "Date"]].map(([k, l]) => (
            <button key={k} onClick={() => setSort(k)} style={{ padding: "6px 11px", borderRadius: 7, fontSize: 11, fontWeight: 600, cursor: "pointer", ...font, background: sort === k ? c.primary + "1F" : "transparent", color: sort === k ? c.primary : c.sub, border: `1px solid ${sort === k ? c.primary + "55" : c.border}` }}>Sort: {l}</button>
          ))}</div>
        </div>
        {rows.map((w, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "60px 1fr auto", gap: 12, padding: "12px 0", borderBottom: `1px solid ${c.border}`, alignItems: "center" }}>
            <div style={{ width: 60, height: 42, borderRadius: 7, background: "#0A0F1A", border: `1px solid ${c.border}`, display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
              <svg viewBox="0 0 60 42" style={{ width: "100%" }}><rect width="60" height="42" fill="#101826" /><ellipse cx="30" cy="18" rx="10" ry="13" fill="#2A3A52" /><rect x="14" y="30" width="32" height="10" fill="#22304a" /><rect x="4" y="4" width="18" height="4" rx="2" fill={c.danger} opacity=".5" /></svg>
            </div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 12.5, fontWeight: 700, color: c.text }}>{w.name} <Badge c={c} tone={w.status === "live" ? "danger" : "sub"} dot={w.status === "live"}>{w.status}</Badge></div>
              <div style={{ fontSize: 10.5, color: c.dim, fontFamily: "monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{w.url}</div>
              <div style={{ display: "flex", gap: 10, marginTop: 4, fontSize: 10, color: c.sub, flexWrap: "wrap" }}>
                <span>domain rep: <b style={{ color: w.rep === "Low" ? c.danger : c.warning }}>{w.rep}</b></span>
                <span>{w.cat}</span><span>{w.country}</span><span>first {w.firstSeen} · last {w.lastSeen}</span><span>source: {w.cred}</span>
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 20, fontWeight: 800, color: w.risk > 80 ? c.danger : c.warning, fontFamily: "'Space Grotesk', Inter, sans-serif" }}>{w.risk}</div>
              <div style={{ fontSize: 9.5, color: c.sub }}>risk · {w.sim}% match</div>
              <Badge c={c} tone={w.threat === "HIGH" ? "danger" : "warning"}>{w.threat}</Badge>
            </div>
          </div>
        ))}
      </Card>
    </div>
  );
};

const SentinelEvidence = ({ c }) => (
  <div style={{ display: "grid", gap: 14 }}>
    <Card c={c}>
      <SectionTitle c={c} icon={GitCompare} title="Side-by-side evidence comparison" right={<Badge c={c} tone="danger">98% match · exact copy</Badge>} />
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 16, alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 11, color: c.sub, marginBottom: 6, fontWeight: 600 }}>ORIGINAL EXHIBIT (EV-01)</div>
          <FaceExhibit c={c} overlay="heat" />
          <div style={{ fontSize: 10.5, color: c.dim, marginTop: 6, fontFamily: "monospace" }}>pHash f0e1d2c3…9687 · verdict FAKE 96%</div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 22, fontWeight: 800, color: c.accent, fontFamily: "'Space Grotesk', Inter, sans-serif" }}>≈</div>
          <div style={{ fontSize: 10, color: c.sub, marginTop: 4 }}>pHash Δ3<br />CLIP 0.97<br />fingerprint ✓</div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: c.sub, marginBottom: 6, fontWeight: 600 }}>CRAWLED COPY (@quick_finance_tips)</div>
          <FaceExhibit c={c} overlay="attn" />
          <div style={{ fontSize: 10.5, color: c.dim, marginTop: 6, fontFamily: "monospace" }}>videosite.example · 48.2K views · live</div>
        </div>
      </div>
    </Card>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
      <Card c={c}>
        <SectionTitle c={c} icon={Fingerprint} title="Metadata comparison" />
        {[["Dimensions", "3840×2160", "1280×720", true], ["Software tag", "PIL 10.1", "absent", true], ["Created", "Jul 8 10:48", "Jul 6 (re-encoded)", true], ["Color profile", "sRGB stripped", "sRGB", false]].map(([k, a, b, diff]) => (
          <div key={k} style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, padding: "8px 0", borderBottom: `1px solid ${c.border}`, fontSize: 11.5 }}>
            <span style={{ color: c.sub }}>{k}</span><span style={{ color: c.text, fontFamily: "monospace", fontSize: 10.5 }}>{a}</span><span style={{ color: diff ? c.warning : c.text, fontFamily: "monospace", fontSize: 10.5 }}>{b}</span>
          </div>
        ))}
      </Card>
      <Card c={c}>
        <SectionTitle c={c} icon={ScanFace} title="Manipulated region highlighting" />
        <FaceExhibit c={c} overlay="ela" />
        <div style={{ fontSize: 11.5, color: c.sub, marginTop: 10, lineHeight: 1.6 }}>Same manipulated region (jawline + mouth) appears in every crawled copy — confirms all instances derive from the one fabricated exhibit, not independent fakes.</div>
        <button style={{ marginTop: 12, display: "flex", gap: 7, alignItems: "center", background: c.card2, border: `1px solid ${c.border}`, borderRadius: 8, padding: "8px 14px", color: c.text, fontSize: 12, fontWeight: 600, cursor: "pointer", ...font }}><Download size={13} color={c.primary} /> Download evidence bundle</button>
      </Card>
    </div>
  </div>
);

const SentinelPerson = ({ c, notify }) => {
  const [scanning, setScanning] = useState(false);
  const [done, setDone] = useState(false);
  const scan = () => { setScanning(true); setDone(false); setTimeout(() => { setScanning(false); setDone(true); notify("Exposure scan complete"); }, 2200); };
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <Card c={c}>
        <SectionTitle c={c} icon={ScanFace} title="Person monitoring — reverse deepfake search" right={<Badge c={c} tone="accent">Protect an individual</Badge>} />
        <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: 18, alignItems: "center" }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ width: 140, height: 140, margin: "0 auto", borderRadius: 14, background: "#0A0F1A", border: `2px dashed ${c.border}`, display: "flex", alignItems: "center", justifyContent: "center", position: "relative", overflow: "hidden" }}>
              <svg viewBox="0 0 140 140" style={{ width: "100%" }}><rect width="140" height="140" fill="#101826" /><ellipse cx="70" cy="60" rx="30" ry="36" fill="#2A3A52" /><ellipse cx="70" cy="125" rx="46" ry="28" fill="#22304a" /></svg>
              {scanning && <div style={{ position: "absolute", left: 0, right: 0, height: 3, background: c.accent, boxShadow: `0 0 12px ${c.accent}`, animation: "tlscan 1.4s linear infinite" }} />}
            </div>
            <button onClick={scan} disabled={scanning} style={{ marginTop: 12, display: "flex", gap: 7, alignItems: "center", justifyContent: "center", width: "100%", background: `linear-gradient(135deg, ${c.primary}, #3B7BE0)`, border: "none", borderRadius: 9, padding: "10px 0", color: "#fff", fontSize: 12.5, fontWeight: 700, cursor: scanning ? "wait" : "pointer", ...font }}>{scanning ? <><RefreshCw size={14} style={{ animation: "tlspin 1s linear infinite" }} /> Scanning…</> : <><Search size={14} /> Scan for exposure</>}</button>
          </div>
          <div>
            {!done ? <div style={{ fontSize: 13, color: c.sub, lineHeight: 1.7 }}>Upload a person's face to search across supported sources for visually similar images, deepfakes, face-swaps, edited videos, and public appearances. Generates an exposure score, match counts, platforms and a risk assessment.</div>
              : <div style={{ display: "grid", gap: 12, animation: "tlfadein .4s" }}>
                <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                  <Ring c={c} value={67} color={c.danger} size={90} label="exposure" />
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 8, flex: 1 }}>
                    {[["14", "matches"], ["3", "deepfakes"], ["5", "videos"], ["6", "platforms"]].map(([v, l]) => (
                      <div key={l} style={{ background: c.card2, borderRadius: 9, padding: "10px 6px", textAlign: "center" }}><div style={{ fontSize: 19, fontWeight: 800, color: c.text, fontFamily: "'Space Grotesk', Inter, sans-serif" }}>{v}</div><div style={{ fontSize: 10, color: c.sub }}>{l}</div></div>
                    ))}
                  </div>
                </div>
                <div style={{ background: c.danger + "14", borderRadius: 9, padding: 11, fontSize: 12, color: c.danger, fontWeight: 600 }}>HIGH RISK — synthetic media of this person is in active circulation across 6 platforms.</div>
              </div>}
          </div>
        </div>
      </Card>
    </div>
  );
};

const COMPLAINT_ROWS = [
  { id: "CMP-707F22BD", case: "TL-2026-0342", status: "under_investigation", portal: "NCRP", ack: "NCRP-2026-482913", submitted: "Jul 8, 11:04", officer: "Insp. R. Gill", jurisdiction: "Cyber Cell — Punjab", updated: "1h ago" },
  { id: "CMP-6612AA01", case: "TL-2026-0339", status: "acknowledged", portal: "NCRP", ack: "NCRP-2026-471882", submitted: "Jul 6, 15:22", officer: "Insp. P. Kaur", jurisdiction: "Cyber Cell — Punjab", updated: "1d ago" },
  { id: "CMP-5501CD77", case: "TL-2026-0329", status: "action_taken", portal: "NCRP", ack: "NCRP-2026-455910", submitted: "Jul 1, 09:40", officer: "Insp. R. Gill", jurisdiction: "Cyber Cell — Punjab", updated: "2d ago" },
];
const cStatusTone = (s) => ({ under_investigation: "warning", acknowledged: "primary", action_taken: "success", filed: "accent", rejected: "danger" }[s] || "sub");
const SentinelComplaints = ({ c, notify }) => {
  const [q, setQ] = useState("");
  const rows = COMPLAINT_ROWS.filter(r => r.id.toLowerCase().includes(q.toLowerCase()) || r.case.toLowerCase().includes(q.toLowerCase()));
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 14 }}>
        <StatCard c={c} icon={FileText} label="Total complaints" value="7" delta="+2" up tone="primary" />
        <StatCard c={c} icon={Clock} label="Under investigation" value="3" delta="" up tone="warning" />
        <StatCard c={c} icon={CheckCircle2} label="Action taken" value="2" delta="+1" up tone="accent" />
      </div>
      <Card c={c}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, background: c.card2, border: `1px solid ${c.border}`, borderRadius: 9, padding: "8px 12px", marginBottom: 14, maxWidth: 320 }}>
          <Search size={14} color={c.dim} /><input value={q} onChange={e => setQ(e.target.value)} placeholder="Search complaints by ID or case…" style={{ background: "transparent", border: "none", outline: "none", color: c.text, fontSize: 12.5, width: "100%", ...font }} />
        </div>
        <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead><tr>{["Complaint", "Case", "Portal", "Status", "Officer", "Submitted", "Updated", ""].map(h => <th key={h} style={{ textAlign: "left", padding: "8px 10px", color: c.dim, fontSize: 10, fontWeight: 700, letterSpacing: .6, textTransform: "uppercase", borderBottom: `1px solid ${c.border}` }}>{h}</th>)}</tr></thead>
          <tbody>{rows.map(r => (
            <tr key={r.id}>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}><div style={{ fontWeight: 700, color: c.text, fontFamily: "monospace" }}>{r.id}</div><div style={{ fontSize: 10, color: c.dim }}>{r.ack}</div></td>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}`, color: c.sub, fontFamily: "monospace" }}>{r.case}</td>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}><Badge c={c} tone="primary">{r.portal}</Badge></td>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}><Badge c={c} tone={cStatusTone(r.status)}>{r.status.replace(/_/g, " ")}</Badge></td>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}`, color: c.sub }}>{r.officer}</td>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}`, color: c.sub }}>{r.submitted}</td>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}`, color: c.dim }}>{r.updated}</td>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}><button onClick={() => notify("Complaint copy downloaded")} style={{ background: c.card2, border: `1px solid ${c.border}`, borderRadius: 7, padding: "5px 8px", cursor: "pointer", color: c.sub }}><Download size={13} /></button></td>
            </tr>
          ))}</tbody>
        </table></div>
        <div style={{ fontSize: 11, color: c.dim, marginTop: 12 }}>Portal-agnostic store — additional reporting portals plug in as CyberCrimeProvider adapters without changing this table.</div>
      </Card>
    </div>
  );
};

const SentinelCampaign = ({ c }) => (
  <div style={{ display: "grid", gap: 14 }}>
    <Card c={c}>
      <SectionTitle c={c} icon={Network} title="Campaign intelligence — correlated investigations" right={<Badge c={c} tone="danger">CAMP-2026-017 · 3 cases</Badge>} />
      <CorrelationGraph c={c} />
      <div style={{ fontSize: 12, color: c.sub, lineHeight: 1.6, marginTop: 10 }}>Three investigations share the same generator fingerprint and payment infrastructure — likely one operator. Dashed edges are inferred links pending analyst confirmation.</div>
    </Card>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
      <Card c={c}>
        <SectionTitle c={c} icon={Layers} title="Shared indicators" />
        {[["Generator fingerprint", "GEN-FP-0342 · 3 cases", c.danger], ["Payment domain", "pay-fastsettle.example", c.warning], ["Posting cadence", "10:40–10:50 window", c.primary], ["Manipulation technique", "diffusion face-insert", c.accent]].map(([k, v, col]) => (
          <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "9px 0", borderBottom: `1px solid ${c.border}`, fontSize: 12 }}><span style={{ color: c.sub }}>{k}</span><span style={{ color: col, fontWeight: 600, fontFamily: "monospace", fontSize: 11 }}>{v}</span></div>
        ))}
      </Card>
      <Card c={c}>
        <SectionTitle c={c} icon={History} title="Campaign timeline" />
        {[["Jul 1", "First case — TL-2026-0329"], ["Jul 6", "Second case — TL-2026-0339"], ["Jul 8", "Third case — TL-2026-0342"], ["Jul 8", "Campaign correlation confirmed"]].map(([d, e], i) => (
          <div key={i} style={{ display: "flex", gap: 10, padding: "8px 0", alignItems: "center" }}>
            <span style={{ fontSize: 10.5, color: c.dim, fontFamily: "monospace", minWidth: 40 }}>{d}</span>
            <span style={{ width: 7, height: 7, borderRadius: 99, background: c.primary, flexShrink: 0 }} />
            <span style={{ fontSize: 12, color: c.text }}>{e}</span>
          </div>
        ))}
      </Card>
    </div>
  </div>
);

const SentinelPage = ({ c, notify }) => {
  const [tab, setTab] = useState("overview");
  const tabs = [
    ["overview", "Overview", LayoutDashboard], ["investigation", "Live Investigation", Zap],
    ["evidence", "Evidence Explorer", GitCompare], ["websites", "Websites", Globe],
    ["person", "Person Monitoring", ScanFace], ["campaign", "Campaign", Network],
    ["complaints", "Complaints", FileText],
  ];
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>Sentinel Threat Intelligence</h1>
          <Badge c={c} tone="accent">Enterprise SOC · plug-and-play extension</Badge>
        </div>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub }}>Post-investigation platform: hunt the confirmed deepfake across the internet, correlate campaigns, and file cybercrime complaints with human-in-the-loop control.</p>
      </div>
      <div style={{ display: "flex", gap: 4, borderBottom: `1px solid ${c.border}`, overflowX: "auto", paddingBottom: 0 }}>
        {tabs.map(([k, l, I]) => (
          <button key={k} onClick={() => setTab(k)} style={{ display: "flex", gap: 7, alignItems: "center", padding: "10px 15px", background: "none", border: "none", borderBottom: `2px solid ${tab === k ? c.primary : "transparent"}`, color: tab === k ? c.text : c.sub, fontSize: 12.5, fontWeight: tab === k ? 700 : 500, cursor: "pointer", whiteSpace: "nowrap", ...font }}>
            <I size={14} color={tab === k ? c.primary : c.sub} /> {l}
          </button>
        ))}
      </div>
      <div key={tab} style={{ animation: "tlfadein .3s" }}>
        {tab === "overview" && <SentinelOverview c={c} />}
        {tab === "investigation" && <SentinelInvestigation c={c} notify={notify} />}
        {tab === "evidence" && <SentinelEvidence c={c} />}
        {tab === "websites" && <SentinelWebsites c={c} />}
        {tab === "person" && <SentinelPerson c={c} notify={notify} />}
        {tab === "campaign" && <SentinelCampaign c={c} />}
        {tab === "complaints" && <SentinelComplaints c={c} notify={notify} />}
      </div>
    </div>
  );
};


/* ============================================================
   LOGIN PAGE (JWT-style, gates the whole app)
   ============================================================ */
const LoginPage = ({ c, onLogin }) => {
  const [u, setU] = useState("admin");
  const [p, setP] = useState("Admin@123");
  const [remember, setRemember] = useState(true);
  const [err, setErr] = useState("");
  const [mode, setMode] = useState("login"); // login | reset
  const submit = () => {
    const found = SEED_USERS.find(x => x.username === u);
    if (!found || !found.enabled) { setErr("Invalid credentials or account disabled."); return; }
    if (found.locked) { setErr("Account locked. Contact an administrator."); return; }
    onLogin(found);
  };
  return (
    <div style={{ minHeight: "100vh", display: "flex", background: c.bg, ...font }}>
      {/* left brand panel */}
      <div style={{ flex: 1, position: "relative", display: "flex", flexDirection: "column", justifyContent: "center", padding: "0 8%", overflow: "hidden" }}>
        <div aria-hidden style={{ position: "absolute", inset: 0, background: `radial-gradient(700px 500px at 20% 20%, ${c.primary}22, transparent 60%), radial-gradient(600px 400px at 80% 90%, ${c.accent}18, transparent 60%)` }} />
        <div style={{ position: "relative" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 28 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, background: `linear-gradient(135deg, ${c.primary}, ${c.accent})`, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: `0 8px 30px ${c.primary}55` }}><Shield size={24} color="#fff" /></div>
            <div>
              <div style={{ fontSize: 20, fontWeight: 800, color: c.text, fontFamily: "'Space Grotesk', Inter, sans-serif" }}>TruthLens AI</div>
              <div style={{ fontSize: 12, color: c.sub }}>Autonomous Multimedia Forensics</div>
            </div>
          </div>
          <h1 style={{ fontSize: 38, fontWeight: 800, color: c.text, lineHeight: 1.1, margin: "0 0 16px", maxWidth: 480 }}>Explainable deepfake investigation, end to end.</h1>
          <p style={{ fontSize: 14, color: c.sub, lineHeight: 1.7, maxWidth: 440 }}>Seven-agent forensic pipeline, calibrated confidence, and post-investigation threat intelligence with human-in-the-loop cybercrime filing.</p>
          <div style={{ display: "flex", gap: 20, marginTop: 32 }}>
            {[["98.1%", "detection accuracy"], ["<48s", "time to verdict"], ["ISO 27042", "report standard"]].map(([v, l]) => (
              <div key={l}><div style={{ fontSize: 22, fontWeight: 800, color: c.text, fontFamily: "'Space Grotesk', Inter, sans-serif" }}>{v}</div><div style={{ fontSize: 11, color: c.sub }}>{l}</div></div>
            ))}
          </div>
        </div>
      </div>
      {/* right form */}
      <div style={{ width: 460, maxWidth: "100%", display: "flex", alignItems: "center", justifyContent: "center", padding: 32, background: c.card, borderLeft: `1px solid ${c.border}` }}>
        <div style={{ width: "100%", maxWidth: 340 }}>
          <div style={{ fontSize: 20, fontWeight: 800, color: c.text, marginBottom: 6 }}>{mode === "login" ? "Sign in" : "Reset password"}</div>
          <div style={{ fontSize: 12.5, color: c.sub, marginBottom: 24 }}>{mode === "login" ? "Access your investigation workspace." : "We'll email a reset link to your registered address."}</div>
          {mode === "login" ? <>
            <label style={{ fontSize: 11.5, fontWeight: 700, color: c.sub }}>Username</label>
            <input value={u} onChange={e => { setU(e.target.value); setErr(""); }} style={{ width: "100%", boxSizing: "border-box", margin: "6px 0 14px", background: c.card2, border: `1px solid ${c.border}`, borderRadius: 9, padding: "11px 12px", color: c.text, fontSize: 13, outline: "none", ...font }} />
            <label style={{ fontSize: 11.5, fontWeight: 700, color: c.sub }}>Password</label>
            <input type="password" value={p} onChange={e => { setP(e.target.value); setErr(""); }} onKeyDown={e => e.key === "Enter" && submit()} style={{ width: "100%", boxSizing: "border-box", margin: "6px 0 14px", background: c.card2, border: `1px solid ${c.border}`, borderRadius: 9, padding: "11px 12px", color: c.text, fontSize: 13, outline: "none", ...font }} />
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
              <label style={{ display: "flex", gap: 7, alignItems: "center", fontSize: 12, color: c.sub, cursor: "pointer" }}>
                <input type="checkbox" checked={remember} onChange={e => setRemember(e.target.checked)} style={{ accentColor: c.primary }} /> Remember me
              </label>
              <button onClick={() => setMode("reset")} style={{ background: "none", border: "none", color: c.primary, fontSize: 12, fontWeight: 600, cursor: "pointer", ...font }}>Forgot password?</button>
            </div>
            {err && <div style={{ background: c.danger + "18", color: c.danger, borderRadius: 8, padding: "9px 12px", fontSize: 12, marginBottom: 14 }}>{err}</div>}
            <button onClick={submit} style={{ width: "100%", background: `linear-gradient(135deg, ${c.primary}, #3B7BE0)`, border: "none", borderRadius: 10, padding: "12px 0", color: "#fff", fontSize: 14, fontWeight: 700, cursor: "pointer", boxShadow: `0 6px 20px ${c.primary}4D`, ...font }}>Sign in securely</button>
            <div style={{ fontSize: 10.5, color: c.dim, marginTop: 14, lineHeight: 1.6, textAlign: "center" }}>Protected by JWT session management · route-level RBAC.<br/>Demo: <b style={{ color: c.sub }}>admin / Admin@123</b> · try r.verma, s.iyer, k.nair, m.das</div>
          </> : <>
            <label style={{ fontSize: 11.5, fontWeight: 700, color: c.sub }}>Email or username</label>
            <input placeholder="you@agency.gov" style={{ width: "100%", boxSizing: "border-box", margin: "6px 0 16px", background: c.card2, border: `1px solid ${c.border}`, borderRadius: 9, padding: "11px 12px", color: c.text, fontSize: 13, outline: "none", ...font }} />
            <button onClick={() => { setMode("login"); }} style={{ width: "100%", background: c.primary, border: "none", borderRadius: 10, padding: "12px 0", color: "#fff", fontSize: 14, fontWeight: 700, cursor: "pointer", ...font }}>Send reset link</button>
            <button onClick={() => setMode("login")} style={{ width: "100%", background: "none", border: "none", color: c.sub, fontSize: 12.5, cursor: "pointer", marginTop: 12, ...font }}>← Back to sign in</button>
          </>}
        </div>
      </div>
    </div>
  );
};

/* ============================================================
   ADMIN PAGES (RBAC)
   ============================================================ */
const AdminUsers = ({ c, notify, currentUser }) => {
  const [users, setUsers] = useState(SEED_USERS);
  const [q, setQ] = useState("");
  const canManage = roleCan(currentUser.role, "user_management", "edit");
  const rows = users.filter(u => u.username.includes(q.toLowerCase()) || u.name.toLowerCase().includes(q.toLowerCase()));
  const toggle = (uname, field) => { setUsers(us => us.map(u => u.username === uname ? { ...u, [field]: !u[field] } : u)); notify(`${uname}: ${field} toggled`); };
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 10 }}>
        <div><h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>User Management</h1><p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub }}>{users.length} users · {users.filter(u=>u.enabled).length} active</p></div>
        {canManage && <button onClick={() => notify("New user form (demo)")} style={{ display: "flex", gap: 7, alignItems: "center", background: `linear-gradient(135deg, ${c.primary}, #3B7BE0)`, border: "none", borderRadius: 9, padding: "9px 15px", color: "#fff", fontSize: 12.5, fontWeight: 700, cursor: "pointer", boxShadow: `0 4px 16px ${c.primary}4D`, ...font }}><Plus size={14} /> Create user</button>}
      </div>
      {!canManage && <Card c={c} pad={14} style={{ borderLeft: `3px solid ${c.warning}` }}><div style={{ fontSize: 12.5, color: c.sub, display: "flex", gap: 8, alignItems: "center" }}><Lock size={14} color={c.warning} /> Your role (<b style={{color:c.text}}>{ROLE_DEFS[currentUser.role].label}</b>) has view-only access to user management.</div></Card>}
      <Card c={c}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, background: c.card2, border: `1px solid ${c.border}`, borderRadius: 9, padding: "8px 12px", marginBottom: 14, maxWidth: 320 }}>
          <Search size={14} color={c.dim} /><input value={q} onChange={e => setQ(e.target.value)} placeholder="Search users…" style={{ background: "transparent", border: "none", outline: "none", color: c.text, fontSize: 12.5, width: "100%", ...font }} />
        </div>
        <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
          <thead><tr>{["User", "Role", "Status", "Last login", "Actions"].map(h => <th key={h} style={{ textAlign: "left", padding: "8px 10px", color: c.dim, fontSize: 10.5, fontWeight: 700, letterSpacing: .8, textTransform: "uppercase", borderBottom: `1px solid ${c.border}` }}>{h}</th>)}</tr></thead>
          <tbody>{rows.map(u => (
            <tr key={u.username}>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}><div style={{ display: "flex", alignItems: "center", gap: 9 }}><div style={{ width: 30, height: 30, borderRadius: 99, background: `linear-gradient(135deg, ${c.primary}, ${c.accent})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800, color: "#fff" }}>{u.name.split(" ").map(x=>x[0]).join("")}</div><div><div style={{ fontWeight: 700, color: c.text }}>{u.name}</div><div style={{ fontSize: 10.5, color: c.dim, fontFamily: "monospace" }}>{u.username}</div></div></div></td>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}><Badge c={c} tone="primary">{ROLE_DEFS[u.role].label}</Badge></td>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}>{u.locked ? <Badge c={c} tone="danger" dot>Locked</Badge> : u.enabled ? <Badge c={c} tone="success" dot>Active</Badge> : <Badge c={c} tone="sub">Disabled</Badge>}</td>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}`, color: c.sub }}>{u.last}</td>
              <td style={{ padding: "11px 10px", borderBottom: `1px solid ${c.border}` }}>{canManage ? <div style={{ display: "flex", gap: 6 }}>
                <button onClick={() => toggle(u.username, "enabled")} title="Enable/disable" style={{ background: c.card2, border: `1px solid ${c.border}`, borderRadius: 7, padding: "5px 9px", fontSize: 11, color: c.sub, cursor: "pointer", ...font }}>{u.enabled ? "Disable" : "Enable"}</button>
                <button onClick={() => toggle(u.username, "locked")} title="Lock/unlock" style={{ background: c.card2, border: `1px solid ${c.border}`, borderRadius: 7, padding: "5px 9px", fontSize: 11, color: c.sub, cursor: "pointer", ...font }}>{u.locked ? "Unlock" : "Lock"}</button>
                <button onClick={() => notify(`Password reset link sent for ${u.username}`)} style={{ background: c.card2, border: `1px solid ${c.border}`, borderRadius: 7, padding: "5px 9px", fontSize: 11, color: c.sub, cursor: "pointer", ...font }}>Reset</button>
              </div> : <span style={{ color: c.dim, fontSize: 11 }}>—</span>}</td>
            </tr>
          ))}</tbody>
        </table></div>
      </Card>
    </div>
  );
};

const AdminRoles = ({ c, currentUser, notify }) => {
  const [role, setRole] = useState("investigator");
  const canEdit = roleCan(currentUser.role, "role_management", "edit");
  const d = ROLE_DEFS[role];
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div><h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>Role Management</h1><p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub }}>{Object.keys(ROLE_DEFS).length} roles · module-level permission matrix</p></div>
      <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: 14, alignItems: "start" }}>
        <Card c={c} pad={10}>{Object.entries(ROLE_DEFS).map(([k, v]) => (
          <button key={k} onClick={() => setRole(k)} style={{ display: "flex", alignItems: "center", gap: 9, width: "100%", padding: "10px 12px", background: role === k ? c.primary + "14" : "transparent", border: `1px solid ${role === k ? c.primary + "44" : "transparent"}`, borderRadius: 9, cursor: "pointer", marginBottom: 3, textAlign: "left", ...font }}>
            <Users size={15} color={role === k ? c.primary : c.sub} /><div style={{ flex: 1 }}><div style={{ fontSize: 12.5, fontWeight: 700, color: c.text }}>{v.label}</div>{v.all && <div style={{ fontSize: 10, color: c.accent }}>full access</div>}</div>
          </button>
        ))}</Card>
        <Card c={c}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <div><div style={{ fontSize: 15, fontWeight: 800, color: c.text }}>{d.label} · permission matrix</div><div style={{ fontSize: 11, color: c.sub }}>{d.all ? "This role has every permission on every module." : "Green = granted. " + (canEdit ? "Click to toggle." : "View-only for your role.")}</div></div>
            {canEdit && <button onClick={() => notify(`Cloned ${d.label}`)} style={{ background: c.card2, border: `1px solid ${c.border}`, borderRadius: 8, padding: "7px 12px", fontSize: 12, color: c.text, fontWeight: 600, cursor: "pointer", ...font }}>Clone role</button>}
          </div>
          <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11.5 }}>
            <thead><tr><th style={{ textAlign: "left", padding: "7px 8px", color: c.dim, fontSize: 10, textTransform: "uppercase", letterSpacing: .6, borderBottom: `1px solid ${c.border}` }}>Module</th>{RBAC_PERMS.map(p => <th key={p} style={{ padding: "7px 4px", color: c.dim, fontSize: 9.5, textTransform: "uppercase", borderBottom: `1px solid ${c.border}` }}>{p}</th>)}</tr></thead>
            <tbody>{RBAC_MODULES.map(m => (
              <tr key={m}><td style={{ padding: "8px", color: c.text, fontWeight: 600, borderBottom: `1px solid ${c.border}`, whiteSpace: "nowrap" }}>{m.replace(/_/g, " ")}</td>{RBAC_PERMS.map(p => { const on = roleCan(role, m, p); return <td key={p} style={{ textAlign: "center", padding: "8px 4px", borderBottom: `1px solid ${c.border}` }}><span style={{ display: "inline-block", width: 16, height: 16, borderRadius: 5, background: on ? c.success + "26" : c.card2, border: `1px solid ${on ? c.success : c.border}`, cursor: canEdit && !d.all ? "pointer" : "default", lineHeight: "14px", fontSize: 10, color: c.success }} onClick={() => canEdit && !d.all && notify(`${role}: ${m}.${p} toggled`)}>{on ? "✓" : ""}</span></td>; })}</tr>
            ))}</tbody>
          </table></div>
        </Card>
      </div>
    </div>
  );
};

const AUDIT_SEED = [
  ["login", "admin", "role=super_admin remember=true", "just now"],
  ["complaint_filed", "k.nair", "CMP-707F22BD → NCRP-2026-482913", "6m ago"],
  ["report_download", "r.verma", "TL-2026-0342-R1.pdf", "18m ago"],
  ["permission_changed", "admin", "analyst: sentinel.export=true", "42m ago"],
  ["user_created", "admin", "user=gaurav role=analyst", "1h ago"],
  ["failed_login", "s.iyer", "bad password from 10.2.4.19", "2h ago"],
  ["account_locked", "s.iyer", "too many failures", "2h ago"],
  ["role_changed", "admin", "k.nair → human_reviewer", "3h ago"],
];
const AdminAudit = ({ c }) => {
  const [f, setF] = useState("All");
  const cats = ["All", "login", "failed_login", "complaint_filed", "report_download", "permission_changed", "user_created"];
  const rows = AUDIT_SEED.filter(r => f === "All" || r[0] === f);
  const tone = (a) => a.includes("fail") || a.includes("lock") ? "danger" : a.includes("login") ? "success" : a.includes("complaint") ? "accent" : "primary";
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div><h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: c.text }}>Audit Logs</h1><p style={{ margin: "4px 0 0", fontSize: 13, color: c.sub }}>Immutable · every security-relevant action · 7-year retention</p></div>
      <Card c={c}>
        <div style={{ display: "flex", gap: 7, marginBottom: 14, flexWrap: "wrap" }}>{cats.map(x => <button key={x} onClick={() => setF(x)} style={{ padding: "7px 12px", borderRadius: 8, fontSize: 11.5, fontWeight: 600, cursor: "pointer", ...font, background: f === x ? c.primary + "1F" : "transparent", color: f === x ? c.primary : c.sub, border: `1px solid ${f === x ? c.primary + "55" : c.border}` }}>{x.replace(/_/g, " ")}</button>)}</div>
        <div style={{ fontFamily: "monospace" }}>{rows.map((r, i) => (
          <div key={i} style={{ display: "flex", gap: 12, alignItems: "center", padding: "9px 4px", borderBottom: `1px solid ${c.border}`, fontSize: 11.5 }}>
            <span style={{ color: c.dim, minWidth: 66 }}>{r[3]}</span>
            <Badge c={c} tone={tone(r[0])}>{r[0].replace(/_/g, " ")}</Badge>
            <span style={{ color: c.text, minWidth: 70, fontWeight: 700 }}>{r[1]}</span>
            <span style={{ color: c.sub }}>{r[2]}</span>
          </div>
        ))}</div>
      </Card>
    </div>
  );
};

/* ============================================================
   APP SHELL
   ============================================================ */
export default function TruthLensAI() {
  const [dark, setDark] = useState(true);
  const [currentUser, setCurrentUser] = useState(null);
  const [page, setPage] = useState("dashboard");
  const [collapsed, setCollapsed] = useState(false);
  const [toast, setToast] = useState(null);
  const c = dark ? T.dark : T.light;
  const notify = (m) => { setToast(m); setTimeout(() => setToast(null), 2600); };

  useEffect(() => {
    const onKey = (e) => {
      if (e.key.toLowerCase() === "n" && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) setPage("new");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);


  if (!currentUser) {
    return (<><style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap'); h1{font-family:'Space Grotesk',Inter,sans-serif} input::placeholder{color:${c.dim}}`}</style><LoginPage c={c} onLogin={(u) => { setCurrentUser(u); setPage("dashboard"); notify(`Welcome, ${u.name}`); }} /></>);
  }
  const goPage = (p) => {
    const nav = NAV.find(n => n.id === p);
    if (nav && nav.mod && !roleCan(currentUser.role, nav.mod, "view")) { notify("You don't have access to that module."); return; }
    setPage(p);
  };
  const pages = {
    dashboard: <Dashboard c={c} setPage={setPage} notify={notify} />,
    users: <AdminUsers c={c} notify={notify} currentUser={currentUser} />,
    roles: <AdminRoles c={c} notify={notify} currentUser={currentUser} />,
    audit: <AdminAudit c={c} />,
    new: <NewInvestigation c={c} setPage={setPage} notify={notify} />,
    workspace: <Workspace c={c} setPage={setPage} />,
    console: <AgentConsole c={c} />,
    explain: <Explainability c={c} />,
    report: <Report c={c} notify={notify} />,
    sentinel: <SentinelPage c={c} notify={notify} />,
    history: <CaseHistory c={c} setPage={setPage} />,
    analytics: <Analytics c={c} />,
    settings: <SettingsPage c={c} dark={dark} setDark={setDark} />,
  };
  const safePage = pages[page] || pages.dashboard;

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: c.bg, position: "relative", isolation: "isolate", ...font }}>
      {dark && <div aria-hidden style={{
        position: "fixed", inset: 0, zIndex: -1, pointerEvents: "none",
        background: `radial-gradient(900px 520px at 12% -8%, ${c.primary}14, transparent 60%),
                     radial-gradient(760px 480px at 105% 108%, ${c.accent}0F, transparent 60%)`,
      }} />}
      {dark && <div aria-hidden style={{
        position: "fixed", inset: 0, zIndex: -1, pointerEvents: "none", opacity: .5,
        backgroundImage: `linear-gradient(${c.border}33 1px, transparent 1px), linear-gradient(90deg, ${c.border}33 1px, transparent 1px)`,
        backgroundSize: "44px 44px",
        maskImage: "radial-gradient(1200px 700px at 30% 0%, black, transparent 75%)",
        WebkitMaskImage: "radial-gradient(1200px 700px at 30% 0%, black, transparent 75%)",
      }} />}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap');
        h1 { font-family: 'Space Grotesk', 'Inter', sans-serif; letter-spacing: -0.025em !important; }
        ::selection { background: ${c.primary}55; }
        ::-webkit-scrollbar { width: 9px; height: 9px; }
        ::-webkit-scrollbar-thumb { background: ${c.border}; border-radius: 99px; border: 2px solid ${c.bg}; }
        ::-webkit-scrollbar-track { background: transparent; }
        table td, table th { font-variant-numeric: tabular-nums; }
        * { scrollbar-width: thin; scrollbar-color: ${c.border} transparent; }
        @keyframes tlshimmer { 0% { background-position: 200% 0 } 100% { background-position: -200% 0 } }
        @keyframes tlfadein { from { opacity: 0; transform: translateY(4px) } to { opacity: 1; transform: none } }
        @keyframes tlpulse { 0%,100% { opacity: 1 } 50% { opacity: .35 } }
        @keyframes tlbounce { 0%,80%,100% { transform: translateY(0) } 40% { transform: translateY(-5px) } }
        @keyframes tlslideup { from { opacity: 0; transform: translateY(12px) } to { opacity: 1; transform: none } }
        @keyframes tldash { to { stroke-dashoffset: -24 } }
        @keyframes tlspin { to { transform: rotate(360deg) } }
        @keyframes tlslidein { from { transform: translateX(100%) } to { transform: none } }
        @keyframes tlscan { 0% { top: 0 } 100% { top: 100% } }
        @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important } }
        select option { background: ${c.card}; color: ${c.text}; }
        input::placeholder, textarea::placeholder { color: ${c.dim}; }
        button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible { outline: 2px solid ${c.primary}; outline-offset: 2px; }
      `}</style>
      <Sidebar c={c} page={page} setPage={goPage} collapsed={collapsed} setCollapsed={setCollapsed} currentUser={currentUser} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <Topbar c={c} dark={dark} setDark={setDark} setPage={goPage} notify={notify} currentUser={currentUser} onLogout={() => { setCurrentUser(null); notify("Signed out"); }} />
        <main key={page} style={{ padding: 24, animation: "tlfadein .3s ease" }}>
          {safePage}
        </main>
      </div>
      <Toast c={c} toast={toast} />
    </div>
  );
}
