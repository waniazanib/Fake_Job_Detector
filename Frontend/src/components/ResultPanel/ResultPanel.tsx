import { motion } from 'framer-motion'
import { AlertTriangle, CheckCircle, AlertCircle, Info, Cpu, FileText } from 'lucide-react'
import type { AnalyzeResponse, FraudLabel, ConfidenceLevel } from '@/types/api'
import { LABEL_COPY, CONFIDENCE_COPY } from '@/types/api'
import ScoreGauge from '@/components/ScoreGauge/ScoreGauge'
import ShapChart  from '@/components/ShapChart/ShapChart'
import styles from './ResultPanel.module.css'

// ── Icon per label ────────────────────────────────────────────

const LABEL_ICON: Record<FraudLabel, React.ReactNode> = {
  LEGITIMATE: <CheckCircle  size={18} aria-hidden="true" />,
  CAUTION:    <AlertCircle  size={18} aria-hidden="true" />,
  SUSPICIOUS: <AlertTriangle size={18} aria-hidden="true" />,
}

// ── Confidence icon ───────────────────────────────────────────

const CONFIDENCE_ICON: Record<ConfidenceLevel, React.ReactNode> = {
  HIGH:   <Info size={14} aria-hidden="true" />,
  MEDIUM: <Info size={14} aria-hidden="true" />,
  LOW:    <AlertCircle size={14} aria-hidden="true" />,
}

// ── Props ─────────────────────────────────────────────────────

interface ResultPanelProps {
  result:    AnalyzeResponse
  onReset:   () => void
}

// ── Component ─────────────────────────────────────────────────

export default function ResultPanel({ result, onReset }: ResultPanelProps) {
  const {
    fraud_score,
    label,
    confidence,
    text_score,
    struct_score,
    shap_signals,
    summary,
  } = result

  return (
    <motion.section
      className={styles.panel}
      data-label={label}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      aria-label="Fraud analysis result"
    >
      {/* ── Top strip — coloured border per state ── */}
      <div className={styles.strip} aria-hidden="true" />

      {/* ── Header row ── */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={`badge badge--${label.toLowerCase()}`}>
            {LABEL_ICON[label]}
            {LABEL_COPY[label]}
          </span>
          <span className={styles.confidence} data-level={confidence}>
            {CONFIDENCE_ICON[confidence]}
            {CONFIDENCE_COPY[confidence]}
          </span>
        </div>
        <button
          className={styles.resetBtn}
          onClick={onReset}
          aria-label="Analyse another posting"
        >
          Analyse another
        </button>
      </div>

      {/* ── Summary sentence ── */}
      <p className={styles.summary}>{summary}</p>

      {/* ── Main content grid ── */}
      <div className={styles.grid}>

        {/* Left col — gauge */}
        <div className={styles.gaugeCol}>
          <ScoreGauge score={fraud_score} label={label} />
        </div>

        {/* Right col — branch breakdown */}
        <div className={styles.branchCol}>
          <h3 className={styles.branchTitle}>Model breakdown</h3>
          <p className={styles.branchSubtitle}>
            Two independent branches — fused for the final score
          </p>

          <div className={styles.branches}>
            {/* Text branch */}
            <div className={styles.branch}>
              <div className={styles.branchHeader}>
                <FileText size={15} className={styles.branchIcon} aria-hidden="true" />
                <span className={styles.branchName}>Text analysis</span>
                <span
                  className={styles.branchScore}
                  style={{ color: branchColor(text_score) }}
                >
                  {pct(text_score)}
                </span>
              </div>
              <div className={styles.branchTrack}>
                <motion.div
                  className={styles.branchFill}
                  style={{ background: branchColor(text_score) }}
                  initial={{ width: 0 }}
                  animate={{ width: `${text_score * 100}%` }}
                  transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
                />
              </div>
              <p className={styles.branchLabel}>DistilBERT — job description language</p>
            </div>

            {/* Structural branch */}
            <div className={styles.branch}>
              <div className={styles.branchHeader}>
                <Cpu size={15} className={styles.branchIcon} aria-hidden="true" />
                <span className={styles.branchName}>Structural signals</span>
                <span
                  className={styles.branchScore}
                  style={{ color: branchColor(struct_score) }}
                >
                  {pct(struct_score)}
                </span>
              </div>
              <div className={styles.branchTrack}>
                <motion.div
                  className={styles.branchFill}
                  style={{ background: branchColor(struct_score) }}
                  initial={{ width: 0 }}
                  animate={{ width: `${struct_score * 100}%` }}
                  transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1], delay: 0.35 }}
                />
              </div>
              <p className={styles.branchLabel}>XGBoost — salary, logo, location, links…</p>
            </div>

            {/* Fusion row */}
            <div className={styles.fusionRow}>
              <span className={styles.fusionLabel}>Fused fraud score</span>
              <span
                className={styles.fusionScore}
                style={{ color: branchColor(fraud_score) }}
              >
                {pct(fraud_score)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── SHAP chart ── */}
      {shap_signals.length > 0 && (
        <div className={styles.shapSection}>
          <hr className={styles.divider} />
          <ShapChart signals={shap_signals} />
        </div>
      )}
    </motion.section>
  )
}

// ── Helpers ───────────────────────────────────────────────────

function pct(score: number): string {
  return `${Math.round(score * 100)}%`
}

function branchColor(score: number): string {
  if (score < 0.35) return 'var(--color-score-low)'
  if (score < 0.65) return 'var(--color-score-mid)'
  if (score < 0.85) return 'var(--color-score-high)'
  return 'var(--color-score-critical)'
}