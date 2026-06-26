import { useEffect, useRef } from 'react'
import { motion, useMotionValue, useTransform, animate } from 'framer-motion'
import type { FraudLabel } from '@/types/api'
import styles from './ScoreGauge.module.css'

// ── Arc geometry ──────────────────────────────────────────────
// Half-circle: starts at 180° (left), ends at 0° (right)
// SVG arc drawn on a 200×120 viewBox — gauge sits in top half

const CX = 100   // centre x
const CY = 110   // centre y (below mid so arc crown is visible)
const R  = 80    // radius

/** Convert a score [0,1] to an SVG arc path clipped at that fraction. */
function describeArc(score: number): string {
  const clipped = Math.max(0, Math.min(1, score))
  
  // Start at bottom-left (210°) sweep to bottom-right (330°)
  // This gives a proper speedometer shape
  const startDeg = 180
  const totalDeg = 180
  const endDeg   = startDeg + clipped * totalDeg

  const startRad = (startDeg * Math.PI) / 180
  const endRad   = (endDeg   * Math.PI) / 180

  const x1 = CX + R * Math.cos(startRad)
  const y1 = CY + R * Math.sin(startRad)
  const x2 = CX + R * Math.cos(endRad)
  const y2 = CY + R * Math.sin(endRad)

  const largeArc = clipped > 0.5 ? 1 : 0

  // sweep-flag = 1 means clockwise
  return `M ${x1} ${y1} A ${R} ${R} 0 ${largeArc} 1 ${x2} ${y2}`
}

/** Full background track arc (always 180°). */
const TRACK_PATH = `M ${CX - R} ${CY} A ${R} ${R} 0 1 1 ${CX + R} ${CY}`

/** Score → stroke colour token */
function scoreToColor(score: number): string {
  if (score < 0.35) return 'var(--color-score-low)'
  if (score < 0.65) return 'var(--color-score-mid)'
  if (score < 0.85) return 'var(--color-score-high)'
  return 'var(--color-score-critical)'
}

// ── Label copy inside gauge ───────────────────────────────────

const LABEL_LINES: Record<FraudLabel, [string, string]> = {
  LEGITIMATE: ['Likely',      'Legitimate'],
  CAUTION:    ['Proceed with','Caution'],
  SUSPICIOUS: ['Likely',      'Fraudulent'],
}

// ── Props ─────────────────────────────────────────────────────

interface ScoreGaugeProps {
  score:      number      // [0, 1]
  label:      FraudLabel
  animate?:   boolean     // default true
}

// ── Component ─────────────────────────────────────────────────

export default function ScoreGauge({
  score,
  label,
  animate: shouldAnimate = true,
}: ScoreGaugeProps) {
  const motionScore = useMotionValue(shouldAnimate ? 0 : score)
  const arcRef      = useRef<SVGPathElement>(null)
  const pctRef      = useRef<SVGTextElement>(null)

  // Animate score counter and arc simultaneously
  useEffect(() => {
    if (!shouldAnimate) {
      motionScore.set(score)
      return
    }

    const controls = animate(motionScore, score, {
      duration: 1.1,
      ease:     [0.34, 1.06, 0.64, 1],   // spring-like overshoot
    })

    // Subscribe to drive arc path and percentage text imperatively
    // (avoids re-rendering the whole component on every animation frame)
    const unsub = motionScore.on('change', (v) => {
      if (arcRef.current) {
        arcRef.current.setAttribute('d', describeArc(v))
        arcRef.current.setAttribute('stroke', scoreToColor(v))
      }
      if (pctRef.current) {
        pctRef.current.textContent = `${Math.round(v * 100)}%`
      }
    })

    return () => {
      controls.stop()
      unsub()
    }
  }, [score, shouldAnimate, motionScore])

  const color      = scoreToColor(score)
  const [line1, line2] = LABEL_LINES[label]

  return (
    <div className={styles.wrapper} role="img" aria-label={`Fraud score ${Math.round(score * 100)}%`}>
      <svg
        viewBox="0 0 200 120"
        className={styles.svg}
        aria-hidden="true"
      >
        {/* ── Gradient def ── */}
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="var(--color-score-low)" />
            <stop offset="50%"  stopColor="var(--color-score-mid)" />
            <stop offset="100%" stopColor="var(--color-score-critical)" />
          </linearGradient>
        </defs>

        {/* ── Background track ── */}
        <path
          d={TRACK_PATH}
          fill="none"
          stroke="var(--color-border)"
          strokeWidth="10"
          strokeLinecap="round"
        />

        {/* ── Animated score arc ── */}
        <path
          ref={arcRef}
          d={describeArc(shouldAnimate ? 0 : score)}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
        />

        {/* ── Needle dot at arc tip ── */}
        <NeedleDot score={score} shouldAnimate={shouldAnimate} />

        {/* ── Percentage label ── */}
        <text
          ref={pctRef}
          x={CX}
          y={CY - 18}
          textAnchor="middle"
          dominantBaseline="auto"
          className={styles.pctText}
          fill={color}
        >
          {shouldAnimate ? '0%' : `${Math.round(score * 100)}%`}
        </text>

        {/* ── Fraud label lines ── */}
        <text
          x={CX}
          y={CY - 2}
          textAnchor="middle"
          dominantBaseline="auto"
          className={styles.labelLine1}
          fill="var(--color-text-muted)"
        >
          {line1}
        </text>
        <text
          x={CX}
          y={CY + 14}
          textAnchor="middle"
          dominantBaseline="auto"
          className={styles.labelLine2}
          fill="var(--color-text-primary)"
        >
          {line2}
        </text>

        {/* ── Scale ticks: 0%, 50%, 100% ── */}
        <text x={CX - R - 4} y={CY + 4} textAnchor="end"    className={styles.tick}>0%</text>
        <text x={CX}         y={CY - R - 8} textAnchor="middle" className={styles.tick}>50%</text>
        <text x={CX + R + 4} y={CY + 4} textAnchor="start"  className={styles.tick}>100%</text>
      </svg>
    </div>
  )
}

// ── Needle dot — separate component for clean motion value ────

interface NeedleDotProps {
  score:         number
  shouldAnimate: boolean
}

function NeedleDot({ score, shouldAnimate }: NeedleDotProps) {
  const motionScore = useMotionValue(shouldAnimate ? 0 : score)

  // x/y of tip: angle = 180 - score * 180 degrees
  const dotX = useTransform(motionScore, (v) => {
    const deg = 180 + v * 180
    return CX + R * Math.cos((deg * Math.PI) / 180)
  })
  const dotY = useTransform(motionScore, (v) => {
    const deg = 180 + v * 180
    return CY + R * Math.sin((deg * Math.PI) / 180)
  })

  useEffect(() => {
    if (!shouldAnimate) {
      motionScore.set(score)
      return
    }
    const controls = animate(motionScore, score, {
      duration: 1.1,
      ease:     [0.34, 1.06, 0.64, 1],
    })
    return () => controls.stop()
  }, [score, shouldAnimate, motionScore])

  const color = scoreToColor(score)

  return (
    <motion.circle
      cx={dotX as unknown as number}
      cy={dotY as unknown as number}
      r={5}
      fill={color}
      stroke="white"
      strokeWidth={2}
    />
  )
}