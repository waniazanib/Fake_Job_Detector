import { useEffect, useRef } from 'react'
import { motion, useMotionValue, useTransform, animate } from 'framer-motion'
import type { FraudLabel } from '@/types/api'
import styles from './ScoreGauge.module.css'

// ── Geometry ──────────────────────────────────────────────────
// Clock-style full circle gauge.
// 0%   = top    (270° in SVG)
// 25%  = right  (0°   in SVG)
// 50%  = bottom (90°  in SVG)
// 75%  = left   (180° in SVG)
// 100% = top again (full circle)
// Sweep direction: clockwise (sweep-flag=1)
// No Y-axis negation needed — SVG clockwise = visual clockwise

const CX = 110   // centre x
const CY = 110   // centre y
const R  = 80    // radius

function degToRad(deg: number): number {
  return (deg * Math.PI) / 180
}

// Convert score [0,1] to SVG angle in degrees
// score=0 → 270° (top), increases clockwise
function scoreToAngleDeg(score: number): number {
  return 270 + score * 360
}

// Get x,y on the circle for a given angle
function pointOnCircle(angleDeg: number): { x: number; y: number } {
  const rad = degToRad(angleDeg)
  return {
    x: CX + R * Math.cos(rad),
    y: CY + R * Math.sin(rad),
  }
}

function describeArc(score: number): string {
  // Clamp to avoid degenerate zero-length arc
  const clipped = Math.max(0.001, Math.min(0.999, score))

  const start    = pointOnCircle(270)                      // always top
  const end      = pointOnCircle(scoreToAngleDeg(clipped)) // clockwise from top
  const largeArc = clipped > 0.5 ? 1 : 0
  // sweep-flag=1 = clockwise in SVG = clockwise visually
  return `M ${start.x} ${start.y} A ${R} ${R} 0 ${largeArc} 1 ${end.x} ${end.y}`
}

function scoreToColor(score: number): string {
  if (score < 0.35) return 'var(--color-score-low)'
  if (score < 0.65) return 'var(--color-score-mid)'
  if (score < 0.85) return 'var(--color-score-high)'
  return 'var(--color-score-critical)'
}

const LABEL_LINES: Record<FraudLabel, [string, string]> = {
  LEGITIMATE: ['Likely',       'Legitimate'],
  CAUTION:    ['Proceed with', 'Caution'],
  SUSPICIOUS: ['Likely',       'Fraudulent'],
}

interface ScoreGaugeProps {
  score:    number
  label:    FraudLabel
  animate?: boolean
}

export default function ScoreGauge({
  score,
  label,
  animate: shouldAnimate = true,
}: ScoreGaugeProps) {
  const motionScore = useMotionValue(shouldAnimate ? 0 : score)
  const arcRef      = useRef<SVGPathElement>(null)
  const pctRef      = useRef<SVGTextElement>(null)

  useEffect(() => {
    if (!shouldAnimate) {
      motionScore.set(score)
      return
    }
    const controls = animate(motionScore, score, {
      duration: 1.1,
      ease:     [0.34, 1.06, 0.64, 1],
    })
    const unsub = motionScore.on('change', (v) => {
      if (arcRef.current) {
        arcRef.current.setAttribute('d', describeArc(v))
        arcRef.current.setAttribute('stroke', scoreToColor(v))
      }
      if (pctRef.current) {
        pctRef.current.textContent = `${Math.round(v * 100)}%`
      }
    })
    return () => { controls.stop(); unsub() }
  }, [score, shouldAnimate, motionScore])

  const color          = scoreToColor(score)
  const [line1, line2] = LABEL_LINES[label]

  return (
    <div className={styles.wrapper} role="img" aria-label={`Fraud score ${Math.round(score * 100)}%`}>
      <svg viewBox="0 0 220 240" className={styles.svg} aria-hidden="true">

        {/* ── Full circle track ── */}
        <circle
          cx={CX}
          cy={CY}
          r={R}
          fill="none"
          stroke="var(--color-border)"
          strokeWidth="10"
        />

        {/* ── Score arc — fills clockwise from top ── */}
        <path
          ref={arcRef}
          d={describeArc(shouldAnimate ? 0.001 : score)}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
        />

        {/* ── Needle dot at arc tip ── */}
        <NeedleDot score={score} shouldAnimate={shouldAnimate} />

        {/* ── Percentage text ── */}
        <text
          ref={pctRef}
          x={CX}
          y={CY - 10}
          textAnchor="middle"
          dominantBaseline="auto"
          className={styles.pctText}
          fill={color}
        >
          {shouldAnimate ? '0%' : `${Math.round(score * 100)}%`}
        </text>

        {/* ── Label lines ── */}
        <text x={CX} y={CY + 8} textAnchor="middle" dominantBaseline="auto"
              className={styles.labelLine1} fill="var(--color-text-muted)">
          {line1}
        </text>
        <text x={CX} y={CY + 24} textAnchor="middle" dominantBaseline="auto"
              className={styles.labelLine2} fill="var(--color-text-primary)">
          {line2}
        </text>

        {/* ── Tick labels ── */}
        {/* 0%  → top    (110, 30)  */}
        <text x={CX}       y={CY - R - 10} textAnchor="middle" className={styles.tick}>0%</text>
        {/* 25% → right  (190, 110) */}
        <text x={CX + R + 10} y={CY + 4}  textAnchor="start"  className={styles.tick}>25%</text>
        {/* 50% → bottom (110, 190) */}
        <text x={CX}       y={CY + R + 18} textAnchor="middle" className={styles.tick}>50%</text>
        {/* 75% → left   (30, 110)  */}
        <text x={CX - R - 10} y={CY + 4}  textAnchor="end"    className={styles.tick}>75%</text>

      </svg>
    </div>
  )
}

// ── Needle dot ────────────────────────────────────────────────

interface NeedleDotProps {
  score:         number
  shouldAnimate: boolean
}

function NeedleDot({ score, shouldAnimate }: NeedleDotProps) {
  const motionScore = useMotionValue(shouldAnimate ? 0 : score)

  const dotX = useTransform(motionScore, (v) => {
    return pointOnCircle(scoreToAngleDeg(v)).x
  })
  const dotY = useTransform(motionScore, (v) => {
    return pointOnCircle(scoreToAngleDeg(v)).y
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

  return (
    <motion.circle
      cx={dotX as unknown as number}
      cy={dotY as unknown as number}
      r={5}
      fill={scoreToColor(score)}
      stroke="white"
      strokeWidth={2}
    />
  )
}