import { useEffect, useRef } from 'react'
import { motion, useMotionValue, useTransform, animate } from 'framer-motion'
import type { FraudLabel } from '@/types/api'
import styles from './ScoreGauge.module.css'

// ── Arc geometry ──────────────────────────────────────────────
const CX = 100   // centre x
const CY = 100   // centre y
const R  = 75    // radius

// Angle in degrees: score=0 -> 180° (left). Clockwise progression.
function scoreToAngleDeg(score: number): number {
  return 180 - score * 360
}

function degToRad(deg: number): number {
  return (deg * Math.PI) / 180
}

function pointOnCircle(angleDeg: number): { x: number; y: number } {
  const rad = degToRad(angleDeg)
  return {
    x: CX + R * Math.cos(rad),
    y: CY + R * Math.sin(rad), // SVG Y-axis is flipped
  }
}

function describeArc(score: number): string {
  // 1. Handle exact boundaries cleanly
  if (score <= 0) {
    return `M ${CX - R} ${CY}`
  }
  if (score >= 1) {
    // Two half-circle arcs create a perfectly seamless 100% complete circle
    return `M ${CX - R} ${CY} A ${R} ${R} 0 1 0 ${CX + R} ${CY} A ${R} ${R} 0 1 0 ${CX - R} ${CY}`
  }

  // 2. Intermediate progress
  const startAngle = 180 
  const endAngle   = scoreToAngleDeg(score)

  const start    = pointOnCircle(startAngle)
  const end      = pointOnCircle(endAngle)
  const largeArc = score > 0.5 ? 1 : 0

  // sweep-flag=0 handles clockwise visual drawing with flipped SVG coordinates
  return `M ${start.x} ${start.y} A ${R} ${R} 0 ${largeArc} 0 ${end.x} ${end.y}`
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
    return () => {
      controls.stop()
      unsub()
    }
  }, [score, shouldAnimate, motionScore])

  const color          = scoreToColor(score)
  const [line1, line2] = LABEL_LINES[label]

  return (
    <div className={styles.wrapper} role="img" aria-label={`Fraud score ${Math.round(score * 100)}%`}>
      <svg viewBox="0 0 200 210" className={styles.svg} aria-hidden="true">

        {/* ── Background Track ── */}
        <circle
          cx={CX}
          cy={CY}
          r={R}
          fill="none"
          stroke="var(--color-border)"
          strokeWidth="10"
        />

        {/* ── Animated Active Score Arc ── */}
        <path
          ref={arcRef}
          d={describeArc(shouldAnimate ? 0 : score)}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
        />

        {/* ── Moving Needle Dot ── */}
        <NeedleDot score={score} shouldAnimate={shouldAnimate} />

        {/* ── Center Percentage ── */}
        <text
          ref={pctRef}
          x={CX}
          y={CY - 8}
          textAnchor="middle"
          dominantBaseline="auto"
          className={styles.pctText}
          fill={color}
        >
          {shouldAnimate ? '0%' : `${Math.round(score * 100)}%`}
        </text>

        {/* ── Label Context Lines ── */}
        <text x={CX} y={CY + 10} textAnchor="middle" dominantBaseline="auto"
              className={styles.labelLine1} fill="var(--color-text-muted)">
          {line1}
        </text>
        <text x={CX} y={CY + 26} textAnchor="middle" dominantBaseline="auto"
              className={styles.labelLine2} fill="var(--color-text-primary)">
          {line2}
        </text>

        {/* ── Structural Progress Markers ── */}
        <text x={CX - R - 6} y={CY + 4}   textAnchor="end"    className={styles.tick}>0%</text>
        <text x={CX + R + 6} y={CY + 4}   textAnchor="start"  className={styles.tick}>50%</text>
        <text x={CX}         y={CY - R - 8} textAnchor="middle" className={styles.tick}>25%</text>
        <text x={CX}         y={CY + R + 16} textAnchor="middle" className={styles.tick}>75%</text>

      </svg>
    </div>
  )
}

interface NeedleDotProps {
  score:         number
  shouldAnimate: boolean
}

function NeedleDot({ score, shouldAnimate }: NeedleDotProps) {
  const motionScore = useMotionValue(shouldAnimate ? 0 : score)

  const dotX = useTransform(motionScore, (v) => pointOnCircle(scoreToAngleDeg(v)).x)
  const dotY = useTransform(motionScore, (v) => pointOnCircle(scoreToAngleDeg(v)).y)

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