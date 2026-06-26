import { useEffect, useRef } from 'react'
import { motion, useMotionValue, useTransform, animate } from 'framer-motion'
import type { FraudLabel } from '@/types/api'
import styles from './ScoreGauge.module.css'

// ── Arc geometry ──────────────────────────────────────────────
// Full circle gauge. Starts at left (180°), sweeps clockwise
// upward through top, right, bottom, back to left.
// score=0   → left   (180°)
// score=0.25 → top   (90° visually = 270° SVG math)
// score=0.5  → right (0°)
// score=0.75 → bottom (90°)
// score=1.0  → left again (full circle)

const CX = 100   // centre x
const CY = 100   // centre y — moved up from 110 so full circle fits
const R  = 75    // radius

// Angle in degrees for a given score, measured from SVG's 0° (right),
// going clockwise visually = counterclockwise in SVG math.
// At score=0: angle=180 (left). Each +score rotates clockwise by 360°.
function scoreToAngleDeg(score: number): number {
  return 180 - score * 360
}

function degToRad(deg: number): number {
  return (deg * Math.PI) / 180
}

// Endpoint coords for a given angle
function pointOnCircle(angleDeg: number): { x: number; y: number } {
  const rad = degToRad(angleDeg)
  return {
    x: CX + R * Math.cos(rad),
    y: CY - R * Math.sin(rad),   // negate sin because SVG Y is flipped
  }
}

function describeArc(score: number): string {
  const clipped = Math.max(0.001, Math.min(0.999, score))

  const startAngle = 180                          // always left
  const endAngle   = scoreToAngleDeg(clipped)     // clockwise from left

  const start    = pointOnCircle(startAngle)
  const end      = pointOnCircle(endAngle)
  const largeArc = clipped > 0.5 ? 1 : 0
  // sweep-flag=0: counterclockwise in SVG math = clockwise visually (Y-flipped)
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

  // Tick positions — verified:
  // 0%   → left   (180°): x=CX-R=25,  y=CY=100
  // 50%  → right  (0°):   x=CX+R=175, y=CY=100
  // 100% → left again, same as 0%

  return (
    <div className={styles.wrapper} role="img" aria-label={`Fraud score ${Math.round(score * 100)}%`}>
      <svg viewBox="0 0 200 210" className={styles.svg} aria-hidden="true">

        {/* ── Full circle track ── */}
        <circle
          cx={CX}
          cy={CY}
          r={R}
          fill="none"
          stroke="var(--color-border)"
          strokeWidth="10"
        />

        {/* ── Animated score arc ── */}
        <path
          ref={arcRef}
          d={describeArc(shouldAnimate ? 0.001 : score)}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
        />

        {/* ── Needle dot ── */}
        <NeedleDot score={score} shouldAnimate={shouldAnimate} />

        {/* ── Percentage ── */}
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

        {/* ── Label lines ── */}
        <text x={CX} y={CY + 10} textAnchor="middle" dominantBaseline="auto"
              className={styles.labelLine1} fill="var(--color-text-muted)">
          {line1}
        </text>
        <text x={CX} y={CY + 26} textAnchor="middle" dominantBaseline="auto"
              className={styles.labelLine2} fill="var(--color-text-primary)">
          {line2}
        </text>

        {/* ── Tick labels ── */}
        {/* 0% at left (180°): x=25, y=100 */}
        <text x={CX - R - 6} y={CY + 4} textAnchor="end"    className={styles.tick}>0%</text>
        {/* 50% at right (0°): x=175, y=100 */}
        <text x={CX + R + 6} y={CY + 4} textAnchor="start"  className={styles.tick}>50%</text>
        {/* 25% at top (90° visual): x=100, y=25 */}
        <text x={CX}         y={CY - R - 8} textAnchor="middle" className={styles.tick}>25%</text>
        {/* 75% at bottom (270° visual): x=100, y=175 */}
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

  const dotX = useTransform(motionScore, (v) => {
    const p = pointOnCircle(scoreToAngleDeg(v))
    return p.x
  })
  const dotY = useTransform(motionScore, (v) => {
    const p = pointOnCircle(scoreToAngleDeg(v))
    return p.y
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