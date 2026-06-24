import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  Tooltip,
  ResponsiveContainer,
  type TooltipProps,
} from 'recharts'
import { motion } from 'framer-motion'
import type { ShapSignal } from '@/types/api'
import styles from './ShapChart.module.css'

// ── Colours ───────────────────────────────────────────────────

const COLOR_FRAUD = 'var(--color-shap-fraud)'   // brick-red
const COLOR_SAFE  = 'var(--color-shap-safe)'    // steel-blue

// ── Custom tooltip ────────────────────────────────────────────

function CustomTooltip({ active, payload }: TooltipProps<number, string> & { payload?: Array<{ payload: ShapSignal }> }) {
  if (!active || !payload?.length) return null
  const signal = payload[0].payload

  return (
    <div className={styles.tooltip}>
      <p className={styles.tooltipName}>{signal.display_name}</p>
      <p className={styles.tooltipExplanation}>{signal.explanation}</p>
      <p className={styles.tooltipImpact}>
        Impact:{' '}
        <span style={{ color: signal.direction === 'fraud' ? COLOR_FRAUD : COLOR_SAFE }}>
          {signal.direction === 'fraud' ? '+' : '−'}
          {(signal.impact * 100).toFixed(1)}pp
        </span>
      </p>
    </div>
  )
}

// ── Custom Y-axis tick ────────────────────────────────────────

interface TickProps {
  x?: number
  y?: number
  payload?: { value: string }
}

function YAxisTick({ x = 0, y = 0, payload }: TickProps) {
  // Truncate long display names to fit the axis
  const label = payload?.value ?? ''
  const truncated = label.length > 26 ? label.slice(0, 24) + '…' : label

  return (
    <g transform={`translate(${x},${y})`}>
      <text
        x={-6}
        y={0}
        dy={4}
        textAnchor="end"
        className={styles.axisTick}
      >
        {truncated}
      </text>
    </g>
  )
}

// ── Props ─────────────────────────────────────────────────────

interface ShapChartProps {
  signals: ShapSignal[]
}

// ── Component ─────────────────────────────────────────────────

export default function ShapChart({ signals }: ShapChartProps) {
  if (!signals.length) {
    return (
      <div className={styles.empty}>
        <p>No explanation signals available.</p>
      </div>
    )
  }

  // Recharts renders top-to-bottom; reverse so highest impact is at top
  const data = [...signals].reverse()

  return (
    <motion.div
      className={styles.wrapper}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: 0.15 }}
    >
      <h3 className={styles.title}>What drove this score</h3>
      <p className={styles.subtitle}>
        Top signals from the structural analysis — hover for details
      </p>

      {/* Legend */}
      <div className={styles.legend}>
        <span className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: COLOR_FRAUD }} />
          Pushes toward fraud
        </span>
        <span className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: COLOR_SAFE }} />
          Pushes toward legitimate
        </span>
      </div>

      <ResponsiveContainer width="100%" height={signals.length * 52 + 16}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 24, bottom: 4, left: 160 }}
          barSize={18}
        >
          <XAxis
            type="number"
            domain={[0, 'dataMax']}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}pp`}
            tick={{ fontFamily: 'var(--font-mono)', fontSize: 11, fill: 'var(--color-text-muted)' }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="display_name"
            width={156}
            tick={<YAxisTick />}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{ fill: 'rgba(0,48,73,0.04)' }}
          />
          <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
            {data.map((signal, index) => (
              <Cell
                key={`cell-${index}`}
                fill={signal.direction === 'fraud' ? COLOR_FRAUD : COLOR_SAFE}
                opacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Signal cards — text explanation below chart */}
      <ul className={styles.signalList}>
        {signals.map((signal) => (
          <li
            key={signal.feature}
            className={styles.signalItem}
            data-direction={signal.direction}
          >
            <span
              className={styles.signalBar}
              style={{
                background: signal.direction === 'fraud' ? COLOR_FRAUD : COLOR_SAFE,
              }}
            />
            <div className={styles.signalBody}>
              <span className={styles.signalName}>{signal.display_name}</span>
              <span className={styles.signalExplanation}>{signal.explanation}</span>
            </div>
            <span
              className={styles.signalImpact}
              style={{
                color: signal.direction === 'fraud' ? COLOR_FRAUD : COLOR_SAFE,
              }}
            >
              {signal.direction === 'fraud' ? '+' : '−'}
              {(signal.impact * 100).toFixed(1)}
              <span className={styles.signalImpactUnit}>pp</span>
            </span>
          </li>
        ))}
      </ul>
    </motion.div>
  )
}