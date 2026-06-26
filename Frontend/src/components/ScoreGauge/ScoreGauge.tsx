import React from 'react';
import styles from './ScoreGauge.module.css';

interface ScoreGaugeProps {
  score: number; // Percentage from 0 to 100
  labelLine1?: string;
  labelLine2?: string;
}

export const ScoreGauge: React.FC<ScoreGaugeProps> = ({
  score = 0,
  labelLine1 = "Score",
  labelLine2 = "Overall"
}) => {
  // Ensure score stays bounded between 0 and 100
  const clappedScore = Math.max(0, Math.min(100, score));

  // Geometry Constants
  const size = 200;
  const center = size / 2;
  const radius = 85; // leaves room for stroke width and dot radius
  const circumference = 2 * Math.PI * radius;

  // 1. Calculate stroke offset for progress (fills clockwise)
  const strokeDashoffset = circumference - (clappedScore / 100) * circumference;

  // 2. Calculate Dot position using corrected Trig angles (0% = top center)
  const angleInRadians = (clappedScore / 100) * (2 * Math.PI) - Math.PI / 2;
  const dotX = center + radius * Math.cos(angleInRadians);
  const dotY = center + radius * Math.sin(angleInRadians);

  // Determine dynamic color based on score thresholds (matching your image styles)
  let strokeColor = 'var(--color-danger, #e53e3e)'; // Default red
  if (clappedScore < 30) {
    strokeColor = '#fef08a'; // Light yellow for low scores like 18%
  } else if (clappedScore < 70) {
    strokeColor = '#ff3b30'; // Bright red/orange for mid scores like 42%, 64%
  } else {
    strokeColor = '#990000'; // Deep crimson maroon for high scores like 98%
  }

  return (
    <div className={styles.wrapper}>
      <svg 
        viewBox={`0 0 ${size} ${size}`} 
        className={styles.svg}
      >
        {/* Background Circle (Beige Track) */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="#f7f7e8" /* Light beige track */
          strokeWidth="12"
        />

        {/* Progress Circle */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={strokeColor}
          strokeWidth="12"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          transform={`rotate(-90 ${center} ${center})`} /* Rotates start to 12 o'clock */
          style={{ transition: 'stroke-dashoffset 0.35s ease-in-out' }}
        />

        {/* Active Indicator Dot */}
        <circle
          cx={dotX}
          cy={dotY}
          r={7}
          fill={strokeColor}
          stroke="#000000"
          strokeWidth="2"
          style={{ transition: 'cx 0.35s ease, cy 0.35s ease' }}
        />

        {/* Center Text Typography */}
        <text
          x={center}
          y={center + 8}
          textAnchor="middle"
          className={styles.pctText}
          fill="#000000"
        >
          {clappedScore}%
        </text>

        {labelLine1 && (
          <text
            x={center}
            y={center - 24}
            textAnchor="middle"
            className={styles.labelLine1}
            fill="var(--color-text-muted)"
          >
            {labelLine1}
          </text>
        )}
        
        {labelLine2 && (
          <text
            x={center}
            y={center + 28}
            textAnchor="middle"
            className={styles.labelLine2}
            fill="var(--color-text)"
          >
            {labelLine2}
          </text>
        )}
      </svg>
    </div>
  );
};