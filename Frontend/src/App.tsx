import { useState, useEffect, useCallback } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import type { AnalyzeResponse, HealthResponse } from '@/types/api'
import { fetchHealth } from '@/api/analyze'
import Header     from '@/components/Header/Header'
import JobForm    from '@/components/JobForm/JobForm'
import ResultPanel from '@/components/ResultPanel/ResultPanel'
import styles from './App.module.css'

// ── View states ───────────────────────────────────────────────
// 'form'   — empty state, JobForm visible
// 'result' — analysis complete, ResultPanel visible

type View = 'form' | 'result'

export default function App() {
  const [view,    setView]    = useState<View>('form')
  const [result,  setResult]  = useState<AnalyzeResponse | null>(null)
  const [health,  setHealth]  = useState<HealthResponse | null>(null)
  const [healthChecked, setHealthChecked] = useState(false)

  // ── Health check on mount ───────────────────────────────────

  useEffect(() => {
    let cancelled = false

    async function check() {
      const h = await fetchHealth()
      if (!cancelled) {
        setHealth(h)
        setHealthChecked(true)
      }
    }

    check()
    return () => { cancelled = true }
  }, [])

  // ── Handlers ────────────────────────────────────────────────

  const handleResult = useCallback((r: AnalyzeResponse) => {
    setResult(r)
    setView('result')
    // Scroll to top so the result panel is immediately visible
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [])

  const handleReset = useCallback(() => {
    setView('form')
    setResult(null)
    window.scrollTo({ top: 0, behavior: 'instant' })
  }, [])

  // ── Render ──────────────────────────────────────────────────

  return (
    <div className={styles.app}>
      <Header />

      {/* ── Models not ready banner ── */}
      <AnimatePresence>
        {healthChecked && health !== null && !health.models_loaded && (
          <motion.div
            className={styles.banner}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22 }}
            role="status"
            aria-live="polite"
          >
            <span className={styles.bannerDot} aria-hidden="true" />
            <span>
              {!health.xgb_ready && !health.bert_ready
                ? 'Models are not loaded — run '
                : !health.xgb_ready
                ? 'XGBoost model not ready — run '
                : 'DistilBERT model not ready — run '}
              <code className={styles.bannerCode}>python src/train.py</code>
              {' '}then restart the server.
            </span>
          </motion.div>
        )}

        {healthChecked && health === null && (
          <motion.div
            className={`${styles.banner} ${styles.bannerError}`}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22 }}
            role="alert"
          >
            <span className={styles.bannerDot} aria-hidden="true" />
            Backend not reachable — make sure{' '}
            <code className={styles.bannerCode}>uvicorn main:app --port 8000</code>
            {' '}is running.
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Main content ── */}
      <main className={styles.main}>
        <AnimatePresence mode="wait">
          {view === 'form' ? (
            <motion.div
              key="form"
              className={styles.viewWrapper}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            >
              <JobForm onResult={handleResult} />
            </motion.div>
          ) : (
            <motion.div
              key="result"
              className={`${styles.viewWrapper} ${styles.resultWrapper}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            >
              {result && (
                <ResultPanel result={result} onReset={handleReset} />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* ── Footer ── */}
      <footer className={styles.footer}>
        <p>
          JobGuard uses{' '}
          <span className={styles.footerAccent}>DistilBERT</span>
          {' '}+{' '}
          <span className={styles.footerAccent}>XGBoost</span>
          {' '}trained on the EMSCAD dataset.
          Explanations powered by{' '}
          <span className={styles.footerAccent}>SHAP</span>.
        </p>
      </footer>
    </div>
  )
}