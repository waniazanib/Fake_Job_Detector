import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, ChevronDown, ChevronUp, Loader2 } from 'lucide-react'
import type { AnalyzeResponse, JobPostingRequest } from '@/types/api'
import { EMPTY_REQUEST } from '@/types/api'
import { analyzePosting, JobGuardApiError } from '@/api/analyze'
import styles from './JobForm.module.css'

// ── Select options ────────────────────────────────────────────

const EMPLOYMENT_TYPES = [
  'Full-time', 'Part-time', 'Contract', 'Temporary',
  'Internship', 'Other',
]

const EXPERIENCE_LEVELS = [
  'Not Applicable', 'Internship', 'Entry level',
  'Associate', 'Mid-Senior level', 'Director', 'Executive',
]

const EDUCATION_LEVELS = [
  'Unspecified', 'High School or equivalent', 'Some College Coursework Completed',
  'Vocational', 'Associate Degree', "Bachelor's Degree",
  "Master's Degree", 'Doctorate', 'Professional',
]

// ── Props ─────────────────────────────────────────────────────

interface JobFormProps {
  onResult: (result: AnalyzeResponse) => void
}

// ── Component ─────────────────────────────────────────────────

export default function JobForm({ onResult }: JobFormProps) {
  const [form,        setForm]        = useState<JobPostingRequest>({ ...EMPTY_REQUEST })
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState<string | null>(null)
  const [expanded,    setExpanded]    = useState(false)   // optional fields section

  // ── Field updaters ──────────────────────────────────────────

  const setText = useCallback(
    (field: keyof JobPostingRequest) =>
      (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        setForm(prev => ({ ...prev, [field]: e.target.value || null }))
        if (error) setError(null)
      },
    [error],
  )

  const setBool = useCallback(
    (field: keyof JobPostingRequest) =>
      (e: React.ChangeEvent<HTMLInputElement>) => {
        setForm(prev => ({ ...prev, [field]: e.target.checked }))
      },
    [],
  )

  // ── Submit ──────────────────────────────────────────────────

  const handleSubmit = useCallback(async () => {
    if (!form.title && !form.description) {
      setError('Please provide at least a job title or description.')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const result = await analyzePosting(form)
      onResult(result)
    } catch (err) {
      if (err instanceof JobGuardApiError) {
        setError(err.message)
      } else {
        setError('Something went wrong. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }, [form, onResult])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        handleSubmit()
      }
    },
    [handleSubmit],
  )

  // ── Render ──────────────────────────────────────────────────

  return (
    <section className={styles.wrapper} aria-label="Job posting analysis form">

      {/* ── Hero copy ── */}
      <div className={styles.hero}>
        <h1 className={styles.heroTitle}>Is this job posting real?</h1>
        <p className={styles.heroSub}>
          Paste any job listing below. JobGuard analyses the text and structure
          using a dual ML model to flag fraud signals instantly.
        </p>
      </div>

      {/* ── Form card ── */}
      <div className={styles.card}>

        {/* Title */}
        <div className={styles.field}>
          <label className={styles.label} htmlFor="title">
            Job title
            <span className={styles.required} aria-hidden="true">*</span>
          </label>
          <input
            id="title"
            type="text"
            className={styles.input}
            placeholder="e.g. Senior Software Engineer"
            value={form.title ?? ''}
            onChange={setText('title')}
            onKeyDown={handleKeyDown}
            disabled={loading}
            autoComplete="off"
          />
        </div>

        {/* Description */}
        <div className={styles.field}>
          <label className={styles.label} htmlFor="description">
            Job description
            <span className={styles.required} aria-hidden="true">*</span>
          </label>
          <textarea
            id="description"
            className={styles.textarea}
            placeholder="Paste the full job description here…"
            rows={7}
            value={form.description ?? ''}
            onChange={setText('description')}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <p className={styles.hint}>
            The more text you provide, the more accurate the analysis.
          </p>
        </div>

        {/* Requirements */}
        <div className={styles.field}>
          <label className={styles.label} htmlFor="requirements">
            Requirements
          </label>
          <textarea
            id="requirements"
            className={styles.textarea}
            placeholder="Skills, qualifications, experience required…"
            rows={4}
            value={form.requirements ?? ''}
            onChange={setText('requirements')}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
        </div>

        {/* ── Optional fields toggle ── */}
        <button
          type="button"
          className={styles.expandBtn}
          onClick={() => setExpanded(v => !v)}
          aria-expanded={expanded}
          aria-controls="optional-fields"
        >
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          {expanded ? 'Hide' : 'Add'} optional details
          <span className={styles.expandHint}>
            (salary, company info, benefits — improves accuracy)
          </span>
        </button>

        <AnimatePresence initial={false}>
          {expanded && (
            <motion.div
              id="optional-fields"
              key="optional"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
              style={{ overflow: 'hidden' }}
            >
              <div className={styles.optionalGrid}>

                {/* Benefits */}
                <div className={`${styles.field} ${styles.spanFull}`}>
                  <label className={styles.label} htmlFor="benefits">Benefits</label>
                  <textarea
                    id="benefits"
                    className={styles.textarea}
                    placeholder="Health insurance, equity, remote work…"
                    rows={3}
                    value={form.benefits ?? ''}
                    onChange={setText('benefits')}
                    disabled={loading}
                  />
                </div>

                {/* Company profile */}
                <div className={`${styles.field} ${styles.spanFull}`}>
                  <label className={styles.label} htmlFor="company_profile">
                    Company description
                  </label>
                  <textarea
                    id="company_profile"
                    className={styles.textarea}
                    placeholder="About the company…"
                    rows={3}
                    value={form.company_profile ?? ''}
                    onChange={setText('company_profile')}
                    disabled={loading}
                  />
                </div>

                {/* Location */}
                <div className={styles.field}>
                  <label className={styles.label} htmlFor="location">Location</label>
                  <input
                    id="location"
                    type="text"
                    className={styles.input}
                    placeholder="New York, NY — or Anywhere"
                    value={form.location ?? ''}
                    onChange={setText('location')}
                    disabled={loading}
                  />
                </div>

                {/* Salary */}
                <div className={styles.field}>
                  <label className={styles.label} htmlFor="salary_range">
                    Salary range
                  </label>
                  <input
                    id="salary_range"
                    type="text"
                    className={styles.input}
                    placeholder="80000-100000 or $50/hr"
                    value={form.salary_range ?? ''}
                    onChange={setText('salary_range')}
                    disabled={loading}
                  />
                </div>

                {/* Employment type */}
                <div className={styles.field}>
                  <label className={styles.label} htmlFor="employment_type">
                    Employment type
                  </label>
                  <select
                    id="employment_type"
                    className={styles.select}
                    value={form.employment_type ?? ''}
                    onChange={setText('employment_type')}
                    disabled={loading}
                  >
                    <option value="">— Select —</option>
                    {EMPLOYMENT_TYPES.map(t => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>

                {/* Experience */}
                <div className={styles.field}>
                  <label className={styles.label} htmlFor="required_experience">
                    Experience level
                  </label>
                  <select
                    id="required_experience"
                    className={styles.select}
                    value={form.required_experience ?? ''}
                    onChange={setText('required_experience')}
                    disabled={loading}
                  >
                    <option value="">— Select —</option>
                    {EXPERIENCE_LEVELS.map(l => (
                      <option key={l} value={l}>{l}</option>
                    ))}
                  </select>
                </div>

                {/* Education */}
                <div className={styles.field}>
                  <label className={styles.label} htmlFor="required_education">
                    Education required
                  </label>
                  <select
                    id="required_education"
                    className={styles.select}
                    value={form.required_education ?? ''}
                    onChange={setText('required_education')}
                    disabled={loading}
                  >
                    <option value="">— Select —</option>
                    {EDUCATION_LEVELS.map(e => (
                      <option key={e} value={e}>{e}</option>
                    ))}
                  </select>
                </div>

                {/* Boolean toggles */}
                <div className={`${styles.field} ${styles.spanFull}`}>
                  <p className={styles.label}>Posting attributes</p>
                  <div className={styles.toggleRow}>
                    <Toggle
                      id="has_company_logo"
                      label="Has company logo"
                      checked={form.has_company_logo}
                      onChange={setBool('has_company_logo')}
                      disabled={loading}
                    />
                    <Toggle
                      id="has_questions"
                      label="Includes screening questions"
                      checked={form.has_questions}
                      onChange={setBool('has_questions')}
                      disabled={loading}
                    />
                    <Toggle
                      id="telecommuting"
                      label="Remote / telecommuting"
                      checked={form.telecommuting}
                      onChange={setBool('telecommuting')}
                      disabled={loading}
                    />
                  </div>
                </div>

              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Error ── */}
        <AnimatePresence>
          {error && (
            <motion.p
              className={styles.error}
              role="alert"
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
            >
              {error}
            </motion.p>
          )}
        </AnimatePresence>

        {/* ── Submit ── */}
        <button
          type="button"
          className={styles.submitBtn}
          onClick={handleSubmit}
          disabled={loading}
          aria-busy={loading}
        >
          {loading ? (
            <>
              <Loader2 size={17} className={styles.spinner} aria-hidden="true" />
              Analysing…
            </>
          ) : (
            <>
              <Send size={17} aria-hidden="true" />
              Analyse posting
            </>
          )}
        </button>

      </div>
    </section>
  )
}

// ── Toggle sub-component ──────────────────────────────────────

interface ToggleProps {
  id:       string
  label:    string
  checked:  boolean
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  disabled: boolean
}

function Toggle({ id, label, checked, onChange, disabled }: ToggleProps) {
  return (
    <label className={styles.toggle} htmlFor={id}>
      <input
        id={id}
        type="checkbox"
        className={styles.toggleInput}
        checked={checked}
        onChange={onChange}
        disabled={disabled}
      />
      <span className={styles.toggleTrack} aria-hidden="true">
        <span className={styles.toggleThumb} />
      </span>
      <span className={styles.toggleLabel}>{label}</span>
    </label>
  )
}