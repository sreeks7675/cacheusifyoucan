import { useEffect, useMemo, useState } from 'react';
import { AlertCircle, CheckCircle2, Loader2, Search, ShieldCheck, Sparkles } from 'lucide-react';
import { checkBackendHealth, createInvestigation } from './api';

const initialForm = {
  topic: '',
  question: '',
  description: '',
  sourceType: 'url',
  sourceUrl: '',
};

const featureCards = [
  { title: 'Cross-source intelligence', detail: 'Merge evidence from documents, URLs, and transcripts into one timeline.' },
  { title: 'Conflict detection', detail: 'Highlight contradictions and identify the strongest supporting evidence.' },
  { title: 'Structured reporting', detail: 'Turn the investigation into a readable report for stakeholders.' },
];

function TruthLensApp() {
  const [backendStatus, setBackendStatus] = useState('checking');
  const [form, setForm] = useState(initialForm);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    let active = true;

    const verifyBackend = async () => {
      try {
        const response = await checkBackendHealth();
        if (active) {
          setBackendStatus(response?.status === 'ok' ? 'connected' : 'error');
        }
      } catch (error) {
        console.error('Backend health check failed', error);
        if (active) {
          setBackendStatus('error');
        }
      }
    };

    verifyBackend();
    return () => {
      active = false;
    };
  }, []);

  const statusTone = useMemo(() => {
    if (backendStatus === 'connected') {
      return { icon: <CheckCircle2 size={16} />, label: 'Live backend connected', color: '#16a34a' };
    }

    if (backendStatus === 'error') {
      return { icon: <AlertCircle size={16} />, label: 'Backend unavailable', color: '#dc2626' };
    }

    return { icon: <Loader2 size={16} className="spin" />, label: 'Checking backend…', color: '#2563eb' };
  }, [backendStatus]);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!form.topic.trim()) {
      setMessage('Please add a topic before submitting an investigation.');
      return;
    }

    setIsSubmitting(true);
    setMessage('');

    try {
      const payload = {
        title: form.topic.trim(),
        description: form.description.trim(),
        topic: form.topic.trim(),
        question: form.question.trim(),
        source_type: form.sourceType,
        source_url: form.sourceUrl.trim(),
      };

      const result = await createInvestigation(payload);
      setMessage(`Investigation queued successfully with ID ${result.id}.`);
      setForm(initialForm);
    } catch (error) {
      console.error('Failed to submit investigation', error);
      setMessage('The investigation could not be submitted. Please verify the backend is running.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div style={styles.appShell}>
      <header style={styles.header}>
        <div>
          <p style={styles.eyebrow}>TruthLens</p>
          <h1 style={styles.title}>Investigate faster with connected evidence.</h1>
          <p style={styles.subtitle}>Route incoming claims through retrieval, reasoning, and reporting in one place.</p>
        </div>
        <div style={{ ...styles.statusChip, borderColor: statusTone.color }}>
          <span style={{ color: statusTone.color }}>{statusTone.icon}</span>
          <span style={{ color: statusTone.color, marginLeft: 8 }}>{statusTone.label}</span>
        </div>
      </header>

      <section style={styles.heroGrid}>
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <Sparkles size={18} color="#7c3aed" />
            <strong>Workspace overview</strong>
          </div>
          <div style={styles.metricRow}>
            <div style={styles.metricBox}>
              <strong style={styles.metricNumber}>24</strong>
              <span style={styles.metricLabel}>Live signals</span>
            </div>
            <div style={styles.metricBox}>
              <strong style={styles.metricNumber}>87%</strong>
              <span style={styles.metricLabel}>Source confidence</span>
            </div>
          </div>
          <p style={styles.cardText}>This view is now backed by the orchestrator API for health checks and investigation submission.</p>
        </div>

        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <ShieldCheck size={18} color="#0f766e" />
            <strong>New investigation</strong>
          </div>
          <form onSubmit={handleSubmit} style={styles.form}>
            <label style={styles.label}>
              Topic
              <input style={styles.input} name="topic" value={form.topic} onChange={handleChange} placeholder="e.g. Viral rumor around a policy claim" />
            </label>
            <label style={styles.label}>
              Question
              <input style={styles.input} name="question" value={form.question} onChange={handleChange} placeholder="What should the investigation answer?" />
            </label>
            <label style={styles.label}>
              Description
              <textarea style={{ ...styles.input, minHeight: 84, resize: 'vertical' }} name="description" value={form.description} onChange={handleChange} placeholder="Add context for the analysis pipeline." />
            </label>
            <div style={styles.row}>
              <label style={styles.label}>
                Source type
                <select style={styles.input} name="sourceType" value={form.sourceType} onChange={handleChange}>
                  <option value="url">URL</option>
                  <option value="document">Document</option>
                  <option value="transcript">Transcript</option>
                </select>
              </label>
              <label style={styles.label}>
                Source URL
                <input style={styles.input} name="sourceUrl" value={form.sourceUrl} onChange={handleChange} placeholder="https://example.com" />
              </label>
            </div>
            <button style={styles.primaryButton} type="submit" disabled={isSubmitting}>
              {isSubmitting ? <Loader2 size={16} className="spin" /> : <Search size={16} />}
              <span style={{ marginLeft: 8 }}>{isSubmitting ? 'Submitting…' : 'Submit investigation'}</span>
            </button>
            {message ? <p style={styles.message}>{message}</p> : null}
          </form>
        </div>
      </section>

      <section style={styles.featureGrid}>
        {featureCards.map((feature) => (
          <div key={feature.title} style={styles.featureCard}>
            <h3 style={styles.featureTitle}>{feature.title}</h3>
            <p style={styles.featureText}>{feature.detail}</p>
          </div>
        ))}
      </section>
    </div>
  );
}

const styles = {
  appShell: {
    minHeight: '100vh',
    padding: '32px',
    background: 'linear-gradient(135deg, #0f172a 0%, #111827 100%)',
    color: '#f8fafc',
    fontFamily: 'Inter, system-ui, sans-serif',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 16,
    marginBottom: 24,
    flexWrap: 'wrap',
  },
  eyebrow: {
    margin: 0,
    fontSize: 12,
    letterSpacing: '0.2em',
    textTransform: 'uppercase',
    color: '#8b5cf6',
  },
  title: {
    margin: '6px 0 8px',
    fontSize: 34,
    lineHeight: 1.1,
  },
  subtitle: {
    margin: 0,
    maxWidth: 700,
    color: '#cbd5e1',
    fontSize: 16,
  },
  statusChip: {
    display: 'flex',
    alignItems: 'center',
    padding: '10px 14px',
    border: '1px solid',
    borderRadius: 999,
    background: 'rgba(255,255,255,0.04)',
  },
  heroGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: 20,
    marginBottom: 20,
  },
  card: {
    background: 'rgba(15, 23, 42, 0.85)',
    border: '1px solid rgba(148, 163, 184, 0.2)',
    borderRadius: 20,
    padding: 20,
    boxShadow: '0 18px 45px rgba(2, 6, 23, 0.35)',
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  metricRow: {
    display: 'flex',
    gap: 12,
    marginBottom: 12,
    flexWrap: 'wrap',
  },
  metricBox: {
    flex: 1,
    minWidth: 120,
    padding: 14,
    borderRadius: 14,
    background: 'rgba(79, 70, 229, 0.16)',
  },
  metricNumber: {
    display: 'block',
    fontSize: 22,
    marginBottom: 4,
  },
  metricLabel: {
    color: '#cbd5e1',
    fontSize: 13,
  },
  cardText: {
    margin: 0,
    color: '#cbd5e1',
    lineHeight: 1.5,
  },
  form: {
    display: 'grid',
    gap: 12,
  },
  label: {
    display: 'grid',
    gap: 6,
    fontSize: 13,
    color: '#e2e8f0',
  },
  input: {
    border: '1px solid rgba(148, 163, 184, 0.25)',
    borderRadius: 10,
    padding: '10px 12px',
    background: 'rgba(15, 23, 42, 0.65)',
    color: '#f8fafc',
    fontSize: 14,
  },
  row: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
    gap: 12,
  },
  primaryButton: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 8,
    border: 'none',
    borderRadius: 999,
    padding: '12px 16px',
    background: 'linear-gradient(90deg, #8b5cf6 0%, #3b82f6 100%)',
    color: '#fff',
    cursor: 'pointer',
    fontWeight: 600,
  },
  message: {
    margin: 0,
    color: '#fef3c7',
    fontSize: 13,
  },
  featureGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
    gap: 16,
  },
  featureCard: {
    borderRadius: 16,
    padding: 16,
    background: 'rgba(15, 23, 42, 0.7)',
    border: '1px solid rgba(148, 163, 184, 0.16)',
  },
  featureTitle: {
    margin: '0 0 8px',
    fontSize: 16,
  },
  featureText: {
    margin: 0,
    color: '#cbd5e1',
    lineHeight: 1.45,
  },
};

export default TruthLensApp;