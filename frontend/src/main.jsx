import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { createRoot } from 'react-dom/client'

import {

  Activity, BarChart3, Bot, ChevronDown, Cpu, Gauge, Leaf, Play,

  Power, RefreshCw, RotateCcw, Server, Settings, ShieldCheck, Square,

  Thermometer, Zap, History, LayoutDashboard, SlidersHorizontal, AlertTriangle, Search, ExternalLink, MonitorCog

} from 'lucide-react'

import {

  Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer,

  Legend, ReferenceLine, Tooltip, XAxis, YAxis

} from 'recharts'

import './styles.css'



const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api'



async function api(path, options = {}) {

  const res = await fetch(`${API}${path}`, {

    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },

    ...options,

  })

  const data = await res.json().catch(() => ({}))

  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)

  return data

}



function toLocalDate(value) {
  if (!value) return new Date(NaN)
  const raw = String(value)
  // Backend simulator timestamps are naive UTC; parse them as UTC, then display local time.
  return new Date(/(?:Z|[+-]\d{2}:?\d{2})$/i.test(raw) ? raw : `${raw}Z`)
}

function fmt(value, digits = 0) {

  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A'

  return Number(value).toFixed(digits)

}



function pct(oldValue, newValue, lowerIsBetter = true) {

  if (!oldValue || newValue === undefined || newValue === null) return null

  const change = ((Number(oldValue) - Number(newValue)) / Number(oldValue)) * 100

  return lowerIsBetter ? change : -change

}



function App() {

  const [data, setData] = useState(null)
  const [environmentMonitor, setEnvironmentMonitor] = useState(null)
  const [environmentScanning, setEnvironmentScanning] = useState(false)

  const [error, setError] = useState('')

  const [busy, setBusy] = useState(false)

  const [metric, setMetric] = useState('power')

  const [range, setRange] = useState(30)

  const initialPage = ['dashboard','optimization','alerts','hardware','settings','sustainability','fleet','workloads','analytics'].includes(window.location.hash.slice(1)) ? window.location.hash.slice(1) : 'dashboard'
  const [activeNav, setActiveNav] = useState(initialPage)
  const [activePage, setActivePage] = useState(initialPage)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true)

  const [settingsOpen, setSettingsOpen] = useState(false)

  const [interval, setIntervalValue] = useState(1)

  const [quality, setQuality] = useState(null)

  const [qualityThreshold, setQualityThreshold] = useState(2)
  const [anomalyToast, setAnomalyToast] = useState(null)
  const lastAlertKey = useRef(null)



  const refresh = useCallback(async () => {

    try {

      const next = await api('/dashboard')
      setData(next)
      // Keep the TinyFish monitor independent so a missing API key never breaks the main dashboard.
      try { setEnvironmentMonitor(await api('/environment/monitor')) } catch { /* optional panel */ }
      setError('')

    } catch (err) {

      setError(err.message)

    }

  }, [])



  useEffect(() => {
    const syncRoute = () => {
      const key = window.location.hash.slice(1)
      const next = ['dashboard','optimization','alerts','hardware','settings','sustainability','fleet','workloads','analytics'].includes(key) ? key : 'dashboard'
      setActivePage(next)
      setActiveNav(next)
    }
    window.addEventListener('hashchange', syncRoute)
    return () => window.removeEventListener('hashchange', syncRoute)
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 1200)
    return () => clearInterval(id)
  }, [refresh])

  useEffect(() => {
    const anomaly = data?.anomaly
    if (!anomaly || anomaly.status !== 'DETECTED' || !anomaly.last_detected_at) return
    const alertKey = `${anomaly.last_detected_at}-${(anomaly.sample_ids || []).slice(-1)[0] || ''}`
    if (lastAlertKey.current === alertKey) return
    lastAlertKey.current = alertKey
    const detectedAt = toLocalDate(anomaly.last_detected_at)
    const timeText = Number.isNaN(detectedAt.getTime())
      ? new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      : detectedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    setAnomalyToast({ key: alertKey, text: `Anomaly detected at this time: ${timeText}` })
    const timeout = setTimeout(() => setAnomalyToast(current => current?.key === alertKey ? null : current), 6000)
    return () => clearTimeout(timeout)
  }, [data?.anomaly])



  const run = async fn => {

    setBusy(true)

    setError('')

    try {

      await fn()

      await refresh()

    } catch (err) {

      setError(err.message)

    } finally {

      setBusy(false)

    }

  }



  const start = () => run(() => api(`/simulation/start?tick_interval_s=${interval}`, { method: 'POST' }))

  const stop = () => run(() => api('/simulation/stop', { method: 'POST' }))

  const reset = () => run(() => api('/simulation/reset', { method: 'POST' }))

  const scanEnvironment = () => run(async () => {
    setEnvironmentScanning(true)
    try {
      const result = await api('/environment/scan', { method: 'POST' })
      setEnvironmentMonitor(result)
    } finally {
      setEnvironmentScanning(false)
    }
  })

  const triggerAnomalySpike = () => run(async () => {
    if (!data?.simulation?.running) {
      await api(`/simulation/start?tick_interval_s=${interval}`, { method: 'POST' })
    }
    await api('/simulation/spike', {
      method: 'POST',
      body: JSON.stringify({ peak_rate: 1450, spike_duration_ticks: 8, peak_hold_ticks: 4 }),
    })
  })

  // Separate manual spike control; automatic spikes and the existing anomaly button remain unchanged.
  const triggerManualSpike = () => run(async () => {
    if (!data?.simulation?.running) {
      await api(`/simulation/start?tick_interval_s=${interval}`, { method: 'POST' })
    }
    await api('/simulation/spike', {
      method: 'POST',
      body: JSON.stringify({ peak_rate: 1250, spike_duration_ticks: 6, peak_hold_ticks: 3 }),
    })
  })

  const evaluateQuality = () => run(async () => {
    const result = await api('/quality/evaluate', { method: 'POST', body: JSON.stringify({ reference_precision: 'FP16', candidate_precision: 'INT8', allowed_degradation_pct: Number(qualityThreshold) }) })
    setQuality(result)
  })

  const toggleAutopilot = () => run(() =>

    data?.autonomous_mode?.enabled

      ? api('/autopilot/stop', { method: 'POST' })

      : api('/autopilot/start?interval_s=2', { method: 'POST' })

  )



  const history = data?.history || []

  const visibleHistory = useMemo(() => {

    const seconds = range * 60

    return history.slice(-Math.max(2, Math.min(history.length, seconds)))

  }, [history, range])



  const chartData = useMemo(() => {

    const applied = (data?.recent_events || [])

      .filter(e => e.type === 'configuration_applied' || e.event === 'configuration_applied')

      .slice(-1)[0]

    const appliedAt = applied ? toLocalDate(applied.timestamp).getTime() : null
    const anomalyIds = new Set(data?.anomaly?.sample_ids || [])

    return visibleHistory.map(s => {

      const ts = toLocalDate(s.timestamp).getTime()

      const row = {

        timestamp: s.timestamp,

        time: toLocalDate(s.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),

        power: Number(s.total_power),

        temperature: Number(s.temperature),

        utilization: Number(s.gpu_utilization),

        requestRate: Number(s.request_rate),

        throughput: Number(s.throughput),
        isAnomaly: anomalyIds.has(s.sample_id),

      }
      row.anomalyMetric = row.isAnomaly ? row[metric === 'power' ? 'power' : 'temperature'] : null

      row.before = appliedAt && ts >= appliedAt ? null : row[metric]

      row.after = appliedAt && ts >= appliedAt ? row[metric] : null

      return row

    })

  }, [visibleHistory, data, metric])



  const eventMarkers = useMemo(() => {

    const ids = new Set()

    return (data?.recent_events || []).filter(e => {

      const key = `${e.timestamp}-${e.type}`

      if (ids.has(key)) return false

      ids.add(key)

      const ts = toLocalDate(e.timestamp).getTime()

      return chartData.length && ts >= toLocalDate(chartData[0].timestamp).getTime()

    })

  }, [data, chartData])



  const telemetry = data?.telemetry

  const opt = data?.optimization

  const current = opt?.current_configuration

  const recommended = opt?.recommended_configuration

  const impact = opt?.expected_impact?.changes || {}

  const slaPass = data?.sla?.status === 'PASS'

  const running = data?.simulation?.running

  const auto = data?.autonomous_mode?.enabled
  const lastCycle = data?.autonomous_mode?.last_cycle
  const recommendationApplied = lastCycle?.action === 'APPLIED' &&
    JSON.stringify(lastCycle?.applied_configuration) === JSON.stringify(lastCycle?.recommended_configuration)



  const nav = [
    ['dashboard', 'Dashboard', LayoutDashboard],
    ['optimization', 'Optimization Center', Cpu],
    ['alerts', 'SLA & Alerts', AlertTriangle],
    ['hardware', 'Hardware & Software Monitor', MonitorCog],
    ['settings', 'Settings', SlidersHorizontal],
    ['sustainability', 'Sustainability', Leaf],
    ['fleet', 'GPU Fleet', Server],
    ['workloads', 'AI Workloads', Bot],
    ['analytics', 'Analytics & History', History],
  ]

  const pageTitles = {
    dashboard: ['Dashboard', 'Live infrastructure health and autonomous optimization'],
    optimization: ['Optimization Center', 'Compare configurations, quality gates, and optimization evidence'],
    alerts: ['SLA & Alerts', 'Telemetry anomaly status and real controller event history'],
    hardware: ['Hardware & Software Monitor', 'Installed inventory and public release checks'],
    settings: ['Settings', 'Simulator cadence, controller status, and SLA configuration'],
    sustainability: ['Sustainability', 'Estimated energy efficiency and thermal impact'],
    fleet: ['GPU Fleet', 'Live status and allocation across five simulated GPUs'],
    workloads: ['AI Workloads', 'Current serving demand, latency, and workload protection'],
    analytics: ['Analytics & History', 'Telemetry trends and recorded controller decisions'],
  }

  const go = (key) => {
    setActiveNav(key)
    setActivePage(key)
    window.location.hash = key
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }



  return (

    <div className="app-shell">

      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : 'expanded'}`}>

        <div className="brand">

          <div className="brand-mark"><Leaf size={27} strokeWidth={2.3}/></div>

          <div className="brand-copy"><div className="brand-name">GPUPilot</div><div className="brand-sub">Sustainable AI Infrastructure</div></div>
          <button className="sidebar-toggle" onClick={() => setSidebarCollapsed(value => !value)} aria-label={sidebarCollapsed ? 'Expand navigation' : 'Collapse navigation'} title={sidebarCollapsed ? 'Expand navigation' : 'Collapse navigation'}>{sidebarCollapsed ? '›' : '‹'}</button>

        </div>



        <nav className="side-nav">

          {nav.map(([key, label, Icon, id]) => (

            <button key={key} className={`nav-item ${activeNav === key ? 'active' : ''}`} onClick={() => go(key)}>

              <Icon size={19}/><span>{label}</span>

            </button>

          ))}



        </nav>



        <div className="health-card">
          <div className="sim-controls-label">SIMULATION CONTROLS</div>

          <div className="health-title"><span className={`health-dot ${running ? '' : 'offline'}`}/><strong>{running ? 'System Healthy' : 'Simulation Idle'}</strong></div>

          <p>{running ? 'All services are running' : 'Start the simulator to receive live telemetry'}</p>

          <button className="autonomous-row" onClick={toggleAutopilot} disabled={busy}>

            <Cpu size={17}/><span>Autonomous Mode: <b>{auto ? 'ON' : 'OFF'}</b></span>

            <span className={`toggle ${auto ? 'on' : ''}`}><i/></span>

          </button>

          <button className="side-action primary" onClick={start} disabled={busy}><Play size={17} fill="currentColor"/>Start Simulation</button>

          <button className="side-action warning side-spike-trigger" onClick={triggerAnomalySpike} disabled={busy}><Zap size={17} fill="currentColor"/>Trigger Anomaly Spike</button>

          <button className="side-action manual-spike-trigger" onClick={triggerManualSpike} disabled={busy}><Zap size={17}/>Manual Spike</button>

          <button className="side-action reset" onClick={reset} disabled={busy}><RotateCcw size={17}/>Reset</button>

          <div className="last-run">Last simulation<br/><b>{running ? 'Live' : 'Idle'}</b></div>

        </div>

      </aside>



      <main className={`main-content ${activePage} ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`} id="dashboard-top">

        <header className="page-header">

          <div>

            <h1>{pageTitles[activePage]?.[0] || 'Dashboard'}</h1>

            <p>{pageTitles[activePage]?.[1] || 'Live infrastructure health and autonomous optimization'}</p>

          </div>

          <div className="header-status">

            <span className={`system-status ${running ? '' : 'muted'}`}><i/>{running ? 'System Healthy' : 'Simulation Idle'}</span>

            <span className="divider"/>

            <span>{new Date().toLocaleDateString('en-IN', { month:'short', day:'numeric', year:'numeric' })}</span>

            <span>{new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'})}</span>

          </div>

        </header>



        {error && <div className="error-banner"><AlertTriangle size={17}/><span>Backend connection: {error}</span><button onClick={refresh}>Retry</button></div>}

        <section className={`anomaly-panel ${data?.anomaly?.status === 'DETECTED' ? 'detected' : data?.anomaly?.status === 'RECOVERING' ? 'recovering' : 'normal'}`} id="sla-anomaly">
          <div className="anomaly-icon"><AlertTriangle size={21}/></div>
          <div className="anomaly-copy">
            <div className="anomaly-title-row"><h2>Telemetry Anomaly Detection</h2><span className={`anomaly-status ${String(data?.anomaly?.status || 'NORMAL').toLowerCase()}`}>{data?.anomaly?.status || 'WAITING'}</span></div>
            <p>{data?.anomaly?.reason || 'Waiting for enough normal telemetry to establish a baseline.'}</p>
            {data?.anomaly?.status !== 'NORMAL' && <small>Detected metrics: {(data?.anomaly?.metrics || []).join(', ') || 'telemetry deviation'} · Severity: {data?.anomaly?.severity || 'warning'}</small>}
          </div>
        </section>

        <section className="panel environment-monitor" id="environment-monitor">
          <div className="panel-header environment-header">
            <div><h2><MonitorCog size={18}/> Hardware &amp; Software Change Monitor</h2><p>TinyFish live release search compared with simulated installed versions</p></div>
            <button className="environment-scan" onClick={scanEnvironment} disabled={busy || environmentScanning}><Search size={15}/>{environmentScanning ? 'Scanning…' : 'Scan for Updates'}</button>
          </div>
          <div className="environment-meta">
            <span className={`environment-connection ${environmentMonitor?.configured ? 'connected' : 'not-connected'}`}><i/>{environmentMonitor?.configured ? 'TinyFish API key configured' : 'TinyFish API key not configured'}</span>
            <span>Inventory: simulated demo values</span>
            {environmentMonitor?.last_scan_at && <span>Last scan: {toLocalDate(environmentMonitor.last_scan_at).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'})}</span>}
          </div>
          <div className="environment-inventory">
            {(environmentMonitor?.inventory || []).map(item => <div className="environment-version" key={item.component}><span>{item.component}</span><strong>{item.installed_version}</strong></div>)}
          </div>
          <p className="environment-message">{environmentMonitor?.message || 'Configure TINYFISH_API_KEY in the backend environment, then scan to retrieve live release information.'}</p>
          {environmentMonitor?.findings?.length > 0 && <div className="environment-findings">
            {environmentMonitor.findings.map((finding, index) => <div className={`environment-finding ${finding.status}`} key={`${finding.component}-${index}`}>
              <div className="environment-finding-head"><strong>{finding.component}</strong><span>{finding.status.replaceAll('_', ' ')}</span></div>
              <p>{finding.summary}</p>
              {finding.source_url && <a href={finding.source_url} target="_blank" rel="noreferrer">{finding.source_title || 'Open source'} <ExternalLink size={12}/></a>}
            </div>)}
          </div>}
          <small className="environment-disclaimer">This feature checks public release information only. It does not read or change a physical GPU driver, and update candidates require manual compatibility verification.</small>
        </section>

        <section className="kpi-grid">

          <Kpi icon={Power} label="Total GPU Power" value={fmt(telemetry?.total_power)} unit=" W" tone="power" history={chartData} dataKey="power"/>

          <Kpi icon={Thermometer} label="Temperature" value={fmt(telemetry?.temperature)} unit=" °C" tone="temperature" history={chartData} dataKey="temperature"/>

          <Kpi icon={Cpu} label="GPU Utilization" value={fmt(telemetry?.gpu_utilization)} unit=" %" tone="util" history={chartData} dataKey="utilization"/>

          <Kpi icon={BarChart3} label="Request Rate" value={fmt(telemetry?.request_rate)} unit=" req/min" tone="request" history={chartData} dataKey="requestRate"/>

        </section>



        <section className="optimization-card" id="ai-controller">

          <div className="config-side">

            <div className="section-title"><Cpu size={19}/><h2>Current <span>→</span> Optimized Configuration</h2></div>

            <div className="config-compare">

              <ConfigBox title="Current (Before)" config={current}/>

              <div className="arrow">→</div>

              <ConfigBox title={recommendationApplied ? "Optimized (Applied by AI)" : "AI Recommendation (Not Applied)"} config={recommended} optimized/>

            </div>

            <div className="config-evolution">
              <div className="section-title"><History size={18}/><h3>Configuration Evolution</h3><span className="live-badge"><i/>Live</span></div>
              <p className="config-evolution-note">Recent unique configurations selected from live workload conditions.</p>
              <div className="config-evolution-grid">
                {(data?.configuration_history || []).map((item, i) => (
                  <div className="config-mini" key={`${item.timestamp}-${i}`}>
                    <div className="config-mini-head"><b>#{i + 1}</b><span>{String(item.workload_state).toUpperCase()}</span></div>
                    <strong>{item.configuration.precision} · B{item.configuration.batch_size}</strong>
                    <div>GPU {item.configuration.gpu_allocation} · R{item.configuration.replicas} · C{item.configuration.concurrency}</div>
                    <small>{fmt(item.request_rate)} req/min</small>
                  </div>
                ))}
              </div>
            </div>

          </div>

          <div className="why-side">

            <div className="section-title"><Zap size={19}/><h2>Why this optimization?</h2></div>

            <ul>{(opt?.reasons || ['Awaiting optimizer evaluation.']).map((r,i) => <li key={i}>{r}</li>)}</ul>

            <div className="optimizer-status-line">
              <span>Background optimizer</span><b>{data?.optimizer_status?.evaluation_running ? 'Evaluating' : (data?.optimizer_status?.worker_running ? 'Idle / scheduled' : 'Starting')}</b>
              <small>Last evaluation: {data?.optimizer_status?.last_evaluation_at ? new Date(data.optimizer_status.last_evaluation_at).toLocaleTimeString() : 'waiting for first cycle'}</small>
            </div>
            <div className="performance-metrics">
              <div><span>Primary SLA Latency</span><strong>{fmt(opt?.current_estimate?.latency, 1)} <small>ms current</small></strong><em>{fmt(opt?.recommended_estimate?.latency, 1)} ms optimized</em></div>
              <div><span>Throughput</span><strong>{fmt(opt?.current_estimate?.throughput)} <small>req/min current</small></strong><em>{fmt(opt?.recommended_estimate?.throughput)} req/min optimized</em></div>
            </div>
            <div className="impact-heading">Expected Impact <span>(vs. Current)</span></div>

            <div className="expected-grid">

              <ImpactStat label="Power" value={impact?.power_reduction_pct} suffix="%" direction="down"/>

              <ImpactStat label="Temperature" value={impact?.temperature_reduction_c} suffix="°C" direction="down"/>

              <ImpactStat label="Throughput" value={impact?.throughput_change_req_min} suffix=" req/min" direction="up" neutral/>

              <ImpactStat label="SLA" value={slaPass ? 'Maintained' : 'Failed'} sla/>

            </div>

            {anomalyToast && <div className="anomaly-toast" role="alert" aria-live="assertive"><AlertTriangle size={16}/><span>{anomalyToast.text}</span><button onClick={() => setAnomalyToast(null)} aria-label="Dismiss alert">×</button></div>}

            <div className={`autonomous-banner ${auto ? '' : 'inactive'}`} onClick={toggleAutopilot}>

              <ShieldCheck size={21}/><div><strong>{auto ? 'Autonomous Optimization Active' : 'Autonomous Optimization Inactive'}</strong><span>{auto ? 'Optimization automatically applied by AI Controller.' : 'Enable Autonomous Mode to apply feasible recommendations.'}</span></div>

            </div>

          </div>

        </section>



        <section className="panel metrics-panel" id="live-monitoring">

          <div className="panel-header">

            <div><h2>GPU Metrics Over Time</h2><p>Live simulator telemetry · 5 simulated GPUs</p></div>

            <select value={range} onChange={e => setRange(Number(e.target.value))}>

              <option value="5">Last 5 minutes</option><option value="15">Last 15 minutes</option>

              <option value="30">Last 30 minutes</option><option value="60">Last 1 hour</option>

            </select>

          </div>

          <div className="chart-toolbar">

            <div className="metric-tabs">

              <button className={metric==='power'?'selected':''} onClick={()=>setMetric('power')}>Power (W)</button>

              <button className={metric==='temperature'?'selected':''} onClick={()=>setMetric('temperature')}>Temperature (°C)</button>

            </div>

            <div className="chart-legend">

              <span><i className="legend before"/>Before Optimization</span>

              <span><i className="legend after"/>After Optimization</span>

              <span><i className="legend target"/>Target</span>

            </div>

          </div>

          <div className="main-chart">

            <ResponsiveContainer width="100%" height="100%">

              <LineChart data={chartData} margin={{top: 10, right: 16, left: 5, bottom: 4}}>

                <CartesianGrid strokeDasharray="3 5"/>

                <XAxis dataKey="time" minTickGap={42}/>

                <YAxis domain={['auto','auto']} tickFormatter={v=>metric==='power'?`${v}`:`${v}°`}/>

                <Tooltip formatter={(v, name)=>[`${fmt(v,1)} ${metric==='power'?'W':'°C'}`, name]}/>

                <Legend verticalAlign="top" height={28} />

                {metric === 'power' && Number.isFinite(Number(opt?.recommended_estimate?.total_power)) && Number(opt?.recommended_estimate?.total_power) > 0 && (
                  <ReferenceLine y={Number(opt.recommended_estimate.total_power)} stroke="#64748b" strokeDasharray="6 5" strokeWidth={2} label={{ value: 'Target', position: 'insideTopRight', fill: '#64748b', fontSize: 11 }} />
                )}

                {metric === 'temperature' && Number.isFinite(Number(data?.sla?.configuration?.max_temperature_c)) && Number(data?.sla?.configuration?.max_temperature_c) > 0 && (
                  <ReferenceLine y={Number(data.sla.configuration.max_temperature_c)} stroke="#64748b" strokeDasharray="6 5" strokeWidth={2} label={{ value: 'Target / SLA limit', position: 'insideTopRight', fill: '#64748b', fontSize: 11 }} />
                )}

                <Line type="linear" name="Before Optimization" dataKey="before" dot={false} stroke="#ef4444" strokeWidth={2.5} activeDot={{ r: 4 }} connectNulls={false} isAnimationActive={false}/>

                <Line type="linear" name="Detected anomaly" dataKey="anomalyMetric" stroke="#e03439" strokeWidth={0} dot={{ r: 5, fill: '#e03439', stroke: '#fff', strokeWidth: 2 }} activeDot={{ r: 7 }} connectNulls={false} isAnimationActive={false}/>

                <Line type="linear" name="After Optimization" dataKey="after" dot={false} stroke="#1683ff" strokeWidth={2.5} activeDot={{ r: 4 }} connectNulls={false} isAnimationActive={false}/>
              {eventMarkers.slice(-4).map((e, i) => {
                const eventTime = toLocalDate(e.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                return (
                  <ReferenceLine
                    key={`${e.timestamp}-${i}`}
                    x={eventTime}
                    stroke={e.type === 'configuration_applied' ? '#0a9561' : '#e22e35'}
                    strokeDasharray="5 4"
                    label={{ value: e.title, position: 'insideTop', fontSize: 9 }}
                  />
                )
              })}
            </LineChart>
          </ResponsiveContainer>
          </div>

          <div className="chart-note">

            <span>Dashed target line shows the optimizer's power estimate or the configured temperature SLA limit.</span>

            <span>{data?.simulation?.workload_state?.toUpperCase() || 'N/A'} · {data?.cluster?.active_count ?? '—'} active simulated GPUs</span>

          </div>

        </section>



        <section className="panel quality-panel" id="precision-quality">
          <div className="panel-header">
            <div><h2>FP16 vs INT8 Quality Gate</h2><p>Simulated quality evaluation · no real model or dataset is loaded</p></div>
            <span className={`quality-pill ${(quality?.result || 'READY') === 'PASS' ? 'pass' : (quality?.result === 'FAIL' ? 'fail' : '')}`}>{quality?.result || 'READY'}</span>
          </div>
          <div className="quality-controls">
            <label>Allowed degradation (percentage points)
              <input type="number" min="0" max="100" step="0.1" value={qualityThreshold} onChange={e => setQualityThreshold(e.target.value)} />
            </label>
            <button onClick={evaluateQuality} disabled={busy}>Evaluate FP16 → INT8</button>
          </div>
          {quality && <div className="quality-results">
            <div><span>FP16 baseline</span><strong>{fmt(quality.baseline_quality_score, 2)}</strong></div>
            <div><span>INT8 candidate</span><strong>{fmt(quality.candidate_quality_score, 2)}</strong></div>
            <div><span>Degradation</span><strong>{fmt(quality.quality_degradation_percentage_points, 2)} pp</strong></div>
            <div><span>Threshold</span><strong>{fmt(quality.allowed_degradation_percentage_points, 2)} pp</strong></div>
            <p>{quality.reason} <small>{quality.disclaimer}</small></p>
          </div>}
        </section>


        <section className="sustainability panel">

          <div className="panel-header simple"><div><h2><Leaf size={20}/> Sustainability Impact</h2><p>Estimated for current workload</p></div></div>

          <div className="sustain-grid">

            <SustainCard icon={Leaf} title="Energy Saved" value={impact?.power_saved_w != null ? fmt(Number(impact.power_saved_w)/1000,2) : 'N/A'} unit="kWh/hr" tone="green" sub="Estimated energy reduction"/>

            <SustainCard icon={Thermometer} title="Thermal Reduction" value={impact?.temperature_reduction_c != null ? `- ${fmt(Math.max(0, impact.temperature_reduction_c),1)}` : 'N/A'} unit="°C" tone="red" sub="Lower cooling load"/>

            <SustainCard icon={BarChart3} title="Energy Efficiency" value={data?.sustainability?.energy_efficiency_req_per_w != null ? fmt(data.sustainability.energy_efficiency_req_per_w,2) : 'N/A'} unit="req/W" tone="purple" sub="AI output per unit energy"/>

          </div>

        </section>



        <section className="bottom-grid">

          <section className="panel fleet-panel" id="gpu-fleet">

            <div className="panel-header"><div><h2>GPU Fleet</h2><p>Exactly 5 simulated GPUs · dynamic allocation</p></div><span className="fleet-count">{data?.cluster?.active_count ?? '—'} Active / 5</span></div>

            <div className="fleet-summary">
              <div><span>Total GPUs</span><strong>{data?.cluster?.gpu_count ?? 5}</strong></div>
              <div><span>Active</span><strong>{(data?.cluster?.gpus || []).filter(g => g.status !== 'Idle').length}</strong></div>
              <div><span>Idle</span><strong>{(data?.cluster?.gpus || []).filter(g => g.status === 'Idle').length}</strong></div>
              <div><span>Overloaded</span><strong>{(data?.cluster?.gpus || []).filter(g => g.status === 'Overloaded').length}</strong></div>
            </div>
            <div className="gpu-cards">
              {(data?.cluster?.gpus || []).map(g => <GpuRow key={g.id} gpu={g}/>)}
              {!data?.cluster?.gpus?.length && <div className="empty">Waiting for GPU fleet telemetry…</div>}
            </div>

          </section>



          <section className="panel workload-panel" id="workloads">

            <div className="panel-header"><div><h2>Active Workloads</h2><p>Current serving demand</p></div></div>

            {(data?.workloads || []).map(w => <div className="workload-row" key={w.id}><div><strong>{w.name}</strong><span>{w.model}</span></div><div><b>{fmt(w.requests_per_min)}</b><span>req/min</span></div><div><b>{fmt(w.latency_ms,0)} ms</b><span>latency</span></div><Pill pass={w.status.includes('Protected')}>{w.status}</Pill></div>)}

          </section>

        </section>



        <section className="panel decision-panel" id="decision-history">

          <div className="panel-header"><div><h2><History size={19}/> Decision History</h2><p>Actual simulator and controller events</p></div><span className={`live-badge ${auto?'':'idle'}`}><i/>{auto ? 'Live' : 'Idle'}</span></div>

          <div className="event-list">

            {(data?.recent_events || []).slice().reverse().slice(0,8).map((e,i) => (

              <div className="decision" key={`${e.timestamp}-${i}`}>

                <span className="decision-time">{toLocalDate(e.timestamp).toLocaleTimeString()}</span>

                <span className={`decision-dot ${e.type || ''}`}/>

                <div><strong>{e.title || String(e.event || '').replaceAll('_',' ')}</strong><p>{e.message}</p></div>

              </div>

            ))}

            {!data?.recent_events?.length && <div className="empty">No controller events yet. Start the simulation; GPUPilot will generate workload spikes automatically.</div>}

          </div>

        </section>



        <section className="control-panel panel">

          <div><span className="eyebrow">SIMULATOR CONTROLS</span><h2>Monitor → Predict → Optimize → Apply → Verify</h2><p>All controls below call the FastAPI backend; no UI-only simulation is used.</p></div>

          <div className="control-actions">

            <button onClick={start} disabled={busy}><Play size={16}/>Start</button>

            <button onClick={stop} disabled={busy}><Square size={16}/>Stop</button>

            <div className="auto-spike-chip"><Zap size={16}/>Automatic spikes enabled</div>

            <button className="warning" onClick={triggerAnomalySpike} disabled={busy}><AlertTriangle size={15}/>Trigger Anomaly Spike</button>

            <button className="autopilot" onClick={toggleAutopilot} disabled={busy}><Cpu size={16}/>{auto ? 'Autopilot ON' : 'Enable Autopilot'}</button>

            <button onClick={reset} disabled={busy}><RotateCcw size={16}/>Reset</button>

          </div>

        </section>



        {(settingsOpen || activePage === 'settings') && <section className="settings-panel panel" id="settings">

          <div className="panel-header simple"><div><h2><SlidersHorizontal size={19}/> Settings</h2><p>Only controls that affect the supplied simulator are exposed.</p></div><button className="close-btn" onClick={()=>setSettingsOpen(false)}>Close</button></div>

          <div className="settings-grid">

            <label>Simulation tick interval

              <select value={interval} onChange={e=>setIntervalValue(Number(e.target.value))}>

                <option value="0.5">0.5 seconds</option><option value="1">1 second</option><option value="2">2 seconds</option><option value="5">5 seconds</option>

              </select>

            </label>

            <div className="setting-state"><span>Autonomous controller</span><b>{auto ? 'Enabled' : 'Disabled'}</b><button onClick={toggleAutopilot}>{auto ? 'Stop' : 'Start'}</button></div>

            <div className="setting-state"><span>Configured SLA</span><b>{data?.sla?.configuration?.max_latency_ms ?? '—'} ms max latency</b><small>{data?.sla?.configuration?.min_throughput_req_min ?? '—'} req/min minimum throughput</small></div>

          </div>

        </section>}



        <footer>GPUPilot prototype · 5 simulated GPUs · GPU telemetry is simulated and not physical hardware telemetry.</footer>

      </main>

    </div>

  )

}



function Kpi({icon:Icon,label,value,unit,tone,history,dataKey}) {

  const first = history?.[0]?.[dataKey]

  const last = history?.[history.length-1]?.[dataKey]

  const delta = first && last ? ((last-first)/first)*100 : null

  return <div className={`kpi ${tone}`}>

    <div className="kpi-icon"><Icon size={22}/></div>

    <div className="kpi-body"><span>{label}</span><strong>{value}<small>{unit}</small></strong>

      <div className="kpi-delta"><b>{delta === null ? '—' : `${delta >= 0 ? '↑' : '↓'} ${Math.abs(delta).toFixed(0)}%`}</b><span>vs. recent history</span></div>

    </div>

    <div className="spark"><ResponsiveContainer width="100%" height="100%"><AreaChart data={history}><Area dataKey={dataKey} type="linear" strokeWidth={2} fillOpacity={0.12} /></AreaChart></ResponsiveContainer></div>

  </div>

}



function ConfigBox({title,config,optimized}) {

  return <div className={`config-box ${optimized?'optimized':''}`}>

    <div className="config-title"><strong>{title}</strong>{optimized && <span className="applied-tag">AI recommendation</span>}</div>

    {config ? <div className="config-values">

      <Row label="Precision" value={config.precision}/><Row label="Batch Size" value={config.batch_size}/>

      <Row label="GPU Allocation" value={config.gpu_allocation}/><Row label="Replicas" value={config.replicas}/>

      <Row label="Concurrency" value={config.concurrency}/>

    </div> : <div className="muted">Awaiting optimizer result.</div>}

  </div>

}

function Row({label,value}) { return <div><span>{label}</span><b>{value ?? 'N/A'}</b></div> }



function ImpactStat({label,value,suffix='',direction,sla,neutral}) {

  if (sla) return <div className="impact-stat"><span>{label}</span><strong className={value==='Maintained'?'positive':'negative'}>{value==='Maintained'?'✓ ':''}{value}</strong></div>

  const num = Number(value)

  return <div className="impact-stat"><span>{label}</span><strong className={num>=0 ? 'positive' : neutral ? 'neutral' : 'negative'}>{value == null ? 'N/A' : `${direction==='down'?'↓':'↑'} ${Math.abs(num).toFixed(1)}${suffix}`}</strong></div>

}

function SustainCard({icon:Icon,title,value,unit,sub,tone}) {

  return <div className={`sustain-card ${tone}`}><div className="sustain-icon"><Icon size={25}/></div><div><span>{title}</span><strong>{value} <small>{unit}</small></strong><p>{sub}</p></div></div>

}

function GpuRow({gpu}) {
  const status = gpu.status || (gpu.active ? 'Active' : 'Idle')
  return <div className={`gpu-row gpu-${String(status).toLowerCase()}`}>
    <div className="gpu-id"><span className={`gpu-status ${gpu.active ? 'active' : ''}`}/><div><strong>{gpu.id}</strong><span>{gpu.model} · {gpu.workload || 'No workload assigned'}</span></div><Pill pass={status === 'Active'}>{status}</Pill></div>
    <div><b>{fmt(gpu.utilization)}%</b><span>Utilization</span><div className="gpu-meter"><i style={{width:`${Math.max(0,Math.min(100,Number(gpu.utilization)||0))}%`}}/></div></div>
    <div><b>{fmt(gpu.power,1)} W</b><span>Power</span></div>
    <div><b>{fmt(gpu.temperature,1)}°C</b><span>Temperature</span></div>
    <div><b>{fmt(gpu.memory,1)} / {fmt(gpu.memory_capacity,0)} GB</b><span>Memory</span><div className="gpu-meter"><i style={{width:`${Math.max(0,Math.min(100,100*(Number(gpu.memory)||0)/(Number(gpu.memory_capacity)||24)))}%`}}/></div></div>
  </div>
}

function Pill({children,pass=false}) { return <span className={`pill ${pass?'pass':''}`}>{children}</span> }



createRoot(document.getElementById('root')).render(<App />)
