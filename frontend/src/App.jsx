import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
    Upload, Loader2, RefreshCcw, ChevronLeft, ChevronRight,
    Activity, Cpu, Server, Layers, Image as ImageIcon,
    CheckCircle, AlertTriangle, ScanLine, Database, Zap,
    GitBranch, FlaskConical, BarChart3, ArrowRight,
    MonitorDot, BookOpen, Box
} from 'lucide-react';
import axios from 'axios';

// ─── Comparison Slider ───────────────────────────────────────────────────────
const ComparisonSlider = ({ before, after }) => {
    const [pos, setPos] = useState(50);
    const containerRef = useRef(null);
    const dragging = useRef(false);

    const handleMove = useCallback((e) => {
        if (!dragging.current || !containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX ?? e.touches?.[0]?.clientX ?? rect.left;
        setPos(Math.min(100, Math.max(0, ((x - rect.left) / rect.width) * 100)));
    }, []);

    const startDrag = (e) => { dragging.current = true; e.preventDefault(); };
    const endDrag = () => { dragging.current = false; };

    useEffect(() => {
        window.addEventListener('mousemove', handleMove);
        window.addEventListener('mouseup', endDrag);
        window.addEventListener('touchmove', handleMove, { passive: false });
        window.addEventListener('touchend', endDrag);
        return () => {
            window.removeEventListener('mousemove', handleMove);
            window.removeEventListener('mouseup', endDrag);
            window.removeEventListener('touchmove', handleMove);
            window.removeEventListener('touchend', endDrag);
        };
    }, [handleMove]);

    return (
        <div className="comparison-wrap">
            <div
                ref={containerRef}
                className="comparison-slider"
                onMouseDown={startDrag}
                onTouchStart={startDrag}
            >
                {/* Dehazed — full bg */}
                <img src={after} alt="Dehazed" draggable={false} style={{ zIndex: 1 }} />
                {/* Original — clipped */}
                <div style={{ position: 'absolute', inset: 0, width: `${pos}%`, overflow: 'hidden', zIndex: 2 }}>
                    <img
                        src={before}
                        alt="Original"
                        draggable={false}
                        style={{
                            position: 'absolute', inset: 0, height: '100%', objectFit: 'contain',
                            width: `${100 / (pos / 100)}%`, maxWidth: 'none'
                        }}
                    />
                </div>
                {/* Divider */}
                <div className="comparison-divider" style={{ left: `${pos}%`, zIndex: 10 }}>
                    <div
                        className="comparison-handle"
                        onMouseDown={startDrag}
                        onTouchStart={startDrag}
                        style={{ left: 0, top: '50%' }}
                    >
                        <ChevronLeft size={12} style={{ marginRight: -4 }} />
                        <ChevronRight size={12} />
                    </div>
                </div>
            </div>
            <div className="comparison-footer">
                <span className="comparison-tag">
                    <span className="comparison-tag-dot" style={{ background: '#f97316' }} />
                    Original &mdash; Hazy Input
                </span>
                <span style={{ color: 'var(--text-dimmer)', fontFamily: 'var(--font-mono)', fontSize: '9px' }}>
                    DRAG TO COMPARE
                </span>
                <span className="comparison-tag">
                    <span className="comparison-tag-dot" style={{ background: '#10b981' }} />
                    Dehazed &mdash; VIM Output
                </span>
            </div>
        </div>
    );
};
// ─── Neural Processing Overlay ───────────────────────────────────────────────
const PIPELINE_STAGES = [
    { id: 0, label: 'Extracting Patches', sub: 'Conv2D · Stride 16 · Tokenization', color: '#f97316' },
    { id: 1, label: 'Forward SSM Scan', sub: 'Bi-VMamba · Linear O(N) Sweep →', color: '#3b82f6' },
    { id: 2, label: 'Backward SSM Scan', sub: 'Bi-VMamba · Reverse Context ←', color: '#3b82f6' },
    { id: 3, label: 'K-Map Prediction', sub: 'Spatial Conv Head · Haze Parameter', color: '#8b5cf6' },
    { id: 4, label: 'AOD Physics Layer', sub: 'J = K·I − K + 1 · Reconstruction', color: '#10b981' },
    { id: 5, label: 'Upsampling Output', sub: 'Lanczos Interp · Native Resolution', color: '#10b981' },
];

const NeuralProcessingOverlay = ({ preview }) => {
    const [activeStage, setActiveStage] = useState(0);
    const [progress, setProgress] = useState(0);
    const [tick, setTick] = useState(0);

    useEffect(() => {
        const iv = setInterval(() => {
            setActiveStage(s => (s + 1) % PIPELINE_STAGES.length);
            setProgress(0);
            setTick(t => t + 1);
        }, 950);
        return () => clearInterval(iv);
    }, []);

    useEffect(() => {
        let start = null;
        const dur = 900;
        const raf = (ts) => {
            if (!start) start = ts;
            const p = Math.min(100, ((ts - start) / dur) * 100);
            setProgress(p);
            if (p < 100) requestAnimationFrame(raf);
        };
        const id = requestAnimationFrame(raf);
        return () => cancelAnimationFrame(id);
    }, [tick]);

    const stage = PIPELINE_STAGES[activeStage];

    return (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {/* Blurred hazy image */}
            <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
                <img src={preview} alt=""
                    style={{
                        width: '100%', height: '100%', objectFit: 'contain',
                        opacity: 0.06, filter: 'blur(6px) grayscale(0.7)'
                    }} />
                <div style={{
                    position: 'absolute', inset: 0,
                    background: 'linear-gradient(to bottom, rgba(5,5,5,0.85), rgba(5,5,5,0.97))'
                }} />
            </div>

            {/* Top scan-line sweep */}
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', zIndex: 1, overflow: 'hidden' }}>
                <div style={{
                    height: '100%',
                    background: `linear-gradient(to right, transparent, ${stage.color}CC, transparent)`,
                    animation: 'scan-sweep 1.4s linear infinite',
                }} />
            </div>

            {/* Main content */}
            <div style={{
                position: 'relative', zIndex: 2, flex: 1,
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center',
                padding: 40, gap: 36,
            }}>
                {/* Hero stage label */}
                <div style={{ textAlign: 'center', maxWidth: 420 }}>
                    <div style={{
                        fontSize: 9, fontFamily: 'var(--font-mono)', letterSpacing: '0.3em',
                        textTransform: 'uppercase', color: 'var(--text-dimmer)', marginBottom: 14
                    }}>
                        Neural Inference In Progress
                    </div>
                    <div style={{
                        fontSize: 28, fontWeight: 700, letterSpacing: '-0.02em',
                        color: stage.color, transition: 'color 0.35s ease',
                        textShadow: `0 0 50px ${stage.color}40`,
                    }}>
                        {stage.label}
                    </div>
                    <div style={{
                        fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)',
                        letterSpacing: '0.08em', marginTop: 10
                    }}>
                        {stage.sub}
                    </div>
                    {/* Stage progress bar */}
                    <div style={{ marginTop: 20, height: 2, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{
                            height: '100%', width: `${progress}%`, borderRadius: 2,
                            background: stage.color, transition: 'width 0.05s linear',
                            boxShadow: `0 0 8px ${stage.color}80`,
                        }} />
                    </div>
                </div>

                {/* Pipeline nodes strip */}
                <div style={{
                    display: 'flex', alignItems: 'flex-start', gap: 0,
                    background: 'rgba(8,8,8,0.9)', border: '1px solid var(--border)',
                    borderRadius: 10, padding: '14px 18px',
                    width: '100%', maxWidth: 700,
                }}>
                    {PIPELINE_STAGES.map((s, i) => {
                        const isDone = i < activeStage;
                        const isActive = i === activeStage;
                        return (
                            <React.Fragment key={s.id}>
                                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 7, flex: 1 }}>
                                    {/* Circle node */}
                                    <div style={{
                                        width: 28, height: 28, borderRadius: '50%',
                                        border: `1.5px solid ${isActive ? s.color : isDone ? '#222' : '#181818'}`,
                                        background: isActive ? `${s.color}15` : 'transparent',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        transition: 'all 0.35s ease',
                                        boxShadow: isActive ? `0 0 14px ${s.color}50` : 'none',
                                    }}>
                                        {isActive ? (
                                            <div style={{
                                                width: 8, height: 8, borderRadius: '50%', background: s.color,
                                                animation: 'node-pulse 0.9s ease-in-out infinite'
                                            }} />
                                        ) : (
                                            <div style={{
                                                width: 6, height: 6, borderRadius: '50%',
                                                background: isDone ? '#2a2a2a' : '#161616'
                                            }} />
                                        )}
                                    </div>
                                    {/* Label */}
                                    <div style={{
                                        fontSize: 8, fontFamily: 'var(--font-mono)', letterSpacing: '0.04em',
                                        textTransform: 'uppercase', textAlign: 'center',
                                        color: isActive ? s.color : isDone ? '#282828' : '#1c1c1c',
                                        transition: 'color 0.35s ease', maxWidth: 74,
                                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                                    }}>
                                        {s.label}
                                    </div>
                                </div>
                                {/* Connector line */}
                                {i < PIPELINE_STAGES.length - 1 && (
                                    <div style={{
                                        flex: 1, height: 1, marginBottom: 22, marginTop: 13,
                                        background: isDone ? '#1e1e1e' : 'var(--border)', position: 'relative', overflow: 'hidden'
                                    }}>
                                        {i === activeStage && (
                                            <div style={{
                                                position: 'absolute', top: 0, bottom: 0, width: '45%',
                                                background: `linear-gradient(to right, transparent, ${PIPELINE_STAGES[i + 1]?.color ?? '#fff'}60, transparent)`,
                                                animation: 'scan-sweep 0.8s linear infinite',
                                            }} />
                                        )}
                                    </div>
                                )}
                            </React.Fragment>
                        );
                    })}
                </div>

                {/* GPU footer */}
                <div style={{
                    fontSize: 9, fontFamily: 'var(--font-mono)', letterSpacing: '0.22em',
                    textTransform: 'uppercase', color: 'var(--text-dimmer)'
                }}>
                    INFERENCE ON GPU · RTX 3050 · CUDA FP16+FP32
                </div>
            </div>
        </div>
    );
};

// ─── Architecture SVG Diagram ────────────────────────────────────────────────
const ArchDiagram = () => (
    <div className="arch-diagram">
        {/* Row 1: Main pipeline */}
        <div className="arch-pipeline">

            <div className="arch-node">
                <span className="arch-node-badge">RAW INPUT</span>
                <div className="arch-box input">
                    <div className="arch-box-label">Hazy Image I</div>
                    <div className="arch-box-sub">RGB · H×W×3</div>
                </div>
            </div>

            <div className="arch-arrow"><ArrowRight size={18} /></div>

            <div className="arch-node">
                <span className="arch-node-badge">STAGE 01</span>
                <div className="arch-box embed">
                    <div className="arch-box-label">Patch Embedding</div>
                    <div className="arch-box-sub">Conv2D · Stride 16</div>
                </div>
            </div>

            <div className="arch-arrow"><ArrowRight size={18} /></div>

            <div className="arch-node">
                <span className="arch-node-badge">STAGE 02 — TRITON CORE</span>
                <div className="arch-box engine">
                    <div style={{ display: 'flex', gap: 6, justifyContent: 'center', marginBottom: 8 }}>
                        <span style={{ fontSize: 8, background: 'rgba(59,130,246,0.2)', color: '#3b82f6', border: '1px solid rgba(59,130,246,0.3)', padding: '2px 6px', borderRadius: 3, fontFamily: 'var(--font-mono)', letterSpacing: '0.1em' }}>SHIFTED WINDOWS</span>
                        <span style={{ fontSize: 8, background: 'rgba(59,130,246,0.2)', color: '#3b82f6', border: '1px solid rgba(59,130,246,0.3)', padding: '2px 6px', borderRadius: 3, fontFamily: 'var(--font-mono)', letterSpacing: '0.1em' }}>BI-DIRECTIONAL SWEEP</span>
                    </div>
                    <div className="arch-box-label" style={{ fontSize: 12 }}>Triton Mamba Array</div>
                    <div className="arch-box-sub">Hardware Accelerated Kernel · O(N)</div>
                </div>
            </div>

            <div className="arch-arrow"><ArrowRight size={18} /></div>

            <div className="arch-node">
                <span className="arch-node-badge">STAGE 03</span>
                <div className="arch-box embed" style={{ background: 'rgba(168,85,247,0.08)', borderColor: 'rgba(168,85,247,0.25)', color: '#c084fc' }}>
                    <div className="arch-box-label">Multi-Scale Fusion</div>
                    <div className="arch-box-sub">5 Spatial Scales (1/16 to 1)</div>
                </div>
            </div>

            <div className="arch-arrow"><ArrowRight size={18} /></div>

            <div className="arch-node">
                <span className="arch-node-badge">PHYSICS LAYER</span>
                <div className="arch-box physics">
                    <div className="arch-box-label">AOD Equation</div>
                    <div className="arch-box-sub">J = K·I − K + 1</div>
                </div>
            </div>

            <div className="arch-arrow"><ArrowRight size={18} /></div>

            <div className="arch-node">
                <span className="arch-node-badge">OUTPUT</span>
                <div className="arch-box output">
                    <div className="arch-box-label">Dehazed J</div>
                    <div className="arch-box-sub">RGB · Restored</div>
                </div>
            </div>
        </div>

        {/* Row 2: Loss pathway */}
        <div style={{ marginTop: 32, paddingTop: 24, borderTop: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ fontSize: 9, letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--text-dim)', fontWeight: 700 }}>
                TRAINING SUPERVISION — DUAL-TRACK LOSS
            </div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                {[
                    { label: 'L1 Pixel Loss', weight: 'w=1.0', desc: 'Pixel-accurate reconstruction via Mean Absolute Error', color: '#3b82f6' },
                    { label: 'SSIM Loss', weight: 'w=0.5', desc: 'Structural & textural similarity enforcement', color: '#8b5cf6' },
                    { label: 'Contrastive (ConvNeXt-T)', weight: 'w=0.1', desc: 'Semantic feature regularization via frozen ConvNeXt-Tiny encoder', color: '#10b981' },
                ].map(l => (
                    <div key={l.label} style={{ flex: '1 1 200px', background: `${l.color}0d`, border: `1px solid ${l.color}26`, borderRadius: 8, padding: '12px 14px' }}>
                        <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: l.color }}>{l.label}</div>
                        <div style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginTop: 3 }}>{l.weight}</div>
                        <div style={{ fontSize: 10, color: 'var(--text-mid)', marginTop: 6, lineHeight: 1.6 }}>{l.desc}</div>
                    </div>
                ))}
            </div>
        </div>
    </div>
);

// ─── System Tab ──────────────────────────────────────────────────────────────
const SystemTab = ({ status }) => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [imgDims, setImgDims] = useState({ w: 0, h: 0 });
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const f = e.target.files[0];
        if (!f) return;
        const url = URL.createObjectURL(f);
        setFile(f); setPreview(url); setResult(null); setError(null);
        const img = new Image();
        img.onload = () => setImgDims({ w: img.naturalWidth, h: img.naturalHeight });
        img.src = url;
    };

    const handleReset = () => {
        setFile(null); setPreview(null); setResult(null); setError(null);
        setImgDims({ w: 0, h: 0 });
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleDehaze = async () => {
        if (!file || loading) return;
        setLoading(true); setResult(null); setError(null);
        const form = new FormData();
        form.append('image', file);
        try {
            const res = await axios.post('/predict', form, { responseType: 'blob' });
            setResult(URL.createObjectURL(res.data));
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Inference failed');
        } finally {
            setLoading(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const f = e.dataTransfer.files[0];
        if (!f) return;
        const dt = new DataTransfer();
        dt.items.add(f);
        if (fileInputRef.current) {
            fileInputRef.current.files = dt.files;
            handleFileChange({ target: fileInputRef.current });
        }
    };

    return (
        <div className="system-layout fade-in">
            {/* ─ Sidebar ─ */}
            <aside className="sidebar">

                {/* Module 01: Input Control */}
                <div className="panel">
                    <div className="panel-header">
                        <Upload size={12} />
                        INPUT CONTROL
                    </div>
                    <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                        <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileChange}
                            accept="image/*,.avif" style={{ display: 'none' }} />

                        {!preview ? (
                            <div
                                className="dropzone"
                                onClick={() => fileInputRef.current?.click()}
                                onDrop={handleDrop}
                                onDragOver={e => e.preventDefault()}
                            >
                                <div className="dropzone-icon"><Upload size={20} /></div>
                                <div>
                                    <div className="dropzone-title">Drop or click to upload</div>
                                    <div className="dropzone-sub">PNG · JPG · WEBP · AVIF · BMP · TIFF</div>
                                </div>
                            </div>
                        ) : (
                            <>
                                <div className="thumb-wrap">
                                    <img src={preview} alt="Preview" />
                                    <div className="thumb-meta">
                                        {imgDims.w}×{imgDims.h}px · {(file?.size / 1024).toFixed(0)} KB
                                    </div>
                                </div>

                                <button
                                    className="btn btn-execute"
                                    id="btn-dehaze"
                                    onClick={handleDehaze}
                                    disabled={loading || !status.model_loaded}
                                >
                                    {loading
                                        ? <><Loader2 size={14} className="animate-spin" /><span>Running…</span></>
                                        : <><ScanLine size={14} /><span>Execute Dehazing</span></>
                                    }
                                </button>

                                <button className="btn btn-ghost" onClick={handleReset}>
                                    <RefreshCcw size={12} />
                                    Reset Session
                                </button>

                                {error && (
                                    <div className="alert error">
                                        <AlertTriangle size={13} style={{ flexShrink: 0, marginTop: 1 }} />
                                        <span>{error}</span>
                                    </div>
                                )}

                                {result && !loading && (
                                    <div className="alert success">
                                        <CheckCircle size={13} style={{ flexShrink: 0, marginTop: 1 }} />
                                        <span>Dehazing complete — drag the slider to compare</span>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>

                {/* Module 02: System Telemetry */}
                <div className="panel">
                    <div className="panel-header">
                        <Activity size={12} />
                        SYSTEM TELEMETRY
                    </div>
                    <div className="panel-body">
                        <div className="telem-row">
                            <span className="telem-label">Backend</span>
                            <span className={`telem-badge ${status.model_loaded ? 'green' : 'red'}`}>
                                <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'currentColor', flexShrink: 0 }} />
                                {status.model_loaded ? 'Operational' : 'Offline'}
                            </span>
                        </div>
                        <div className="telem-row">
                            <span className="telem-label">Compute</span>
                            <span className="telem-badge amber">
                                <Cpu size={10} />
                                {status.device || 'RTX 3050'}
                            </span>
                        </div>
                        <div className="telem-row">
                            <span className="telem-label">Checkpoint</span>
                            <span className="telem-value" style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-mid)', maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {status.checkpoint || '…'}
                            </span>
                        </div>
                        <div className="telem-row">
                            <span className="telem-label">API</span>
                            <span className="telem-value" style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                                :5000/predict
                            </span>
                        </div>
                    </div>
                </div>

                {/* Module 03: Info */}
                <div className="alert info" style={{ margin: 0 }}>
                    <Database size={13} style={{ flexShrink: 0, marginTop: 1 }} />
                    <span>Output is upsampled back to input resolution via Lanczos interpolation post-inference.</span>
                </div>

            </aside>

            {/* ─ Viewport ─ */}
            <div className="viewport">
                {!preview && (
                    <div
                        className="viewport-idle"
                        onClick={() => fileInputRef.current?.click()}
                        onDrop={handleDrop}
                        onDragOver={e => e.preventDefault()}
                    >
                        <div className="viewport-idle-icon">
                            <ImageIcon size={28} />
                        </div>
                        <div style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-dim)', marginBottom: 6 }}>
                                No image loaded
                            </div>
                            <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', letterSpacing: '0.15em', textTransform: 'uppercase', color: 'var(--text-dimmer)' }}>
                                Upload a hazy image to begin
                            </div>
                        </div>
                    </div>
                )}

                {preview && !result && (
                    <div className="preview-await">
                        {loading ? (
                            <NeuralProcessingOverlay preview={preview} />
                        ) : (
                            <>
                                <div className="preview-await-bg"><img src={preview} alt="" /></div>
                                <div className="preview-await-content">
                                    <div style={{
                                        width: 48, height: 48,
                                        background: 'var(--bg-raised)',
                                        border: '1px solid var(--border)',
                                        borderRadius: 12,
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        color: 'var(--text-dim)'
                                    }}>
                                        <Zap size={22} />
                                    </div>
                                    <div className="scan-label">
                                        <div className="scan-label-primary" style={{ color: 'var(--text-mid)', fontSize: 14 }}>
                                            Image Loaded
                                        </div>
                                        <div className="scan-label-sub">Press Execute Dehazing in the sidebar →</div>
                                    </div>
                                </div>
                            </>
                        )}
                    </div>
                )}

                {result && (
                    <ComparisonSlider before={preview} after={result} />
                )}
            </div>
        </div>
    );
};


// ─── Epoch data ────────────────────────────────────────────────────────────
const EPOCH_DATA = [
    { ep: 1, lr: '4.00e-05', loss: '0.1242', psnr: 14.22, ssim: 0.6841, phase: 'Linear Warmup' },
    { ep: 2, lr: '8.00e-05', loss: '0.0815', psnr: 18.45, ssim: 0.7622, phase: 'Stabilizing SSM' },
    { ep: 3, lr: '1.20e-04', loss: '0.0633', psnr: 21.12, ssim: 0.8215, phase: 'Global Context' },
    { ep: 4, lr: '1.60e-04', loss: '0.0512', psnr: 23.08, ssim: 0.8594, phase: 'Edge Refinement' },
    { ep: 5, lr: '2.00e-04', loss: '0.0428', psnr: 24.56, ssim: 0.8912, phase: 'Peak LR Reached' },
];

// ─── Metrics SVG Chart ──────────────────────────────────────────────────────
const MetricsChart = () => {
    const [hovered, setHovered] = useState(null);
    const W = 420, H = 200, PL = 42, PR = 42, PT = 16, PB = 32;
    const iW = W - PL - PR, iH = H - PT - PB;
    const epochs = EPOCH_DATA.map(d => d.ep);
    const psnrMin = 12, psnrMax = 28;
    const ssimMin = 0.6, ssimMax = 1.0;
    const xOf = (ep) => PL + ((ep - 1) / (epochs.length - 1)) * iW;
    const yOfP = (v) => PT + (1 - (v - psnrMin) / (psnrMax - psnrMin)) * iH;
    const yOfS = (v) => PT + (1 - (v - ssimMin) / (ssimMax - ssimMin)) * iH;
    const psnrPts = EPOCH_DATA.map(d => `${xOf(d.ep)},${yOfP(d.psnr)}`).join(' ');
    const ssimPts = EPOCH_DATA.map(d => `${xOf(d.ep)},${yOfS(d.ssim)}`).join(' ');
    const gridYs = [0.25, 0.5, 0.75, 1].map(f => PT + f * iH);

    return (
        <div style={{ position: 'relative', userSelect: 'none' }}>
            <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', overflow: 'visible' }}>
                {/* Grid lines */}
                {gridYs.map((y, i) => (
                    <line key={i} x1={PL} y1={y} x2={W - PR} y2={y}
                        stroke="var(--border)" strokeWidth="0.5" />
                ))}
                {/* PSNR axis labels (left) */}
                {gridYs.map((y, i) => {
                    const v = psnrMax - ((i + 1) / 4) * (psnrMax - psnrMin);
                    return <text key={i} x={PL - 4} y={y + 4} textAnchor="end"
                        fontSize="7" fill="var(--text-dim)" fontFamily="var(--font-mono)">{v.toFixed(0)}</text>;
                })}
                {/* SSIM axis labels (right) */}
                {gridYs.map((y, i) => {
                    const v = ssimMax - ((i + 1) / 4) * (ssimMax - ssimMin);
                    return <text key={i} x={W - PR + 4} y={y + 4} textAnchor="start"
                        fontSize="7" fill="var(--text-dim)" fontFamily="var(--font-mono)">{v.toFixed(2)}</text>;
                })}
                {/* X labels */}
                {EPOCH_DATA.map(d => (
                    <text key={d.ep} x={xOf(d.ep)} y={H - 6} textAnchor="middle"
                        fontSize="7" fill="var(--text-dim)" fontFamily="var(--font-mono)">E{d.ep}</text>
                ))}
                {/* Axis labels */}
                <text x={6} y={PT + iH / 2} textAnchor="middle" fontSize="7" fill="var(--accent-blue)"
                    fontFamily="var(--font-mono)" transform={`rotate(-90, 6, ${PT + iH / 2})`}>PSNR dB</text>
                <text x={W - 6} y={PT + iH / 2} textAnchor="middle" fontSize="7" fill="var(--accent-green)"
                    fontFamily="var(--font-mono)" transform={`rotate(90, ${W - 6}, ${PT + iH / 2})`}>SSIM</text>

                {/* Hover vertical line */}
                {hovered !== null && (
                    <line x1={xOf(hovered + 1)} y1={PT} x2={xOf(hovered + 1)} y2={PT + iH}
                        stroke="rgba(255,255,255,0.15)" strokeWidth="1" strokeDasharray="3,3" />
                )}

                {/* PSNR line */}
                <polyline points={psnrPts} fill="none" stroke="var(--accent-blue)" strokeWidth="1.5"
                    strokeLinejoin="round" strokeLinecap="round" />
                {/* SSIM line */}
                <polyline points={ssimPts} fill="none" stroke="var(--accent-green)" strokeWidth="1.5"
                    strokeLinejoin="round" strokeLinecap="round" />

                {/* Dots + hit zones */}
                {EPOCH_DATA.map((d, i) => (
                    <g key={i}>
                        {/* Invisible wide hit strip */}
                        <rect x={xOf(d.ep) - 20} y={PT} width={40} height={iH}
                            fill="transparent" style={{ cursor: 'crosshair' }}
                            onMouseEnter={() => setHovered(i)}
                            onMouseLeave={() => setHovered(null)} />
                        {/* PSNR dot */}
                        <circle cx={xOf(d.ep)} cy={yOfP(d.psnr)} r={hovered === i ? 4 : 2.5}
                            fill="var(--accent-blue)" style={{ transition: 'r 0.1s' }} />
                        {/* SSIM dot */}
                        <circle cx={xOf(d.ep)} cy={yOfS(d.ssim)} r={hovered === i ? 4 : 2.5}
                            fill="var(--accent-green)" style={{ transition: 'r 0.1s' }} />
                    </g>
                ))}
            </svg>

            {/* Hover tooltip */}
            {hovered !== null && (() => {
                const d = EPOCH_DATA[hovered];
                const onRight = hovered < 3;
                return (
                    <div style={{
                        position: 'absolute',
                        top: '10%',
                        [onRight ? 'left' : 'right']: '4%',
                        background: 'rgba(5,5,5,0.92)',
                        border: '1px solid var(--border-mid)',
                        borderRadius: 8,
                        padding: '10px 14px',
                        pointerEvents: 'none',
                        zIndex: 10,
                        minWidth: 190,
                        backdropFilter: 'blur(12px)',
                    }}>
                        <div style={{
                            fontSize: 9, fontFamily: 'var(--font-mono)', letterSpacing: '0.18em',
                            textTransform: 'uppercase', color: 'var(--text-dim)', marginBottom: 8,
                            borderBottom: '1px solid var(--border)', paddingBottom: 6
                        }}>
                            EPOCH {d.ep} — {d.phase}
                        </div>
                        {[
                            { label: 'Learning Rate', val: d.lr, color: 'var(--accent-blue)' },
                            { label: 'Train Loss', val: d.loss, color: 'var(--text-mid)' },
                            { label: 'Val PSNR', val: `${d.psnr} dB`, color: 'var(--accent-green)' },
                            { label: 'Val SSIM', val: d.ssim.toFixed(4), color: 'var(--accent-green)' },
                        ].map(r => (
                            <div key={r.label} style={{
                                display: 'flex', justifyContent: 'space-between',
                                alignItems: 'center', padding: '3px 0'
                            }}>
                                <span style={{
                                    fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.08em',
                                    textTransform: 'uppercase'
                                }}>{r.label}</span>
                                <span style={{
                                    fontSize: 11, fontFamily: 'var(--font-mono)',
                                    fontWeight: 600, color: r.color
                                }}>{r.val}</span>
                            </div>
                        ))}
                    </div>
                );
            })()}

            {/* Legend */}
            <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginTop: 8 }}>
                {[['PSNR (dB)', 'var(--accent-blue)'], ['SSIM', 'var(--accent-green)']].map(([l, c]) => (
                    <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ width: 20, height: 2, background: c, borderRadius: 1 }} />
                        <span style={{
                            fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)',
                            letterSpacing: '0.1em', textTransform: 'uppercase'
                        }}>{l}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

// ─── Comic Bubble Overlay ───────────────────────────────────────────────────
const BUBBLES = [
    {
        id: 0,
        top: '8%', left: '3%',
        tail: 'bottom-right',
        title: '📄 The System Executive Summary',
        text: "This dashboard is the official technical interface for the Vision Mamba Neural Dehazing framework. The model leverages State Space Models (SSMs) — a massive paradigm shift from traditional CNN architectures. By processing the entire tokenized image globally, it achieves superior haze profile estimation while operating at a fraction of the computational cost of Transformers.",
    },
    {
        id: 1,
        top: '28%', left: '3%',
        tail: 'right',
        title: '🔄 End-to-End Architecture Flow',
        text: "This diagram maps the forward pass topology. A hazy input is first tokenized by the Patch Embedding layer, translating pixels into high-dimensional vectors. These vectors are ingested by the sequential Bi-VMamba blocks which act as feature extractors. Finally, the extracted parameters are fed into the physics reconstruction layer to synthesize the pristine output.",
    },
    {
        id: 2,
        top: '28%', right: '3%',
        tail: 'left',
        title: '🧠 Triton Mamba Engine (The Core)',
        text: "Unlike standard Transformers, the Mamba SSM architecture runs in linear O(N) time. The workstation upgrade uses highly-optimized Triton hardware kernels for brutal speed. To prevent global sequences from blurring fine local textures, it utilizes Shifted Window Partitioning (similar to Swin), running bi-directional sweeps inside alternating windows to capture sharp localized optical depth.",
    },
    {
        id: 3,
        top: '51%', left: '3%',
        tail: 'right',
        title: '⚗️ Multi-Scale Physics Fusion',
        text: "Before passing to the final equation, the Mamba features are unpacked into 5 distinct spatial scales (from 1/16th resolution up to full 1:1). This multi-scale fusion forces the model to synthesize both macro-level global atmospheric density gradients and micro-level transmission edges into a unified K-map, clearing out haze mathematically without hallucinating pixels.",
    },
    {
        id: 4,
        top: '51%', right: '3%',
        tail: 'left',
        title: '📐 Operational Specifications',
        text: "This is the Workstation-Grade deployment. Freed from the constraints of 4GB edge-device VRAM, this architecture leverages the official `mamba_ssm` GPU ops to saturate modern high-end PCI-E bandwidth. The 'd=16' internal computing state dimension acts as its continuous selective memory during the sequence scan.",
    },
    {
        id: 5,
        top: '74%', left: '3%',
        tail: 'right',
        title: '🎯 Tri-Objective Loss Optimization',
        text: "The network optimizes against three independent mathematical judges during training. L1 dictates strict pixel-to-pixel color fidelity. SSIM dictates that structural contours and edge boundaries match human perception. The Contrastive Loss utilizes a frozen ConvNeXt-Tiny model to act as a semantic critic, ensuring reconstructed features look functionally realistic.",
    },
    {
        id: 6,
        top: '74%', right: '3%',
        tail: 'left',
        title: '📈 Performance Telemetry',
        text: "These are automated validation metrics processed after every training epoch. PSNR (Peak Signal-to-Noise Ratio) uses logarithmic decibel scaling where >25dB represents state-of-the-art physical recovery. SSIM validates structural integrity from 0 to 1. The interactive visualization dynamically plots the rapid convergence rate during the initial warm-up phase.",
    },
    {
        id: 7,
        bottom: '2%', left: '34%',
        tail: 'bottom',
        title: '🗄️ Aggregated Datasets',
        text: "Instead of training on a single environment, the model ingests ~21,400 raw hazy-to-clear image pairs from 6 distinct academic benchmarks. By exposing the network to everything from thin remote-sensing clouds (RS-Haze) to dense non-homogeneous smog (NH-HAZE), the global Mamba sequence is forced to learn universal physical scattering laws rather than memorizing dataset biases.",
    },
];

const TAIL_STYLES = {
    'right': {
        right: -10, top: '50%', transform: 'translateY(-50%)',
        borderTop: '8px solid transparent', borderBottom: '8px solid transparent',
        borderLeft: '10px solid rgba(240,240,240,0.9)',
    },
    'left': {
        left: -10, top: '50%', transform: 'translateY(-50%)',
        borderTop: '8px solid transparent', borderBottom: '8px solid transparent',
        borderRight: '10px solid rgba(240,240,240,0.9)',
    },
    'bottom': {
        bottom: -10, left: '50%', transform: 'translateX(-50%)',
        borderLeft: '8px solid transparent', borderRight: '8px solid transparent',
        borderTop: '10px solid rgba(240,240,240,0.9)',
    },
    'bottom-right': {
        bottom: -10, right: 16,
        borderLeft: '8px solid transparent', borderRight: '8px solid transparent',
        borderTop: '10px solid rgba(240,240,240,0.9)',
    },
    'top': {
        top: -10, left: '50%', transform: 'translateX(-50%)',
        borderLeft: '8px solid transparent', borderRight: '8px solid transparent',
        borderBottom: '10px solid rgba(240,240,240,0.9)',
    },
};

const ExplainOverlay = ({ onClose }) => {
    const [visible, setVisible] = useState([]);
    useEffect(() => {
        BUBBLES.forEach((b, i) => {
            setTimeout(() => setVisible(v => [...v, b.id]), i * 90);
        });
    }, []);

    return (
        <div
            style={{
                position: 'fixed', inset: 0, zIndex: 100,
                background: 'rgba(0,0,0,0.72)',
                backdropFilter: 'blur(3px)',
            }}
            onClick={onClose}
        >
            {/* Close button */}
            <button
                onClick={onClose}
                style={{
                    position: 'fixed', top: 16, right: 16, zIndex: 102,
                    background: 'rgba(8,8,8,0.9)',
                    border: '1.5px dashed rgba(255,255,255,0.5)',
                    borderRadius: 8,
                    color: '#f0f0f0',
                    padding: '6px 14px',
                    fontSize: 11,
                    fontFamily: 'var(--font-mono)',
                    letterSpacing: '0.1em',
                    cursor: 'pointer',
                    backdropFilter: 'blur(16px)',
                }}
            >
                ✕ CLOSE
            </button>

            {/* Instruction hint */}
            <div style={{
                position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
                fontSize: 9, fontFamily: 'var(--font-mono)', letterSpacing: '0.2em',
                textTransform: 'uppercase', color: 'rgba(255,255,255,0.25)',
                pointerEvents: 'none',
            }}>
                Click anywhere to close
            </div>

            {/* Bubbles */}
            {BUBBLES.map((b) => {
                const isVisible = visible.includes(b.id);
                const posStyle = {};
                if (b.top) posStyle.top = b.top;
                if (b.bottom) posStyle.bottom = b.bottom;
                if (b.left) posStyle.left = b.left;
                if (b.right) posStyle.right = b.right;

                return (
                    <div
                        key={b.id}
                        onClick={e => e.stopPropagation()}
                        style={{
                            position: 'fixed',
                            ...posStyle,
                            maxWidth: 320,
                            background: 'rgba(5,5,5,0.88)',
                            backdropFilter: 'blur(22px)',
                            WebkitBackdropFilter: 'blur(22px)',
                            border: '1.5px dashed rgba(255,255,255,0.52)',
                            borderRadius: 12,
                            padding: '14px 16px',
                            zIndex: 101,
                            opacity: isVisible ? 1 : 0,
                            transform: isVisible ? 'scale(1)' : 'scale(0.72)',
                            transition: 'opacity 0.28s ease, transform 0.28s cubic-bezier(0.34,1.48,0.64,1)',
                        }}
                    >
                        {/* Comic tail */}
                        <div style={{
                            position: 'absolute',
                            width: 0, height: 0,
                            ...(TAIL_STYLES[b.tail] || {}),
                        }} />

                        <div style={{
                            fontSize: 11, fontWeight: 700, color: '#f0f0f0',
                            marginBottom: 7, lineHeight: 1.3,
                        }}>
                            {b.title}
                        </div>
                        <div style={{
                            fontSize: 10.5, color: 'rgba(200,200,200,0.75)',
                            lineHeight: 1.65,
                        }}>
                            {b.text}
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

// ─── Overview Tab ─────────────────────────────────────────────────────────────
const OverviewTab = () => {
    const [explainMode, setExplainMode] = useState(false);

    return (
        <div className="overview-layout fade-in">
            {explainMode && <ExplainOverlay onClose={() => setExplainMode(false)} />}

            {/* Header */}
            <div className="overview-header">
                <div>
                    <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                        <span className="overview-version-badge">v2.0.0 · VISIONMAMBATRAININGREADY</span>
                        <span className="overview-version-badge" style={{ color: 'var(--accent-green)', borderColor: 'rgba(16,185,129,0.25)' }}>ACTIVE</span>
                    </div>
                    <h1 className="overview-title">
                        Vision Mamba<br />Neural Dehazing System
                    </h1>
                    <p className="overview-sub">
                        A State Space Model (SSM) architecture for single-image haze removal.
                        Combines bi-directional Mamba scanning with AOD physics reconstruction,
                        achieving global context understanding at linear O(N) complexity.
                    </p>
                </div>
                <div style={{ flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'flex-end' }}>
                    {/* Arch card */}
                    <div style={{ background: 'var(--bg-raised)', border: '1px solid var(--border)', borderRadius: 8, padding: '16px 20px', textAlign: 'right' }}>
                        <div style={{ fontSize: 9, letterSpacing: '0.15em', textTransform: 'uppercase', color: 'var(--text-dim)', marginBottom: 4 }}>Architecture</div>
                        <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>End-to-End VMamba</div>
                        <div style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>4× S4-Blocks · d_state=16</div>
                    </div>
                    {/* Explanation trigger */}
                    <button
                        onClick={() => setExplainMode(true)}
                        style={{
                            background: 'var(--bg-raised)', border: '1px solid var(--border)',
                            borderRadius: 8, padding: '12px 20px', textAlign: 'right',
                            cursor: 'pointer', width: '100%',
                            transition: 'border-color 0.18s, background 0.18s',
                        }}
                        onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--border-hi)'; e.currentTarget.style.background = 'var(--bg-subtle)'; }}
                        onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.background = 'var(--bg-raised)'; }}
                    >
                        <div style={{ fontSize: 9, letterSpacing: '0.15em', textTransform: 'uppercase', color: 'var(--text-dim)', marginBottom: 4 }}>Guide</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 6 }}>
                            <span>💬</span> Explanations
                        </div>
                        <div style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>Click to annotate this page</div>
                    </button>
                </div>
            </div>

            {/* Architecture Diagram */}
            <div className="overview-section">
                <div className="overview-section-title">
                    <GitBranch size={12} />
                    Neural Architecture Pipeline
                </div>
                <ArchDiagram />
            </div>

            {/* Core Specifications */}
            <div className="overview-section">
                <div className="overview-section-title">
                    <Database size={12} />
                    Model Specifications
                </div>
                <div className="spec-grid">
                    {[
                        { label: 'Parameters', value: '~12M', unit: 'Trainable Weights' },
                        { label: 'Complexity', value: 'O(N)', unit: 'Linear scaling vs. Transformers' },
                        { label: 'Input Size', value: '256²', unit: 'px — processing resolution' },
                        { label: 'Embed Dim', value: '64', unit: 'd_model channels' },
                        { label: 'SSM State', value: 'd=16', unit: 'Hidden state dimension' },
                        { label: 'Num Blocks', value: '4', unit: 'Bi-VMamba layers' },
                        { label: 'Batch Size', value: '14', unit: 'GPU: RTX 3050 4GB' },
                        { label: 'Grad Clip', value: '0.5', unit: 'max_norm — SSM stability' },
                    ].map(s => (
                        <div className="spec-card" key={s.label}>
                            <div className="spec-card-label">{s.label}</div>
                            <div className="spec-card-value">{s.value}</div>
                            <div className="spec-card-unit">{s.unit}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Loss Functions */}
            <div className="overview-section">
                <div className="overview-section-title">
                    <FlaskConical size={12} />
                    Training Objectives
                </div>
                <div className="loss-grid">
                    <div className="loss-card">
                        <div className="loss-card-title" style={{ color: '#3b82f6' }}>L1 Pixel Loss</div>
                        <div className="loss-card-weight" style={{ color: '#3b82f6' }}>1.0×</div>
                        <div className="loss-card-desc">
                            Mean Absolute Error between predicted J̃ and ground-truth J.
                            Enforces pixel-accurate color and luminance reconstruction.
                        </div>
                    </div>
                    <div className="loss-card">
                        <div className="loss-card-title" style={{ color: '#8b5cf6' }}>SSIM Loss</div>
                        <div className="loss-card-weight" style={{ color: '#8b5cf6' }}>0.5×</div>
                        <div className="loss-card-desc">
                            Structural Similarity Index ensures the model preserves perceived
                            texture contrast and structural edges under heavy haze.
                        </div>
                    </div>
                    <div className="loss-card">
                        <div className="loss-card-title" style={{ color: '#10b981' }}>Contrastive (ConvNeXt-T)</div>
                        <div className="loss-card-weight" style={{ color: '#10b981' }}>0.1×</div>
                        <div className="loss-card-desc">
                            Frozen ConvNeXt-Tiny feature extractor regularizes semantic content.
                            Prevents over-smoothed outputs and enforces high-level scene fidelity.
                        </div>
                    </div>
                </div>
            </div>

            {/* Evaluation Metrics — table + chart side by side */}
            <div className="overview-section">
                <div className="overview-section-title">
                    <BarChart3 size={12} />
                    Evaluation Metrics &amp; Warmup Performance (Epochs 1–5)
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 420px', gap: 16, alignItems: 'start' }}>
                    {/* Table */}
                    <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 10, overflow: 'hidden' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr style={{ background: 'var(--bg-raised)' }}>
                                    {['Epoch', 'Learning Rate', 'Train Loss', 'Val PSNR (dB)', 'Val SSIM', 'Phase'].map(h => (
                                        <th key={h} style={{
                                            padding: '10px 14px', textAlign: 'left',
                                            fontSize: 9, fontWeight: 700, letterSpacing: '0.15em',
                                            textTransform: 'uppercase', color: 'var(--text-dim)',
                                            borderBottom: '1px solid var(--border)'
                                        }}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {EPOCH_DATA.map((r, i) => (
                                    <tr key={r.ep} style={{ borderBottom: '1px solid var(--border)', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)' }}>
                                        <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-primary)', fontWeight: 600 }}>{r.ep}</td>
                                        <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-blue)' }}>{r.lr}</td>
                                        <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-mid)' }}>{r.loss}</td>
                                        <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-green)', fontWeight: 600 }}>{r.psnr}</td>
                                        <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-green)' }}>{r.ssim.toFixed(4)}</td>
                                        <td style={{ padding: '10px 14px', fontSize: 10, color: 'var(--text-dim)' }}>{r.phase}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    {/* Chart */}
                    <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 10, padding: '16px 12px' }}>
                        <div style={{
                            fontSize: 9, letterSpacing: '0.15em', textTransform: 'uppercase',
                            color: 'var(--text-dim)', marginBottom: 10, fontWeight: 700
                        }}>
                            PSNR &amp; SSIM vs Epoch — hover to inspect
                        </div>
                        <MetricsChart />
                    </div>
                </div>
            </div>

            {/* Dataset */}
            <div className="overview-section">
                <div className="overview-section-title">
                    <Database size={12} />
                    Dataset Composition
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12 }}>
                    {[
                        { name: 'Thesis Composite', size: '~15,800 pairs', detail: 'NH-HAZE · I-HAZE · O-HAZE · Dense-Haze · SOTS · BeDDE', color: '#3b82f6' },
                        { name: 'Haze1k', size: '~3,600 pairs', detail: 'Thin · Moderate · Thick variants', color: '#8b5cf6' },
                        { name: 'RS-Haze', size: '~2,000 pairs', detail: 'Remote sensing dehazing benchmark', color: '#f59e0b' },
                    ].map(d => (
                        <div key={d.name} style={{ background: 'var(--bg-raised)', border: '1px solid var(--border)', borderLeft: `2px solid ${d.color}`, borderRadius: 8, padding: '14px 16px' }}>
                            <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>{d.name}</div>
                            <div style={{ fontSize: 18, fontWeight: 700, color: d.color, fontFamily: 'var(--font-mono)', marginBottom: 4 }}>{d.size}</div>
                            <div style={{ fontSize: 10, color: 'var(--text-dim)', lineHeight: 1.5 }}>{d.detail}</div>
                        </div>
                    ))}
                    <div style={{ background: 'var(--bg-raised)', border: '1px solid var(--border)', borderLeft: '2px solid var(--accent-green)', borderRadius: 8, padding: '14px 16px' }}>
                        <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>Total Pipeline</div>
                        <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-green)', fontFamily: 'var(--font-mono)', marginBottom: 4 }}>~21,400 pairs</div>
                        <div style={{ fontSize: 10, color: 'var(--text-dim)', lineHeight: 1.5 }}>80% Train · 10% Val · 10% Test</div>
                    </div>
                </div>
            </div>

            {/* Footer */}
            <div style={{ borderTop: '1px solid var(--border)', paddingTop: 20, marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--text-dimmer)' }}>
                    Vision Mamba Neural Dehazing · Capstone 2026
                </span>
                <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--text-dimmer)' }}>
                    BRANCH: VISIONMAMBATRAININGREADY
                </span>
            </div>
        </div>
    );
};

// ─── Root App ─────────────────────────────────────────────────────────────────
export default function App() {
    const [activeTab, setActiveTab] = useState('system');
    const [status, setStatus] = useState({ model_loaded: false, checkpoint: '…', device: '…' });

    useEffect(() => {
        const poll = async () => {
            try {
                const res = await axios.get('/status');
                setStatus(res.data);
            } catch {
                setStatus(s => ({ ...s, model_loaded: false, checkpoint: 'Offline', device: 'N/A' }));
            }
        };
        poll();
        const id = setInterval(poll, 5000);
        return () => clearInterval(id);
    }, []);

    return (
        <div className="app-shell">

            {/* ─ Navbar ─ */}
            <nav className="navbar">
                <div className="navbar-brand">
                    <div className="brand-mark">
                        <ScanLine size={14} color="#fff" />
                    </div>
                    <div>
                        <div className="brand-title">Neural Dehazing System</div>
                        <div className="brand-sub">VIM-DHZ · 2026 Capstone</div>
                    </div>
                </div>

                <div className="tab-bar">
                    <button
                        id="tab-system"
                        className={`tab-btn ${activeTab === 'system' ? 'active' : ''}`}
                        onClick={() => setActiveTab('system')}
                    >
                        <MonitorDot size={12} />
                        System
                    </button>
                    <button
                        id="tab-overview"
                        className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
                        onClick={() => setActiveTab('overview')}
                    >
                        <BookOpen size={12} />
                        Overview
                    </button>
                </div>

                <div className="sys-status">
                    <div className="status-pill">
                        <span className={`status-dot ${status.model_loaded ? '' : 'offline'}`} />
                        <span style={{ color: status.model_loaded ? 'var(--accent-green)' : '#f87171' }}>
                            {status.model_loaded ? 'Operational' : 'Offline'}
                        </span>
                    </div>
                </div>
            </nav>

            {/* ─ Tab Content ─ */}
            <div className="tab-content">
                {activeTab === 'system' && <SystemTab status={status} />}
                {activeTab === 'overview' && <OverviewTab />}
            </div>
        </div>
    );
}
