import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
    Upload, Sparkles, Loader2, RefreshCcw,
    ChevronLeft, ChevronRight, Activity, Cpu,
    Layers, Info, Image as ImageIcon, CheckCircle, AlertTriangle
} from 'lucide-react';
import axios from 'axios';

// Comparison Slider — both images rendered at exactly the same pixel dimensions
const ComparisonSlider = ({ before, after, width, height }) => {
    const [position, setPosition] = useState(50);
    const containerRef = useRef(null);
    const dragging = useRef(false);

    const handleMove = useCallback((e) => {
        if (!dragging.current || !containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX ?? e.touches?.[0]?.clientX ?? rect.left;
        const pct = Math.min(100, Math.max(0, ((x - rect.left) / rect.width) * 100));
        setPosition(pct);
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

    const style = { width: '100%', maxWidth: width || '100%', aspectRatio: width && height ? `${width}/${height}` : '16/9' };

    return (
        <div
            ref={containerRef}
            className="relative overflow-hidden rounded-2xl border border-white/10 shadow-2xl bg-black cursor-ew-resize select-none group mx-auto"
            style={style}
            onMouseDown={startDrag}
            onTouchStart={startDrag}
        >
            {/* After (dehazed) — full background */}
            <img
                src={after}
                alt="Dehazed"
                className="absolute inset-0 w-full h-full object-contain"
                draggable={false}
            />

            {/* Before (original) — clipped to left portion */}
            <div
                className="absolute inset-0 overflow-hidden"
                style={{ width: `${position}%` }}
            >
                <img
                    src={before}
                    alt="Original"
                    className="absolute inset-0 h-full object-contain"
                    style={{ width: `${100 / (position / 100)}%`, maxWidth: 'none' }}
                    draggable={false}
                />
            </div>

            {/* Divider line */}
            <div
                className="absolute top-0 bottom-0 w-0.5 bg-white z-20 pointer-events-none"
                style={{ left: `${position}%` }}
            >
                <div
                    className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 bg-white rounded-full flex items-center justify-center shadow-xl text-slate-900 pointer-events-auto cursor-ew-resize"
                    onMouseDown={startDrag}
                    onTouchStart={startDrag}
                >
                    <ChevronLeft size={14} />
                    <ChevronRight size={14} />
                </div>
            </div>

            {/* Labels */}
            <div className="absolute bottom-4 left-4 px-3 py-1.5 bg-black/70 backdrop-blur-md rounded-lg text-[10px] font-bold uppercase tracking-wider z-30 pointer-events-none border border-white/10">
                Original
            </div>
            <div className="absolute bottom-4 right-4 px-3 py-1.5 bg-indigo-600/80 backdrop-blur-md rounded-lg text-[10px] font-bold uppercase tracking-wider z-30 pointer-events-none border border-indigo-400/30">
                Dehazed
            </div>
        </div>
    );
};

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [imgDims, setImgDims] = useState({ w: 0, h: 0 });
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [status, setStatus] = useState({ model_loaded: false, checkpoint: '…', device: '…' });
    const fileInputRef = useRef(null);

    // Poll backend status
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

    const handleFileChange = (e) => {
        const f = e.target.files[0];
        if (!f) return;
        const url = URL.createObjectURL(f);
        setFile(f);
        setPreview(url);
        setResult(null);
        setError(null);
        // Read natural image dimensions
        const img = new Image();
        img.onload = () => setImgDims({ w: img.naturalWidth, h: img.naturalHeight });
        img.src = url;
    };

    const handleReset = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
        setImgDims({ w: 0, h: 0 });
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleDehaze = async () => {
        if (!file || loading) return;
        setLoading(true);
        setResult(null);
        setError(null);

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await axios.post('/predict', formData, { responseType: 'blob' });
            const url = URL.createObjectURL(response.data);
            setResult(url);
        } catch (err) {
            const msg = err.response?.data?.detail || err.message || 'Unknown error';
            setError(msg);
        } finally {
            setLoading(false);
        }
    };

    // Drag-and-drop support
    const handleDrop = (e) => {
        e.preventDefault();
        const f = e.dataTransfer.files[0];
        if (!f) return;
        // Simulate file input
        const dt = new DataTransfer();
        dt.items.add(f);
        if (fileInputRef.current) {
            fileInputRef.current.files = dt.files;
            handleFileChange({ target: fileInputRef.current });
        }
    };

    return (
        <div className="min-h-screen px-4 py-10 md:px-10 max-w-[1400px] mx-auto">
            {/* ── Header ─────────────────────────────────────────────── */}
            <header className="text-center mb-14">
                <div className="inline-flex items-center gap-2 px-4 py-1.5 mb-5 rounded-full bg-indigo-500/20 border border-indigo-500/30 text-indigo-400 text-[10px] font-bold uppercase tracking-[0.2em]">
                    <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
                    Capstone 2026 · RTX 3050 · Physics-Informed ML
                </div>
                <h1 className="text-5xl md:text-6xl font-extrabold tracking-tighter mb-3 bg-gradient-to-b from-white to-slate-400 bg-clip-text text-transparent leading-none">
                    AtmosDehaze <span className="text-indigo-400">AI</span>
                </h1>
                <p className="text-slate-400 text-base max-w-xl mx-auto font-light leading-relaxed mt-4">
                    World-first Single Image Dehazing via **Vision Mamba (Vim)**.
                    Leveraging State Space Models (SSM) and AOD physics for stable, global-context image restoration.
                </p>
            </header>

            {/* ── Main Grid ──────────────────────────────────────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-8 items-start">

                {/* ── Sidebar ─────────────────── */}
                <aside className="flex flex-col gap-5">

                    {/* Control Hub */}
                    <div className="glass-card p-6">
                        <h3 className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest text-slate-400 mb-5">
                            <Layers size={16} className="text-indigo-400" /> Control Hub
                        </h3>

                        {!preview ? (
                            <div
                                className="border-2 border-dashed border-slate-700 rounded-xl p-10 flex flex-col items-center gap-4 cursor-pointer transition-all hover:border-indigo-500/60 hover:bg-indigo-500/5 group"
                                onClick={() => fileInputRef.current?.click()}
                                onDrop={handleDrop}
                                onDragOver={e => e.preventDefault()}
                            >
                                <div className="w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center text-slate-500 group-hover:bg-indigo-500/20 group-hover:text-indigo-400 transition-colors">
                                    <Upload size={26} />
                                </div>
                                <div className="text-center">
                                    <p className="font-semibold text-slate-200 text-sm">Drop or click to upload</p>
                                    <p className="text-[10px] text-slate-500 mt-1 uppercase tracking-widest">PNG · JPG · WEBP · AVIF · BMP</p>
                                </div>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    className="hidden"
                                    onChange={handleFileChange}
                                    accept="image/*,.avif"
                                />
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {/* Thumbnail */}
                                <div className="relative rounded-xl overflow-hidden border border-white/5 bg-black">
                                    <img src={preview} alt="Preview" className="w-full object-contain max-h-44" />
                                    <div className="absolute bottom-2 left-2 bg-black/60 backdrop-blur-sm px-2 py-0.5 rounded text-[9px] font-mono text-slate-400">
                                        {imgDims.w} × {imgDims.h}px · {(file?.size / 1024).toFixed(0)} KB
                                    </div>
                                </div>

                                <button
                                    className="btn-primary"
                                    onClick={handleDehaze}
                                    disabled={loading || !status.model_loaded}
                                >
                                    {loading
                                        ? <><Loader2 size={18} className="animate-spin" /><span>Processing…</span></>
                                        : <><Sparkles size={18} /><span>Dehaze Image</span></>
                                    }
                                </button>

                                <button
                                    onClick={handleReset}
                                    className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-slate-800/50 border border-white/5 text-slate-400 text-sm hover:bg-slate-800 hover:text-white transition-all"
                                >
                                    <RefreshCcw size={14} /> Reset
                                </button>

                                {error && (
                                    <div className="flex items-start gap-2 p-3 bg-red-900/20 border border-red-800/30 rounded-xl text-red-400 text-xs">
                                        <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                                        <span>{error}</span>
                                    </div>
                                )}

                                {result && !loading && (
                                    <div className="flex items-center gap-2 p-3 bg-emerald-900/20 border border-emerald-800/30 rounded-xl text-emerald-400 text-xs">
                                        <CheckCircle size={14} className="shrink-0" />
                                        <span>Dehazing complete · Drag slider to compare</span>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Status Panel */}
                    <div className="glass-card p-6">
                        <h3 className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest text-slate-400 mb-5">
                            <Activity size={16} className="text-emerald-400" /> System Status
                        </h3>
                        <div className="grid grid-cols-2 gap-3 mb-3">
                            <div className="bg-slate-900/50 border border-white/5 p-3 rounded-xl">
                                <p className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">GPU</p>
                                <div className="flex items-center gap-1.5 text-xs font-bold text-amber-400">
                                    <Cpu size={12} /> RTX 3050
                                </div>
                            </div>
                            <div className="bg-slate-900/50 border border-white/5 p-3 rounded-xl">
                                <p className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">Backend</p>
                                <div className={`flex items-center gap-1.5 text-xs font-bold ${status.model_loaded ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    <span className={`w-1.5 h-1.5 rounded-full ${status.model_loaded ? 'bg-emerald-500 shadow-[0_0_6px_#10b981]' : 'bg-rose-500'} animate-pulse`} />
                                    {status.model_loaded ? 'Ready' : 'Loading…'}
                                </div>
                            </div>
                        </div>
                        <div className="bg-slate-900/50 border border-white/5 p-3 rounded-xl">
                            <p className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">Model Weights</p>
                            <p className="text-[11px] font-mono text-indigo-300 truncate">{status.checkpoint}</p>
                        </div>
                    </div>

                    {/* Info */}
                    <div className="p-4 bg-indigo-500/5 border border-indigo-500/10 rounded-xl flex gap-3 text-[11px] text-indigo-400/70 leading-relaxed">
                        <Info size={14} className="shrink-0 mt-0.5" />
                        <p>Output is upsampled back to your exact input resolution via Lanczos interpolation post-inference.</p>
                    </div>
                </aside>

                {/* ── Viewport ────────────────── */}
                <main>
                    {preview ? (
                        result ? (
                            <ComparisonSlider
                                before={preview}
                                after={result}
                                width={imgDims.w}
                                height={imgDims.h}
                            />
                        ) : (
                            <div className="relative w-full rounded-2xl overflow-hidden border border-white/10 bg-slate-950 flex items-center justify-center shadow-2xl"
                                style={{ minHeight: '400px' }}
                            >
                                <img
                                    src={preview}
                                    alt="Preview"
                                    className="w-full h-full object-contain opacity-25 blur-[3px] grayscale absolute inset-0"
                                />
                                <div className="relative z-10 flex flex-col items-center gap-6 text-center px-8 py-12">
                                    {loading ? (
                                        <>
                                            <div className="relative">
                                                <Loader2 size={56} className="animate-spin text-indigo-500" />
                                                <div className="absolute inset-0 blur-md opacity-50 text-indigo-500">
                                                    <Loader2 size={56} className="animate-spin" />
                                                </div>
                                            </div>
                                            <div>
                                                <p className="text-lg font-bold text-white tracking-tight">Solving Scattering Matrix</p>
                                                <p className="text-[10px] text-slate-500 uppercase tracking-[0.35em] mt-1">Processing on RTX 3050 GPU…</p>
                                            </div>
                                        </>
                                    ) : (
                                        <div className="space-y-3">
                                            <div className="w-12 h-0.5 bg-indigo-500/40 mx-auto rounded-full" />
                                            <p className="text-xl font-light text-slate-300">Image Loaded</p>
                                            <p className="text-[10px] text-slate-500 uppercase tracking-[0.4em]">Click "Dehaze Image" to process →</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )
                    ) : (
                        <div className="w-full rounded-2xl border-2 border-dashed border-slate-800/60 flex items-center justify-center"
                            style={{ minHeight: '480px' }}
                            onClick={() => fileInputRef.current?.click()}
                            onDrop={handleDrop}
                            onDragOver={e => e.preventDefault()}
                        >
                            <div className="text-center text-slate-700 cursor-pointer group py-8">
                                <ImageIcon size={56} className="mx-auto mb-4 opacity-20 group-hover:opacity-40 transition-opacity" />
                                <p className="text-sm uppercase tracking-[0.25em] opacity-40 group-hover:opacity-70 transition-opacity">
                                    Upload or drop a hazy image
                                </p>
                            </div>
                        </div>
                    )}
                </main>
            </div>

            {/* ── Footer ──────────────────────────────────────────────── */}
            <footer className="mt-20 text-center border-t border-slate-900 pt-8">
                <p className="text-slate-700 text-[10px] font-bold uppercase tracking-[0.4em]">
                    AtmosDehaze AI · Capstone 2026 · Physics-Informed Deep Learning
                </p>
            </footer>
        </div>
    );
}
