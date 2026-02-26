import React, { useState } from 'react';
import { Upload, Image as ImageIcon, Sparkles, Loader2, ArrowRight } from 'lucide-react';
import axios from 'axios';

function App() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreview(URL.createObjectURL(selectedFile));
            setResult(null);
        }
    };

    const handleDehaze = async () => {
        if (!file) return;
        setLoading(true);
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await axios.post('/predict', formData, {
                responseType: 'blob'
            });
            const url = URL.createObjectURL(response.data);
            setResult(url);
        } catch (error) {
            console.error('Error dehazing:', error);
            alert('Failed to dehaze image. Check if backend is running.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container">
            <header className="header">
                <h1>AtmosDehaze AI</h1>
                <p style={{ color: '#94a3b8' }}>Research-grade atmospheric scattering restoration</p>
            </header>

            <main>
                {!preview ? (
                    <div className="upload-card">
                        <div style={{ marginBottom: '2rem' }}>
                            <Upload size={64} color="#6366f1" style={{ margin: '0 auto' }} />
                        </div>
                        <h2 style={{ marginBottom: '1rem' }}>Upload Hazy Image</h2>
                        <p style={{ color: '#94a3b8', marginBottom: '2rem' }}>
                            Select a single image for physics-guided dehazing
                        </p>
                        <input
                            type="file"
                            id="fileInput"
                            style={{ display: 'none' }}
                            onChange={handleFileChange}
                            accept="image/*"
                        />
                        <button className="btn" onClick={() => document.getElementById('fileInput').click()}>
                            Choose Image
                        </button>
                    </div>
                ) : (
                    <div>
                        <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginBottom: '2rem' }}>
                            <button className="btn" onClick={() => setPreview(null)} style={{ background: 'transparent', border: '1px solid var(--border)' }}>
                                Change Image
                            </button>
                            <button className="btn" onClick={handleDehaze} disabled={loading}>
                                {loading ? <Loader2 className="animate-spin" /> : <><Sparkles size={18} style={{ marginRight: '8px' }} /> Dehaze Image</>}
                            </button>
                        </div>

                        <div className="comparison-grid">
                            <div className="image-box">
                                <h3>Original Hazy</h3>
                                <img src={preview} alt="Hazy" />
                            </div>
                            <div className="image-box">
                                <h3>Dehazed Result</h3>
                                {result ? (
                                    <img src={result} alt="Dehazed" />
                                ) : (
                                    <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', background: 'rgba(0,0,0,0.2)', borderRadius: '0.5rem' }}>
                                        {loading ? 'Processing...' : 'Click "Dehaze" to see results'}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </main>

            <footer style={{ marginTop: '5rem', textAlign: 'center', color: '#475569', fontSize: '0.875rem' }}>
                Built with PyTorch & Atmospheric Scattering Physics (RTX 3050 Optimized)
            </footer>
        </div>
    );
}

export default App;
