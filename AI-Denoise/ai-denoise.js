"use strict";

///////////////////////////////////////////////////////////////
//                                                           //
//  AI DENOISER PLUGIN FOR FM-DX-WEBSERVER (V1.1)            //
//                                                           //
//  by Highpoint                last update: 2026-03-14      //
//                                                           //
//  https://github.com/Highpoint2000/AI-Denoise              //
//                                                           //
//  Primary:   DTLN (2 models) via ONNX Runtime Web          //
//  Secondary: RNNoise WASM by Shiguredo (MIT)               //
//  VAD:       Silero VAD v4 (ONNX) – RNNoise only           //
//                                                           //
///////////////////////////////////////////////////////////////

(() => {
    const pluginVersion     = '1.1';
    const pluginName        = 'AI-Denoise';
    const pluginHomepageUrl = 'https://github.com/Highpoint2000/AI-Denoise/releases';
    const pluginUpdateUrl   = 'https://raw.githubusercontent.com/Highpoint2000/AI-Denoise/refs/heads/main/AI-Denoise/ai-denoise.js';
    const CHECK_FOR_UPDATES = false;

    const PLUGIN_BASE   = window.location.origin + '/plugins/AI-Denoise/';
    const DTLN_BASE_URL = PLUGIN_BASE + 'DTLN/';
    const RNNOISE_BASE  = PLUGIN_BASE + 'rnnoise/';
    const DTLN_MODEL_1  = DTLN_BASE_URL + 'model_1.onnx';
    const DTLN_MODEL_2  = DTLN_BASE_URL + 'model_2.onnx';
    const SILERO_URL    = DTLN_BASE_URL + 'silero_vad.onnx';
    const ORT_CDN       = DTLN_BASE_URL + 'ort.min.js';
    const CDN_ESM       = RNNOISE_BASE  + 'rnnoise.js';

    // ── DTLN STFT parameters (must match the model) ──────────────────
    // Model 1 input:  magnitude spectrum  [1, 1, 257]
    // Model 2 input:  cleaned magnitude   [1, 1, 512]  (zero-padded from 257)
    // DTLN standard:  block_len=512, block_shift=128, sample rate=16 kHz
    const DTLN_FFT_SIZE    = 512;
    const DTLN_HOP         = 128;
    const DTLN_BINS        = 257;   // FFT_SIZE / 2 + 1
    const DTLN_SAMPLE_RATE = 16000;

    // Phase accumulation: number of past frames to average over
    // Higher = smoother phase, slightly more latency
    // 4 frames = 4 × 8ms = ~32ms lookahead at 16kHz
    const PHASE_ACC_FRAMES = 4;

    // Silero VAD v4 state size
    const SILERO_STATE_SIZE = 64;

    let currentMode = localStorage.getItem('ai-denoise-mode') || 'dtln';

    let isEnabled  = false;
    let sourceNode = null;
    let nativeCtx  = null;
    let ctx48      = null;
    let nodes      = {};
    let nodes48    = {};

    let rnnoiseObj    = null;
    let denoiseState1 = null;
    let FRAME_SIZE    = 480;

    let ortLoaded     = false;
    let dtlnSession1  = null;
    let dtlnSession2  = null;
    let sileroSession = null;
    let dtlnReady     = false;
    let sileroH       = null;
    let sileroC       = null;

    // Silero key names (discovered at runtime)
    let sileroKeyInput = 'input';
    let sileroKeySr    = 'sr';
    let sileroKeyH     = 'h';
    let sileroKeyC     = 'c';
    let sileroKeyHn    = 'hn';
    let sileroKeyCn    = 'cn';

    // DTLN I/O name arrays (discovered at runtime)
    let dtlnInputNames1  = null;
    let dtlnOutputNames1 = null;
    let dtlnInputNames2  = null;
    let dtlnOutputNames2 = null;

    // Running LSTM states (reset each new session)
    let dtlnState1 = null;
    let dtlnState2 = null;

    // ── Phase accumulation buffers ────────────────────────────────────
    // Stores weighted complex sum of recent clean frames for phase smoothing
    // phaseAccRe[k], phaseAccIm[k] = sum of (cleanMag × cos/sin(phase))
    let phaseAccRe  = null;   // Float32Array[DTLN_BINS]
    let phaseAccIm  = null;   // Float32Array[DTLN_BINS]
    let phaseFrames = [];     // ring buffer of recent {re, im, mag} frames

    function resetPhaseAccumulator() {
        phaseAccRe  = new Float32Array(DTLN_BINS);
        phaseAccIm  = new Float32Array(DTLN_BINS);
        phaseFrames = [];
    }

    // ── Settings ─────────────────────────────────────────────────────
    const DEFAULTS = { strength: 0.5, voice: 0.5, gate: 0.5 };
    function lsKey(mode, param) { return `ai-denoise-${mode}-${param}`; }
    function loadSettings(mode) {
        return {
            strength: parseFloat(localStorage.getItem(lsKey(mode, 'strength')) ?? DEFAULTS.strength),
            voice:    parseFloat(localStorage.getItem(lsKey(mode, 'voice'))    ?? DEFAULTS.voice),
            gate:     parseFloat(localStorage.getItem(lsKey(mode, 'gate'))     ?? DEFAULTS.gate),
        };
    }
    function saveSettings(mode, s) {
        localStorage.setItem(lsKey(mode, 'strength'), s.strength);
        localStorage.setItem(lsKey(mode, 'voice'),    s.voice);
        localStorage.setItem(lsKey(mode, 'gate'),     s.gate);
    }

    let S = loadSettings(currentMode);
    let P = {};

    function calcParams() {
        const str    = S.strength, voi = S.voice, gat = S.gate;
        const isDTLN = currentMode === 'dtln';
        P = {
            // DTLN: blend only active in top 20% of slider (0.8–1.0 → 0.0–1.0)
            // RNNoise: VAD-based blend
            blend:        isDTLN ? Math.max(0, (str - 0.8) / 0.2) : 0.3 + str * 0.7,

            // VAD parameters – RNNoise only
            vadThreshold: 0.15 + (1 - str) * 0.25,
            vadFloor:     (1 - str) * 0.30,
            holdFrames:   Math.round(40 + str * 60),
            fadeFrames:   Math.round(30 + str * 80),

            // Gate – RNNoise only
            gateThresh:   0.002 + (1 - gat) * 0.05,
            gateRatio:    gat * 0.95,

            // Output gain:
            // DTLN: Gain slider (0.5×–5.0×)
            // RNNoise: fixed makeup gain
            outputGain:   isDTLN
                ? 0.5 + gat * 4.5
                : (1.0 + str * 0.25) * 1.778,

            // DTLN mask sharpening – active across full slider range
            // str=0.0 → alpha=0.5 (soft/conservative masks)
            // str=0.5 → alpha=1.0 (original model masks)
            // str=1.0 → alpha=3.0 (aggressive suppression)
            maskAlpha:    isDTLN ? 0.5 + str * 2.5 : 1.0,

            // Phase accumulation weight – how strongly the refined phase is used
            // str=0.0 → 0.0 (noisy original phase, no refinement)
            // str=0.5 → 0.5 (equal mix)
            // str=1.0 → 0.9 (heavily refined phase)
            phaseWeight:  isDTLN ? str * 0.9 : 0.0,

            // EQ – DTLN: tone slider warm (left) ↔ bright (right)
            //      RNNoise: original behaviour
            eqPresence:   isDTLN ? (voi * 6.0) - 1.0       : 2.0 + str * 4.0 + voi * 3.0,
            eqClarity:    isDTLN ? (voi * 4.0) - 1.0       : 1.0 + voi * 2.0,
            eqLow:        isDTLN ? 4.0 - voi * 8.0         : (0.5 - voi) * 4.0,
            eqMid:        isDTLN ? 1.0 + (voi - 0.5) * 2.0 : 1.0 + str * 1.5 + voi * 1.5,
            eqHigh:       isDTLN ? (voi * 8.0) - 2.0       : 3.0 + str * 5.0 + voi * 4.0,
            eqAir:        isDTLN ? (voi * 6.0) - 3.0       : 2.0 + str * 3.0 + voi * 4.0,
        };
    }

    // ==========================================
    // Update Check
    // ==========================================
    function checkUpdate() {
        fetch(pluginUpdateUrl + '?t=' + Date.now(), { cache: 'no-store' })
            .then(r => r.ok ? r.text() : null)
            .then(txt => {
                if (!txt) return;
                const match = txt.match(/const\s+pluginVersion\s*=\s*['"]([^'"]+)['"]/);
                if (!match) return;
                const remoteVer = match[1];
                if (remoteVer === pluginVersion) return;
                console.log(`[${pluginName}] Update available: ${pluginVersion} -> ${remoteVer}`);
                const settings = document.getElementById('plugin-settings');
                if (settings && !settings.innerHTML.includes(pluginHomepageUrl))
                    settings.innerHTML += `<br><a href="${pluginHomepageUrl}" target="_blank">[${pluginName}] Update: ${pluginVersion} -> ${remoteVer}</a>`;
                const navIcon =
                    document.querySelector('.wrapper-outer #navigation .sidenav-content .fa-puzzle-piece') ||
                    document.querySelector('.wrapper-outer .sidenav-content') ||
                    document.querySelector('.sidenav-content');
                if (navIcon && !navIcon.querySelector(`.${pluginName}-update-dot`)) {
                    const dot = document.createElement('span');
                    dot.classList.add(`${pluginName}-update-dot`);
                    dot.style.cssText = 'display:block;width:12px;height:12px;border-radius:50%;background-color:#FE0830;margin-left:82px;margin-top:-12px;';
                    navIcon.appendChild(dot);
                }
            })
            .catch(e => console.log(`[${pluginName}] Update check failed:`, e.message));
    }
    if (CHECK_FOR_UPDATES) checkUpdate();

    // ==========================================
    // Load ORT
    // ==========================================
    function loadORT() {
        if (ortLoaded || window.ort) {
            ortLoaded = true;
            ort.env.wasm.wasmPaths  = DTLN_BASE_URL;
            ort.env.wasm.numThreads = 1;
            return Promise.resolve();
        }
        return new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src     = ORT_CDN;
            s.onload  = () => {
                ortLoaded = true;
                ort.env.wasm.wasmPaths  = DTLN_BASE_URL;
                ort.env.wasm.numThreads = 1;
                resolve();
            };
            s.onerror = () => reject(new Error('ORT load failed: ' + ORT_CDN));
            document.head.appendChild(s);
        });
    }

    // ==========================================
    // Load DTLN + Silero VAD
    // ==========================================
    async function loadDTLN() {
        if (dtlnReady) return true;
        try {
            updateToggleBtn();
            await loadORT();

            console.log(`[${pluginName}] Loading DTLN model 1…`);
            dtlnSession1 = await ort.InferenceSession.create(DTLN_MODEL_1, { executionProviders: ['wasm'] });
            console.log(`[${pluginName}] Loading DTLN model 2…`);
            dtlnSession2 = await ort.InferenceSession.create(DTLN_MODEL_2, { executionProviders: ['wasm'] });

            dtlnInputNames1  = Array.from(dtlnSession1.inputNames);
            dtlnOutputNames1 = Array.from(dtlnSession1.outputNames);
            dtlnInputNames2  = Array.from(dtlnSession2.inputNames);
            dtlnOutputNames2 = Array.from(dtlnSession2.outputNames);
            console.log(`[${pluginName}] DTLN m1 in:`, dtlnInputNames1, 'out:', dtlnOutputNames1);
            console.log(`[${pluginName}] DTLN m2 in:`, dtlnInputNames2, 'out:', dtlnOutputNames2);

            await resetDTLNStates(null, null);

            try {
                console.log(`[${pluginName}] Loading Silero VAD…`);
                sileroSession = await ort.InferenceSession.create(SILERO_URL, { executionProviders: ['wasm'] });
                const inNames  = Array.from(sileroSession.inputNames);
                const outNames = Array.from(sileroSession.outputNames);
                console.log(`[${pluginName}] Silero VAD ready. inputs:`, inNames, 'outputs:', outNames);
                sileroKeyInput = inNames[0]  || 'input';
                sileroKeySr    = inNames[1]  || 'sr';
                sileroKeyH     = inNames[2]  || 'h';
                sileroKeyC     = inNames[3]  || 'c';
                sileroKeyHn    = outNames[1] || 'hn';
                sileroKeyCn    = outNames[2] || 'cn';
            } catch (err) {
                console.warn(`[${pluginName}] Silero VAD unavailable – using energy VAD:`, err);
            }
            resetSileroState();
            resetPhaseAccumulator();

            dtlnReady = true;
            console.log(`[${pluginName}] DTLN ready`);
            updateToggleBtn();
            return true;
        } catch (e) {
            console.error(`[${pluginName}] DTLN load error:`, e);
            updateToggleBtn(true);
            return false;
        }
    }

    // ==========================================
    // Probe DTLN LSTM state shape
    // ==========================================
    async function probeDTLNStateShape(session, inputNames, audioInputShape) {
        const audioTensor = new ort.Tensor('float32',
            new Float32Array(audioInputShape.reduce((a, b) => a * b, 1)), audioInputShape);

        for (let rank = 2; rank <= 5; rank++) {
            const testShape = new Array(rank).fill(1);
            try {
                const feeds = {};
                feeds[inputNames[0]] = audioTensor;
                feeds[inputNames[1]] = new ort.Tensor('float32',
                    new Float32Array(testShape.reduce((a, b) => a * b, 1)), testShape);
                await session.run(feeds);
                console.log(`[${pluginName}] Probed state shape OK:`, testShape);
                return testShape;
            } catch (e) {
                const msg = e.message || '';
                const rankMismatch = msg.match(/Invalid rank.*Expected:\s*(\d+)/i);
                if (rankMismatch) {
                    const expectedRank = parseInt(rankMismatch[1]);
                    if (expectedRank !== rank) { rank = expectedRank - 1; continue; }
                }
                const shape    = new Array(rank).fill(1);
                const dimRegex = /index:\s*(\d+)\s+Got:\s*\d+\s+Expected:\s*(\d+)/g;
                let m, foundAny = false;
                while ((m = dimRegex.exec(msg)) !== null) {
                    shape[parseInt(m[1])] = parseInt(m[2]);
                    foundAny = true;
                }
                if (foundAny) {
                    try {
                        const feeds2 = {};
                        feeds2[inputNames[0]] = audioTensor;
                        feeds2[inputNames[1]] = new ort.Tensor('float32',
                            new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
                        await session.run(feeds2);
                        console.log(`[${pluginName}] Probed state shape OK (parsed from error):`, shape);
                        return shape;
                    } catch (e2) {
                        console.warn(`[${pluginName}] Parsed shape ${JSON.stringify(shape)} still failed:`, e2.message);
                    }
                }
            }
        }
        console.warn(`[${pluginName}] Could not probe state shape, defaulting to [1,2,128,2]`);
        return [1, 2, 128, 2];
    }

    async function resetDTLNStates(s1shape, s2shape) {
        const audioShape1 = [1, 1, DTLN_BINS];
        const audioShape2 = [1, 1, DTLN_FFT_SIZE];
        if (!s1shape) s1shape = await probeDTLNStateShape(dtlnSession1, dtlnInputNames1, audioShape1);
        if (!s2shape) s2shape = await probeDTLNStateShape(dtlnSession2, dtlnInputNames2, audioShape2);
        const sz1 = s1shape.reduce((a, b) => a * b, 1);
        const sz2 = s2shape.reduce((a, b) => a * b, 1);
        dtlnState1 = new ort.Tensor('float32', new Float32Array(sz1), s1shape);
        dtlnState2 = new ort.Tensor('float32', new Float32Array(sz2), s2shape);
        console.log(`[${pluginName}] DTLN state shapes: m1=${JSON.stringify(s1shape)}, m2=${JSON.stringify(s2shape)}`);
    }

    // ==========================================
    // Silero VAD v4  (RNNoise path only)
    // ==========================================
    function resetSileroState() {
        if (!window.ort) return;
        const sz = 2 * 1 * SILERO_STATE_SIZE;
        sileroH = new ort.Tensor('float32', new Float32Array(sz), [2, 1, SILERO_STATE_SIZE]);
        sileroC = new ort.Tensor('float32', new Float32Array(sz), [2, 1, SILERO_STATE_SIZE]);
    }

    async function sileroVAD(frame16k) {
        if (!sileroSession) return energyVAD(frame16k);
        try {
            const feeds = {
                [sileroKeyInput]: new ort.Tensor('float32', frame16k, [1, frame16k.length]),
                [sileroKeySr]:    new ort.Tensor('int64', BigInt64Array.from([16000n]), [1]),
                [sileroKeyH]:     sileroH,
                [sileroKeyC]:     sileroC,
            };
            const out = await sileroSession.run(feeds);
            sileroH = out[sileroKeyHn];
            sileroC = out[sileroKeyCn];
            return out.output.data[0];
        } catch (e) {
            if (!sileroVAD._errLogged) {
                console.warn(`[${pluginName}] Silero VAD error (falling back to energy VAD):`, e.message);
                sileroVAD._errLogged = true;
            }
            return energyVAD(frame16k);
        }
    }

    function energyVAD(frame) {
        let sum = 0;
        for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
        return Math.min(1, Math.sqrt(sum / frame.length) / 0.02);
    }

    // ==========================================
    // DTLN two-stage spectral inference
    //
    // Phase reconstruction strategy:
    //   The noisy input phase is inherently corrupted by noise.
    //   We improve it by accumulating a weighted complex sum of recent frames,
    //   where each frame is weighted by its clean magnitude.
    //   This is inspired by the "Phase-Sensitive Masking" approach and is
    //   computationally cheap (no extra model inference needed).
    //
    //   phaseWeight=0.0 → pure noisy phase (V1.6 behaviour)
    //   phaseWeight=0.5 → 50% refined / 50% noisy
    //   phaseWeight=0.9 → heavily refined phase (recommended at high strength)
    //
    // Mask sharpening (maskAlpha):
    //   mask^(1/alpha) pushes values toward 0 or 1
    //   alpha < 1 → softer suppression
    //   alpha = 1 → original model output
    //   alpha > 1 → more aggressive suppression
    // ==========================================
    async function dtlnProcessSpectrum(magSpec, re, im) {
        const invAlpha = 1.0 / P.maskAlpha;

        // ── Stage 1: magnitude mask ───────────────────────────────────
        const feeds1 = {};
        feeds1[dtlnInputNames1[0]] = new ort.Tensor('float32',
            Float32Array.from(magSpec), [1, 1, DTLN_BINS]);
        feeds1[dtlnInputNames1[1]] = dtlnState1;

        const out1  = await dtlnSession1.run(feeds1);
        dtlnState1  = out1[dtlnOutputNames1[1]];
        const mask1 = out1[dtlnOutputNames1[0]].data;

        // Apply sharpened mask1 → cleaned magnitude
        const magClean = new Float32Array(DTLN_BINS);
        for (let k = 0; k < DTLN_BINS; k++) {
            const m = Math.max(0, Math.min(1, mask1[k]));
            magClean[k] = magSpec[k] * Math.pow(m, invAlpha);
        }

        // ── Stage 2: refine cleaned magnitude ────────────────────────
        const in2 = new Float32Array(DTLN_FFT_SIZE);
        for (let k = 0; k < DTLN_BINS; k++) in2[k] = magClean[k];

        const feeds2 = {};
        feeds2[dtlnInputNames2[0]] = new ort.Tensor('float32', in2, [1, 1, DTLN_FFT_SIZE]);
        feeds2[dtlnInputNames2[1]] = dtlnState2;

        const out2  = await dtlnSession2.run(feeds2);
        dtlnState2  = out2[dtlnOutputNames2[1]];
        const mask2 = out2[dtlnOutputNames2[0]].data;

        // Apply sharpened mask2 → final magnitude (never exceeds original)
        const magFinal = new Float32Array(DTLN_BINS);
        for (let k = 0; k < DTLN_BINS; k++) {
            const m = Math.max(0, Math.min(1, mask2[k]));
            magFinal[k] = Math.min(
                magClean[k] * Math.pow(m, invAlpha),
                magSpec[k]
            );
        }

        // ── Phase reconstruction ──────────────────────────────────────
        //
        // Step 1: Update phase accumulator ring buffer
        //   Add current frame weighted by its CLEAN magnitude into the accum.
        //   This effectively votes: bins with strong clean signal contribute
        //   more phase information than bins dominated by noise.
        //
        // Step 2: Derive refined phase from accumulated complex sum
        //   refinedAngle[k] = atan2(phaseAccIm[k], phaseAccRe[k])
        //
        // Step 3: Blend refined phase with original noisy phase
        //   finalAngle = lerp(noisyAngle, refinedAngle, phaseWeight)
        //
        const outRe = new Float32Array(DTLN_BINS);
        const outIm = new Float32Array(DTLN_BINS);

        if (P.phaseWeight > 0.0 && phaseAccRe !== null) {
            // Decay accumulator (exponential moving average over PHASE_ACC_FRAMES)
            const decay = 1.0 - (1.0 / PHASE_ACC_FRAMES);
            for (let k = 0; k < DTLN_BINS; k++) {
                phaseAccRe[k] = phaseAccRe[k] * decay + re[k] * magFinal[k];
                phaseAccIm[k] = phaseAccIm[k] * decay + im[k] * magFinal[k];
            }

            for (let k = 0; k < DTLN_BINS; k++) {
                // Noisy phase angle
                const noisyAngle   = Math.atan2(im[k], re[k]);

                // Refined phase angle from accumulator
                const accMag = Math.sqrt(phaseAccRe[k] * phaseAccRe[k] + phaseAccIm[k] * phaseAccIm[k]);
                const refinedAngle = accMag > 1e-10
                    ? Math.atan2(phaseAccIm[k], phaseAccRe[k])
                    : noisyAngle;   // fall back if accumulator too weak

                // Interpolate between noisy and refined phase
                const w     = P.phaseWeight;
                const angle = noisyAngle * (1 - w) + refinedAngle * w;

                outRe[k] = magFinal[k] * Math.cos(angle);
                outIm[k] = magFinal[k] * Math.sin(angle);
            }
        } else {
            // phaseWeight=0: keep original noisy phase (V1.6 behaviour)
            for (let k = 0; k < DTLN_BINS; k++) {
                const norm = magSpec[k] > 1e-8 ? magFinal[k] / magSpec[k] : 0;
                outRe[k]   = re[k] * norm;
                outIm[k]   = im[k] * norm;
            }
        }

        return { outRe, outIm };
    }

    // ==========================================
    // Hann window
    // ==========================================
    function makeHann(n) {
        const w = new Float32Array(n);
        for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / n));
        return w;
    }
    const hannWindow = makeHann(DTLN_FFT_SIZE);

    // ==========================================
    // Fast Cooley-Tukey FFT
    // In-place on interleaved [re0,im0,re1,im1,...] of length 2N
    // ==========================================
    function fftCT(buf, invert) {
        const N = buf.length >> 1;
        for (let i = 1, j = 0; i < N; i++) {
            let bit = N >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                [buf[2*i],   buf[2*j]]   = [buf[2*j],   buf[2*i]];
                [buf[2*i+1], buf[2*j+1]] = [buf[2*j+1], buf[2*i+1]];
            }
        }
        for (let len = 2; len <= N; len <<= 1) {
            const ang = 2 * Math.PI / len * (invert ? 1 : -1);
            const wRe = Math.cos(ang), wIm = Math.sin(ang);
            for (let i = 0; i < N; i += len) {
                let curRe = 1, curIm = 0;
                for (let j = 0; j < len / 2; j++) {
                    const uRe  = buf[2*(i+j)];
                    const uIm  = buf[2*(i+j)+1];
                    const tvRe = buf[2*(i+j+len/2)] * curRe - buf[2*(i+j+len/2)+1] * curIm;
                    const tvIm = buf[2*(i+j+len/2)] * curIm + buf[2*(i+j+len/2)+1] * curRe;
                    buf[2*(i+j)]         = uRe + tvRe;
                    buf[2*(i+j)+1]       = uIm + tvIm;
                    buf[2*(i+j+len/2)]   = uRe - tvRe;
                    buf[2*(i+j+len/2)+1] = uIm - tvIm;
                    const newRe = curRe * wRe - curIm * wIm;
                    curIm = curRe * wIm + curIm * wRe;
                    curRe = newRe;
                }
            }
        }
        if (invert) {
            const s = 1 / N;
            for (let i = 0; i < buf.length; i++) buf[i] *= s;
        }
    }

    function rfftFast(frame) {
        const N   = DTLN_FFT_SIZE;
        const buf = new Float32Array(N * 2);
        for (let i = 0; i < N; i++) buf[2*i] = frame[i];
        fftCT(buf, false);
        const re = new Float32Array(N / 2 + 1);
        const im = new Float32Array(N / 2 + 1);
        for (let k = 0; k <= N / 2; k++) { re[k] = buf[2*k]; im[k] = buf[2*k+1]; }
        return { re, im };
    }

    function irfftFast(re, im) {
        const N   = DTLN_FFT_SIZE;
        const buf = new Float32Array(N * 2);
        buf[0] = re[0]; buf[1] = 0;
        for (let k = 1; k < N / 2; k++) {
            buf[2*k]       = re[k];  buf[2*k+1]       =  im[k];
            buf[2*(N-k)]   = re[k];  buf[2*(N-k)+1]   = -im[k];
        }
        buf[N] = re[N / 2]; buf[N+1] = 0;
        fftCT(buf, true);
        const out = new Float32Array(N);
        for (let i = 0; i < N; i++) out[i] = buf[2*i];
        return out;
    }

    // ==========================================
    // Load RNNoise
    // ==========================================
    async function loadRNNoise() {
        if (rnnoiseObj) return true;
        try {
            console.log(`[${pluginName}] Fetching RNNoise from: ${CDN_ESM}`);
            const resp = await fetch(CDN_ESM, { cache: 'no-store' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status} – ${CDN_ESM}`);
            const code      = await resp.text();
            const moduleUrl = URL.createObjectURL(new Blob([code], { type: 'application/javascript' }));
            console.log(`[${pluginName}] Importing RNNoise module…`);
            const mod  = await import(/* webpackIgnore: true */ moduleUrl);
            rnnoiseObj = await mod.Rnnoise.load();
            FRAME_SIZE    = rnnoiseObj.frameSize;
            denoiseState1 = rnnoiseObj.createDenoiseState();
            console.log(`[${pluginName}] RNNoise ready, frameSize=${FRAME_SIZE}`);
            updateToggleBtn();
            return true;
        } catch (e) {
            console.error(`[${pluginName}] RNNoise error:`, e);
            updateToggleBtn(true);
            return false;
        }
    }

    // ==========================================
    // RMS + Soft-Gate  (RNNoise only)
    // ==========================================
    function calcRMS(buf) {
        let sum = 0;
        for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
        return Math.sqrt(sum / buf.length);
    }
    function softGate(frame) {
        const rms = calcRMS(frame);
        if (rms >= P.gateThresh) return frame;
        const g   = (1.0 - P.gateRatio) + P.gateRatio * (rms / P.gateThresh);
        const out = new Float32Array(frame.length);
        for (let i = 0; i < frame.length; i++) out[i] = frame[i] * g;
        return out;
    }

    // ==========================================
    // Apply EQ live
    // ==========================================
    function applyEQ() {
        if (!ctx48) return;
        const t = ctx48.currentTime;
        nodes48.eqPresence?.gain.setTargetAtTime(P.eqPresence, t, 0.02);
        nodes48.eqClarity?.gain.setTargetAtTime(P.eqClarity,   t, 0.02);
        nodes48.eqLow?.gain.setTargetAtTime(P.eqLow,           t, 0.02);
        nodes48.eqMid?.gain.setTargetAtTime(P.eqMid,           t, 0.02);
        nodes48.eqHigh?.gain.setTargetAtTime(P.eqHigh,         t, 0.02);
        nodes48.eqAir?.gain.setTargetAtTime(P.eqAir,           t, 0.02);
    }

    // ==========================================
    // DSP Chain
    // ==========================================
    async function initDSPChain() {
        if (currentMode === 'dtln') {
            if (!dtlnReady && !await loadDTLN()) return;
        } else {
            if (!rnnoiseObj && !await loadRNNoise()) return;
        }
        if (ctx48) { try { ctx48.close(); } catch (_) {} ctx48 = null; }
        calcParams();
        ctx48 = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });

        nodes.hpf = nativeCtx.createBiquadFilter();
        nodes.hpf.type            = 'highpass';
        nodes.hpf.frequency.value = 80;
        nodes.hpf.Q.value         = 0.5;

        nodes.streamDest  = nativeCtx.createMediaStreamDestination();
        nodes48.streamSrc = ctx48.createMediaStreamSource(nodes.streamDest.stream);

        if (currentMode === 'dtln') await buildDTLNProcessor();
        else buildRNNoiseProcessor();

        buildEQChain();
    }

    // ==========================================
    // DTLN ScriptProcessor (STFT-domain)
    //
    // Audio path:
    //   48 kHz → downsample 3:1 → 16 kHz
    //   Sliding window: hop=128, window=512
    //   STFT → two-stage DTLN mask → phase refinement → ISTFT → OLA → upsample → 48 kHz
    //
    // No VAD: suppression always active.
    // Strength slider controls mask sharpening (0–80%) and dry/wet blend (80–100%).
    // Phase refinement is controlled by phaseWeight (derived from strength slider).
    // ==========================================
    async function buildDTLNProcessor() {
        dtlnState1 = null;
        dtlnState2 = null;
        await resetDTLNStates(null, null);
        resetSileroState();
        resetPhaseAccumulator();

        const RATIO    = 3;
        const MAX_BUF  = 48000 * 4;
        const OLA_NORM = 1.5;   // Hann 75% overlap normalisation factor

        let inBuf16  = new Float32Array(MAX_BUF);
        let inLen16  = 0;
        let outBuf48 = new Float32Array(MAX_BUF);
        let outLen48 = 0;
        let olaSynth = new Float32Array(DTLN_FFT_SIZE);

        function downsample3(b48) {
            const n   = Math.floor(b48.length / RATIO);
            const out = new Float32Array(n);
            for (let i = 0; i < n; i++)
                out[i] = (b48[i*RATIO] + b48[i*RATIO+1] + b48[i*RATIO+2]) / 3;
            return out;
        }
        function upsample3(b16) {
            const out = new Float32Array(b16.length * RATIO);
            for (let i = 0; i < b16.length; i++)
                out[i*RATIO] = out[i*RATIO+1] = out[i*RATIO+2] = b16[i];
            return out;
        }

        nodes48.processor = ctx48.createScriptProcessor(2048, 1, 1);
        nodes48.processor.onaudioprocess = (e) => {
            const input  = e.inputBuffer.getChannelData(0);
            const output = e.outputBuffer.getChannelData(0);

            // Downsample 48 kHz → 16 kHz
            const chunk16 = downsample3(input);
            if (inLen16 + chunk16.length > MAX_BUF) inLen16 = 0;
            inBuf16.set(chunk16, inLen16);
            inLen16 += chunk16.length;

            (async () => {
                while (inLen16 >= DTLN_FFT_SIZE) {
                    // Windowed analysis frame
                    const frame = new Float32Array(DTLN_FFT_SIZE);
                    for (let i = 0; i < DTLN_FFT_SIZE; i++)
                        frame[i] = inBuf16[i] * hannWindow[i];

                    // STFT
                    const { re, im } = rfftFast(frame);

                    // Magnitude spectrum
                    const mag = new Float32Array(DTLN_BINS);
                    for (let k = 0; k < DTLN_BINS; k++)
                        mag[k] = Math.sqrt(re[k]*re[k] + im[k]*im[k]);

                    // DTLN two-stage inference + phase refinement
                    let outRe, outIm;
                    try {
                        ({ outRe, outIm } = await dtlnProcessSpectrum(mag, re, im));
                    } catch (err) {
                        if (!buildDTLNProcessor._inferErrLogged) {
                            console.warn(`[${pluginName}] DTLN inference error:`, err.message);
                            buildDTLNProcessor._inferErrLogged = true;
                        }
                        outRe = re; outIm = im;   // pass-through on error
                    }

                    // Blend denoised vs dry
                    // 0–80%: blend=0 (dry), mask sharpening + phase refinement only
                    // 80–100%: blend 0→1 (full DTLN)
                    const blendedRe = new Float32Array(DTLN_BINS);
                    const blendedIm = new Float32Array(DTLN_BINS);
                    for (let k = 0; k < DTLN_BINS; k++) {
                        blendedRe[k] = outRe[k] * P.blend + re[k] * (1 - P.blend);
                        blendedIm[k] = outIm[k] * P.blend + im[k] * (1 - P.blend);
                    }

                    // ISTFT + overlap-add
                    const synth = irfftFast(blendedRe, blendedIm);
                    for (let i = 0; i < DTLN_FFT_SIZE; i++)
                        olaSynth[i] += (synth[i] * hannWindow[i]) / OLA_NORM;

                    // Flush one hop to output ring
                    const hopOut48 = upsample3(olaSynth.slice(0, DTLN_HOP));
                    if (outLen48 + hopOut48.length < MAX_BUF) {
                        outBuf48.set(hopOut48, outLen48);
                        outLen48 += hopOut48.length;
                    }

                    // Shift OLA buffer
                    olaSynth.copyWithin(0, DTLN_HOP);
                    olaSynth.fill(0, DTLN_FFT_SIZE - DTLN_HOP);

                    // Advance input ring
                    inBuf16.copyWithin(0, DTLN_HOP, inLen16);
                    inLen16 -= DTLN_HOP;
                }
            })();

            // Output
            if (outLen48 >= output.length) {
                output.set(outBuf48.subarray(0, output.length));
                outBuf48.copyWithin(0, output.length, outLen48);
                outLen48 -= output.length;
            } else {
                const last = outLen48 > 0 ? outBuf48[outLen48 - 1] : 0;
                for (let i = 0; i < output.length; i++)
                    output[i] = last * (1 - i / output.length);
            }
            for (let i = 0; i < output.length; i++)
                output[i] = Math.max(-1, Math.min(1, output[i] * P.outputGain));
        };
    }

    // ==========================================
    // RNNoise ScriptProcessor
    // ==========================================
    function buildRNNoiseProcessor() {
        let holdCounter = 0, fadeCounter = 0, phase = 'noise';
        function getBlend(vad) {
            if (vad >= P.vadThreshold) {
                phase = 'speech'; holdCounter = P.holdFrames; fadeCounter = 0; return P.blend;
            }
            if (phase === 'speech' || phase === 'hold') {
                phase = 'hold';
                if (--holdCounter <= 0) { phase = 'fade'; fadeCounter = 0; }
                return P.blend;
            }
            if (phase === 'fade') {
                fadeCounter++;
                const t      = Math.min(fadeCounter / P.fadeFrames, 1.0);
                const smooth = 0.5 - 0.5 * Math.cos(t * Math.PI);
                if (fadeCounter >= P.fadeFrames) phase = 'noise';
                return P.blend * (1 - smooth) + P.vadFloor * smooth;
            }
            return P.vadFloor;
        }

        const MAX_BUF    = 4096 * 16;
        let inputBuf     = new Float32Array(MAX_BUF);
        let inputLen     = 0;
        let outputBuf    = new Float32Array(MAX_BUF);
        let outputBufLen = 0;
        const rnInput    = new Float32Array(FRAME_SIZE);

        nodes48.processor = ctx48.createScriptProcessor(2048, 1, 1);
        nodes48.processor.onaudioprocess = (e) => {
            const input  = e.inputBuffer.getChannelData(0);
            const output = e.outputBuffer.getChannelData(0);

            if (inputLen + input.length > MAX_BUF) inputLen = 0;
            inputBuf.set(input, inputLen);
            inputLen += input.length;

            while (inputLen >= FRAME_SIZE) {
                const original = inputBuf.slice(0, FRAME_SIZE);
                for (let i = 0; i < FRAME_SIZE; i++) rnInput[i] = original[i] * 32768;
                const vad = denoiseState1.processFrame(rnInput);
                const denoised = new Float32Array(FRAME_SIZE);
                for (let i = 0; i < FRAME_SIZE; i++) denoised[i] = rnInput[i] / 32768;
                const bf    = getBlend(vad);
                const mixed = new Float32Array(FRAME_SIZE);
                for (let i = 0; i < FRAME_SIZE; i++)
                    mixed[i] = denoised[i] * bf + original[i] * (1 - bf);
                const gated = softGate(mixed);
                if (outputBufLen + FRAME_SIZE < MAX_BUF) {
                    outputBuf.set(gated, outputBufLen);
                    outputBufLen += FRAME_SIZE;
                }
                inputBuf.copyWithin(0, FRAME_SIZE, inputLen);
                inputLen -= FRAME_SIZE;
            }

            if (outputBufLen >= output.length) {
                output.set(outputBuf.subarray(0, output.length));
                outputBuf.copyWithin(0, output.length, outputBufLen);
                outputBufLen -= output.length;
            } else {
                const last = outputBufLen > 0 ? outputBuf[outputBufLen - 1] : 0;
                for (let i = 0; i < output.length; i++)
                    output[i] = last * (1 - i / output.length);
            }
            for (let i = 0; i < output.length; i++)
                output[i] = Math.max(-1, Math.min(1, output[i] * P.outputGain));
        };
    }

    // ==========================================
    // EQ + Limiter
    // ==========================================
    function buildEQChain() {
        nodes48.eqPresence = ctx48.createBiquadFilter();
        nodes48.eqPresence.type            = 'peaking';
        nodes48.eqPresence.frequency.value = 3500;
        nodes48.eqPresence.Q.value         = 1.4;
        nodes48.eqPresence.gain.value      = P.eqPresence;

        nodes48.eqClarity = ctx48.createBiquadFilter();
        nodes48.eqClarity.type            = 'peaking';
        nodes48.eqClarity.frequency.value = 1000;
        nodes48.eqClarity.Q.value         = 1.2;
        nodes48.eqClarity.gain.value      = P.eqClarity;

        nodes48.eqLow = ctx48.createBiquadFilter();
        nodes48.eqLow.type            = 'lowshelf';
        nodes48.eqLow.frequency.value = 300;
        nodes48.eqLow.gain.value      = P.eqLow;

        nodes48.eqMid = ctx48.createBiquadFilter();
        nodes48.eqMid.type            = 'peaking';
        nodes48.eqMid.frequency.value = 2500;
        nodes48.eqMid.Q.value         = 0.9;
        nodes48.eqMid.gain.value      = P.eqMid;

        nodes48.eqHigh = ctx48.createBiquadFilter();
        nodes48.eqHigh.type            = 'highshelf';
        nodes48.eqHigh.frequency.value = 7000;
        nodes48.eqHigh.gain.value      = P.eqHigh;

        nodes48.eqAir = ctx48.createBiquadFilter();
        nodes48.eqAir.type            = 'highshelf';
        nodes48.eqAir.frequency.value = 12000;
        nodes48.eqAir.gain.value      = P.eqAir;

        nodes48.limiter = ctx48.createDynamicsCompressor();
        nodes48.limiter.threshold.value = currentMode === 'dtln' ? -6  : -1;
        nodes48.limiter.knee.value      = currentMode === 'dtln' ?  2  :  3;
        nodes48.limiter.ratio.value     = currentMode === 'dtln' ? 12  : 20;
        nodes48.limiter.attack.value    = 0.001;
        nodes48.limiter.release.value   = currentMode === 'dtln' ? 0.2 : 0.1;

        nodes48.outDest = ctx48.createMediaStreamDestination();
        nodes.outSrc    = nativeCtx.createMediaStreamSource(nodes48.outDest.stream);
    }

    // ==========================================
    // Routing
    // ==========================================
    function connectPipeline() {
        sourceNode.connect(nodes.hpf);
        nodes.hpf.connect(nodes.streamDest);
        nodes48.streamSrc.connect(nodes48.processor);
        nodes48.processor.connect(nodes48.eqPresence);
        nodes48.eqPresence.connect(nodes48.eqClarity);
        nodes48.eqClarity.connect(nodes48.eqLow);
        nodes48.eqLow.connect(nodes48.eqMid);
        nodes48.eqMid.connect(nodes48.eqHigh);
        nodes48.eqHigh.connect(nodes48.eqAir);
        nodes48.eqAir.connect(nodes48.limiter);
        nodes48.limiter.connect(nodes48.outDest);
        if (!nodes.directGain) {
            nodes.directGain = nativeCtx.createGain();
            nodes.directGain.gain.value = 0;
        }
        try { sourceNode.disconnect(nativeCtx.destination); } catch (_) {}
        sourceNode.connect(nodes.directGain);
        nodes.directGain.connect(nativeCtx.destination);
        nodes.outSrc.connect(nativeCtx.destination);
    }

    function disconnectAll() {
        try { nodes.directGain?.disconnect(); } catch (_) {}
        nodes.directGain = null;
        [nodes.hpf, nodes.streamDest, nodes.outSrc]
            .forEach(n => { try { n?.disconnect(); } catch (_) {} });
        [nodes48.streamSrc, nodes48.processor,
         nodes48.eqPresence, nodes48.eqClarity, nodes48.eqLow,
         nodes48.eqMid, nodes48.eqHigh, nodes48.eqAir,
         nodes48.limiter, nodes48.outDest]
            .forEach(n => { try { n?.disconnect(); } catch (_) {} });
        try { sourceNode.connect(nativeCtx.destination); } catch (_) {}
    }

    function toggleDSP() {
        if (!sourceNode || !nodes48.processor) return;
        disconnectAll();
        if (isEnabled) connectPipeline();
        else sourceNode.connect(nativeCtx.destination);
        updateToggleBtn();
    }

    // ==========================================
    // Mode Switch
    // ==========================================
    async function switchMode(newMode) {
        if (newMode === currentMode && (currentMode === 'rnnoise' ? rnnoiseObj : dtlnReady)) return;

        saveSettings(currentMode, S);
        currentMode = newMode;
        localStorage.setItem('ai-denoise-mode', newMode);
        S = loadSettings(newMode);

        const wasEnabled = isEnabled;
        if (isEnabled) { isEnabled = false; if (nodes48.processor) disconnectAll(); }
        if (ctx48) { try { ctx48.close(); } catch (_) {} ctx48 = null; }
        nodes = {}; nodes48 = {};

        refreshSliderUI();
        calcParams();
        updateModeButtons();
        updateToggleBtn();

        if (newMode === 'dtln') await loadDTLN();
        else await loadRNNoise();

        if (sourceNode) {
            await initDSPChain();
            if (wasEnabled) { isEnabled = true; toggleDSP(); }
        }
    }

    // ==========================================
    // Stream Hook
    // ==========================================
    function checkAndHookAudio() {
        if (!Stream?.Fallback?.Player?.Amplification) return;
        const src = Stream.Fallback.Player.Amplification;
        if (src === sourceNode) return;
        sourceNode = src;
        nativeCtx  = sourceNode.context;
        initDSPChain().then(() => { if (isEnabled) toggleDSP(); });
    }

    // ==========================================
    // Styles
    // ==========================================
    const style = document.createElement('style');
    style.innerHTML = `
        #ai-denoise-overlay {
            position: fixed; display: none; width: 230px;
            background-color: var(--color-1); color: #fff;
            font-family: sans-serif; border-radius: 8px;
            z-index: 1500; cursor: move; user-select: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            border: 1px solid #444; font-size: 13px;
        }
        #ai-denoise-header {
            background: rgba(255,255,255,0.1); padding: 8px 10px;
            border-bottom: 1px solid #444; font-weight: bold;
            border-radius: 8px 8px 0 0;
            display: flex; justify-content: space-between; align-items: center;
        }
        #ai-denoise-close {
            cursor: pointer; font-weight: bold; color: #ccc;
            font-size: 18px; line-height: 1; padding: 0 4px; margin-right: -6px;
        }
        #ai-denoise-close:hover { color: #fff; }
        #ai-denoise-content { padding: 10px; display: flex; flex-direction: column; gap: 10px; }
        .ai-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }
        .ai-label { color: #aaa; }
        .ai-val   { color: #fff; font-weight: bold; }
        .ai-slider-wrap { display: flex; flex-direction: column; gap: 2px; }
        .ai-slider-labels { display: flex; justify-content: space-between; font-size: 9px; color: #555; margin-top: 2px; }
        #ai-mode-row { display: flex; gap: 4px; }
        .ai-mode-btn {
            flex: 1; border: 1px solid #444; background: transparent; color: #888;
            border-radius: 5px; padding: 5px 0; cursor: pointer;
            font-size: 11px; font-family: sans-serif; transition: all 0.2s;
        }
        .ai-mode-btn:hover { color: #fff; border-color: #666; }
        .ai-mode-btn.selected { background: #4da6ff; color: #000; border-color: #4da6ff; font-weight: bold; }
        #ai-denoise-overlay input[type=range] {
            width: 100%; height: 6px; cursor: pointer;
            appearance: none; -webkit-appearance: none;
            background: linear-gradient(to right, #4da6ff 50%, #333 0%);
            border-radius: 3px; outline: none; margin: 0; display: block;
        }
        #ai-denoise-overlay input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%;
            background: #fff; cursor: pointer; border: 3px solid #4da6ff;
            box-shadow: 0 0 4px rgba(77,166,255,0.9);
        }
        #ai-denoise-overlay input[type=range]::-moz-range-thumb {
            width: 18px; height: 18px; border: 3px solid #4da6ff;
            border-radius: 50%; background: #fff; cursor: pointer;
        }
        #ai-denoise-onoff-btn {
            width: 100%; border: none; border-radius: 5px; padding: 8px 0;
            cursor: pointer; font-weight: bold; font-size: 12px;
            font-family: sans-serif; transition: all 0.3s;
            background-color: #4da6ff; color: #000;
        }
        #ai-denoise-onoff-btn.active   { background-color: #dc3545 !important; color: #fff !important; }
        #ai-denoise-onoff-btn:disabled { cursor: not-allowed; opacity: 0.6; }
        #ai-denoise-reset-btn {
            width: 100%; background: transparent; color: #555; border: 1px solid #333;
            border-radius: 5px; padding: 5px 0; cursor: pointer;
            font-size: 10px; font-family: sans-serif; transition: all 0.2s;
        }
        #ai-denoise-reset-btn:hover { color: #fff; border-color: #666; }
        #Denoiser-on-off:hover  { color: var(--color-5); filter: brightness(120%); }
        #Denoiser-on-off.active { background-color: var(--color-2) !important; filter: brightness(120%); }
    `;
    document.head.appendChild(style);

    // ==========================================
    // Overlay HTML
    // ==========================================
    const overlay = document.createElement('div');
    overlay.id = 'ai-denoise-overlay';
    overlay.innerHTML = `
        <div id="ai-denoise-header">
            <span>AI Denoiser</span>
            <span id="ai-denoise-close" title="Close">&times;</span>
        </div>
        <div id="ai-denoise-content">
            <button id="ai-denoise-onoff-btn">Loading AI...</button>
            <hr style="border:0;border-top:1px solid #333;margin:0;">
            <div>
                <div class="ai-row" style="margin-bottom:5px;">
                    <span class="ai-label" style="font-size:11px;">AI Engine</span>
                </div>
                <div id="ai-mode-row">
                    <button class="ai-mode-btn" id="ai-mode-dtln"    title="DTLN – dual ONNX model (high quality)">DTLN</button>
                    <button class="ai-mode-btn" id="ai-mode-rnnoise" title="RNNoise – lightweight WASM model">RNNoise</button>
                </div>
            </div>
            <hr style="border:0;border-top:1px solid #333;margin:0;">
            <div class="ai-slider-wrap">
                <div class="ai-row">
                    <span class="ai-label">Strength</span>
                    <span class="ai-val" id="ai-strength-val">50%</span>
                </div>
                <input type="range" id="ai-sl-strength" min="0" max="1" step="0.01" value="0.5">
                <div class="ai-slider-labels"><span>low</span><span>high</span></div>
            </div>
            <div class="ai-slider-wrap">
                <div class="ai-row">
                    <span class="ai-label">Tone</span>
                    <span class="ai-val" id="ai-voice-val">50%</span>
                </div>
                <input type="range" id="ai-sl-voice" min="0" max="1" step="0.01" value="0.5">
                <div class="ai-slider-labels"><span>warm</span><span>bright</span></div>
            </div>
            <div class="ai-slider-wrap">
                <div class="ai-row">
                    <span class="ai-label" id="ai-gate-label">Gate</span>
                    <span class="ai-val" id="ai-gate-val">50%</span>
                </div>
                <input type="range" id="ai-sl-gate" min="0" max="1" step="0.01" value="0.5">
                <div class="ai-slider-labels">
                    <span id="ai-gate-label-min">open</span>
                    <span id="ai-gate-label-max">tight</span>
                </div>
            </div>
            <hr style="border:0;border-top:1px solid #333;margin:0;">
            <button id="ai-denoise-reset-btn">&#8635; Reset</button>
        </div>
    `;
    document.body.appendChild(overlay);

    // ==========================================
    // Mode Buttons
    // ==========================================
    function updateModeButtons() {
        document.getElementById('ai-mode-dtln')   .classList.toggle('selected', currentMode === 'dtln');
        document.getElementById('ai-mode-rnnoise').classList.toggle('selected', currentMode === 'rnnoise');
        updateGateLabel();
    }

    function updateGateLabel() {
        const isDTLN   = currentMode === 'dtln';
        const label    = document.getElementById('ai-gate-label');
        const labelMin = document.getElementById('ai-gate-label-min');
        const labelMax = document.getElementById('ai-gate-label-max');
        if (!label) return;
        if (isDTLN) {
            label.textContent    = 'Gain';
            labelMin.textContent = 'quiet';
            labelMax.textContent = 'loud';
        } else {
            label.textContent    = 'Gate';
            labelMin.textContent = 'open';
            labelMax.textContent = 'tight';
        }
    }

    updateModeButtons();
    document.getElementById('ai-mode-dtln')   .addEventListener('click', () => switchMode('dtln'));
    document.getElementById('ai-mode-rnnoise').addEventListener('click', () => switchMode('rnnoise'));

    // ==========================================
    // Slider Helpers
    // ==========================================
    function setSliderUI(slId, valId, value) {
        const sl  = document.getElementById(slId);
        const val = document.getElementById(valId);
        if (!sl || !val) return;
        const pct = Math.round(value * 100);
        sl.value      = value;
        val.innerText = pct + '%';
        sl.style.background = `linear-gradient(to right,#4da6ff ${pct}%,#333 0%)`;
    }
    function refreshSliderUI() {
        setSliderUI('ai-sl-strength', 'ai-strength-val', S.strength);
        setSliderUI('ai-sl-voice',    'ai-voice-val',    S.voice);
        setSliderUI('ai-sl-gate',     'ai-gate-val',     S.gate);
    }
    function bindSlider(slId, key, valId) {
        setSliderUI(slId, valId, S[key]);
        document.getElementById(slId).addEventListener('input', () => {
            const v = parseFloat(document.getElementById(slId).value);
            S[key]  = v;
            setSliderUI(slId, valId, v);
            localStorage.setItem(lsKey(currentMode, key), v);
            calcParams();
            applyEQ();
        });
    }
    bindSlider('ai-sl-strength', 'strength', 'ai-strength-val');
    bindSlider('ai-sl-voice',    'voice',    'ai-voice-val');
    bindSlider('ai-sl-gate',     'gate',     'ai-gate-val');

    document.getElementById('ai-denoise-onoff-btn').addEventListener('click', () => {
        const ready = currentMode === 'dtln' ? dtlnReady : !!rnnoiseObj;
        if (!ready) { alert('AI is still loading, please wait.'); return; }
        if (typeof Stream === 'undefined' || !Stream) { alert('Please start playback first.'); return; }
        isEnabled = !isEnabled;
        toggleDSP();
    });

    document.getElementById('ai-denoise-reset-btn').addEventListener('click', () => {
        Object.assign(S, DEFAULTS);
        saveSettings(currentMode, S);
        refreshSliderUI();
        calcParams();
        applyEQ();
    });

    document.getElementById('ai-denoise-close').addEventListener('click', () => {
        $('#ai-denoise-overlay').stop(true, true).fadeOut(400);
        const btn = document.getElementById('Denoiser-on-off');
        if (btn) btn.classList.remove('active');
    });

    // ==========================================
    // Toggle Button State
    // ==========================================
    function updateToggleBtn(error) {
        const btn    = document.getElementById('ai-denoise-onoff-btn');
        const navBtn = document.getElementById('Denoiser-on-off');
        if (!btn) return;
        const icon = navBtn ? navBtn.querySelector('i') : null;
        if (icon) icon.style.color = (isEnabled && !error) ? '#FE0830' : '';
        if (error) {
            btn.textContent = 'Error loading AI';
            btn.style.backgroundColor = '#ff4444';
            btn.style.color = '#fff';
            btn.disabled = false;
            btn.classList.remove('active');
            return;
        }
        const ready = currentMode === 'dtln' ? dtlnReady : !!rnnoiseObj;
        if (!ready) {
            btn.textContent = currentMode === 'dtln' ? 'Loading DTLN…' : 'Loading RNNoise…';
            btn.style.backgroundColor = '#ffc107';
            btn.style.color = '#000';
            btn.disabled = true;
            btn.classList.remove('active');
            return;
        }
        const streamActive = (typeof Stream !== 'undefined' && !!Stream);
        if (!streamActive) {
            btn.textContent = 'Start playback first';
            btn.style.backgroundColor = '#555';
            btn.style.color = '#aaa';
            btn.disabled = true;
            btn.classList.remove('active');
            if (icon) icon.style.color = '';
            return;
        }
        btn.disabled = false;
        if (isEnabled) {
            btn.textContent = 'AI ON';
            btn.style.backgroundColor = '';
            btn.style.color = '';
            btn.classList.add('active');
        } else {
            btn.textContent = 'AI OFF';
            btn.style.backgroundColor = '';
            btn.style.color = '';
            btn.classList.remove('active');
        }
    }

    // ==========================================
    // Position + Drag
    // ==========================================
    overlay.style.left = localStorage.getItem('denoiserLeft') || '20px';
    overlay.style.top  = localStorage.getItem('denoiserTop')  || '240px';
    (function () {
        let dragging = false, sx, sy, ox, oy;
        overlay.addEventListener('mousedown', e => {
            if (e.target.id === 'ai-denoise-close'     ||
                e.target.id === 'ai-denoise-onoff-btn' ||
                e.target.id === 'ai-denoise-reset-btn' ||
                e.target.classList.contains('ai-mode-btn') ||
                e.target.tagName === 'INPUT') return;
            dragging = true; sx = e.clientX; sy = e.clientY;
            const r = overlay.getBoundingClientRect(); ox = r.left; oy = r.top;
            e.preventDefault();
        });
        document.addEventListener('mousemove', e => {
            if (!dragging) return;
            overlay.style.left = (ox + e.clientX - sx) + 'px';
            overlay.style.top  = (oy + e.clientY - sy) + 'px';
        });
        document.addEventListener('mouseup', () => {
            if (!dragging) return;
            dragging = false;
            localStorage.setItem('denoiserLeft', overlay.style.left);
            localStorage.setItem('denoiserTop',  overlay.style.top);
        });
    })();

    // ==========================================
    // Toolbar Button
    // ==========================================
    (function () {
        const btnId = 'Denoiser-on-off';
        let found = false;
        const obs = new MutationObserver((_, o) => {
            if (typeof addIconToPluginPanel === 'function') {
                found = true; o.disconnect();
                addIconToPluginPanel(btnId, 'Denoiser', 'solid', 'wand-magic-sparkles',
                    `AI Denoiser v${pluginVersion}`);
                const btnObs = new MutationObserver((_, o2) => {
                    const $btn = $(`#${btnId}`);
                    if ($btn.length) {
                        o2.disconnect();
                        $btn.addClass('hide-phone bg-color-2');
                        $btn.on('click', () => {
                            const visible = $('#ai-denoise-overlay').is(':visible');
                            if (!visible) {
                                $btn.addClass('active');
                                $('#ai-denoise-overlay').stop(true, true).fadeIn(400);
                            } else {
                                $btn.removeClass('active');
                                $('#ai-denoise-overlay').stop(true, true).fadeOut(400);
                            }
                        });
                    }
                });
                btnObs.observe(document.body, { childList: true, subtree: true });
            }
        });
        obs.observe(document.body, { childList: true, subtree: true });
        setTimeout(() => { if (!found) obs.disconnect(); }, 10000);
    })();

    // ==========================================
    // Start
    // ==========================================
    if (currentMode === 'dtln') loadDTLN();
    else loadRNNoise();

    setInterval(() => {
        checkAndHookAudio();
        updateToggleBtn();
    }, 1000);

})();