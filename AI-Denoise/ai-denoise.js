"use strict";

///////////////////////////////////////////////////////////////
//                                                           //
//  AI DENOISER PLUGIN FOR FM-DX-WEBSERVER (V1.0)           //
//                                                           //
//  by Highpoint                last update: 2026-03-12      //
//                                                           //
//  https://github.com/Highpoint2000/AI-Denoise             //
//                                                           //
///////////////////////////////////////////////////////////////

(() => {
    // ------------- Plugin Metadata ----------------
    const pluginVersion    = '1.0';
    const pluginName       = 'AI-Denoise';
    const pluginHomepageUrl = 'https://github.com/Highpoint2000/AI-Denoise/releases';
    const pluginUpdateUrl   = 'https://raw.githubusercontent.com/Highpoint2000/AI-Denoise/main/ai-denoise.js';
    const CHECK_FOR_UPDATES = true;

    // ------------- DSP State ----------------------
    let isEnabled  = false;
    let sourceNode = null;
    let nativeCtx  = null;
    let ctx48      = null;
    let nodes      = {};
    let nodes48    = {};

    let rnnoiseObj    = null;
    let denoiseState1 = null;
    let FRAME_SIZE    = 480;

    const CDN_ESM = 'https://cdn.jsdelivr.net/npm/@shiguredo/rnnoise-wasm@2025.1.5/dist/rnnoise.js';

    const DEFAULTS = { strength: 0.5, voice: 0.5, gate: 0.5 };
    const S = { ...DEFAULTS };

    let P = {};
    function calcParams() {
        const str = S.strength, voi = S.voice, gat = S.gate;
        P = {
            blend:        0.5  + str * 0.45,
            vadThreshold: 0.02 + (1 - str) * 0.15,
            vadFloor:     (1 - str) * 0.25,
            holdFrames:   Math.round(40 + str * 60),
            fadeFrames:   Math.round(30 + str * 80),
            gateThresh:   0.003 + gat * 0.06,
            gateRatio:    1.0  - gat * 0.85,
            eqPresence:   2.0  + str * 4.0 + voi * 3.0,
            eqClarity:    1.0  + voi * 2.0,
            eqLow:        (0.5 - voi) * 4.0,
            eqMid:        1.0  + str * 1.5 + voi * 1.5,
            eqHigh:       3.0  + str * 5.0 + voi * 4.0,
            eqAir:        2.0  + str * 3.0 + voi * 4.0,
            outputGain:   1.0  + str * 0.25,
        };
    }

    // ==========================================
    // Update Check
    // ==========================================
    function checkUpdate() {
        const cleanUrl = pluginUpdateUrl + '?t=' + Date.now();
        fetch(cleanUrl, { cache: 'no-store' })
            .then(r => r.text())
            .then(txt => {
                const match = txt.match(/const\s+pluginVersion\s*=\s*['"]([^'"]+)['"]/);
                if (!match) return;
                const remoteVer = match[1];
                if (remoteVer !== pluginVersion) {
                    console.log(`[${pluginName}] Update available: ${pluginVersion} -> ${remoteVer}`);

                    // Add link to plugin settings page
                    const settings = document.getElementById('plugin-settings');
                    if (settings && !settings.innerHTML.includes(pluginHomepageUrl)) {
                        settings.innerHTML += `<br><a href="${pluginHomepageUrl}" target="_blank">[${pluginName}] Update: ${pluginVersion} -> ${remoteVer}</a>`;
                    }

                    // Add red dot to nav icon
                    const navIcon =
                        document.querySelector('.wrapper-outer #navigation .sidenav-content .fa-puzzle-piece') ||
                        document.querySelector('.wrapper-outer .sidenav-content') ||
                        document.querySelector('.sidenav-content');
                    if (navIcon && !navIcon.querySelector(`.${pluginName}-update-dot`)) {
                        const dot = document.createElement('span');
                        dot.classList.add(`${pluginName}-update-dot`);
                        dot.style.cssText = `
                            display:block;width:12px;height:12px;border-radius:50%;
                            background-color:#FE0830;margin-left:82px;margin-top:-12px;
                        `;
                        navIcon.appendChild(dot);
                    }
                }
            })
            .catch(e => console.warn(`[${pluginName}] Update check failed`, e));
    }
    if (CHECK_FOR_UPDATES) checkUpdate();

    // ==========================================
    // 1. Load RNNoise
    // ==========================================
    async function loadRNNoise() {
        if (rnnoiseObj) return true;
        try {
            console.log(`[${pluginName}] Loading RNNoise:`, CDN_ESM);
            const mod  = await import(/* webpackIgnore: true */ CDN_ESM);
            rnnoiseObj = await mod.Rnnoise.load();
            FRAME_SIZE    = rnnoiseObj.frameSize;
            denoiseState1 = rnnoiseObj.createDenoiseState();
            console.log(`[${pluginName}] Ready, frameSize=${FRAME_SIZE}`);
            updateToggleBtn();
            return true;
        } catch (e) {
            console.error(`[${pluginName}] Error:`, e);
            updateToggleBtn(true);
            return false;
        }
    }

    // ==========================================
    // 2. RMS + Soft-Gate
    // ==========================================
    function calcRMS(buf) {
        let sum = 0;
        for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
        return Math.sqrt(sum / buf.length);
    }

    function softGate(frame) {
        const rms = calcRMS(frame);
        if (rms >= P.gateThresh) return frame;
        const g   = P.gateRatio + (1 - P.gateRatio) * (rms / P.gateThresh);
        const out = new Float32Array(frame.length);
        for (let i = 0; i < frame.length; i++) out[i] = frame[i] * g;
        return out;
    }

    // ==========================================
    // 3. Apply EQ live
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
    // 4. DSP Chain
    // ==========================================
    async function initDSPChain() {
        if (!rnnoiseObj && !await loadRNNoise()) return;
        if (ctx48) { try { ctx48.close(); } catch (_) {} ctx48 = null; }

        calcParams();
        ctx48 = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });

        nodes.hpf = nativeCtx.createBiquadFilter();
        nodes.hpf.type            = 'highpass';
        nodes.hpf.frequency.value = 80;
        nodes.hpf.Q.value         = 0.5;

        nodes.streamDest  = nativeCtx.createMediaStreamDestination();
        nodes48.streamSrc = ctx48.createMediaStreamSource(nodes.streamDest.stream);

        let holdCounter = 0, fadeCounter = 0, phase = 'noise';
        function getBlend(vad) {
            if (vad >= P.vadThreshold) {
                phase = 'speech'; holdCounter = P.holdFrames; fadeCounter = 0;
                return P.blend;
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
        nodes48.limiter.threshold.value = -1;
        nodes48.limiter.knee.value      = 3;
        nodes48.limiter.ratio.value     = 20;
        nodes48.limiter.attack.value    = 0.001;
        nodes48.limiter.release.value   = 0.1;

        nodes48.outDest = ctx48.createMediaStreamDestination();
        nodes.outSrc    = nativeCtx.createMediaStreamSource(nodes48.outDest.stream);
    }

    // ==========================================
    // 5. Routing
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
        nodes.outSrc.connect(nativeCtx.destination);
    }

    function disconnectAll() {
        [sourceNode, nodes.hpf, nodes.streamDest, nodes.outSrc]
            .forEach(n => { try { n?.disconnect(); } catch (_) {} });
        [nodes48.streamSrc, nodes48.processor,
         nodes48.eqPresence, nodes48.eqClarity, nodes48.eqLow,
         nodes48.eqMid, nodes48.eqHigh, nodes48.eqAir,
         nodes48.limiter, nodes48.outDest]
            .forEach(n => { try { n?.disconnect(); } catch (_) {} });
    }

    function toggleDSP() {
        if (!sourceNode || !nodes48.processor) return;
        disconnectAll();
        if (isEnabled) {
            connectPipeline();
        } else {
            sourceNode.connect(nativeCtx.destination);
        }
        updateToggleBtn();
    }

    // ==========================================
    // 6. Stream Hook
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
    // 7. Styles
    // ==========================================
    const style = document.createElement('style');
    style.innerHTML = `
        #ai-denoise-overlay {
            position: fixed;
            display: none;
            width: 220px;
            background-color: var(--color-1);
            color: #fff;
            font-family: sans-serif;
            border-radius: 8px;
            z-index: 1500;
            cursor: move;
            user-select: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            border: 1px solid #444;
            font-size: 13px;
        }
        #ai-denoise-header {
            background: rgba(255,255,255,0.1);
            padding: 8px 10px;
            border-bottom: 1px solid #444;
            font-weight: bold;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        #ai-denoise-close {
            cursor: pointer;
            font-weight: bold;
            color: #ccc;
            font-size: 18px;
            line-height: 1;
            padding: 0 4px;
            margin-right: -6px;
        }
        #ai-denoise-close:hover { color: #fff; }
        #ai-denoise-content {
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .ai-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2px;
        }
        .ai-label { color: #aaa; }
        .ai-val   { color: #fff; font-weight: bold; }
        .ai-slider-wrap { display: flex; flex-direction: column; gap: 2px; }
        .ai-slider-labels {
            display: flex;
            justify-content: space-between;
            font-size: 9px;
            color: #555;
            margin-top: 2px;
        }
        #ai-denoise-overlay input[type=range] {
            width: 100%;
            height: 6px;
            cursor: pointer;
            appearance: none;
            -webkit-appearance: none;
            background: linear-gradient(to right, #4da6ff 50%, #333 0%);
            border-radius: 3px;
            outline: none;
            margin: 0;
            display: block;
        }
        #ai-denoise-overlay input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #ffffff;
            cursor: pointer;
            border: 3px solid #4da6ff;
            box-shadow: 0 0 4px rgba(77,166,255,0.9);
        }
        #ai-denoise-overlay input[type=range]::-moz-range-thumb {
            width: 18px;
            height: 18px;
            border: 3px solid #4da6ff;
            border-radius: 50%;
            background: #ffffff;
            cursor: pointer;
        }
        #ai-denoise-onoff-btn {
            width: 100%;
            border: none;
            border-radius: 5px;
            padding: 8px 0;
            cursor: pointer;
            font-weight: bold;
            font-size: 12px;
            font-family: sans-serif;
            transition: all 0.3s;
            background-color: #dc3545;
            color: #fff;
        }
        #ai-denoise-onoff-btn.active {
            background-color: #4da6ff !important;
            color: #000 !important;
        }
        #ai-denoise-reset-btn {
            width: 100%;
            background: transparent;
            color: #555;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 5px 0;
            cursor: pointer;
            font-size: 10px;
            font-family: sans-serif;
            transition: all 0.2s;
        }
        #ai-denoise-reset-btn:hover { color: #fff; border-color: #666; }
        #Denoiser-on-off:hover  { color: var(--color-5); filter: brightness(120%); }
        #Denoiser-on-off.active { background-color: var(--color-2) !important; filter: brightness(120%); }
    `;
    document.head.appendChild(style);

    // ==========================================
    // 8. Overlay HTML
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
                    <span class="ai-label">Gate</span>
                    <span class="ai-val" id="ai-gate-val">50%</span>
                </div>
                <input type="range" id="ai-sl-gate" min="0" max="1" step="0.01" value="0.5">
                <div class="ai-slider-labels"><span>open</span><span>tight</span></div>
            </div>

            <hr style="border:0;border-top:1px solid #333;margin:0;">

            <button id="ai-denoise-reset-btn">&#8635; Reset</button>

        </div>
    `;
    document.body.appendChild(overlay);

    // ==========================================
    // 9. Slider Events
    // ==========================================
    function bindSlider(id, key, valId) {
        const sl  = document.getElementById(id);
        const val = document.getElementById(valId);
        sl.addEventListener('input', () => {
            const v   = parseFloat(sl.value);
            S[key]    = v;
            const pct = Math.round(v * 100);
            val.innerText = pct + '%';
            sl.style.background =
                `linear-gradient(to right,#4da6ff ${pct}%,#333 0%)`;
            calcParams();
            applyEQ();
        });
    }
    bindSlider('ai-sl-strength', 'strength', 'ai-strength-val');
    bindSlider('ai-sl-voice',    'voice',    'ai-voice-val');
    bindSlider('ai-sl-gate',     'gate',     'ai-gate-val');

    // On/Off button inside overlay
    document.getElementById('ai-denoise-onoff-btn').addEventListener('click', () => {
        if (!rnnoiseObj) { alert('AI is still loading, please wait.'); return; }
        isEnabled = !isEnabled;
        toggleDSP();
    });

    // Reset button
    document.getElementById('ai-denoise-reset-btn').addEventListener('click', () => {
        Object.assign(S, DEFAULTS);
        [['ai-sl-strength','strength','ai-strength-val'],
         ['ai-sl-voice',   'voice',   'ai-voice-val'],
         ['ai-sl-gate',    'gate',    'ai-gate-val']
        ].forEach(([slId, key, valId]) => {
            const sl  = document.getElementById(slId);
            const val = document.getElementById(valId);
            const pct = Math.round(S[key] * 100);
            sl.value      = S[key];
            val.innerText = pct + '%';
            sl.style.background =
                `linear-gradient(to right,#4da6ff ${pct}%,#333 0%)`;
        });
        calcParams(); applyEQ();
    });

    // Close button
    document.getElementById('ai-denoise-close').addEventListener('click', () => {
        $('#ai-denoise-overlay').stop(true, true).fadeOut(400);
        const btn = document.getElementById('Denoiser-on-off');
        if (btn) btn.classList.remove('active');
    });

    // ==========================================
    // 10. Toggle Button State
    // ==========================================
    function updateToggleBtn(error) {
        const btn = document.getElementById('ai-denoise-onoff-btn');
        if (!btn) return;
        if (error) {
            btn.textContent = 'Error loading AI';
            btn.style.backgroundColor = '#ff4444';
            btn.style.color = '#fff';
            btn.classList.remove('active');
            return;
        }
        if (!rnnoiseObj) {
            btn.textContent = 'Loading AI...';
            btn.style.backgroundColor = '#ffc107';
            btn.style.color = '#000';
            btn.classList.remove('active');
            return;
        }
        if (isEnabled) {
            btn.textContent = 'AI ON';
            btn.classList.add('active');
        } else {
            btn.textContent = 'AI OFF';
            btn.classList.remove('active');
        }
    }

    // ==========================================
    // 11. Position + Drag
    // ==========================================
    overlay.style.left = localStorage.getItem('denoiserLeft') || '20px';
    overlay.style.top  = localStorage.getItem('denoiserTop')  || '240px';

    (function () {
        let dragging = false, sx, sy, ox, oy;
        overlay.addEventListener('mousedown', e => {
            if (e.target.id === 'ai-denoise-close'     ||
                e.target.id === 'ai-denoise-onoff-btn' ||
                e.target.id === 'ai-denoise-reset-btn' ||
                e.target.tagName === 'INPUT') return;
            dragging = true;
            sx = e.clientX; sy = e.clientY;
            const r = overlay.getBoundingClientRect();
            ox = r.left; oy = r.top;
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
    // 12. Toolbar Button
    // ==========================================
    (function () {
        const btnId = 'Denoiser-on-off';
        let found = false;
        const obs = new MutationObserver((_, o) => {
            if (typeof addIconToPluginPanel === 'function') {
                found = true; o.disconnect();
                // 'waveform-lines' as AI/audio icon
                addIconToPluginPanel(btnId, 'Denoiser', 'solid', 'waveform-lines',
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
    // 13. Start
    // ==========================================
    loadRNNoise();
    setInterval(checkAndHookAudio, 1000);

})();