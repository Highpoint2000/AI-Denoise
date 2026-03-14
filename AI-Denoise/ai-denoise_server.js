///////////////////////////////////////////////////////////////
//                                                           //
//  AI DENOISER SERVER PLUGIN FOR FM-DX-WEBSERVER (V1.1)     //
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

const fs   = require('fs');
const path = require('path');
const { logInfo, logError } = require('./../../server/console');

function copyAllClientFiles() {

    const srcDir  = __dirname;
    const destDir = path.join(__dirname, '..', '..', 'web', 'plugins', 'AI-Denoise');

    logInfo('[AI-Denoise] Deploying plugin files to web directory:', destDir);

    try {
        fs.mkdirSync(destDir, { recursive: true });
        fs.chmodSync(destDir, 0o775);
    } catch (e) {
        logError('[AI-Denoise] Failed to create destination directory:', e.message);
        return;
    }

    // Copy sub-directories: DTLN, rnnoise, DeepFilterNet
    const subdirectories = ['DTLN', 'rnnoise', 'DeepFilterNet'];

    subdirectories.forEach((folder) => {
        const folderSrc  = path.join(srcDir,  folder);
        const folderDest = path.join(destDir, folder);

        if (!fs.existsSync(folderSrc)) return;

        try {
            fs.mkdirSync(folderDest, { recursive: true });
            fs.chmodSync(folderDest, 0o775);
        } catch (err) {
            logError(`[AI-Denoise] Failed to create subdirectory ${folderDest}:`, err.message);
            return;
        }

        const files = fs.readdirSync(folderSrc);
        files.forEach((file) => {
            const s = path.join(folderSrc,  file);
            const d = path.join(folderDest, file);
            try {
                fs.copyFileSync(s, d);
                fs.chmodSync(d, 0o664);
            } catch (err) {
                logError(`[AI-Denoise] Error copying ${folder}/${file}:`, err.message);
            }
        });

        logInfo(`[AI-Denoise] Deployed subdirectory: ${folder}/`);
    });

    // Copy root plugin files
    const rootFiles = ['ai-denoise.js'];

    rootFiles.forEach((file) => {
        const s = path.join(srcDir,  file);
        const d = path.join(destDir, file);
        if (!fs.existsSync(s)) return;
        try {
            fs.copyFileSync(s, d);
            fs.chmodSync(d, 0o664);
            logInfo(`[AI-Denoise] Deployed: ${file}`);
        } catch (err) {
            logError(`[AI-Denoise] Failed to copy ${file}:`, err.message);
        }
    });

    logInfo('[AI-Denoise] Web deployment finished.');
}

copyAllClientFiles();