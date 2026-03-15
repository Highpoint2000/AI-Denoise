# AI-Denoise
AI-based models and a real-time DSP chain to remove background noise for FM-DX Webserver

<img width="1511" height="856" alt="Image" src="https://github.com/user-attachments/assets/850ae6b0-712f-4023-b842-24fd3ea074c3" />

## Version 1.1

- Added alternate AI model: DTLN via ONNX Runtime Web
- Option to switch between the two AI models (DTLN <> RNNoise)
- Minor optimizations

## Installation notes:

1. [Download](https://github.com/Highpoint2000/AI-Denoise/releases) the last repository as a zip
2. Unpack all files from the plugins folder to ..fm-dx-webserver-main\plugins\ 
3. Stop or close the fm-dx-webserver
4. Start/Restart the fm-dx-webserver with "npm run webserver" on node.js console, check the console informations
5. Activate the sysinfo plugin in the settings
6. Stop or close the fm-dx-webserver
7. Start/Restart the fm-dx-webserver with "npm run webserver" on node.js console, check the console informations 
8. Reload the browser

## How to use:     
                                         
- Click the AI toolbar button to open the control panel 
- Choose your preferred AI model 
- Start audio playback first, then click AI ON          
- Strength: controls denoising intensity                
- Tone: warm (left) to bright (right) sound character    
- Gate (only RNNoise AI): open (subtle) to tight (aggressive) gating  
- Gain (only DTLN AI): Adjust the volume     
- Red toolbar icon = AI denoiser is active               
- Panel position and settings are saved automatically

## Notes: 

- This filter is only suitable for noise reduction in speech
- When used correctly, it's possible to filter out intelligible speech even from extremely noisy environments
- The simultaneous use of the cEQ and iMS filters must be tested individually. It can improve or worsen the result

## Contact

If you have any questions, would like to report problems, or have suggestions for improvement, please feel free to contact me! You can reach me by email at highpoint2000@googlemail.com. I look forward to hearing from you!

<a href="https://www.buymeacoffee.com/Highpoint" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

<details>
<summary>History</summary>

### Version 1.0

- RNNoise WASM neural network denoiser (VAD + soft-gate)
- internal automatic 6-band EQ, high-pass filter & brickwall limiter
- Strength / Tone / Gate controls with persistent state
- Non-destructive routing (MetricsMonitor compatible)
- WASM & settings cached in localStorage
- Uses RNNoise WASM by Shiguredo (MIT License) https://github.com/shiguredo/rnnoise-wasm  
