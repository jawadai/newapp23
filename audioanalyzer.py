import os
import json
import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Union, Optional
from scipy import signal

@dataclass
class AnalyzerConfig:
    """Configuration for audio analysis parameters"""
    # VAD parameters
    vad_threshold: float = 0.3
    min_silence_duration_ms: int = 50
    min_speech_duration_ms: int = 100
    
    # Audio processing parameters
    sampling_rate: int = 16000
    frame_length_ms: int = 30
    energy_threshold_db: float = -60
    window_size_ms: int = 500
    noise_percentile: float = 0.4
    
    # Frequency analysis parameters
    nfft: int = 2048
    hop_length_ms: int = 10
    freq_bands: List[tuple] = None
    
    # Device selection
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Initialize default frequency bands if none provided"""
        if self.freq_bands is None:
            self.freq_bands = [
                (20, 300),     # Low frequency
                (300, 1000),   # Low-mid frequency
                (1000, 3000),  # Mid frequency
                (3000, 8000),  # High-mid frequency
                (8000, 20000)  # High frequency
            ]

class AudioAnalyzer:
    """
    Comprehensive audio analysis tool with capabilities for:
    - Voice Activity Detection (VAD)
    - Signal-to-Noise Ratio (SNR) calculation
    - Silence analysis
    - Frequency spectrum analysis
    - Amplitude and dynamics analysis
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """Initialize analyzer with configuration"""
        self.config = config or AnalyzerConfig()
        self.logger = self._setup_logger()
        self.model, self.utils = self._load_silero_vad()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger('AudioAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        return logger
    
    def _load_silero_vad(self):
        """Load and initialize Silero VAD model"""
        self.logger.info("Loading Silero VAD model...")
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=True
            )
            model = model.to(self.config.device)
            return model, utils
        except Exception as e:
            self.logger.error(f"Failed to load Silero VAD model: {str(e)}")
            raise
    
    def _safe_db_conversion(
        self,
        value: float,
        floor_db: float = -120.0
    ) -> Union[float, str]:
        """Convert power ratio to dB with safety checks"""
        try:
            if value <= 0:
                return floor_db
            return max(20 * np.log10(value), floor_db)
        except:
            return "N/A"
    
    def _safe_mean(self, tensor: torch.Tensor) -> Union[float, str]:
        """Calculate mean with safety checks"""
        try:
            if tensor.numel() == 0:
                return "N/A"
            return float(torch.mean(tensor).item())
        except:
            return "N/A"
    
    def _calculate_snr(
        self,
        speech_audio: torch.Tensor,
        non_speech_audio: torch.Tensor
    ) -> Dict[str, Union[float, str]]:
        """Calculate Signal-to-Noise Ratio and related metrics"""
        try:
            if len(speech_audio) == 0 or len(non_speech_audio) == 0:
                # Fallback to energy-based segmentation
                audio = torch.cat([speech_audio, non_speech_audio])
                window_size = int(self.config.window_size_ms * 
                                self.config.sampling_rate / 1000)
                hop_size = window_size // 2
                
                # Create overlapping frames
                frames = audio.unfold(-1, window_size, hop_size)
                frame_energies = torch.mean(frames**2, dim=1)
                
                # Separate speech and noise based on energy threshold
                threshold = torch.quantile(frame_energies, 
                                         self.config.noise_percentile)
                speech_frames = frames[frame_energies > threshold]
                noise_frames = frames[frame_energies <= threshold]
                
                speech_audio = speech_frames.reshape(-1)
                non_speech_audio = noise_frames.reshape(-1)

            # Calculate power metrics
            speech_power = self._safe_mean(speech_audio**2)
            noise_power = self._safe_mean(non_speech_audio**2)

            if isinstance(speech_power, str) or isinstance(noise_power, str):
                return {
                    "snr": "N/A",
                    "signal_power": "N/A",
                    "noise_power": "N/A"
                }

            # Calculate SNR
            if noise_power > 0 and speech_power > 0:
                snr = 10 * np.log10(speech_power / noise_power)
                return {
                    "snr": float(snr),
                    "signal_power": float(speech_power),
                    "noise_power": float(noise_power)
                }
            
            return {
                "snr": "N/A",
                "signal_power": float(speech_power),
                "noise_power": float(noise_power)
            }
            
        except Exception as e:
            self.logger.error(f"SNR calculation failed: {str(e)}")
            return {
                "snr": "N/A",
                "signal_power": "N/A",
                "noise_power": "N/A"
            }

    def _analyze_silence(
        self,
        speech_audio: torch.Tensor,
        non_speech_audio: torch.Tensor
    ) -> Dict[str, Union[float, str]]:
        """Analyze silence segments and their characteristics"""
        try:
            if non_speech_audio.numel() == 0:
                return {
                    "silence_duration": 0.0,
                    "silence_percentage": 0.0,
                    "mean_silence_energy_db": "N/A"
                }

            # Create analysis frames
            frame_length = int(self.config.frame_length_ms * 
                             self.config.sampling_rate / 1000)
            frames = non_speech_audio.unfold(-1, frame_length, frame_length // 2)

            if frames.numel() == 0:
                return {
                    "silence_duration": 0.0,
                    "silence_percentage": 0.0,
                    "mean_silence_energy_db": "N/A"
                }

            # Calculate frame energies
            frame_energies = torch.mean(frames**2, dim=1)
            energy_db = 10 * torch.log10(frame_energies + 1e-10)
            silence_mask = energy_db < self.config.energy_threshold_db

            if not torch.any(silence_mask):
                return {
                    "silence_duration": 0.0,
                    "silence_percentage": 0.0,
                    "mean_silence_energy_db": "N/A"
                }

            audio = torch.cat([speech_audio, non_speech_audio])
            # Calculate silence metrics
            total_duration = len(audio) / self.config.sampling_rate
            effective_frame_step = frame_length // 2
            silence_duration = float(torch.sum(silence_mask).item() * 
                                   effective_frame_step / self.config.sampling_rate)
            silence_percentage = (silence_duration / total_duration * 100 
                                if total_duration > 0 else 0)
            silence_energy = self._safe_mean(energy_db[silence_mask])

            return {
                "silence_duration": silence_duration,
                "silence_percentage": silence_percentage,
                "mean_silence_energy_db": silence_energy
            }
            
        except Exception as e:
            self.logger.error(f"Silence analysis failed: {str(e)}")
            return {
                "silence_duration": 0.0,
                "silence_percentage": 0.0,
                "mean_silence_energy_db": "N/A"
            }

    # def _analyze_frequency(
    #     self,
    #     audio: torch.Tensor
    # ) -> Dict[str, Union[Dict[str, float], str]]:
    #     """Analyze frequency content and spectral characteristics"""
    #     try:
    #         audio_np = audio.numpy()
            
    #         if len(audio_np) == 0:
    #             return {
    #                 "band_energies": {
    #                     band: "N/A" for band in [
    #                         'low', 'low_mid', 'mid', 'high_mid', 'high'
    #                     ]
    #                 },
    #                 "spectral_centroid": "N/A"
    #             }

    #         # Calculate power spectral density
    #         freqs, psd = signal.welch(
    #             audio_np,
    #             fs=self.config.sampling_rate,
    #             nperseg=self.config.nfft,
    #             scaling='spectrum'
    #         )
            
    #         if np.sum(psd) == 0:
    #             return {
    #                 "band_energies": {
    #                     band: "N/A" for band in [
    #                         'low', 'low_mid', 'mid', 'high_mid', 'high'
    #                     ]
    #                 },
    #                 "spectral_centroid": "N/A"
    #             }

    #         # Convert to dB scale
    #         psd_db = 10 * np.log10(psd / np.max(psd) + 1e-10)
            
    #         # Calculate band energies
    #         band_energies = {}
    #         for band_name, (low_freq, high_freq) in zip(
    #             ['low', 'low_mid', 'mid', 'high_mid', 'high'],
    #             self.config.freq_bands
    #         ):
    #             band_mask = (freqs >= low_freq) & (freqs <= high_freq)
    #             if np.any(band_mask):
    #                 band_energy = float(np.mean(psd[band_mask]))
    #                 band_energy_db = 10 * np.log10(
    #                     band_energy / np.sum(psd) + 1e-10
    #                 )
    #                 band_energies[f"{band_name}_band"] = band_energy_db
    #             else:
    #                 band_energies[f"{band_name}_band"] = "N/A"

    #         # Calculate spectral centroid
    #         spectral_centroid = (float(np.sum(freqs * psd) / np.sum(psd))
    #                            if np.sum(psd) > 0 else "N/A")

    #         return {
    #             "band_energies": band_energies,
    #             "spectral_centroid": spectral_centroid
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"Frequency analysis failed: {str(e)}")
    #         return {
    #             "band_energies": {
    #                 band: "N/A" for band in [
    #                     'low', 'low_mid', 'mid', 'high_mid', 'high'
    #                 ]
    #             },
    #             "spectral_centroid": "N/A"
    #         }

    # def _analyze_amplitude(
    #     self,
    #     audio: torch.Tensor
    # ) -> Dict[str, Union[float, str]]:
    #     """Analyze amplitude and dynamic characteristics"""
    #     try:
    #         audio_np = audio.numpy()
            
    #         if len(audio_np) == 0:
    #             return {
    #                 "rms_amplitude": "N/A",
    #                 "peak_amplitude": "N/A",
    #                 "crest_factor": "N/A",
    #                 "dynamic_range_db": "N/A"
    #             }

    #         # Calculate basic amplitude metrics
    #         rms = float(np.sqrt(np.mean(audio_np**2)))
    #         peak = float(np.max(np.abs(audio_np)))
            
    #         # Calculate crest factor
    #         crest_factor = peak / rms if rms > 0 else "N/A"

    #         # Calculate dynamic range
    #         non_zero_samples = audio_np[audio_np != 0]
    #         dynamic_range = (
    #             self._safe_db_conversion(peak / np.mean(np.abs(non_zero_samples)))
    #             if len(non_zero_samples) > 0 else "N/A"
    #         )

    #         return {
    #             "rms_amplitude": rms,
    #             "peak_amplitude": peak,
    #             "crest_factor": crest_factor,
    #             "dynamic_range_db": dynamic_range
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"Amplitude analysis failed: {str(e)}")
    #         return {
    #             "rms_amplitude": "N/A",
    #             "peak_amplitude": "N/A",
    #             "crest_factor": "N/A",
    #             "dynamic_range_db": "N/A"
    #         }




    # def _analyze_frequency(self, audio: torch.Tensor) -> Dict[str, Union[Dict[str, float], str]]:
    #     try:
    #         audio_np = audio.numpy()
            
    #         if len(audio_np) == 0:
    #             return self._get_empty_frequency_result()

    #         # Calculate power spectral density with improved parameters
    #         freqs, psd = signal.welch(
    #             audio_np,
    #             fs=self.config.sampling_rate,
    #             nperseg=self.config.nfft,
    #             scaling='spectrum',
    #             window='hann'
    #         )
            
    #         # Ensure positive values and normalize
    #         psd = np.abs(psd) + 1e-10
    #         total_energy = np.sum(psd)
            
    #         if total_energy == 0:
    #             return self._get_empty_frequency_result()

    #         # Calculate band energies with improved method
    #         band_energies = {}
    #         for band_name, (low_freq, high_freq) in zip(
    #             ['low', 'low_mid', 'mid', 'high_mid', 'high'],
    #             self.config.freq_bands
    #         ):
    #             band_mask = (freqs >= low_freq) & (freqs <= high_freq)
    #             if np.any(band_mask):
    #                 band_energy = np.sum(psd[band_mask])
    #                 band_energy_db = 10 * np.log10(band_energy / total_energy)
    #                 band_energies[f"{band_name}_band"] = band_energy_db

    #         # Calculate spectral centroid with improved stability
    #         spectral_centroid = float(np.sum(freqs * psd) / total_energy)

    #         return {
    #             "band_energies": band_energies,
    #             "spectral_centroid": spectral_centroid
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"Frequency analysis failed: {str(e)}")
    #         return self._get_empty_frequency_result()


    def _analyze_frequency(self, audio: torch.Tensor) -> Dict[str, Union[Dict[str, float], str]]:
        try:
            audio_np = audio.numpy()
            
            if len(audio_np) == 0:
                return self._get_empty_frequency_result()

            # Normalize audio
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-10)
            
            # Calculate power spectral density with improved parameters
            freqs, psd = signal.welch(
                audio_np,
                fs=self.config.sampling_rate,
                nperseg=min(self.config.nfft, len(audio_np)),
                noverlap=self.config.nfft // 2,
                scaling='density',
                detrend='constant',
                window='hamming'
            )
            
            # Ensure positive values and normalize
            psd = np.abs(psd)
            psd_smoothed = signal.savgol_filter(psd, 5, 2)  # Smooth the spectrum
            total_energy = np.sum(psd_smoothed)
            
            if total_energy < 1e-10:
                return self._get_empty_frequency_result()

            # Calculate band energies with reference level
            reference_level = np.max(psd_smoothed)
            band_energies = {}
            
            for band_name, (low_freq, high_freq) in zip(
                ['low', 'low_mid', 'mid', 'high_mid', 'high'],
                self.config.freq_bands
            ):
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(band_mask):
                    band_energy = np.sum(psd_smoothed[band_mask])
                    if band_energy > 0:
                        band_energy_db = 10 * np.log10(band_energy / reference_level)
                        band_energies[f"{band_name}_band"] = float(band_energy_db)
                    else:
                        band_energies[f"{band_name}_band"] = -60.0  # Minimum threshold
                else:
                    band_energies[f"{band_name}_band"] = -60.0

            # Calculate spectral centroid
            if np.sum(psd_smoothed) > 0:
                spectral_centroid = float(np.sum(freqs * psd_smoothed) / np.sum(psd_smoothed))
            else:
                spectral_centroid = 0.0

            return {
                "band_energies": band_energies,
                "spectral_centroid": spectral_centroid
            }
            
        except Exception as e:
            self.logger.error(f"Frequency analysis failed: {str(e)}")
            return self._get_empty_frequency_result()

    def _get_empty_frequency_result(self):
        return {
            "band_energies": {
                f"{band}_band": "N/A" 
                for band in ['low', 'low_mid', 'mid', 'high_mid', 'high']
            },
            "spectral_centroid": "N/A"
        }

    def _analyze_amplitude(self, audio: torch.Tensor) -> Dict[str, Union[float, str]]:
        try:
            audio_np = audio.numpy()
            
            if len(audio_np) == 0:
                return self._get_empty_amplitude_result()

            # Calculate RMS with improved stability
            rms = np.sqrt(np.mean(np.square(audio_np)) + 1e-10)
            peak = np.max(np.abs(audio_np))

            if rms < 1e-6 or peak < 1e-6:
                return self._get_empty_amplitude_result()

            # Calculate metrics with improved precision
            crest_factor = peak / rms
            
            # Use percentile-based dynamic range
            non_zero_mask = np.abs(audio_np) > 1e-6
            if np.any(non_zero_mask):
                floor = np.percentile(np.abs(audio_np[non_zero_mask]), 5)
                dynamic_range = 20 * np.log10(peak / floor)
            else:
                dynamic_range = 0

            return {
                "rms_amplitude": float(rms),
                "peak_amplitude": float(peak),
                "crest_factor": float(crest_factor),
                "dynamic_range_db": float(dynamic_range)
            }
            
        except Exception as e:
            self.logger.error(f"Amplitude analysis failed: {str(e)}")
            return self._get_empty_amplitude_result()

    def _get_empty_amplitude_result(self):
        return {
            "rms_amplitude": "N/A",
            "peak_amplitude": "N/A",
            "crest_factor": "N/A",
            "dynamic_range_db": "N/A"
        }

    




    

    def analyze_file(self, file_path: str) -> Dict:


        
        """
        Perform comprehensive analysis of an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            self.logger.info(f"Analyzing file: {file_path}")


            try:
                audio, sampling_rate = torchaudio.load(
                    file_path,
                    backend="soundfile"  # Try soundfile backend first
                )
            except:
                try:
                    # Fallback to sox_io backend
                    audio, sampling_rate = torchaudio.load(
                        file_path,
                        backend="sox_io"
                    )
                except:
                    # Final fallback to default backend
                    audio, sampling_rate = torchaudio.load(file_path)

            if audio.shape[0] > 1:  # Convert stereo to mono
                audio = torch.mean(audio, dim=0, keepdim=True)


            # Resample if necessary
            if sampling_rate != self.config.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    sampling_rate,
                    self.config.sampling_rate
                )
                audio = resampler(audio)
                sampling_rate = self.config.sampling_rate

            # Perform VAD analysis
            speech_timestamps = self.utils[0](
                audio,
                self.model,
                sampling_rate=sampling_rate,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                threshold=self.config.vad_threshold
            )

            # Create speech/non-speech masks
            speech_mask = torch.zeros(audio.shape[1], dtype=torch.bool)
            for segment in speech_timestamps:
                speech_mask[segment['start']:segment['end']] = True

            speech_audio = audio.squeeze()[speech_mask]
            non_speech_audio = audio.squeeze()[~speech_mask]

            # Perform all analyses
            snr_results = self._calculate_snr(speech_audio, non_speech_audio)
            silence_stats = self._analyze_silence(speech_audio, non_speech_audio)
            freq_analysis = self._analyze_frequency(audio.squeeze())
            amp_analysis = self._analyze_amplitude(audio.squeeze())

            # Format results
            results = {
                "filename": Path(file_path).name,
                "analysis_timestamp": datetime.now().isoformat(),
                "audio_duration": f"{round(float(audio.shape[1] / sampling_rate), 2)}s",
                "sample_rate": f"{sampling_rate}Hz",
                "signal_power": (
                    f"{self._safe_db_conversion(snr_results['signal_power'])}dB"
                    if isinstance(snr_results['signal_power'], (int, float))
                    else "N/A"
                ),
                "noise_power": (
                    f"{self._safe_db_conversion(snr_results['noise_power'])}dB"
                    if isinstance(snr_results['noise_power'], (int, float))
                    else "N/A"
                ),
                "snr_analysis": {
                    "overall": (
                        f"{round(float(snr_results['snr']), 1)}dB"
                        if isinstance(snr_results['snr'], (int, float))
                        else "N/A"
                    )
                },
                "silence_analysis": {
                    "duration": f"{round(silence_stats['silence_duration'], 2)}s",
                    "percentage": f"{round(silence_stats['silence_percentage'], 1)}%",
                    "mean_energy": (
                        f"{round(float(silence_stats['mean_silence_energy_db']), 1)}dB"
                        if isinstance(silence_stats['mean_silence_energy_db'], (int, float))
                        else "N/A"
                    )
                },
                "frequency_analysis": {
                    "band_energies": {
                        k: f"{v:.1f}dB" if isinstance(v, (int, float)) else "N/A"
                        for k, v in freq_analysis["band_energies"].items()
                    },
                    "spectral_centroid": (
                        f"{freq_analysis['spectral_centroid']:.1f}Hz"
                        if isinstance(freq_analysis['spectral_centroid'], (int, float))
                        else "N/A"
                    )
                },
                "amplitude_analysis": {
                    "rms_level": (
                        f"{self._safe_db_conversion(amp_analysis['rms_amplitude']):.1f}dB"
                        if isinstance(amp_analysis['rms_amplitude'], (int, float))
                        else "N/A"
                    ),
                    "peak_level": (
                        f"{self._safe_db_conversion(amp_analysis['peak_amplitude']):.1f}dB"
                        if isinstance(amp_analysis['peak_amplitude'], (int, float))
                        else "N/A"
                    ),
                    "crest_factor": (
                        f"{amp_analysis['crest_factor']:.1f}"
                        if isinstance(amp_analysis['crest_factor'], (int, float))
                        else "N/A"
                    ),
                    "dynamic_range": (
                        f"{amp_analysis['dynamic_range_db']:.1f}dB"
                        if isinstance(amp_analysis['dynamic_range_db'], (int, float))
                        else "N/A"
                    )
                }
            }
            return results

        except Exception as e:
            self.logger.error(f"Analysis failed for {file_path}: {str(e)}")
            return {
                "filename": Path(file_path).name,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
                "snr_analysis": {"overall": "N/A"},
                "silence_analysis": {
                    "duration": "0.0s",
                    "percentage": "0.0%",
                    "mean_energy": "N/A"
                },
                "frequency_analysis": {
                    "band_energies": {
                        "low_band": "N/A",
                        "low_mid_band": "N/A",
                        "mid_band": "N/A",
                        "high_mid_band": "N/A",
                        "high_band": "N/A"
                    },
                    "spectral_centroid": "N/A"
                },
                "amplitude_analysis": {
                    "rms_level": "N/A",
                    "peak_level": "N/A",
                    "crest_factor": "N/A",
                    "dynamic_range": "N/A"
                }
            }

    def process_directory(
        self,
        input_dir: str,
        output_json: str = 'analysis_results.json',
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Process all audio files in a directory and save results to JSON.
        
        Args:
            input_dir: Directory containing audio files
            output_json: Path to save JSON results
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing analysis results for all files
        """
        try:
            input_path = Path(input_dir)
            self.logger.info(f"Processing directory: {input_path}")

            # Find all supported audio files
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac']:
                audio_files.extend(list(input_path.rglob(f"*{ext}")))

            if not audio_files:
                self.logger.warning(f"No audio files found in {input_path}")
                return {}

            self.logger.info(f"Found {len(audio_files)} audio files")
            results = {}

            # Process each file
            for i, file_path in enumerate(audio_files, 1):
                self.logger.info(f"Processing file {i}/{len(audio_files)}: {file_path}")
                
                try:
                    results[str(file_path)] = self.analyze_file(str(file_path))
                    
                    # Update progress if callback provided
                    if progress_callback:
                        progress_callback(i / len(audio_files))
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {str(e)}")
                    results[str(file_path)] = {
                        "filename": file_path.name,
                        "error": str(e),
                        "analysis_timestamp": datetime.now().isoformat()
                    }

            # Save results to JSON
            if output_json:
                with open(output_json, 'w') as f:
                    json.dump(results, f, indent=4)
                self.logger.info(f"Results saved to {output_json}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Directory processing failed: {str(e)}")
            raise

    def __str__(self) -> str:
        """String representation of the analyzer configuration"""
        return (
            f"AudioAnalyzer(sampling_rate={self.config.sampling_rate}Hz, "
            f"vad_threshold={self.config.vad_threshold}, "
            f"device={self.config.device})"
        )