# import streamlit as st
# import torch
# import torchaudio
# from pathlib import Path
# from audioanalyzer import AudioAnalyzer, AnalyzerConfig
# import concurrent.futures
# import tempfile
# import os

# DEFAULT_CONFIG = AnalyzerConfig(
#     vad_threshold=0.3,
#     min_silence_duration_ms=50,
#     min_speech_duration_ms=100,
#     energy_threshold_db=-60
# )


import streamlit as st
import warnings
import torch
import torchaudio
from pathlib import Path
from audioanalyzer import AudioAnalyzer, AnalyzerConfig
import concurrent.futures
import tempfile
import os

# Filter out torch path warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')

DEFAULT_CONFIG = AnalyzerConfig(
    vad_threshold=0.3,
    min_silence_duration_ms=50,
    min_speech_duration_ms=100,
    energy_threshold_db=-60
)

def analyze_file_wrapper(file_path):
    analyzer = AudioAnalyzer(DEFAULT_CONFIG)
    return analyzer.analyze_file(str(file_path))

def round_value(value, places=5):
    if isinstance(value, str):
        try:
            # Extract numeric value and unit
            import re
            match = re.match(r'(-?\d*\.?\d+)\s*([a-zA-Z%]*)', value)
            if match:
                num, unit = match.groups()
                rounded = round(float(num), places)
                return f"{rounded}{unit}"
        except:
            return value
    return value

def display_analysis_results(results):
    tabs = st.tabs(["Overview", "Signal Analysis", "Silence Analysis", "Frequency Analysis", "Amplitude Analysis"])
    
    with tabs[0]:
        st.subheader("File Information")
        cols = st.columns(4)
        cols[0].metric("Duration", round_value(results['audio_duration']))
        cols[1].metric("Sample Rate", round_value(results['sample_rate']))
        cols[2].metric("Signal Power", round_value(results['signal_power']))
        cols[3].metric("Noise Power", round_value(results['noise_power']))
    
    with tabs[1]:
        st.subheader("Signal-to-Noise Ratio Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall SNR", round_value(results['snr_analysis']['overall']))
        col2.metric("Signal Power", round_value(results['signal_power']))
        col3.metric("Noise Power", round_value(results['noise_power']))
    
    with tabs[2]:
        st.subheader("Silence Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", round_value(results['silence_analysis']['duration']))
        col2.metric("Percentage", round_value(results['silence_analysis']['percentage']))
        col3.metric("Mean Energy", round_value(results['silence_analysis']['mean_energy']))
    
    with tabs[3]:
        st.subheader("Frequency Analysis")
        st.metric("Spectral Centroid", round_value(results['frequency_analysis']['spectral_centroid']))
        
        cols = st.columns(5)
        band_energies = results['frequency_analysis']['band_energies']
        for i, (band, energy) in enumerate(band_energies.items()):
            cols[i].metric(band.replace('_', ' ').title(), round_value(energy))
    
    with tabs[4]:
        st.subheader("Amplitude Analysis")
        col1, col2 = st.columns(2)
        metrics = results['amplitude_analysis']
        for i, (key, value) in enumerate(metrics.items()):
            (col1 if i < 2 else col2).metric(key.replace('_', ' ').title(), round_value(value))

def main():
    st.set_page_config(page_title="Audio Analyzer", layout="wide")
    st.title("Audio Analysis Dashboard")
    
    uploaded_files = st.file_uploader(
        "Upload Audio Files", 
        type=['wav', 'mp3', 'flac'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        temp_dir = tempfile.mkdtemp()
        temp_paths = []
        
        for uploaded_file in uploaded_files:
            temp_path = Path(temp_dir) / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            temp_paths.append(temp_path)
        
        if st.button("Analyze Files", type="primary"):
            with st.spinner("Analyzing audio files..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_file = {
                        executor.submit(analyze_file_wrapper, path): path 
                        for path in temp_paths
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            results = future.result()
                            with st.expander(f"Results for {file_path.name}", expanded=True):
                                display_analysis_results(results)
                        except Exception as e:
                            st.error(f"Error analyzing {file_path.name}: {str(e)}")
            
            for path in temp_paths:
                try:
                    os.remove(path)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

if __name__ == "__main__":
    main()