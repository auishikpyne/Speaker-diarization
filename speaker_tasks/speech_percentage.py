from pydub import AudioSegment
import glob
from tqdm import tqdm

output_file = open('/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/speech_percentage.txt', 'w', encoding='UTF-8')
output_file.write("File path\tTotal duration\tSpeech duration sum\tspeech_percentage\n")

def write_speech_stats_to_file(file_path, total_duration, duration_sum, speech_percentage):
    output_file.write(f"{file_path}\t{total_duration}\t{duration_sum}\t{speech_percentage}\n")
        


def calculate_speech_percentage(rttm_file_path, audio_duration):
    speech_duration_sum = 0.0
    
    with open(rttm_file_path, 'r') as rttm_file:
        for line in rttm_file:
            parts = line.strip().split()
            if parts[0] == 'SPEAKER':
                start_time = float(parts[3])
                duration = float(parts[4])
                speech_duration_sum += duration
                
    speech_percentage = (speech_duration_sum / audio_duration) * 100
    
    return speech_duration_sum, speech_percentage

            
            
if __name__ == "__main__":
    rttm_files = glob.glob('/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/oracle_vad/youtube_rttms/*')
    for rttm_file in tqdm(rttm_files):
        
        rttm_file_path = rttm_file
        file_id = (rttm_file_path.split('/')[-1]).split('.')[0]
        audio_file_path = '/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/data/youtube_data/' + file_id + '.flac'
        audio_duration = AudioSegment.from_file(audio_file_path).duration_seconds
        
        duration_sum, speech_percentage = calculate_speech_percentage(rttm_file_path, audio_duration)
        
        write_speech_stats_to_file( audio_file_path, audio_duration, duration_sum, speech_percentage)