import json 

# rttm_file = '/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/oracle_vad/crowd_test_rttms/45a4c830-ee95-4713-ba3b-70e0ba70668e.rttm'


def get_speaker_segments(rttm_file):
    
    speaker_segments = {}
    
    with open(rttm_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            speaker = parts[7]
            
            start_time = float(parts[3])
            end_time = start_time + float(parts[4])
            
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
                # speaker_segments.update({speaker:[]})
            speaker_segments[speaker].append((start_time, end_time))
            
    return speaker_segments
        
# print(speaker_segments)
if __name__ == "__main__":
    result = get_speaker_segments('/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/oracle_vad/pred_rttms/-ahockJS-cA.rttm')
    print(result)
# json_data = json.dumps(speaker_segments, indent=4)
# print(json.dumps(speaker_segments, indent=4))
# with open('speaker_segments.json', 'w', encoding='utf-8') as json_file:
#     json.dump(speaker_segments, json_file, indent=4, ensure_ascii=False, sort_keys=True)
        
        