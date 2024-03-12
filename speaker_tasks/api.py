from fastapi import FastAPI, UploadFile
import uvicorn
from sd_msdd_inference import infer_msdd
from pydub import AudioSegment
import os
import traceback
from rttm_to_spk_segments import get_speaker_segments

app = FastAPI()


@app.post("/diarization_infer/")
async def diar_infer(file: UploadFile):
    try:
        filename = file.filename.split('/')[-1]
        file_path = f"/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/data/{filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1)
        audio.export(file_path, format='wav')
        
        infer_msdd(file_path)
        file_name = filename.split('.')[0]
        rttm_file_path = f'/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/oracle_vad/pred_rttms/{file_name}.rttm'
        
        speaker_segments = get_speaker_segments(rttm_file_path)
        print(speaker_segments)
        
        os.remove(file_path)
        os.remove(rttm_file_path)
        
        return speaker_segments
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        with open("error_traceback.txt", "a") as f:
            f.write(error_traceback)
            
        return []


if __name__ == "__main__":
    
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
