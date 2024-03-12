import json
import os
from omegaconf import OmegaConf
import glob
from tqdm import tqdm

def infer_msdd(file):

    data_dir = '/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/data'
    ROOT = '/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks'
    meta = {
                'audio_filepath': file,
                'offset': 0,
                'duration': None,
                'label': 'infer',
                'text': '-',
                'num_speakers': None,
                'rttm_filepath': None,
                'uem_filepath': None
            }
    
    with open('data/input_manifest.json','w') as fp:
        json.dump(meta,fp)
        fp.write('\n')
        
    output_dir = os.path.join(ROOT, 'oracle_vad')
    # os.makedirs(output_dir,exist_ok=True)

    
    MODEL_CONFIG = os.path.join(data_dir,'diar_infer_telephonic.yaml')

    config = OmegaConf.load(MODEL_CONFIG)
   
    config.diarizer.manifest_filepath = 'data/input_manifest.json'
    config.diarizer.out_dir = output_dir # Directory to store intermediate files and prediction outputs
    pretrained_speaker_model = 'titanet_large'
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5] 
    config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1] 
    config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1] 
    config.diarizer.oracle_vad = False # ----> ORACLE VAD 
    config.diarizer.clustering.parameters.oracle_num_speakers = False


    #for MSDD
    config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic' # Telephonic speaker diarization model 
    config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1]
    
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer


    oracle_vad_msdd_model = NeuralDiarizer(cfg=config)
    
    oracle_vad_msdd_model.diarize()

if __name__ == "__main__":
    # files = glob.glob('/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/data/test/*.flac')
    # for file in files:
    infer_msdd('/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/data/multi_speaker_conversation/e5129809-3652-49b7-8b0f-5cce27255073.flac')

    print('done')

