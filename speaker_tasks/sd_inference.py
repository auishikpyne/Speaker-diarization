import json
from omegaconf import OmegaConf
import os
import time
from clustering_diarizer import ClusteringDiarizer
import statistics
from pydub import AudioSegment
import glob
from tqdm import tqdm

class OracleVADClusteringDiarizer:
    def __init__(self, data_dir, output_dir, model_config, pretrained_speaker_model):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_config = model_config
        self.pretrained_speaker_model = pretrained_speaker_model

    def create_manifest(self, audio_filepath):
        meta = {
            'audio_filepath': audio_filepath,
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'text': '-',
            'num_speakers': None,
            'rttm_filepath': None,
            'uem_filepath': None
        }
        # If you want to write manifest in a json file uncomment it
        with open(os.path.join(self.data_dir, 'input_manifest.json'), 'w') as fp:
            json.dump(meta, fp)
            fp.write('\n')

    def diarize(self):
        config = OmegaConf.load(self.model_config)

        config.diarizer.manifest_filepath = os.path.join(self.data_dir, 'input_manifest.json')
        config.diarizer.out_dir = self.output_dir
        config.diarizer.speaker_embeddings.model_path = self.pretrained_speaker_model
        config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5, 1.25, 1.0, 0.75, 0.5]
        config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75, 0.625, 0.5, 0.375, 0.1]
        config.diarizer.speaker_embeddings.parameters.multiscale_weights = [1, 1, 1, 1, 1]
        config.diarizer.oracle_vad = False
        config.diarizer.clustering.parameters.oracle_num_speakers = False

        oracle_vad_clusdiar_model = ClusteringDiarizer(cfg=config)
        # print('oracle_vad_clusdiar_model......', oracle_vad_clusdiar_model)
        oracle_vad_clusdiar_model.diarize()

        print('done.')

if __name__ == '__main__':
    
    # required_time = []
    
    for file in tqdm(glob.glob('/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/data/denoised_crowd_test/*flac')):
    
        audio_file = file
        audio = AudioSegment.from_file(audio_file)
        channel = audio.channels
        
        if channel == 2:
            audio = audio.set_channels(1)
            audio.export(audio_file, format='flac')
            
        else:
            pass
        
        
        data_dir = '/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/data'
        output_dir = '/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/oracle_vad'
        model_config = os.path.join(data_dir, 'diar_infer_telephonic.yaml')
        pretrained_speaker_model = 'titanet_large'

        # start = time.time()
        
        diarizer = OracleVADClusteringDiarizer(data_dir, output_dir, model_config, pretrained_speaker_model)
        
        diarizer.create_manifest(audio_file)
        
        diarizer.diarize()
            
        # end = time.time()
            
        #     required_time.append(end - start)
        
        # print('list:',required_time)
        # print('min:', min(required_time))
        # print('max:', max(required_time))
        # print('mean:', statistics.mean(required_time))
        # print(end -start)
    
    
