import numpy as np
import time
import websockets
import asyncio
import json
import os
from pydub.silence import split_on_silence
from pydub import AudioSegment

# For VAD
import wave
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from torch.utils.data import DataLoader
import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
from scipy.io.wavfile import read
from tqdm import tqdm
import copy
import base64

### Trim Silence for VAD
def clear_silence(audio):
    dBFS = audio.dBFS
    chunks = split_on_silence(audio,
        min_silence_len = 150,
        silence_thresh = dBFS-20,
        keep_silence = 50
    )
    return chunks

### VAD CLASSes and Functions
class FrameVAD:
    
    def __init__(self, vad_path , sample_rate,
                 frame_len=2, frame_overlap=2.5, 
                 offset=10, device = 'cuda'):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        
        
        os.environ["CUDA_VISIBLE_DEVICES"]="0"

        vad_model = nemo_asr.models.EncDecClassificationModel.restore_from(vad_path, map_location = device)
        cfg_vad = copy.deepcopy(vad_model._cfg)

        vad_model.preprocessor = vad_model.from_config_dict(cfg_vad.preprocessor)
        vad_model.eval()
        
        self.vad_model = vad_model.to(vad_model.device)
        self.data_layer_vad = AudioDataLayer(sample_rate=cfg_vad.train_ds.sample_rate)
        self.data_loader_vad = DataLoader(self.data_layer_vad, batch_size=1, collate_fn=self.data_layer_vad.collate_fn)


        self.vocab = list(cfg_vad.labels)
        self.vocab.append('_')
        
        self.sr = sample_rate
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = cfg_vad.preprocessor['window_stride']
        
        for block in cfg_vad.encoder['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.reset()
        
    def _decode(self, frame, offset=0):
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits = self.infer_signal_vad().cpu().numpy()[0]
        decoded = self._greedy_decoder(
            logits,
            self.vocab
        )
        return decoded  
    
    def infer_signal_vad(self):
        self.data_layer_vad.set_signal(self.buffer)
        batch = next(iter(self.data_loader_vad))
        audio_signal, audio_signal_len = batch
        audio_signal, audio_signal_len = audio_signal.to(self.vad_model.device), audio_signal_len.to(self.vad_model.device)
        logits = self.vad_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        return logits

    @torch.no_grad()
    def transcribe(self, frame=None):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged = self._decode(frame, self.offset)
        return unmerged
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = []
        if logits.shape[0]:
            probs = torch.softmax(torch.as_tensor(logits), dim=-1)
            probas, preds = torch.max(probs, dim=-1)
            s = [preds.item(), str(vocab[preds]), probs[0].item(), probs[1].item(), str(logits)]
        return s

class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)/32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1


class Append():
    def __init__(self):
        self.speech = np.array([], dtype='int16')
        self.labels = []
        self.StEn_timestamps = []
        self.speech_in_buffer = 0
        self.total_in_buffer = 0
        self.percent_of_speech = 0
        self.active_voice = None
        
    def activated_voice(self, active = None):
        self.active_voice = active
        
    def buffer_for_speech(self, bytes_speech = None, switch = None):
        if not switch:
            self.speech = np.concatenate((self.speech, bytes_speech), axis=None)
        else:
            self.speech = np.array([], dtype='int16')
        return self.speech
    
    def buffer_for_labels(self, label_predict = None, switch = None):
        if not switch:
            self.labels.append(label_predict)
            
            if label_predict == 'speech':
                self.speech_in_buffer +=1
                self.total_in_buffer +=1
            else:
                self.total_in_buffer +=1

            self.percent_of_speech = self.speech_in_buffer / self.total_in_buffer

        elif switch == 'send':
            self.labels = []
            self.speech_in_buffer = 0
            self.total_in_buffer = 0
            self.percent_of_speech = 0
            
        elif switch == 'refresh':
            
            self.labels = []
            self.buffer_for_labels(switch = 'send')
            self.buffer_for_speech(switch = 'clear')
            self.buffer_for_sten_timestamps(switch = 'clear')
                        
        elif switch == 'move_window':
            self.labels = self.labels[1:]

        return self.labels
    
    def buffer_for_sten_timestamps(self, times = None, switch = None):
        if not switch:
            self.StEn_timestamps.append(times)
        else:
            self.StEn_timestamps = []
        return self.StEn_timestamps

def encode_and_create_request(speech, StEn_timestamps, increase_volume, request_id, ASR = True, emotion = True, speaker_id = True):
    encoded = base64.b64encode(speech)
    wav_byte = encoded.decode('ascii')

    data = {"start": StEn_timestamps[0], "speech": wav_byte, "increase_volume" : increase_volume // 1.29 , 'request_id': request_id, 'ASR': ASR, 'emotion': emotion, 'speaker_id': speaker_id}
    return data

# ## End VAD Classes and Functions

class Buffer_for_ASR():
    def __init__(self):
        self.data_for_save = {}
        
    def save_data(self, data=None, switch=None, waiter=None):
        if not switch:
            if waiter:
                return self.data_for_save
            self.data_for_save = data
        else:
            self.data_for_save = {}
        return self.data_for_save

class Buffer_Text():
    def __init__(self):
        self.text = []
        self.start_ts = []
        self.end_ts = []
        self.count = 0
    def save_text(self, data=None):
        self.text.append(data)
        return self.text
    def get_text(self):
        return self.text
    def save_timestamps(self, start=None, end=None):
        if start:
            self.start_ts.append(start)
        if end:
            self.end_ts.append(end)
    def get_timestamps(self):
        return self.start_ts[0], self.start_ts[0] + sum(self.end_ts)
    def del_index(self):
        if len(self.text) != 1:
            self.text = self.text[:-1]
    def count_check(self, count = None, hold = False):
        if not hold:
            self.count += 1
        return self.count
    def reset_data(self):
        self.text = []
        self.count = 0
