
###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import os
import argparse
import json
import sys
import numpy as np
import torch

import time 

from flowtron import Flowtron
from torch.utils.data import DataLoader
from data import Data
from train import update_params

sys.path.insert(0, "tacotron2")
sys.path.insert(0, "tacotron2/waveglow")
from glow import WaveGlow
from scipy.io.wavfile import write



def infer(flowtron_path, waveglow_path, text, speaker_id, n_frames, sigma,
          seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # load waveglow
    waveglow = torch.load(waveglow_path)['model'].cuda().eval()
    waveglow.cuda().half()
    for k in waveglow.convinv:
        k.float()
    waveglow.eval()

    # load flowtron
    model = Flowtron(**model_config).cuda()
    cpt_dict = torch.load(flowtron_path )
    if 'model' in cpt_dict:
        dummy_dict = cpt_dict['model'].state_dict()
    else:
        dummy_dict = cpt_dict['state_dict']
    model.load_state_dict(dummy_dict)
    model.eval()

    print("Loaded checkpoint '{}')" .format(flowtron_path))

    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

    tic_prep = time.time()

    str_text = text 
    num_char = len(str_text)
    num_word = len(str_text.split() )

    speaker_vecs = trainset.get_speaker_id(speaker_id).cuda()
    text = trainset.get_text(text).cuda()

    speaker_vecs = speaker_vecs[None]
    text = text[None]
    toc_prep = time.time()

    ############## warm up   ########### to measure exact flowtron inference time 

         
    with torch.no_grad():
        tic_warmup = time.time()
        residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma    
        mels, attentions = model.infer(residual, speaker_vecs, text)
        toc_warmup = time.time()    


    tic_flowtron = time.time()
    with torch.no_grad():
        tic_residual = time.time()
        residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
        toc_residual = time.time()        
        mels, attentions = model.infer(residual, speaker_vecs, text)
        toc_flowtron = time.time()    

    for k in range(len(attentions)):
        attention = torch.cat(attentions[k]).cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        axes[0].imshow(mels[0].cpu().numpy(), origin='bottom', aspect='auto')
        axes[1].imshow(attention[:, 0].transpose(), origin='bottom', aspect='auto')
        fig.savefig('sid{}_sigma{}_attnlayer{}.png'.format(speaker_id, sigma, k))
        plt.close("all")

    tic_waveglow = time.time()
    audio = waveglow.infer(mels.half(), sigma=0.8).float()
    toc_waveglow = time.time()


    audio = audio.cpu().numpy()[0]
    # normalize audio for now
    audio = audio / np.abs(audio).max()

    len_audio = len(audio)
    dur_audio = len_audio / 22050
    num_frames = int(len_audio / 256)
    
    dur_prep = toc_prep - tic_prep
    dur_residual = toc_residual - tic_residual
    dur_flowtron_in = toc_flowtron - toc_residual
    dur_warmup = toc_warmup - tic_warmup 
    dur_flowtron_out = toc_flowtron - tic_residual
    dur_waveglow = toc_waveglow - tic_waveglow        
    dur_total = dur_prep + dur_flowtron_out + dur_waveglow 

    RTF =  dur_audio / dur_total

    str_text =  "\n text : " + str_text
    str_num   = "\n text {:d} char {:d} words  ".format(num_char, num_word ) 
    str_audio = "\n generated audio : {:2.3f} samples  {:2.3f} sec  with  {:d} mel frames ".format( len_audio, dur_audio, num_frames ) 
    str_perf  = "\n total time {:2.3f} = text prep {:2.3f} + flowtron{:2.3f} + wg {:2.3f}  ".format( dur_total, dur_prep, dur_flowtron_out, dur_waveglow ) 
    str_flow   ="\n total flowtron {:2.3f} = residual cal {:2.3f} + flowtron {:2.3f}  " .format(dur_flowtron_out, dur_residual, dur_flowtron_in  ) 
    str_rtf   = "\n RTF is {:2.3f} x  with warm up {:2.3f} ".format(RTF, dur_warmup ) 

    print(str_text,  str_num, str_audio, str_perf, str_flow, str_rtf  )  


    write("sid{}_sigma{}.wav".format(speaker_id, sigma),
          data_config['sampling_rate'], audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-f', '--flowtron_path',
                        help='Path to flowtron state dict', type=str)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow state dict', type=str)
    parser.add_argument('-t', '--text', help='Text to synthesize', type=str)
    parser.add_argument('-i', '--id', help='Speaker id', type=int)
    parser.add_argument('-n', '--n_frames', help='Number of frames',
                        default=400, type=int)
    parser.add_argument('-o', "--output_dir", default="results/")
    parser.add_argument("-s", "--sigma", default=0.5, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)

    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    infer(args.flowtron_path, args.waveglow_path, args.text, args.id,
          args.n_frames, args.sigma, args.seed)

