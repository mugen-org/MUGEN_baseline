## Video-Audio-Text Generation

### Tokenization
#### Video VQVAE
```
sh generation/experiments/vqvae/VideoVQVAE_L8.sh
sh generation/experiments/vqvae/VideoVQVAE_L16.sh
sh generation/experiments/vqvae/VideoVQVAE_L32.sh
```
#### Audio VQVAE
```
sh generation/experiments/vqvae/AudioVQVAE_128x.sh
sh generation/experiments/vqvae/AudioVQVAE_256x.sh
sh generation/experiments/vqvae/AudioVQVAE_512x.sh
sh generation/experiments/vqvae/AudioVQVAE_1024x.sh
```

#### Text Tokenization
```
python generation/train_tokenizer.py
```
Our provided tokenizer is in ```datasets/coinrun/tokenizers/tokenizer-coinrun_1024.json``` 


### Video-to-Text Generation
#### Training
``` 
sh generation/experiments/gpt/V2T/VideoTextGPT_L8.sh # 8 frames
sh generation/experiments/gpt/V2T/VideoTextGPT_L16.sh # 16 frames
sh generation/experiments/gpt/V2T/VideoTextGPT_L32.sh # 32 frames
```

#### Sampling
``` 
sh generation/experiments/gpt/V2T/eval/sampling/L8.sh # 8 frames
sh generation/experiments/gpt/V2T/eval/sampling/L16.sh # 16 frames
sh generation/experiments/gpt/V2T/eval/sampling/L32.sh # 32 frames
```

#### Evaluation by similarity
``` 
sh generation/experiments/gpt/V2T/eval/sim/L8.sh # 8 frames
sh generation/experiments/gpt/V2T/eval/sim/L16.sh # 16 frames
sh generation/experiments/gpt/V2T/eval/sim/L32.sh # 32 frames
sh generation/experiments/sim/VT/gt.sh # GT similarity
python generation/compute_sim.py --task V2T
```

#### Evaluation by captioning
``` 
sh generation/experiments/gpt/V2T/eval/cap/L8.sh # 8 frames
sh generation/experiments/gpt/V2T/eval/cap/L16.sh # 16 frames
sh generation/experiments/gpt/V2T/eval/cap/L32.sh # 32 frames
```

### Audio-to-Text Generation
#### Training
```
sh generation/experiments/gpt/A2T/AudioTextGPT_1024x.sh
sh generation/experiments/gpt/A2T/AudioTextGPT_512x.sh
sh generation/experiments/gpt/A2T/AudioTextGPT_256x.sh
sh generation/experiments/gpt/A2T/AudioTextGPT_128x.sh
```
#### Sampling
``` 
sh generation/experiments/gpt/A2T/eval/sampling/1024x.sh
sh generation/experiments/gpt/A2T/eval/sampling/512x.sh
sh generation/experiments/gpt/A2T/eval/sampling/256x.sh
sh generation/experiments/gpt/A2T/eval/sampling/128x.sh
```

#### Evaluation by similarity
``` 
sh generation/experiments/gpt/A2T/eval/sim/1024x.sh
sh generation/experiments/gpt/A2T/eval/sim/512x.sh
sh generation/experiments/gpt/A2T/eval/sim/256x.sh
sh generation/experiments/gpt/A2T/eval/sim/128x.sh
sh generation/experiments/sim/VT/gt.sh # GT similarity
python generation/compute_sim.py --task A2T
```

#### Evaluation by captioning
``` 
sh generation/experiments/gpt/A2T/eval/cap/1024x.sh
sh generation/experiments/gpt/A2T/eval/cap/512x.sh
sh generation/experiments/gpt/A2T/eval/cap/256x.sh
sh generation/experiments/gpt/A2T/eval/cap/128x.sh
```

### Text-to-Video Generation
#### Training
```
sh generation/experiments/gpt/T2V/TextVideoGPT_L8.sh
sh generation/experiments/gpt/T2V/TextVideoGPT_L16.sh
sh generation/experiments/gpt/T2V/TextVideoGPT_L32.sh
sh generation/experiments/gpt/T2V/TextVideoGPT_L32_A.sh
sh generation/experiments/gpt/T2V/TextVideoGPT_L32_M+A.sh
sh generation/experiments/gpt/T2V/TextVideoGPT_L8_Down.sh
sh generation/experiments/gpt/T2V/TextVideoGPT_L16_Down.sh
sh generation/experiments/gpt/T2V/TextVideoGPT_L32_Down.sh
sh generation/experiments/gpt/T2V/TextVideoGPT_L32_Down_A.sh
sh generation/experiments/gpt/T2V/TextVideoGPT_L32_Down_M+A.sh
```

#### Sampling
```
sh generation/experiments/gpt/T2V/eval/sampling/L8.sh
sh generation/experiments/gpt/T2V/eval/sampling/L16.sh
sh generation/experiments/gpt/T2V/eval/sampling/L32.sh
sh generation/experiments/gpt/T2V/eval/sampling/L32_A.sh
sh generation/experiments/gpt/T2V/eval/sampling/L32_M+A.sh
sh generation/experiments/gpt/T2V/eval/sampling/L8_Down.sh
sh generation/experiments/gpt/T2V/eval/sampling/L16_Down.sh
sh generation/experiments/gpt/T2V/eval/sampling/L32_Down.sh
sh generation/experiments/gpt/T2V/eval/sampling/L32_Down_A.sh
sh generation/experiments/gpt/T2V/eval/sampling/L32_Down_M+A.sh
```

#### Evaluation by FVD
```
sh generation/experiments/gpt/T2V/eval/fvd/L8.sh
sh generation/experiments/gpt/T2V/eval/fvd/L16.sh
sh generation/experiments/gpt/T2V/eval/fvd/L32.sh
sh generation/experiments/gpt/T2V/eval/fvd/L32_A.sh
sh generation/experiments/gpt/T2V/eval/fvd/L32_M+A.sh
sh generation/experiments/gpt/T2V/eval/fvd/L8_Down.sh
sh generation/experiments/gpt/T2V/eval/fvd/L16_Down.sh
sh generation/experiments/gpt/T2V/eval/fvd/L32_Down.sh
sh generation/experiments/gpt/T2V/eval/fvd/L32_Down_A.sh
sh generation/experiments/gpt/T2V/eval/fvd/L32_Down_M+A.sh
```

#### Evaluation by similarity
```
sh generation/experiments/gpt/T2V/eval/sim/L8.sh
sh generation/experiments/gpt/T2V/eval/sim/L16.sh
sh generation/experiments/gpt/T2V/eval/sim/L32.sh
sh generation/experiments/gpt/T2V/eval/sim/L32_A.sh
sh generation/experiments/gpt/T2V/eval/sim/L32_M+A.sh
sh generation/experiments/gpt/T2V/eval/sim/L8_Down.sh
sh generation/experiments/gpt/T2V/eval/sim/L16_Down.sh
sh generation/experiments/gpt/T2V/eval/sim/L32_Down.sh
sh generation/experiments/gpt/T2V/eval/sim/L32_Down_A.sh
sh generation/experiments/gpt/T2V/eval/sim/L32_Down_M+A.sh
sh generation/experiments/sim/VT/gt.sh # GT similarity
python generation/compute_sim.py --task T2V
```

#### Evaluation by captioning
```
sh generation/experiments/gpt/T2V/eval/cap/L8.sh
sh generation/experiments/gpt/T2V/eval/cap/L16.sh
sh generation/experiments/gpt/T2V/eval/cap/L32.sh
sh generation/experiments/gpt/T2V/eval/cap/L32_A.sh
sh generation/experiments/gpt/T2V/eval/cap/L32_M+A.sh
sh generation/experiments/gpt/T2V/eval/cap/L8_Down.sh
sh generation/experiments/gpt/T2V/eval/cap/L16_Down.sh
sh generation/experiments/gpt/T2V/eval/cap/L32_Down.sh
sh generation/experiments/gpt/T2V/eval/cap/L32_Down_A.sh
sh generation/experiments/gpt/T2V/eval/cap/L32_Down_M+A.sh
```


### Audio-to-Video Generation
#### Training
```
sh generation/experiments/gpt/A2V/AudioVideoGPT_L32_1024x.sh
sh generation/experiments/gpt/A2V/AudioVideoGPT_L32_512x.sh
sh generation/experiments/gpt/A2V/AudioVideoGPT_L32_256x.sh
sh generation/experiments/gpt/A2V/AudioVideoGPT_L32_128x.sh
```

#### Sampling
```
sh generation/experiments/gpt/A2V/eval/sampling/L32_1024x.sh
sh generation/experiments/gpt/A2V/eval/sampling/L32_512x.sh
sh generation/experiments/gpt/A2V/eval/sampling/L32_256x.sh
sh generation/experiments/gpt/A2V/eval/sampling/L32_128x.sh
```

#### Evaluation by FVD
```
sh generation/experiments/gpt/A2V/eval/fvd/L32_1024x.sh
sh generation/experiments/gpt/A2V/eval/fvd/L32_512x.sh
sh generation/experiments/gpt/A2V/eval/fvd/L32_256x.sh
sh generation/experiments/gpt/A2V/eval/fvd/L32_128x.sh
```

#### Evaluation by Similarity
```
sh generation/experiments/gpt/A2V/eval/sim/L32_1024x.sh
sh generation/experiments/gpt/A2V/eval/sim/L32_512x.sh
sh generation/experiments/gpt/A2V/eval/sim/L32_256x.sh
sh generation/experiments/gpt/A2V/eval/sim/L32_128x.sh
sh generation/experiments/sim/AV/gt.sh # GT similarity
python generation/compute_sim.py --task A2V
```

### Video-to-Audio Generation
#### Training
```
sh generation/experiments/gpt/V2A/VideoAudioGPT_L32_1024x.sh
sh generation/experiments/gpt/V2A/VideoAudioGPT_L32_512x.sh
sh generation/experiments/gpt/V2A/VideoAudioGPT_L32_256x.sh
sh generation/experiments/gpt/V2A/VideoAudioGPT_L32_128x.sh
```
#### Sampling
```
sh generation/experiments/gpt/V2A/eval/sampling/L32_1024x.sh
sh generation/experiments/gpt/V2A/eval/sampling/L32_512x.sh
sh generation/experiments/gpt/V2A/eval/sampling/L32_256x.sh
sh generation/experiments/gpt/V2A/eval/sampling/L32_128x.sh
```
#### Evaluation by FAD
```
sh generation/experiments/gpt/V2A/eval/fad/L32_1024x.sh
sh generation/experiments/gpt/V2A/eval/fad/L32_512x.sh
sh generation/experiments/gpt/V2A/eval/fad/L32_256x.sh
sh generation/experiments/gpt/V2A/eval/fad/L32_128x.sh
```
#### Evaluation by similarity
```
sh generation/experiments/gpt/V2A/eval/sim/L32_1024x.sh
sh generation/experiments/gpt/V2A/eval/sim/L32_512x.sh
sh generation/experiments/gpt/V2A/eval/sim/L32_256x.sh
sh generation/experiments/gpt/V2A/eval/sim/L32_128x.sh
sh generation/experiments/sim/AV/gt.sh # GT similarity
python generation/compute_sim.py --task V2A
```

### Text-to-Audio Generation

#### Training
```
sh generation/experiments/gpt/T2A/TextAudioGPT_1024x.sh
sh generation/experiments/gpt/T2A/TextAudioGPT_512x.sh
sh generation/experiments/gpt/T2A/TextAudioGPT_256x.sh
sh generation/experiments/gpt/T2A/TextAudioGPT_128x.sh
```
#### Sampling
```
sh generation/experiments/gpt/T2A/eval/sampling/1024x.sh
sh generation/experiments/gpt/T2A/eval/sampling/512x.sh
sh generation/experiments/gpt/T2A/eval/sampling/256x.sh
sh generation/experiments/gpt/T2A/eval/sampling/128x.sh
```

#### Evaluation by FAD
```
sh generation/experiments/gpt/T2A/eval/fad/1024x.sh
sh generation/experiments/gpt/T2A/eval/fad/512x.sh
sh generation/experiments/gpt/T2A/eval/fad/256x.sh
sh generation/experiments/gpt/T2A/eval/fad/128x.sh
```

#### Evaluation by similarity
```
sh generation/experiments/gpt/T2A/eval/sim/1024x.sh
sh generation/experiments/gpt/T2A/eval/sim/512x.sh
sh generation/experiments/gpt/T2A/eval/sim/256x.sh
sh generation/experiments/gpt/T2A/eval/sim/128x.sh
sh generation/experiments/sim/AT/gt.sh # GT similarity
python generation/compute_sim.py --task T2A
```

#### Evaluation by captioning
```
sh generation/experiments/gpt/T2A/eval/cap/1024x.sh
sh generation/experiments/gpt/T2A/eval/cap/512x.sh
sh generation/experiments/gpt/T2A/eval/cap/256x.sh
sh generation/experiments/gpt/T2A/eval/cap/128x.sh
```
