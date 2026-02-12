<div align="center">

<img src="https://africa.dlnlp.ai/simba/images/VoC_logo.png" alt="VoC Logo">

[![EMNLP 2025 Paper](https://img.shields.io/badge/EMNLP_2025-Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=B31B1B&labelColor=FFCDD2)](https://aclanthology.org/2025.emnlp-main.559/)
[![Official Website](https://img.shields.io/badge/Official-Website-2EA44F?style=for-the-badge&logo=googlechrome&logoColor=2EA44F&labelColor=C8E6C9)](https://africa.dlnlp.ai/simba/)
[![SimbaBench](https://img.shields.io/badge/SimbaBench-Benchmark-8A2BE2?style=for-the-badge&logo=googlecharts&logoColor=8A2BE2&labelColor=E1BEE7)](https://huggingface.co/spaces/UBC-NLP/SimbaBench)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=181717&labelColor=E0E0E0)](https://github.com/UBC-NLP/simba)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E?style=for-the-badge&logoColor=181717&labelColor=FFF9C4)](https://huggingface.co/collections/UBC-NLP/simba-speech-series)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E?style=for-the-badge&logoColor=181717&labelColor=FFF9C4)](https://huggingface.co/datasets/UBC-NLP/SimbaBench_dataset)

</div>

## *Bridging the Digital Divide for African AI*

**Voice of a Continent** is a comprehensive open-source ecosystem designed to bring African languages to the forefront of artificial intelligence. By providing a unified suite of benchmarking tools and state-of-the-art models, we ensure that the future of speech technology is inclusive, representative, and accessible to over a billion people.

## Best-in-Class Multilingual Models

<img src="https://africa.dlnlp.ai/simba/images/VoC_simba" alt="VoC Simba Models Logo">

Introduced in our EMNLP 2025 paper *[Voice of a Continent](https://aclanthology.org/2025.emnlp-main.559/)*, the **Simba Series** represents the current state-of-the-art for African speech AI.

- **Unified Suite:** Models optimized for African languages.
- **Superior Accuracy:** Outperforms generic multilingual models by leveraging SimbaBench's high-quality, domain-diverse datasets.
- **Multitask Capability:** Designed for high performance in ASR (Automatic Speech Recognition) and TTS (Text-to-Speech).
- **Inclusion-First:** Specifically built to mitigate the "digital divide" by empowering speakers of underrepresented languages.

The **Simba** family consists of state-of-the-art models fine-tuned using SimbaBench. These models achieve superior performance by leveraging dataset quality, domain diversity, and language family relationships.

### 🗣️✍️ Simba-ASR
> **The New Standard for African Speech-to-Text**

**🎯 Task** `Automatic Speech Recognition` — Powering high-accuracy transcription across the continent.

**🌍 Language Coverage (43 African languages)**
>  **Amharic** (`amh`), **Arabic** (`ara`), **Asante Twi** (`asanti`), **Bambara** (`bam`), **Baoulé** (`bau`), **Bemba** (`bem`), **Ewe** (`ewe`), **Fanti** (`fat`), **Fon** (`fon`), **French** (`fra`), **Ganda** (`lug`), **Hausa** (`hau`), **Igbo** (`ibo`), **Kabiye** (`kab`), **Kinyarwanda** (`kin`), **Kongo** (`kon`), **Lingala** (`lin`), **Luba-Katanga** (`lub`), **Luo** (`luo`), **Malagasy** (`mlg`), **Mossi** (`mos`), **Northern Sotho** (`nso`), **Nyanja** (`nya`), **Oromo** (`orm`), **Portuguese** (`por`), **Shona** (`sna`), **Somali** (`som`), **Southern Sotho** (`sot`), **Swahili** (`swa`), **Swati** (`ssw`), **Tigrinya** (`tir`), **Tsonga** (`tso`), **Tswana** (`tsn`), **Twi** (`twi`), **Umbundu** (`umb`), **Venda** (`ven`), **Wolof** (`wol`), **Xhosa** (`xho`), **Yoruba** (`yor`), **Zulu** (`zul`), **Tamazight** (`tzm`), **Sango** (`sag`), **Dinka** (`din`).

**🏗️ Base Architectures**

  -  **Simba-S** (SeamlessM4T-v2-MT) — *Top Performer*
  - **Simba-W** (Whisper-v3-large)
  - **Simba-X** (Wav2Vec2-XLS-R-2b)
  - **Simba-M** (MMS-1b-all)
  - **Simba-H** (AfriHuBERT)
      
🌐 Explore the Frontier

| **ASR Models**   | **Architecture**  | **#Parameters** | **🤗 Hugging Face Model Card** | **Status** |
|---------|:------------------:| :------------------:| :------------------:|:------------------:|    
| 🔥**Simba-S**🔥|    SeamlessM4T-v2  |  2.3B | 🤗 [https://huggingface.co/UBC-NLP/Simba-S](https://huggingface.co/UBC-NLP/Simba-S) | ✅ Released |
| 🔥**Simba-W**🔥|    Whisper         |  1.5B | 🤗 [https://huggingface.co/UBC-NLP/Simba-W](https://huggingface.co/UBC-NLP/Simba-W) | ✅ Released | 
| 🔥**Simba-X**🔥|    Wav2Vec2        |  1B | 🤗 [https://huggingface.co/UBC-NLP/Simba-X](https://huggingface.co/UBC-NLP/Simba-X) | ✅ Released |   
| 🔥**Simba-M**🔥|    MMS             |  1B | 🤗 [https://huggingface.co/UBC-NLP/Simba-M](https://huggingface.co/UBC-NLP/Simba-M) | ✅ Released |   
| 🔥**Simba-H**🔥|    HuBERT          |  94M | 🤗 [https://huggingface.co/UBC-NLP/Simba-H](https://huggingface.co/UBC-NLP/Simba-H) | ✅ Released |   

* **Simba-S** emerged as the best-performing ASR model overall.


**🧩 Usage Example**

You can easily run inference using the Hugging Face `transformers` library.

```python
from transformers import pipeline

# Load Simba-S for ASR
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="UBC-NLP/Simba-S" #Simba mdoels `UBC-NLP/Simba-S`, `UBC-NLP/Simba-W`, `UBC-NLP/Simba-X`, `UBC-NLP/Simba-H`, `UBC-NLP/Simba-M`
)

##### Load the multilingual African adapter (Only for  `UBC-NLP/Simba-M`)
asr_pipeline.model.load_adapter("multilingual_african")  # Only for  `UBC-NLP/Simba-M`
###########################

# Transcribe audio from file
result = asr_pipeline("https://africa.dlnlp.ai/simba/audio/afr_Lwazi_afr_test_idx3889.wav")
print(result["text"])


# Transcribe audio from audio array
result = asr_pipeline({
    "array": audio_array,
    "sampling_rate": 16_000
})
print(result["text"])

```

#### Example Outputs

Using the same audio file with different Simba models:

```python
# Simba-S
{'text': 'watter verontwaardiging sou daar, in ons binneste gewees het.'}
```

```python
# Simba-W
{'text': 'watter veronwaardigingsel daar, in ons binneste gewees het.'}
```

```python
# Simba-X
{'text': 'fator fr on ar taamsodr is'}
```

```python
# Simba-M
{'text': 'watter veronwaardiging sodaar in ons binniste gewees het'}
```

```python
# Simba-H
{'text': 'watter vironwaardiging so daar in ons binneste geweeshet'}
```
Get started with Simba models in minutes using our interactive Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/UBC-NLP/simba/blob/main/simba_models.ipynb)


### 🔊 Simba-TTS (Text-to-Speech)
* **🎯 Task:** `Text-to-Speech` — Natural Voice Synthesis.
**🌍 Language Coverage (7 African languages)**
> **Afrikaans** (`afr`), **Asante Twi** (`asanti`), **Akuapem Twi** (`akuapem`), **Lingala** (`lin`), **Southern Sotho** (`sot`), **Tswana** (`tsn`), **Xhosa** (`xho`)

| **TTS Model** | **Architecture** | **Hugging Face Card** | **Status** |
| :--- | :--- | :---: | :---: |
| **Simba-TTS-afr** 🔊 | MMS-TTS |  🤗 [https://huggingface.co/UBC-NLP/Simba-TTS-afr](https://huggingface.co/UBC-NLP/Simba-TTS-afr) | ✅ Released |
| **Simba-TTS-twi-asanti** 🔊 | MMS-TTS |  🤗 [https://huggingface.co/UBC-NLP/Simba-TTS-twi-asanti](https://huggingface.co/UBC-NLP/Simba-TTS-twi-asanti) | ✅ Released |
| **Simba-TTS-twi-akuapem** 🔊 | MMS-TTS |  🤗 [https://huggingface.co/UBC-NLP/Simba-TTS-twi-akuapem](https://huggingface.co/UBC-NLP/Simba-TTS-twi-akuapem) | ✅ Released |
| **Simba-TTS-lin** 🔊 | MMS-TTS |  🤗 [https://huggingface.co/UBC-NLP/Simba-TTS-lin](https://huggingface.co/UBC-NLP/Simba-TTS-lin) | ✅ Released |
| **Simba-TTS-sot** 🔊 | MMS-TTS |  🤗 [https://huggingface.co/UBC-NLP/Simba-TTS-sot](https://huggingface.co/UBC-NLP/Simba-TTS-sot) | ✅ Released |
| **Simba-TTS-tsn** 🔊 | MMS-TTS |  🤗 [https://huggingface.co/UBC-NLP/Simba-TTS-tsn](https://huggingface.co/UBC-NLP/Simba-TTS-tsn) | ✅ Released |
| **Simba-TTS-xho** 🔊 | MMS-TTS |  🤗 [https://huggingface.co/UBC-NLP/Simba-TTS-xho](https://huggingface.co/UBC-NLP/Simba-TTS-xho) | ✅ Released |

**🧩 Usage Example**

You can easily run inference using the Hugging Face `transformers` library.

```python
from transformers import VitsModel, AutoTokenizer
import torch

model_name="Simba-TTS-afr" ## Simba-TTS-twi-asanti, Simba-TTS-twi-akuapem, Simba-TTS-lin, Simba-TTS-sot, Simba-TTS-tsn, Simba-TTS-xho
model = VitsModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Ons noem hierdie deeltjies sub-atomiese deeltjies" #example of Afrikaans (afr) language 
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

```
The resulting waveform can be saved as a .wav file:
```python
scipy.io.wavfile.write("outputfile.wav", rate=model.config.sampling_rate, data=output.float().numpy())

```
Or displayed in a Jupyter Notebook / Google Colab:

```python
from IPython.display import Audio

Audio(output.numpy(), rate=model.config.sampling_rate)


```

### 🔍 Simba-SLID (Spoken Language Identification)
* **🎯 Task:** `Spoken Language Identification` — Intelligent input routing.
* **🌍 Language Coverage (49 African languages)**
  > **Akuapim Twi** (`Akuapim-twi`), **Asante Twi** (`Asante-twi`), **Tunisian Arabic** (`aeb`), **Afrikaans** (`afr`), **Amharic** (`amh`), **Arabic** (`ara`), **Basaa** (`bas`), **Bemba** (`bem`), **Taita** (`dav`), **Dyula** (`dyu`), **English** (`eng`), **Nigerian Pidgin** (`eng-zul`), **Ewe** (`ewe`), **Fanti** (`fat`), **Fon** (`fon`), **Pulaar** (`fuc`), **Pular** (`fuf`), **Ga** (`gaa`), **Hausa** (`hau`), **Igbo** (`ibo`), **Kabyle** (`kab`), **Kinyarwanda** (`kin`), **Kalenjin** (`kln`), **Lingala** (`lin`), **Lozi** (`loz`), **Luganda** (`lug`), **Luo** (`luo`), **Western Maninkakan** (`mlq`), **South Ndebele** (`nbl`), **Northern Sotho** (`nso`), **Chichewa** (`nya`), **Southern Sotho** (`sot`), **Serer** (`srr`), **Swati** (`ssw`), **Susu** (`sus`), **Kiswahili** (`swa`), **Swahili** (`swh`), **Tigre** (`tig`), **Tigrinya** (`tir`), **Tonga** (`toi`), **Tswana** (`tsn`), **Tsonga** (`tso`), **Twi** (`twi`), **Venda** (`ven`), **Wolof** (`wol`), **Xhosa** (`xho`), **Yoruba** (`yor`), **Standard Moroccan Tamazight** (`zgh`), **Zulu** (`zul`)

| **SLID Model** | **Architecture** | **Hugging Face Card** | **Status** |
| :--- | :--- | :---: | :---: |
| **Simba-SLID** 🔍 | HuBERT | 🤗 [https://huggingface.co/UBC-NLP/Simba-SLIS_49](https://huggingface.co/UBC-NLPSimba-SLIS_49) | ✅ Released |


**🧩 Usage Example**

You can easily run inference using the Hugging Face `transformers` library.

```python
from transformers import (
    HubertForSequenceClassification,
    AutoFeatureExtractor,
    AutoProcessor
)
import torch

model_id = "UBC-NLP/Simba-SLIS_49"
model = HubertForSequenceClassification.from_pretrained(model_id).to("cuda")
# HuBERT models can use either processor or feature extractor depending on the specific model
try:
    processor = AutoProcessor.from_pretrained(model_id)
    print("Loaded Simba-SLIS_49 model with AutoProcessor")
except:
    processor = AutoFeatureExtractor.from_pretrained(model_id)
    print("Loaded Simba-SLIS_49 model with AutoFeatureExtractor")

# Optimize model for inference
model.eval()
audio_arrays = [] ### add your audio array
sample_rate=16000

nputs = processor(audio_arrays, sampling_rate=sample_rate, return_tensors="pt", padding=True).to("cuda")
    
# Different models might have slightly different input formats
try:
    logits = model(**inputs).logits
except Exception as e:
    # Try alternative input format if the first attempt fails
    if "input_values" in inputs:
        logits = model(input_values=inputs.input_values).logits
    else:
        raise e

# Calculate softmax probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)

# Get the maximum probability (confidence) for each prediction
confidence_values, pred_ids = torch.max(probs, dim=-1)

# Convert to Python lists
pred_ids = pred_ids.tolist()
confidence_values = confidence_values.cpu().tolist()
# Get labels from IDs
pred_labels = [model.config.id2label[i] for i in pred_ids]


print(pred_labels, confidence_values)
```

## SibmaBench Data Release & Benchmarking

To evaluate your model on **SimbaBench** across all supported tasks (ASR, TTS, and SLID), simply load the corresponding configuration for the task and language you wish to benchmark.

Each task is organized by configuration name (e.g., `asr_test_afr`, `tts_test_wol`, `slid_61_test`). Loading a configuration provides the standardized evaluation split for that specific benchmark.

Example:

```python
from datasets import load_dataset

data = load_dataset("UBC-NLP/SimbaBench_dataset", "asr_test_afr")
```
```
DatasetDict({
    test: Dataset({
        features: ['split', 'benchmark_id', 'audio', 'text', 'duration_s', 'lang_iso3', 'lang_name'],
        num_rows: 1000
    })
})

```
``` python
data['test'][0]
```
```
{'split': 'test',
 'benchmark_id': 'afr_Lwazi_afr_test_idx3889',
 'audio': {'path': None,
  'array': array([ 4.27246094e-04,  7.62939453e-04,  6.71386719e-04, ...,
         -3.05175781e-04, -2.13623047e-04, -6.10351562e-05]),
  'sampling_rate': 16000},
 'text': 'watter, verontwaardiging sou daar, in ons binneste gewees het?',
 'duration_s': 5.119999885559082,
 'lang_iso3': 'afr',
 'lang_name': 'Afrikaans'}

```

### 📌 ASR Evaluation Configurations

| Config Name | Language | ISO | # Samples | # Hours |
|-------------|----------|-----|----------|--------|
| asr_test_Akuapim-twi | Akuapim-twi | Akuapim-twi | 1,000 | 1.35 |
| asr_test_Asante-twi | Asante-twi | Asante-twi | 1,000 | 0.97 |
| asr_test_afr | Afrikaans | afr | 1,000 | 0.87 |
| asr_test_amh | Amharic | amh | 581 | 1.12 |
| asr_test_bas | Basaa | bas | 582 | 0.76 |
| asr_test_bem | Bemba | bem | 1,000 | 2.15 |
| asr_test_dav | Taita | dav | 878 | 1.17 |
| asr_test_dyu | Dyula | dyu | 59 | 0.10 |
| asr_test_fat | Fanti | fat | 1,000 | 1.38 |
| asr_test_fon | Fon | fon | 1,000 | 0.66 |
| asr_test_fuc | Pulaar | fuc | 100 | 0.10 |
| asr_test_fuf | Pular | fuf | 129 | 0.03 |
| asr_test_gaa | Ga | gaa | 1,000 | 1.52 |
| asr_test_hau | Hausa | hau | 681 | 0.89 |
| asr_test_ibo | Igbo | ibo | 5 | 0.01 |
| asr_test_kab | Kabyle | kab | 1,000 | 1.05 |
| asr_test_kin | Kinyarwanda | kin | 1,000 | 1.50 |
| asr_test_kln | Kalenjin | kln | 1,000 | 1.50 |
| asr_test_loz | Lozi | loz | 399 | 0.91 |
| asr_test_lug | Ganda | lug | 1,000 | 1.65 |
| asr_test_luo | Luo (Kenya and Tanzania) | luo | 1,000 | 1.31 |
| asr_test_mlq | Western Maninkakan | mlq | 182 | 0.04 |
| asr_test_nbl | South Ndebele | nbl | 1,000 | 1.12 |
| asr_test_nso | Northern Sotho | nso | 1,000 | 0.88 |
| asr_test_nya | Nyanja | nya | 428 | 1.31 |
| asr_test_sot | Southern Sotho | sot | 1,000 | 0.82 |
| asr_test_srr | Serer | srr | 899 | 2.84 |
| asr_test_ssw | Swati | ssw | 1,000 | 0.93 |
| asr_test_sus | Susu | sus | 210 | 0.05 |
| asr_test_swa | Swahili | swa | 1,000 | 1.23 |
| asr_test_tig | Tigre | tig | 185 | 0.33 |
| asr_test_tir | Tigrinya | tir | 7 | 0.01 |
| asr_test_toi | Tonga (Zambia) | toi | 463 | 1.47 |
| asr_test_tsn | Tswana | tsn | 1,000 | 0.82 |
| asr_test_tso | Tsonga | tso | 1,000 | 0.99 |
| asr_test_twi | Twi | twi | 12 | 0.02 |
| asr_test_ven | Venda | ven | 1,000 | 0.92 |
| asr_test_wol | Wolof | wol | 1,000 | 1.19 |
| asr_test_xho | Xhosa | xho | 1,000 | 0.92 |
| asr_test_yor | Yoruba | yor | 359 | 0.42 |
| asr_test_zgh | Standard Moroccan Tamazight | zgh | 197 | 0.22 |
| asr_test_zul | Zulu | zul | 1,000 | 1.10 |

## 📌 TTS Evaluation Configurations

| Config Name           | Language        | ISO          | # Samples | # Hours |
|----------------------|----------------|-------------|----------|--------|
| tts_test_ewe         | Ewe            | ewe         | 66       | 0.29   |
| tts_test_kin         | Kinyarwanda    | kin         | 1,053    | 1.30   |
| tts_test_Asante-twi  | Asante-twi     | Asante-twi  | 64       | 0.18   |
| tts_test_yor         | Yoruba         | yor         | 40       | 0.13   |
| tts_test_wol         | Wolof          | wol         | 4,001    | 4.12   |
| tts_test_hau         | Hausa          | hau         | 124      | 0.24   |
| tts_test_lin         | Lingala        | lin         | 63       | 0.28   |
| tts_test_xho         | Xhosa          | xho         | 242      | 0.31   |
| tts_test_tsn         | Tswana         | tsn         | 238      | 0.36   |
| tts_test_afr         | Afrikaans      | afr         | 293      | 0.34   |
| tts_test_sot         | Southern Sotho | sot         | 210      | 0.33   |
| tts_test_Akuapim-twi | Akuapim-twi    | Akuapim-twi | 83       | 0.22   |


### 📌 SLID Evaluation

| Config Name   | Language Scope | # Samples | # Hours |
|--------------|---------------|----------|--------|
| slid_61_test | 61 Languages  | 21,817       | 34.36 |


## Citation

If you use the Simba models or SimbaBench  benchmark for your scientific publication, or if you find the resources in this website useful, please cite our paper.

```bibtex

@inproceedings{elmadany-etal-2025-voice,
    title = "Voice of a Continent: Mapping {A}frica{'}s Speech Technology Frontier",
    author = "Elmadany, AbdelRahim A.  and
      Kwon, Sang Yun  and
      Toyin, Hawau Olamide  and
      Alcoba Inciarte, Alcides  and
      Aldarmaki, Hanan  and
      Abdul-Mageed, Muhammad",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.559/",
    doi = "10.18653/v1/2025.emnlp-main.559",
    pages = "11039--11061",
    ISBN = "979-8-89176-332-6",
}

```


