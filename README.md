<div align="center">

<img src="https://africa.dlnlp.ai/simba/images/VoC_logo.png" alt="VoC Logo">

[![EMNLP 2025 Paper](https://img.shields.io/badge/EMNLP_2025-Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=B31B1B&labelColor=FFCDD2)](https://aclanthology.org/2025.emnlp-main.559/)
[![Official Website](https://img.shields.io/badge/Official-Website-2EA44F?style=for-the-badge&logo=googlechrome&logoColor=2EA44F&labelColor=C8E6C9)](https://africa.dlnlp.ai/simba/)
[![SimbaBench](https://img.shields.io/badge/SimbaBench-Benchmark-8A2BE2?style=for-the-badge&logo=googlecharts&logoColor=8A2BE2&labelColor=E1BEE7)](#simbabench)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E?style=for-the-badge&logoColor=black&labelColor=FFF9C4)](https://huggingface.co/collections/UBC-NLP/simba-speech-series)

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
| **Simba-TTS** 🔊 | MMS-TTS |  | 🛠️ In Progress  |
| **Simba-TTS** 🔊 | MMS-TTS |  | 🛠️ In Progress  |
| **Simba-TTS** 🔊 | MMS-TTS |  | 🛠️ In Progress  |
| **Simba-TTS** 🔊 | MMS-TTS |  | 🛠️ In Progress  |
| **Simba-TTS** 🔊 | MMS-TTS | | 🛠️ In Progress  |
| **Simba-TTS** 🔊 | MMS-TTS |  | 🛠️ In Progress  |
| **Simba-TTS** 🔊 | MMS-TTS |  | 🛠️ In Progress  |


### 🔍 Simba-SLID (Spoken Language Identification)
* **🎯 Task:** `Spoken Language Identification` — Intelligent input routing.
**🌍 Language Coverage (32 African languages)**
> **Afrikaans** (`afr`), **Amharic** (`amh`), **Arabic** (`ara`), **Asante Twi** (`asanti`), **Bambara** (`bam`), **Baoulé** (`bau`), **Bemba** (`bem`), **Dinka** (`din`), **Ewe** (`ewe`), **Fanti** (`fat`), **Fon** (`fon`), **French** (`fra`), **Ganda** (`lug`), **Hausa** (`hau`), **Igbo** (`ibo`), **Kabiye** (`kab`), **Kinyarwanda** (`kin`), **Kongo** (`kon`), **Lingala** (`lin`), **Luba-Katanga** (`lub`), **Luo** (`luo`), **Malagasy** (`mlg`), **Mossi** (`mos`), **Northern Sotho** (`nso`), **Nyanja** (`nya`), **Oromo** (`orm`), **Portuguese** (`por`), **Sango** (`sag`), **Shona** (`sna`), **Somali** (`som`), **Southern Sotho** (`sot`), **Swahili** (`swa`), **Swati** (`ssw`), **Tamazight** (`tzm`), **Tigrinya** (`tir`), **Tsonga** (`tso`), **Tswana** (`tsn`), **Twi** (`twi`), **Umbundu** (`umb`), **Venda** (`ven`), **Wolof** (`wol`), **Xhosa** (`xho`), **Yoruba** (`yor`), **Zulu** (`zul`)

| **SLID Model** | **Architecture** | **Hugging Face Card** | **Status** |
| :--- | :--- | :---: | :---: |
| **Simba-SLID** 🔍 | AfriHuBERT | — | 🛠️ In Progress |



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


