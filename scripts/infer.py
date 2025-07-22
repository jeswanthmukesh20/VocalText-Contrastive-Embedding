from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor, PreTrainedTokenizer, PreTrainedModel
import chromadb
from dataclasses import dataclass
import torchaudio
import torch
from torch.functional import F
from model import AudioEncoder
from fastrtc import Stream, ReplyOnPause
import numpy as np
import sys

sys.path.append("./models/csm")
from generator import load_csm_1b, Segment

@dataclass
class Embedder(EmbeddingFunction):
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    def name(self):
        return "default"

    def __call__(self, input: Documents) -> Embeddings:
        model_inputs = self.tokenizer(input, return_tensors="pt", padding='max_length', max_length=512)
        attention_mask = model_inputs["attention_mask"]
        token_embeddings = self.model(**model_inputs).last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return F.normalize(sum_embeddings / sum_mask, p=2, dim=1).detach().numpy()

collection_name = "my_collection"
chroma_client = chromadb.Client()
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")
text_embedding = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
weights = torch.load("./models/text_encoder.pth", map_location=torch.device('cpu'))
audio_embedding_head_weight = torch.load("./models/audio_head.pth", map_location=torch.device('cpu'))

weights = {key[11:]:value for key, value in weights.items()}

text_embedding.load_state_dict(weights, strict=True)
audio_encoder = AudioEncoder()
audio_encoder.embedding_head.load_state_dict(audio_embedding_head_weight, strict=True)
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=Embedder(tokenizer, text_embedding))


generator = load_csm_1b(device)
conversation_audio, sample_rate = torchaudio.load("./models/csm/csm-1b/prompts/conversational_a.wav")
conversation_a = (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
)
audio_tensor = conversation_audio.squeeze(0)
audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=generator.sample_rate
)



@torch.no_grad()
def echo(audio: tuple[int, np.ndarray]):
    generated_segments = []
    prompt_segments = [Segment(text=conversation_a, speaker=0, audio=audio_tensor)]
    sample_rate = audio[0]

    resampled_audio = torchaudio.functional.resample(
            torch.from_numpy(audio[-1]).to(torch.float32), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    
    audio_inputs = feature_extractor(
        raw_speech=resampled_audio.squeeze(0),
        return_tensors="pt",
        sampling_rate=16000
    )

    with torch.no_grad():
        output_embeds = audio_encoder(audio_inputs)

    response = collection.query(
        query_embeddings=[output_embeds.squeeze(0).detach().numpy()],
        n_results=1,
    )
    if not response["documents"][0]:
        input_message = "Hey no documents found, can you please try again?"
    else:
        response_text = "\n".join(response["documents"][0])
        input_message = f"Did you mean '{response_text}'"
    

    audio_tensor = generator.generate(
        text=input_message,
        speaker=0,
        context=prompt_segments + generated_segments,
        max_audio_length_ms=10_000,
    )
    generated_segments.append(Segment(text=input_message, speaker=0, audio=audio_tensor))
    all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
    torchaudio.save(
        "response_audio.wav",
        all_audio.unsqueeze(0).cpu(),
        generator.sample_rate
    )
    yield (generator.sample_rate, all_audio.unsqueeze(0).cpu().detach().numpy())


stream = Stream(
    handler=ReplyOnPause(echo),
    modality="audio",
    mode="send-receive",
)
stream.ui.launch()