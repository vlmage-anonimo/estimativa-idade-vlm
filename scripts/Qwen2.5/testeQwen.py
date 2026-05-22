from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
import torch
import pandas as pd

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
path = '../models/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    path,
    dtype=torch.bfloat16,
    quantization_config=quantization_config,
#    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(path, min_pixels=min_pixels, max_pixels=max_pixels)
processor.tokenizer.padding_side = "left"

### data load
df = pd.read_csv('../LAGENDA/lagenda_annotation.csv')

imgs_todas_idades_ok = (
    df.groupby('img_name')['age']
      .apply(lambda x: (x != -1).all())
      .reset_index()
)

# Filtra as imagens em que TODAS as idades são conhecidas
imgs_todas_idades_ok = imgs_todas_idades_ok[imgs_todas_idades_ok['age'] == True]

# tem uma imagem faltando nos arquivos (olhar depois)
imgs_todas_idades_ok = imgs_todas_idades_ok[imgs_todas_idades_ok['img_name'] != 'lag_benchmark/b5c8ef090f134ad5.jpg.jpg']

print(f"{len(imgs_todas_idades_ok)} imagens têm todas as idades conhecidas.")
print(imgs_todas_idades_ok)

prompt= """<image>
Você recebe uma imagem que pode conter uma ou mais pessoas. 
Para CADA pessoa, determine a partir da aparência se a pessoa é uma criança com menos de 14 anos.
Use APENAS as seguintes categorias: ["criança", "adulto"]
- "criança": idade menor que 14 anos
- "adulto": idade maior ou igual a 14 anos

Regras:
- Responda APENAS com JSON válido
- Nenhum texto extra antes ou depois

Formato de saída:
{
  "número de pessoas": <inteiro>,
  "pessoas": [
    {"id": 1, "categoria": <classificação da idade> }, 
  ]
}"""

#prompt = """Você recebe uma imagem que pode conter uma ou mais pessoas. 
#Para CADA pessoa determine a sua idade exata a partir de sua aparência.
#
#Regras:
#- Responda APENAS com JSON válido
#- Nenhum texto extra antes ou depois
#- A idade deve ser um número inteiro.
#- Estime individualmente a idade de CADA pessoa.
#
#Formato de saída:
#{
#  "número de pessoas": <inteiro>,
#  "pessoas": [
#    {"id": 1, "idade": <idade da pessoa> }, 
#  ]
#}
#"""


from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm  # <-- barra de progresso
from codecarbon import EmissionsTracker

########## code carbon ##########
tracker = EmissionsTracker(tracking_mode='process', log_level='critical', output_file='emissions_qwen2.5_prompt1_json_noface.csv')
tracker.start()
#################################

# pasta_imagens = "../LAGENDA/images"
pasta_imagens = "../LAGENDA/images_noface"

# prompt = prompts_todos[0]
resultados = []
batch_size = 16 #### numero de imagens
cont = 0

torch.cuda.empty_cache()
for i in tqdm(range(0, len(imgs_todas_idades_ok), batch_size), desc="Processando imagens"):
    batch_imgs = imgs_todas_idades_ok['img_name'][i:i+batch_size]
    
    batch_images = []
    batch_nomes_arquivo = []

    messages = []

    for img_name in batch_imgs:
        nome_arquivo = img_name.split("/")[-1]
        batch_nomes_arquivo.append(nome_arquivo)
        caminho = f"{pasta_imagens}/{nome_arquivo}"
        imagem = Image.open(caminho)
        batch_images.append(imagem)

        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": imagem,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        messages.append(message)

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    generated_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Process results
    for img_name, generated_text in zip(batch_nomes_arquivo, generated_texts):
        resultados.append({
            "img_name": img_name,
            "resposta_modelo": generated_text
        })
    
    del inputs, generated_ids
    torch.cuda.empty_cache()
    cont += 1
    if cont % 10 == 0: print(generated_text)

########## code carbon ##########
emissions = tracker.stop()
print(f"Emissions: {emissions} kg CO₂")
#################################
    
# visualizar
df_resultados = pd.DataFrame(resultados)
print(df_resultados)

df_resultados.to_csv('resultados_qwen2.5_prompt1_json_noface.csv', index=False)
