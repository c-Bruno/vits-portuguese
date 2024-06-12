# Importção das bibliotecas necessárias
import os
import librosa
import pandas as pd
import soundfile as sf
from tqdm.notebook import tqdm_notebook as tqdm


full_dataset = pd.DataFrame(columns=['path', 'sentence'])

path_custom_dataset = 'data/custom_dataset'


# Aqui vamos ler o arquivo de texto com o nome dos audios e suas descrições
df_dataset_tts_portuguese = pd.read_csv(f'{path_custom_dataset}/texts.csv', header=None, sep='==')
df_dataset_tts_portuguese = df_dataset_tts_portuguese.rename(columns={0: 'path', 1: 'sentence'})

# Aqui adicionamos o caminho para acessar os audios considerando a pasta do fork como raiz
df_dataset_tts_portuguese['path'] = f'{path_custom_dataset}/' + df_dataset_tts_portuguese["path"]

full_dataset = pd.concat([full_dataset, df_dataset_tts_portuguese], ignore_index=True)


# Printa os caminhos que não existe o arquivo (se não printar nada, significa que está tudo ok) e 
# marca para serem removidos do dataset
for idx, path in enumerate(full_dataset['path']):
  if not os.path.exists(path):
    print(idx, path)
    full_dataset.loc[idx, 'sentence'] = None

full_dataset[full_dataset['sentence'] == ''].count()

full_dataset[full_dataset['sentence'] == ''] = None
full_dataset = full_dataset.dropna(subset=['sentence'])

# Salva dataset
all_times = [] # Aqui vamos fazer uma analise dos audios coletando o tempo de duraçção
for path in tqdm(full_dataset['path']):
    all_times.append(librosa.get_duration(filename=path))
full_dataset['time'] = all_times

# Salva o dataset com a coluna 'time' que contem a informação de tempo de cada audio
full_dataset.to_csv(f'{path_custom_dataset}/dataset_custom.tsv', sep='\t', index=False)

# Limpar o dataset
pd_dataset_dict = pd.read_csv(f'{path_custom_dataset}/dataset_custom.tsv', sep='\t')  # Aqui vamos ler o dataset com a coluna 'time'

#Remover áudios muito grande
pd_dataset_dict['time'].sort_values()

ranges_time = []  # Separamos os ranges de tempo dos audios de 5 em 5 segundos para fazer o levantamento de durações
ranges_time.append(pd_dataset_dict['time'].between(0, 5, inclusive='left').sum())
ranges_time.append(pd_dataset_dict['time'].between(5, 10, inclusive='left').sum())
ranges_time.append(pd_dataset_dict['time'].between(10, 15, inclusive='left').sum())
ranges_time.append(pd_dataset_dict['time'].between(15, 20, inclusive='left').sum())
ranges_time.append(pd_dataset_dict['time'].between(20, 25, inclusive='left').sum())
ranges_time.append(pd_dataset_dict['time'].between(25, 30, inclusive='left').sum())
ranges_time.append((pd_dataset_dict['time'] >= 30).sum())

# Aqui vamos remover todos os aduiso com mais de 15 segundos do dataset final
pd_dataset_dict.loc[pd_dataset_dict['time'] >= 15, 'sentence'] = None
pd_dataset_dict = pd_dataset_dict.dropna(subset=['sentence'])

pd_dataset_dict.to_csv(f'{path_custom_dataset}/dataset_cleaned.tsv', sep='\t', index=False)

#Seperar dataset para treinamento
from sklearn.model_selection import train_test_split

# Aqui vamos carregar novamente o dataset para dropar a coluna de tempo e separar o dataset em treino e validação
pd_dataset = pd.read_csv(f'{path_custom_dataset}/dataset_cleaned.tsv', sep='\t')
pd_dataset = pd_dataset.drop(columns=['time'])

# Aqui separamos 80% dos audios para treinamento e 20% para validação
train, valid = train_test_split(pd_dataset, test_size=0.2, shuffle=True)
#valid, test = train_test_split(valid, test_size=0.5, shuffle=True)

print(len(train))
print(len(valid))
#print(len(test))

# Criação dos aquivos de train e valid com base no que foi gerado
train.to_csv(f'{path_custom_dataset}/train.csv', sep='|', index=False, header=False)
valid.to_csv(f'{path_custom_dataset}/valid.csv', sep='|', index=False, header=False)
#test.to_csv('data/test.tsv', sep='|', index=False)