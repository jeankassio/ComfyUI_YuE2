# ComfyUI_YuE2

[English](README.md)

Nodes para ComfyUI usando **o pipeline Python oficial do YuE2**: carregamento dos modelos, estilo/letra, composição simbólica, tokens semânticos, síntese acústica, áudio estéreo e covers condicionados por partitura. O motor está fixado no [YuE2 v0.1.6](https://github.com/multimodal-art-projection/YuE/tree/1647252d68b70cbead046ffd3f7caf036dde47ee).

A inferência usa a implementação Python original e os pesos oficiais do Hugging Face. O ComfyUI fornece o grafo, as entradas/saídas de áudio e a exibição de progresso. A matemática do modelo, o tokenizer, a amostragem, o guidance e o solver midpoint continuam no motor oficial.

## Instalação no Windows

Coloque o repositório em `ComfyUI/custom_nodes/ComfyUI_YuE2`. Exemplo de instalação portable:

```text
ComfyUI_NOVO/
├── python_embeded/
└── ComfyUI/
    ├── models/
    └── custom_nodes/ComfyUI_YuE2/
```

Execute os arquivos nesta ordem; todos mantêm o terminal visível:

1. `setup-runtime.bat` — instala o motor oficial fixado em `.venv`, com Python 3.12 e versões próprias de Torch/Transformers.
2. `scripts/download-models.bat` — baixa explicitamente os modelos nas revisões fixadas e verifica SHA256.
3. `convert-models.bat` — organiza os arquivos oficiais verificados nas categorias de modelos do ComfyUI e cria o índice local.
4. Para covers a partir de áudio, execute também `scripts/setup-sheetsage.bat` — cria o ambiente separado `.venv-sheetsage`. O instalador prepara uma cópia privada do `uv` quando necessário; também aceita `--uv C:\caminho\uv.exe`.
5. Reinicie o ComfyUI e arraste um workflow de exemplo para a tela.

O adaptador usa NumPy, SoundFile e tqdm no Python existente do ComfyUI. Se estiverem ausentes, instale o `requirements.txt` deste repositório com esse Python. As dependências pesadas de geração/transcrição ficam nos ambientes isolados; os instaladores não substituem Torch ou Transformers do ComfyUI.

No Windows, mantenha `backend=torch-eager`: a distribuição oficial de Torch usada aqui não inclui o kernel FlashAttention exigido pelo backend CUDA Graph. O modo eager preserva pesos, precisão e algoritmo oficiais. Os índices CUDA correspondem às GPUs visíveis ao processo; para expor ambas no ComfyUI, use `--cuda-device all`.

## Instalação no Linux e macOS

Abra um terminal em `ComfyUI/custom_nodes/ComfyUI_YuE2`. O bootstrap precisa de Python 3 com `venv`/`pip`; no Debian/Ubuntu, instale `python3-venv` se o instalador informar sua ausência. Os launchers também encontram automaticamente o `.venv` ou `venv` do ComfyUI. Execute:

```sh
sh setup-runtime.sh
sh scripts/download-models.sh
sh convert-models.sh
# Opcional: transcrição para covers a partir de áudio
sh scripts/setup-sheetsage.sh
```

Os scripts preservam os argumentos, funcionam a partir de qualquer diretório e usam ambientes isolados. O launcher de download baixa os arquivos por padrão; `--check` verifica os arquivos locais sem rede:

```sh
sh scripts/download-models.sh --models yue2 vae
sh scripts/download-models.sh --check
```

| Plataforma | Instalação e configuração dos nodes |
|---|---|
| Linux com NVIDIA | Torch oficial CUDA 12.8 para YuE2; Torch/Torchaudio CUDA 12.6 para SheetSage2. Selecione o dispositivo CUDA disponível. O ponto de partida oficialmente recomendado é GPU com suporte a BF16 e 24 GB de VRAM. |
| Linux sem CUDA | Adicione `--cpu` aos dois launchers de setup e selecione `device=cpu` nos nodes. |
| macOS com Apple Silicon | Pacotes oficiais para macOS pelo PyPI, incluindo CPU/MPS. No Model Loader, selecione `device=auto` ou `mps` e `backend=torch-eager`; no SheetSage2 Transcribe, selecione `cpu`. BF16 no MPS exige macOS 14 ou posterior. |
| macOS Intel/Python x86 traduzido | As versões oficiais fixadas de Torch não têm wheels para macOS Intel; o setup informa isso claramente. No Apple Silicon, use Python ARM64 nativo. |

Os instaladores mantêm Python e dependências locais, incluindo um bootstrap privado de `uv` quando necessário, sem modificar o Python do sistema ou as dependências do ComfyUI. Ambos aceitam `--torch-index-url URL` para escolher explicitamente outro índice de wheels. No macOS, `--cpu` usa a mesma distribuição oficial CPU/MPS; selecione `cpu` no node para execução em CPU. O backend opcional vLLM exige Linux com GPU: `sh setup-runtime.sh --fast`.

Após instalar, reinicie o ComfyUI e abra um workflow de `example_workflows`. A seleção de pacotes Linux/macOS e os launchers foram testados sem instalar nesses sistemas; a geração foi validada no Windows/RTX 3090.

## Modelos e pastas

Os cinco conjuntos de modelos e arquivos necessários somam **11,08 GB**, sem contar as dependências dos ambientes.

| Modelo | Função | Destino adaptado dentro de `ComfyUI/models/` |
|---|---|---|
| [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) | Backbone AR/NAR compartilhado: planejamento ABC, tokens semânticos e latentes acústicos | `diffusion_models/YuE2/YuE2-3B/` |
| `qwen.tiktoken` | Tokenizer oficial; os componentes aprendidos de texto estão dentro do YuE2-3B | `text_encoders/YuE2/qwen.tiktoken` e a pasta do modelo |
| [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) | Decoder atual para ouvir as gerações; seleção padrão | `vae/YuE2/YuE2-Vae/` |
| [YuE2-Vae-legacy](https://huggingface.co/m-a-p/YuE2-Vae-legacy) | Decoder para reproduzir o protocolo publicado de avaliação | `vae/YuE2/YuE2-Vae-legacy/` |
| [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) | Extrai uma partitura ABC estruturada da música usada como referência | `audio_encoders/YuE2/SheetSage2/` |
| [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) | Backbone de áudio obrigatório do SheetSage2 | `audio_encoders/YuE2/MERT-v2-FullSong/` |

O YuE2 usa um transformer conjunto; sua arquitetura não é composta por um UNet de imagem e um checkpoint independente de text encoder. A adaptação das pastas mantém **pesos e configurações oficiais idênticos, byte a byte**: não quantiza, não renomeia tensores e não troca o motor por uma implementação nativa do ComfyUI. Hard links evitam uma segunda cópia no mesmo sistema de arquivos; quando indisponíveis, os arquivos verificados são copiados. Os downloads originais permanecem nos destinos registrados em `model_manifest.json`.

Para baixar somente modelos selecionados ou verificar todos os arquivos sem acessar a internet:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --download --models yue2 vae
.\.venv\Scripts\python.exe scripts\download_models.py --check
```

A inferência funciona com o Hugging Face em modo local. Os downloads só acontecem quando você executa o downloader. O `model_manifest.json` fixa repositórios, revisões, tamanhos e SHA256, incluindo a revisão exata do MERT esperada pelo SheetSage2.

## Workflows de exemplo

Os exemplos em **JSON de tela ComfyUI v1.0** estão em [example_workflows/](example_workflows/).

| Workflow | Uso |
|---|---|
| [01-text-to-music.json](example_workflows/01-text-to-music.json) | Quatro nodes: Model Loader → Text Encode → Generate Music → Save Audio. Comece por este. |
| [02-staged-generation.json](example_workflows/02-staged-generation.json) | Controles completos e etapas exatas: Generate ABC → Semantic Tokens → Synthesize → Decode Audio → Save Audio. |
| [03-edited-abc-score.json](example_workflows/03-edited-abc-score.json) | Edite uma composição ABC curta e original e gere uma interpretação dela. |
| [04-audio-to-cover.json](example_workflows/04-audio-to-cover.json) | Load Audio → SheetSage2 Transcribe → Text Encode com ABC da melodia → Generate Music → Save Audio. |

No workflow 04, **envie e selecione uma música real no Load Audio antes de executar**. `select-source-audio.wav` é um nome de exemplo; não há gravação incluída com esse nome. Preencha a letra/estilo de destino no Text Encode. Use `mode=melody` no Transcribe e `cot=melody` no Text Encode para permitir que o acompanhamento se adapte ao cover.

O texto de exemplo vem do [examples/song.json oficial](https://github.com/multimodal-art-projection/YuE/blob/1647252d68b70cbead046ffd3f7caf036dde47ee/examples/song.json). O workflow 03 contém uma composição nova e curta no node ABC Score Input. Edite notas, durações, harmonia ou andamento conforme sua intenção; a partitura condiciona uma nova interpretação.

## Nodes e controles oficiais

Os nodes ficam na categoria **audio/YuE2 Official**, com nomes e campos em inglês.

| Node | Entradas e resultado |
|---|---|
| Official Model Loader | Modelo YuE2, VAE listening/legacy, dispositivo, orçamento de memória, `torch`/`torch-eager`/`vllm`, quantização, offload AR e tamanho dos blocos do VAE. Retorna configurações de execução/modelos. |
| Official Text Encode | Estilo, letra, modo de planejamento, seed, CFG automático/manual e ID da música; aceita partitura ABC opcional. Cria o pedido oficial. |
| Official Generation Options | Amostragem independente de ABC e tokens semânticos, passos ODE acústicos e decode completo/em blocos. |
| Official Generate Music | Pipeline completo; retorna AUDIO estéreo de 48 kHz, ABC e relatório JSON. |
| Official Generate ABC | Salva um plano simbólico com IDs exatos dos tokens; retorna plano, ABC legível e relatório. |
| Official Semantic Tokens | Usa diretamente o plano salvo e preserva seus IDs exatos. |
| Official Synthesize | Converte tokens semânticos em latentes acústicos com o solver midpoint oficial. |
| Official Decode Audio | Decodifica os latentes salvos com o VAE oficial escolhido. |
| ABC Score Input | Campo editável para ABC no dialeto nativo. |
| Official SheetSage2 Transcribe | Controles de transcrição, melody/full, BF16/FP32, presets default/paper, janelas automáticas/manuais, prompts de tarefas e opções oficiais de exportação/renderização. |

Os modos de planejamento seguem a API original:

- `full`: planeja melodia e acordes; padrão para músicas novas.
- `melody`: planeja a melodia e deixa o acompanhamento livre; recomendado para covers.
- `off`: gera diretamente a partir de letra e estilo.
- ABC informado pelo usuário exige `full` ou `melody`. Use as duas vozes `Vocal`/`Ins` do [guia oficial de ABC](https://github.com/multimodal-art-projection/YuE/blob/1647252d68b70cbead046ffd3f7caf036dde47ee/skills/yue2-music/references/abc-editing.md).

Generation Options expõe `temperature`, `top_p`, `top_k`, `repetition_penalty`, `penalty_window`, `min_tokens` e `max_tokens` **separadamente para ABC e tokens semânticos**. Os valores padrão seguem a configuração oficial, incluindo **32 passos ODE midpoint**. O CFG automático usa o padrão oficial do modo escolhido. A seleção padrão de qualidade usa **geração BF16 sem quantização e decode VAE FP32**, com o VAE listening. FP8 é uma escolha opcional explícita e não é ativado para contornar falta de memória.

O áudio semântico usa **25 tokens por segundo**. `semantic_max_tokens` é um teto de duração, não uma duração exata: o modelo pode terminar antes. Por exemplo, 9000 tokens permitem aproximadamente 360 segundos. Escreva andamento, idioma, voz, instrumentos e gênero em `style`; coloque palavras cantáveis e tags de seção em `lyrics`.

No SheetSage2, `max_seconds=0` processa a gravação inteira; um valor positivo limita explicitamente a entrada àquela quantidade de segundos. `automatic_window_settings=true` deixa o preset oficial definir as janelas. Desative para usar os campos de segundos. Prompts de tarefas vazios usam os padrões oficiais. As opções extras de exportação geram artefatos; a renderização opcional precisa da pasta oficial `render_assets/` e das dependências instaladas pelo `setup_render.py` do SheetSage2. O downloader fornece os arquivos de inferência; os recursos do renderer são uma configuração opcional separada.

## Saídas, progresso e comparação

Conecte AUDIO ao Save Audio (Advanced) do ComfyUI; os exemplos usam **FLAC**. O áudio é estéreo em **48.000 Hz**. O adaptador não aplica normalização, efeitos ou masterização. O progresso aparece na interface e no console do ComfyUI, usando as mesmas barras `tqdm` dos samplers. A geração de tokens mostra contagem e velocidade; os passos acústicos e blocos do decoder mostram seus totais reais. O carregamento mostra tempo decorrido. Cancelar interrompe o processo isolado e fecha a barra do console.

Antes de iniciar o motor separado, o adaptador libera os modelos do cache ComfyUI apenas no dispositivo escolhido, para abrir espaço para os pesos oficiais.

Os artefatos temporários ficam em `ComfyUI/temp/yue2/`: partitura, plano exato, tokens semânticos, latentes e metadados de geração. Os relatórios incluem truncamento por limite de tokens e tempos. Um áudio reproduzível ainda pode ter `truncated=true`; confira o relatório antes de considerar a música completa.

Os nodes de etapas trocam apenas o ID da pasta do artefato, resolvido dentro do diretório temporário configurado no ComfyUI. Os caminhos dos modelos vêm das chaves oficiais registradas, nunca de caminhos recebidos nos dicionários dos nodes. Caminhos absolutos, com unidade, UNC e travessias de diretórios são rejeitados, assim como links simbólicos e junctions do Windows nos caminhos de artefatos/modelos. Após atualizar, reinicie o ComfyUI e execute novamente as etapas anteriores para substituir saídas em cache do formato antigo.

Reutilize diretamente o `YUE2_PLAN` sem alterações entre as etapas. Para editar a composição, copie/edite o ABC e crie um novo pedido. Para comparar os decoders listening e legacy, decodifique **os mesmos latentes** duas vezes, sem gerar duas músicas diferentes. Seeds ajudam nas comparações, mas alterações de dispositivo, ambiente ou configuração podem mudar a reprodução do resultado. Cada execução gera um candidato; os resultados publicados best-of-N usam uma avaliação separada para escolher candidatos.

Os covers do SheetSage2 criam uma nova interpretação a partir da partitura. **Não clonam uma pessoa a partir de sua gravação de voz e não trocam apenas o cantor preservando a forma de onda original.** Este node não adiciona um modelo de identidade vocal ao YuE2.

Os arquivos baixados passam na verificação SHA256; os exemplos passam no schema publicado ComfyUI v1.0 e na conferência das conexões. Essas verificações não medem qualidade perceptiva nem garantem reprodução fiel da partitura/letra. Ouça os resultados que gerar.

Validado localmente no Windows/RTX 3090: treze testes de execução, os quatro prompts pelo validador real do ComfyUI, geração de 20 segundos pelo ComfyUI com progresso nativo e saída FLAC, decode completo dos latentes salvos e transcrição SheetSage2 desse áudio gerado. Os testes do console cobrem contagens reais, falhas e cancelamento do ComfyUI. O teste curto usou deliberadamente um teto de 500 tokens semânticos; os workflows entregues mantêm os padrões oficiais completos.

## Licenças e atribuição

O adaptador mantém a [licença MIT existente](LICENSE). O motor oficial fixado preserva a [licença Apache 2.0](runtime/vendor/LICENSE), os [termos dos modelos](runtime/vendor/MODEL_LICENSE) e os [avisos de terceiros](runtime/vendor/THIRD_PARTY_NOTICES.md).

Os pesos oficiais usam **CC BY-NC 4.0 com permissão adicional para criadores individuais**. Pelos termos dos licenciadores, pessoas físicas podem publicar e monetizar as próprias gerações. Essa permissão não autoriza empresas a usar comercialmente os pesos; para isso, é preciso tratar de uma licença comercial com os licenciadores. Leia as condições exatas na [licença oficial](https://huggingface.co/m-a-p/YuE2-3B/blob/8f24312f187763b9854adeec87e437e2b03bbff1/LICENSE). Os modelos de transcrição mantêm suas próprias licenças baixadas. A criação deste node local não o implanta no Somora.

Referências primárias: [guia oficial de geração](https://github.com/multimodal-art-projection/YuE/blob/1647252d68b70cbead046ffd3f7caf036dde47ee/docs/generation.md), [release oficial](https://github.com/multimodal-art-projection/YuE/releases/tag/yue2-v0.1.6), [SheetSage2](https://huggingface.co/m-a-p/SheetSage2), [schema de workflows ComfyUI](https://docs.comfy.org/specs/workflow_json).
