# Sales Copilot — probabilidade de fechamento ao vivo com o Jev

Copiloto para calls de vendas. A cada fala da conversa, o app pergunta ao **Jev** (modelo *System One* da TypeSafe,
que devolve decisões tipadas em vez de texto) cerca de 37 perguntas objetivas numa única requisição e, em menos de um
segundo, atualiza:

1. **Probabilidade de fechamento** (0–100 %), num gráfico ao vivo.
2. **Etapa da call** — abertura, descoberta, qualificação, apresentação, objeções, preço, fechamento, próximos passos.
3. **Próxima melhor jogada** — uma das 15 jogadas do playbook, com a frase sugerida ("diga: …").
4. **Objeção em aberto** — preço, timing, autoridade, necessidade, confiança ou concorrente.
5. **Sinais ao vivo** — dor identificada, orçamento, decisor, prazo, sinal de compra, cliente desengajando, vendedor
   falando demais e outros.
6. **Frase personalizada** (opcional) — um LLM pequeno reescreve a frase do playbook para *esta* conversa, e o Jev
   confere se ela não inventou nenhum dado.

O Jev não escreve texto: ele só responde às perguntas. Toda a conta (probabilidade, médias, proporção de fala) é feita
no código, em `copilot/engine.py`.

> Baseado em [moritzkremb/jev-sales-copilot](https://github.com/moritzkremb/jev-sales-copilot). Esta versão está em
> português, usa o Jev pelo OpenRouter, traz o playbook da Playbook Lab, chamada de vídeo e deploy na Vercel.

## Como rodar no seu computador

Você precisa de:

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (gerenciador de Python; ele instala o Python 3.13 sozinho);
- uma chave do [OpenRouter](https://openrouter.ai/settings/keys) com crédito.

```bash
git clone https://github.com/felipenunes07/playbook-sales-copilot.git
cd playbook-sales-copilot
cp .env.example .env      # abra o .env e cole sua OPENROUTER_API_KEY
```

Depois:

- **Windows:** dê dois cliques em `iniciar.bat`. O navegador abre em http://localhost:8000. Para parar, feche a janela preta.
- **Mac/Linux:** `./run.sh` e abra http://localhost:8000.

### Quanto custa

Com o OpenRouter, uma call de 30 minutos custa menos de US$ 0,05 no Jev, menos de US$ 0,01 em frases personalizadas e
cerca de US$ 0,05 por hora de transcrição na chamada de vídeo. Respostas do Jev ficam guardadas em `.cache/jev`:
repetir uma call já analisada é de graça.

## Modos (abas no topo)

| Aba | Para que serve | Quem fala é identificado… |
| --- | --- | --- |
| **Replay** | Toca uma call de exemplo, com velocidade ajustável | automaticamente (vem na transcrição) |
| **Vídeo** | Assiste a uma call gravada com o copiloto analisando no tempo do vídeo; pode avançar e voltar | automaticamente |
| **Microfone ao vivo** | Transcreve o seu microfone em português (só no Chrome) | manualmente: aperte **S** a cada troca |
| **Digitar** | Escreve as falas para treinar | você escolhe vendedor/prospect |
| **📹 Chamada de vídeo** | Cria uma sala, você manda o link ao cliente e faz a call ali mesmo | automaticamente: seu microfone = vendedor, áudio do cliente = prospect |

**Chamada de vídeo no seu computador:** para o link funcionar fora da sua máquina, instale o
[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
(`winget install --id Cloudflare.cloudflared`). O app cria um link público e gratuito na hora. Na Vercel isso não é
necessário. Use fone de ouvido.

## Adaptar para o seu negócio

- **Playbook** (`data/playbook.json`): as 15 jogadas, cada uma com três frases. O `what`/`not_for` em inglês é o que o
  Jev lê para escolher a jogada; o `what_pt`, o `title` e as `phrasings` são o que aparece na tela. As frases atuais
  são da Playbook Lab (Sales OS, squad de automação, prospecção no LinkedIn). Troque pelas suas.
- **Pesos e limites** (`copilot/constants.py`): peso de cada sinal na probabilidade, limiares de confiança e o texto de
  todas as perguntas feitas ao Jev. Os pesos também podem ser testados na tela, em "Pesos do composto", sem novas
  chamadas ao Jev.

### Adicionar uma call gravada (aba Vídeo)

Cada call é um JSON em `data/calls/` com as falas (`speaker`: `rep` ou `prospect`, `text`, `t` e `t_end` em segundos) e
o vídeo em `"video": {"file": "nome.mp4"}` (arquivo em `data/media/`), `{"url": "https://…"}` ou
`{"youtube_id": "…"}`. Veja `data/calls/cloudtalk.json` como exemplo.

**Calls reais de clientes não devem ir para o GitHub** (este repositório é público). Deixe o JSON e o vídeo fora do
git (veja `.gitignore`) e, em produção, coloque-os num armazenamento privado e informe o endereço do JSON em
`PRIVATE_CALLS_URLS`.

## Publicar na Vercel

O projeto já vem pronto para a Vercel (`app.py`, `vercel.json`, `.vercelignore`).

1. Importe o repositório num projeto da Vercel (o framework detectado é **FastAPI**).
2. Em **Settings → Environment Variables**, defina:
   - `OPENROUTER_API_KEY` — obrigatória;
   - `APP_PASSWORD` e `APP_SECRET` — para exigir senha da equipe; **ou** `OPEN_ACCESS=1` para deixar sem login
     (qualquer pessoa com o endereço usa o app e gasta a sua chave);
   - `PRIVATE_CALLS_URLS` — opcional, endereços dos JSON de calls privadas.
3. Para a **Chamada de vídeo**, adicione **Upstash Redis** pelo Marketplace da Vercel (plano grátis). Sem ele, você e
   o cliente podem cair em servidores diferentes e a chamada não conecta.

**Publicação automática:** `.github/workflows/deploy-vercel.yml` publica em produção a cada push na `main`. Crie
um token em [vercel.com/account/tokens](https://vercel.com/account/tokens), salve no repositório em **Settings →
Secrets and variables → Actions** com o nome `VERCEL_TOKEN` e troque `VERCEL_ORG_ID`/`VERCEL_PROJECT_ID` no arquivo
pelos do seu projeto. Sem o token, a automação não faz nada.

Como a Vercel encerra conexões longas (cerca de 5 minutos no plano grátis), o navegador reconecta sozinho e devolve a
call ao servidor, que continua a análise de onde parou sem chamar o Jev de novo.

## Outros provedores do Jev

O app escolhe o provedor pela chave que encontrar, nesta ordem: `OPENROUTER_API_KEY` (recomendado), `BEATAPI_API_KEY`,
`JEVMODEL_API_KEY` (ou uma chave `sk-…` do jevmodel.org) e `TYPESAFE_API_KEY` (API oficial). Com chaves de cota
gratuita pequena, `COPILOT_ECONOMY=1` corta o número de perguntas por fala pela metade. Detalhes em `.env.example`.

## Estrutura

```
static/index.html      tela do copiloto (Chart.js, WebSocket, microfone, vídeo, chamada)
static/sala.html       página que o cliente abre na chamada de vídeo
copilot/server.py      FastAPI: páginas, WebSocket do copiloto, sala da chamada, login
copilot/engine.py      estado da call, sinais, probabilidade, objeção, próxima jogada, reconexão
copilot/constants.py   perguntas ao Jev, pesos, limiares, modelos
copilot/jev_client.py  clientes do Jev (OpenRouter, jevmodel.org, BeatAPI, TypeSafe) e cache
copilot/personalize.py frase personalizada (LLM) verificada pelo Jev
copilot/videocall.py   salas assinadas, sinalização, túnel, transcrição da chamada
copilot/auth.py        login com senha (ou acesso aberto)
data/playbook.json     as 15 jogadas e as frases
data/calls/*.json      calls de exemplo
tests/                 testes (offline por padrão)
```

## Testes

```bash
uv run pytest -q tests/ --ignore=tests/test_integration_jev.py --ignore=tests/test_e2e_replay.py   # offline
./run.sh test                                                                                     # inclui testes com a API real
```

## Limitações

- O microfone ao vivo não identifica quem fala; use a Chamada de vídeo para isso.
- A probabilidade é um sinal para orientar o vendedor, não uma previsão calibrada. Os pesos foram ajustados com
  poucas calls; trate-os como ponto de partida.
- O Jev lê as últimas 12 falas e uma lista de fatos já confirmados, não a call inteira.
- Em redes corporativas que bloqueiam vídeo ponto a ponto, configure um servidor TURN em `ICE_SERVERS`.
