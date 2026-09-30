# Painel diário: Ouro, Brent e dólar

Todo dia às 7h (Brasília) o GitHub Actions baixa as cotações, calcula as
estatísticas e publica `docs/index.html` no GitHub Pages.

| Ativo | Ticker (Yahoo Finance) | Equivalente no Investing.com |
|---|---|---|
| Ouro | `GC=F` | Gold Futures |
| Petróleo Brent | `BZ=F` | Brent Oil Futures |
| Dólar | `BRL=X` | USD/BRL |

## Como colocar no ar (uma vez só)

1. Crie um repositório no GitHub (pode ser privado se o seu plano permitir Pages privado; senão, público).
2. Envie todo o conteúdo desta pasta, incluindo a pasta oculta `.github`.
3. Em **Settings → Pages**, escolha *Deploy from a branch*, branch `main`, pasta `/docs`.
4. Em **Actions**, abra "Atualizar painel" e clique em **Run workflow** para gerar a primeira versão.
5. O endereço do painel aparece em Settings → Pages (algo como `https://seu-usuario.github.io/nome-do-repo/`).

## Rodar no seu computador

```bash
pip install -r requirements.txt
python gerar_dashboard.py
# abre docs/index.html no navegador
```

## Observações

- O agendamento do GitHub pode atrasar alguns minutos em horários de pico.
- Se um ativo falhar, o painel sai com os outros dois e um aviso no lugar do que faltou.
  Se todos falharem, a versão anterior é mantida.
- `docs/dados.json` traz as mesmas estatísticas em formato estruturado, útil para
  integrar com outras bases depois.
- Para mudar o horário, edite a linha `cron` em `.github/workflows/atualizar-dashboard.yml`
  (horário em UTC: 7h de Brasília = 10h UTC).
