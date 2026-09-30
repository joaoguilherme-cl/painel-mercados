#!/usr/bin/env python3
"""
Gera o painel diário de Ouro, Brent e USD/BRL.

Saídas:
  docs/index.html  -> painel (servido pelo GitHub Pages)
  docs/dados.json  -> as mesmas estatísticas em formato estruturado

Fonte dos dados: Yahoo Finance (via yfinance). O Investing.com não tem API
pública e bloqueia coleta automatizada; os tickers abaixo são os mesmos
instrumentos das páginas do Investing (futuros de ouro e Brent, dólar spot).
"""
from __future__ import annotations

import html
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BRT = timezone(timedelta(hours=-3))  # Brasil sem horário de verão desde 2019
SAIDA = Path(__file__).resolve().parent / "docs"
JANELA_CORR = 60  # pregões usados em cada ponto da correlação móvel

ATIVOS = [
    {
        "id": "ouro", "nome": "Ouro", "ticker": "GC=F",
        "unidade": "US$ por onça troy", "instrumento": "Futuro COMEX, 1º vencimento",
        "cor": "#9A7209", "casas": 2, "em_reais": True,
        "investing": "https://www.investing.com/commodities/gold",
    },
    {
        "id": "brent", "nome": "Petróleo Brent", "ticker": "BZ=F",
        "unidade": "US$ por barril", "instrumento": "Futuro ICE, 1º vencimento",
        "cor": "#6E3B1E", "casas": 2, "em_reais": True,
        "investing": "https://www.investing.com/commodities/brent-oil",
    },
    {
        "id": "usdbrl", "nome": "Dólar", "ticker": "BRL=X",
        "unidade": "R$ por US$", "instrumento": "Câmbio à vista",
        "cor": "#1D5C85", "casas": 4, "em_reais": False,
        "investing": "https://www.investing.com/currencies/usd-brl",
    },
]


# ---------------------------------------------------------------- dados

def baixar(ticker: str, tentativas: int = 3) -> pd.DataFrame:
    import yfinance as yf

    erro = None
    for i in range(tentativas):
        try:
            df = yf.download(ticker, period="2y", interval="1d",
                             auto_adjust=False, progress=False, threads=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close"]].dropna()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                if len(df) > 60:
                    return df
        except Exception as e:  # rede, rate limit do Yahoo etc.
            erro = e
        print(f"[{ticker}] tentativa {i + 1} sem dados ({erro})", file=sys.stderr)
        time.sleep(10 * (i + 1))
    raise RuntimeError(f"Não foi possível obter dados de {ticker}: {erro}")


# ---------------------------------------------------------- estatísticas

def _var_desde(close: pd.Series, dias: int) -> float | None:
    limite = close.index[-1] - pd.Timedelta(days=dias)
    base = close[close.index <= limite]
    return None if base.empty else float(close.iloc[-1] / base.iloc[-1] - 1)


def _rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff()
    ganho = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    perda = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = ganho.iloc[-1] / perda.iloc[-1] if perda.iloc[-1] else math.inf
    return float(100 - 100 / (1 + rs))


def estatisticas(df: pd.DataFrame) -> dict:
    c = df["Close"]
    ult, ant = float(c.iloc[-1]), float(c.iloc[-2])
    um_ano = df[df.index > df.index[-1] - pd.Timedelta(days=365)]
    c_ano = um_ano["Close"]
    ret_log = np.log(c).diff().dropna()
    ret_ano = c_ano.pct_change().dropna()

    anterior_ano = c[c.index.year < c.index[-1].year]
    ytd = float(ult / anterior_ano.iloc[-1] - 1) if not anterior_ano.empty else None

    minimo, maximo = float(um_ano["Low"].min()), float(um_ano["High"].max())
    mm = {n: float(c.rolling(n).mean().iloc[-1]) for n in (20, 50, 200)}
    drawdown = (c_ano / c_ano.cummax() - 1)

    return {
        "data": c.index[-1].strftime("%Y-%m-%d"),
        "ultimo": ult,
        "anterior": ant,
        "var_dia": ult / ant - 1,
        "maxima_dia": float(df["High"].iloc[-1]),
        "minima_dia": float(df["Low"].iloc[-1]),
        "var": {
            "1 semana": _var_desde(c, 7),
            "1 mês": _var_desde(c, 30),
            "3 meses": _var_desde(c, 91),
            "No ano": ytd,
            "12 meses": _var_desde(c, 365),
        },
        "min_52s": minimo,
        "max_52s": maximo,
        "posicao_faixa": (ult - minimo) / (maximo - minimo) if maximo > minimo else 0.5,
        "mm": mm,
        "vol_1m": float(ret_log.tail(21).std() * math.sqrt(252)),
        "vol_12m": float(ret_log[ret_log.index > c.index[-1] - pd.Timedelta(days=365)].std() * math.sqrt(252)),
        "drawdown_max_12m": float(drawdown.min()),
        "drawdown_atual": float(drawdown.iloc[-1]),
        "rsi14": _rsi(c),
        "melhor_dia": (ret_ano.idxmax().strftime("%Y-%m-%d"), float(ret_ano.max())),
        "pior_dia": (ret_ano.idxmin().strftime("%Y-%m-%d"), float(ret_ano.min())),
        "dias_alta": float((ret_ano > 0).mean()),
    }


def correlacoes(series: dict[str, pd.Series]) -> pd.DataFrame | None:
    if len(series) < 2:
        return None
    df = pd.concat(series, axis=1, join="inner")
    df = df[df.index > df.index[-1] - pd.Timedelta(days=365)]
    return df.pct_change().dropna().corr()


def correlacao_movel(series: dict[str, pd.Series]) -> dict:
    """Correlação dos retornos diários em janela móvel, para cada par de ativos."""
    if len(series) < 2:
        return {}
    ret = pd.concat(series, axis=1, join="inner").pct_change().dropna()
    corte = ret.index[-1] - pd.Timedelta(days=365)
    ids, pares = list(ret.columns), {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            s = ret[a].rolling(JANELA_CORR).corr(ret[b]).dropna()
            pares[(a, b)] = pares[(b, a)] = s[s.index > corte]
    return pares


# ------------------------------------------------------------ formatação

def num(v: float, casas: int = 2) -> str:
    s = f"{v:,.{casas}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(v: float | None, casas: int = 2, sinal: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    s = num(abs(v) * 100, casas) + "%"
    if not sinal:
        return s
    return ("+" if v > 0 else "−" if v < 0 else "") + s


def classe(v: float | None) -> str:
    if v is None or v == 0:
        return ""
    return "sobe" if v > 0 else "cai"


def data_br(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d/%m/%Y")


# ------------------------------------------------------------- gráfico

def grafico_svg(df: pd.DataFrame, cor: str, casas: int) -> str:
    c_full = df["Close"]
    mm50_full = c_full.rolling(50).mean()
    corte = df.index[-1] - pd.Timedelta(days=365)
    c = c_full[c_full.index > corte]
    mm50 = mm50_full[mm50_full.index > corte]

    W, H, E, D, T, B = 760, 240, 8, 64, 10, 26
    lo = float(min(c.min(), mm50.min(skipna=True)))
    hi = float(max(c.max(), mm50.max(skipna=True)))
    folga = (hi - lo) * 0.08 or abs(hi) * 0.01 or 1
    lo, hi = lo - folga, hi + folga
    n = len(c)

    def x(i): return E + i * (W - E - D) / max(n - 1, 1)
    def y(v): return T + (hi - v) * (H - T - B) / (hi - lo)

    pts = [(x(i), y(v)) for i, v in enumerate(c.values)]
    linha = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    area = linha + f" L{pts[-1][0]:.1f},{H - B} L{pts[0][0]:.1f},{H - B} Z"
    mm_pts = [(x(i), y(v)) for i, v in enumerate(mm50.values) if not math.isnan(v)]
    mm_linha = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in mm_pts) if mm_pts else ""

    grade = []
    for k in range(5):
        v = lo + (hi - lo) * (k + 0.5) / 5
        grade.append(
            f'<line x1="{E}" x2="{W - D}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grade"/>'
            f'<text x="{W - D + 8}" y="{y(v) + 4:.1f}" class="eixo">{num(v, min(casas, 2) if v > 100 else casas)}</text>'
        )
    # marcas de mês no eixo x
    meses = []
    vistos = set()
    nomes = "jan fev mar abr mai jun jul ago set out nov dez".split()
    for i, d in enumerate(c.index):
        chave = (d.year, d.month)
        if chave in vistos or i == 0:
            vistos.add(chave)
            continue
        vistos.add(chave)
        if d.month % 2 == 1 or n < 150:
            rot = nomes[d.month - 1] + (f" {str(d.year)[2:]}" if d.month == 1 else "")
            meses.append(f'<text x="{x(i):.1f}" y="{H - 8}" class="eixo" text-anchor="middle">{rot}</text>')

    ux, uy = pts[-1]
    return f"""<svg viewBox="0 0 {W} {H}" role="img" aria-label="Preço de fechamento nos últimos 12 meses">
  <defs><linearGradient id="g-{cor[1:]}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="{cor}" stop-opacity=".22"/><stop offset="1" stop-color="{cor}" stop-opacity="0"/>
  </linearGradient></defs>
  {''.join(grade)}
  <path d="{area}" fill="url(#g-{cor[1:]})"/>
  <path d="{mm_linha}" class="mm"/>
  <path d="{linha}" fill="none" stroke="{cor}" stroke-width="1.8" stroke-linejoin="round"/>
  <circle cx="{ux:.1f}" cy="{uy:.1f}" r="4" fill="{cor}" stroke="#fff" stroke-width="2"/>
  {''.join(meses)}
</svg>"""


def grafico_correlacao_svg(linhas: list[tuple[str, str, pd.Series]]) -> str:
    """linhas: (nome, cor, série de correlação móvel)."""
    W, H, E, D, T, B = 760, 132, 8, 64, 8, 22
    ini = min(s.index[0] for _, _, s in linhas)
    fim = max(s.index[-1] for _, _, s in linhas)
    total = (fim - ini).total_seconds() or 1

    def x(d): return E + (d - ini).total_seconds() / total * (W - E - D)
    def y(v): return T + (1 - v) * (H - T - B) / 2

    partes = []
    for v in (1, 0.5, 0, -0.5, -1):
        rot = ("+" if v > 0 else "−" if v < 0 else "") + num(abs(v), 1)
        partes.append(
            f'<line x1="{E}" x2="{W - D}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"zero" if v == 0 else "grade"}"/>'
            f'<text x="{W - D + 8}" y="{y(v) + 4:.1f}" class="eixo">{rot}</text>')
    nomes = "jan fev mar abr mai jun jul ago set out nov dez".split()
    for d in pd.date_range(ini, fim, freq="MS")[1:]:
        if d.month % 2 == 1:
            rot = nomes[d.month - 1] + (f" {str(d.year)[2:]}" if d.month == 1 else "")
            partes.append(f'<text x="{x(d):.1f}" y="{H - 6}" class="eixo" text-anchor="middle">{rot}</text>')
    for _, cor, s in linhas:
        caminho = "M" + " L".join(f"{x(d):.1f},{y(v):.1f}" for d, v in s.items())
        partes.append(f'<path d="{caminho}" fill="none" stroke="{cor}" stroke-width="1.6" stroke-linejoin="round"/>')
        partes.append(f'<circle cx="{x(s.index[-1]):.1f}" cy="{y(s.iloc[-1]):.1f}" r="3.5" fill="{cor}" stroke="#fff" stroke-width="1.5"/>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="Correlação móvel com os outros ativos nos últimos 12 meses">{"".join(partes)}</svg>')


# --------------------------------------------------------------- HTML

CSS = """
:root{
  --papel:#F3F4F1; --folha:#FFFFFF; --tinta:#1C2530; --suave:#5F6B78; --linha:#D9DDD8;
  --sobe:#17744F; --cai:#B0303F;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --papel:#151A20; --folha:#1C232B; --tinta:#E6E9EC; --suave:#98A3AE; --linha:#2E3843;
  --sobe:#4CC38A; --cai:#F07482;
}}
:root[data-theme="dark"]{
  --papel:#151A20; --folha:#1C232B; --tinta:#E6E9EC; --suave:#98A3AE; --linha:#2E3843;
  --sobe:#4CC38A; --cai:#F07482;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--papel);color:var(--tinta);
  font-family:"Instrument Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:15px;line-height:1.5;font-variant-numeric:tabular-nums;
  padding:env(safe-area-inset-top,0) 0 env(safe-area-inset-bottom,0)}
main{max-width:1080px;margin:0 auto;padding:40px 20px 64px}
header{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:8px 24px;
  border-bottom:2px solid var(--tinta);padding-bottom:14px;margin-bottom:28px}
h1{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:clamp(28px,5vw,40px);
  letter-spacing:-.02em;margin:0;line-height:1.1}
.carimbo{color:var(--suave);font-size:14px;margin:0}
.carimbo strong{color:var(--tinta);font-weight:600}

.ativo{background:var(--folha);border:1px solid var(--linha);border-left:6px solid var(--cor);
  border-radius:4px;padding:24px 26px;margin-bottom:22px;
  display:grid;grid-template-columns:minmax(230px,300px) 1fr;gap:10px 36px}
.ativo h2{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:24px;margin:0;letter-spacing:-.01em}
.ativo .unid{color:var(--suave);font-size:13px;margin:2px 0 16px}
.preco{font-size:44px;font-weight:600;letter-spacing:-.02em;line-height:1}
.dia{font-size:17px;font-weight:600;margin-top:6px}
.sobe{color:var(--sobe)} .cai{color:var(--cai)}
.detalhe{color:var(--suave);font-size:13px;margin-top:4px}
.reais{margin-top:14px;padding-top:12px;border-top:1px dashed var(--linha);font-size:14px}
.reais b{font-weight:600}

.faixa{margin-top:18px}
.faixa .trilho{position:relative;height:8px;border-radius:4px;
  background:linear-gradient(90deg,color-mix(in srgb,var(--cor) 15%,transparent),color-mix(in srgb,var(--cor) 55%,transparent))}
.faixa .marca{position:absolute;top:-5px;width:4px;height:18px;border-radius:2px;background:var(--tinta);transform:translateX(-2px)}
.faixa .pontas{display:flex;justify-content:space-between;font-size:12px;color:var(--suave);margin-top:6px}
.faixa .titulo{font-size:12px;color:var(--suave);margin-bottom:8px}

.grafico svg{width:100%;height:auto;display:block}
.grafico .grade{stroke:var(--linha);stroke-width:1}
.grafico .eixo{fill:var(--suave);font-size:11px;font-family:inherit}
.grafico .mm{fill:none;stroke:var(--suave);stroke-width:1.2;stroke-dasharray:4 4}
.legenda{font-size:12px;color:var(--suave);margin:4px 0 0}
.corr{margin-top:20px;padding-top:14px;border-top:1px dashed var(--linha)}
.corr h3{font-size:13px;font-weight:600;margin:0 0 2px}
.corr .chaves{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:13px;margin:0 0 4px}
.corr .chave::before{content:"";display:inline-block;width:14px;height:3px;border-radius:2px;
  background:var(--c);vertical-align:middle;margin-right:6px}
.corr .chave b{font-weight:600}
.grafico .zero{stroke:var(--suave);stroke-width:1;opacity:.6}

.numeros{grid-column:1/-1;display:grid;grid-template-columns:1fr 1.3fr 1.1fr 1.1fr;gap:0 28px;
  border-top:1px solid var(--linha);padding-top:16px;margin-top:10px}
.numeros h3{font-size:13px;font-weight:600;margin:0 0 6px;color:var(--suave)}
.numeros dl{margin:0;display:grid;grid-template-columns:1fr auto;row-gap:3px;font-size:14px}
.numeros dt{color:var(--tinta)} .numeros dd{margin:0;text-align:right;font-weight:500;white-space:nowrap;padding-left:8px}
.numeros .nota{font-weight:400;color:var(--suave);font-size:12px}

.erro{grid-column:1/-1;color:var(--cai)}

section.extra{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.bloco{background:var(--folha);border:1px solid var(--linha);border-radius:4px;padding:22px 26px}
.bloco h2{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:20px;margin:0 0 4px}
.bloco p{color:var(--suave);font-size:13px;margin:0 0 14px;max-width:62ch}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--linha)}
th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--suave);font-size:13px}
td.c{font-weight:600}
.fontes a{color:inherit}
.fontes li{margin-bottom:6px;font-size:14px}
.fontes ul{padding-left:18px;margin:0}

@media (max-width:820px){
  .ativo{grid-template-columns:1fr}
  .numeros{grid-template-columns:1fr 1fr;row-gap:18px}
  section.extra{grid-template-columns:1fr}
}
@media (max-width:460px){
  main{padding:24px 14px 48px}
  .ativo{padding:20px 18px}
  .numeros{grid-template-columns:1fr}
  .preco{font-size:38px}
}
"""


def bloco_correlacao(linhas: list) -> str:
    if not linhas:
        return ""
    chaves = "".join(
        f'<span class="chave" style="--c:{cor}">Com {nome}: <b>{num(serie.iloc[-1], 2).replace("-", "−")}</b></span>'
        for nome, cor, serie in linhas)
    return f"""<div class="corr">
      <h3>Correlação móvel com os outros ativos</h3>
      <div class="chaves">{chaves}</div>
      {grafico_correlacao_svg(linhas)}
      <p class="legenda">Cada ponto é a correlação dos retornos diários nos {JANELA_CORR} pregões anteriores àquela data. O valor em destaque é o mais recente.</p>
    </div>"""


def bloco_ativo(a: dict, df: pd.DataFrame | None, s: dict | None,
                usdbrl: float | None, erro: str | None, corr_linhas: list | None = None) -> str:
    cab = (f'<h2>{html.escape(a["nome"])}</h2>'
           f'<p class="unid">{a["unidade"]}. {a["instrumento"]}.</p>')
    if s is None:
        return (f'<article class="ativo" style="--cor:{a["cor"]}"><div>{cab}</div>'
                f'<p class="erro">Os dados de {a["nome"]} não chegaram nesta atualização '
                f'({html.escape(str(erro))}). O painel tenta de novo na próxima execução.</p></article>')

    k = a["casas"]
    reais = ""
    if a["em_reais"] and usdbrl:
        reais = (f'<div class="reais">Em reais: <b>R$ {num(s["ultimo"] * usdbrl, 2)}</b>'
                 f' <span class="detalhe">(câmbio de R$ {num(usdbrl, 4)})</span></div>')

    faixa = f"""<div class="faixa">
      <div class="titulo">Faixa das últimas 52 semanas</div>
      <div class="trilho"><span class="marca" style="left:{s['posicao_faixa'] * 100:.1f}%"></span></div>
      <div class="pontas"><span>{num(s['min_52s'], k)}</span><span>{num(s['max_52s'], k)}</span></div>
    </div>"""

    variacoes = "".join(
        f'<dt>{rot}</dt><dd class="{classe(v)}">{pct(v)}</dd>' for rot, v in s["var"].items())
    medias = "".join(
        f'<dt>Média {n}d</dt><dd>{num(v, k)} '
        f'<span class="nota">({pct(s["ultimo"] / v - 1, 1)})</span></dd>'
        for n, v in s["mm"].items() if not math.isnan(v))
    rsi = s["rsi14"]
    rsi_txt = " sobrecomprado" if rsi >= 70 else " sobrevendido" if rsi <= 30 else ""
    melhor_d, melhor_v = s["melhor_dia"]
    pior_d, pior_v = s["pior_dia"]

    return f"""<article class="ativo" style="--cor:{a['cor']}" id="{a['id']}">
  <div>
    {cab}
    <div class="preco">{num(s['ultimo'], k)}</div>
    <div class="dia {classe(s['var_dia'])}">{pct(s['var_dia'])} no dia</div>
    <div class="detalhe">Fechamento anterior {num(s['anterior'], k)}. Mínima e máxima do dia {num(s['minima_dia'], k)} e {num(s['maxima_dia'], k)}.</div>
    {reais}
    {faixa}
  </div>
  <div class="grafico">
    {grafico_svg(df, a['cor'], k)}
    <p class="legenda">Fechamentos dos últimos 12 meses. A linha tracejada é a média móvel de 50 dias. Última cotação em {data_br(s['data'])}.</p>
    {bloco_correlacao(corr_linhas or [])}
  </div>
  <div class="numeros">
    <div><h3>Variação</h3><dl>{variacoes}</dl></div>
    <div><h3>Tendência</h3><dl>{medias}<dt>IFR 14d</dt><dd>{num(rsi, 1)}<span class="nota">{rsi_txt}</span></dd></dl></div>
    <div><h3>Risco</h3><dl>
      <dt>Volatilidade 1 mês</dt><dd>{pct(s['vol_1m'], 1, False)}</dd>
      <dt>Volatilidade 12 meses</dt><dd>{pct(s['vol_12m'], 1, False)}</dd>
      <dt>Maior queda em 12m</dt><dd class="cai">{pct(s['drawdown_max_12m'], 1)}</dd>
      <dt>Distância do topo</dt><dd>{pct(s['drawdown_atual'], 1)}</dd>
    </dl></div>
    <div><h3>Extremos em 12 meses</h3><dl>
      <dt>Melhor dia <span class="nota">{data_br(melhor_d)}</span></dt><dd class="sobe">{pct(melhor_v)}</dd>
      <dt>Pior dia <span class="nota">{data_br(pior_d)}</span></dt><dd class="cai">{pct(pior_v)}</dd>
      <dt>Dias de alta</dt><dd>{pct(s['dias_alta'], 0, False)}</dd>
    </dl></div>
  </div>
</article>"""


def tabela_correlacao(corr: pd.DataFrame | None, nomes: dict) -> str:
    if corr is None:
        return "<p>Correlação indisponível: faltaram dados de pelo menos um ativo.</p>"
    cols = list(corr.columns)
    cab = "".join(f"<th>{nomes[c]}</th>" for c in cols)
    linhas = ""
    for r in cols:
        cel = ""
        for c in cols:
            v = corr.loc[r, c]
            cel += "<td>–</td>" if r == c else f'<td class="c">{num(v, 2)}</td>'
        linhas += f"<tr><th>{nomes[r]}</th>{cel}</tr>"
    return f"<table><thead><tr><th></th>{cab}</tr></thead><tbody>{linhas}</tbody></table>"


def gerar_html(resultados: list, corr, agora: datetime, moveis: dict | None = None) -> str:
    usdbrl = next((r[2]["ultimo"] for r in resultados if r[0]["id"] == "usdbrl" and r[2]), None)
    nomes = {a["id"]: a["nome"] for a in ATIVOS}
    cores = {a["id"]: a["cor"] for a in ATIVOS}
    moveis = moveis or {}

    def linhas(aid):
        return [(nomes[o], cores[o], moveis[(aid, o)]) for o in nomes
                if o != aid and (aid, o) in moveis and not moveis[(aid, o)].empty]

    blocos = "\n".join(bloco_ativo(a, df, s, usdbrl, e, linhas(a["id"])) for a, df, s, e in resultados)
    fontes = "".join(
        f'<li>{a["nome"]}: Yahoo Finance, ticker {a["ticker"]}. '
        f'<a href="{a["investing"]}" target="_blank" rel="noopener">Ver no Investing.com</a></li>'
        for a in ATIVOS)

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Ouro, Brent e dólar | {agora:%d/%m/%Y}</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📈</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<main>
<header>
  <h1>Ouro, Brent e dólar</h1>
  <p class="carimbo">Atualizado em <strong>{agora:%d/%m/%Y} às {agora:%H:%M}</strong> (Brasília). Próxima atualização amanhã às 7h.</p>
</header>
{blocos}
<section class="extra">
  <div class="bloco">
    <h2>Correlação entre os ativos</h2>
    <p>Correlação dos retornos diários nos últimos 12 meses, usando só os dias em que os três negociaram. Perto de 1, andam juntos. Perto de −1, em direções opostas.</p>
    {tabela_correlacao(corr, nomes)}
  </div>
  <div class="bloco fontes">
    <h2>Como ler este painel</h2>
    <p>Às 7h os futuros de ouro e Brent já estão negociando, então o preço e a variação do dia refletem o pregão em andamento, comparados ao fechamento anterior. Médias 20d, 50d e 200d são médias móveis simples, com a distância do preço atual entre parênteses. Volatilidade é o desvio-padrão dos retornos diários, anualizado. A correlação móvel mostra como essa relação mudou ao longo do ano. IFR acima de 70 costuma ser lido como sobrecompra e abaixo de 30 como sobrevenda.</p>
    <ul>{fontes}</ul>
  </div>
</section>
</main>
</body>
</html>"""


# ---------------------------------------------------------------- main

def executar(baixador=baixar) -> int:
    agora = datetime.now(BRT)
    resultados, closes, dados_json = [], {}, {}
    for a in ATIVOS:
        try:
            df = baixador(a["ticker"])
            s = estatisticas(df)
            resultados.append((a, df, s, None))
            closes[a["id"]] = df["Close"]
            dados_json[a["id"]] = {"nome": a["nome"], "ticker": a["ticker"], **s}
        except Exception as e:
            print(f"[{a['ticker']}] falhou: {e}", file=sys.stderr)
            resultados.append((a, None, None, e))

    if not closes:
        print("Nenhum ativo retornou dados; o painel anterior foi mantido.", file=sys.stderr)
        return 1

    corr = correlacoes(closes)
    moveis = correlacao_movel(closes)
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "index.html").write_text(gerar_html(resultados, corr, agora, moveis), encoding="utf-8")
    dados_json["gerado_em"] = agora.isoformat(timespec="minutes")
    dados_json["correlacao_12m"] = None if corr is None else corr.round(4).to_dict()
    dados_json[f"correlacao_movel_{JANELA_CORR}p_atual"] = {
        f"{a}-{b}": round(float(s.iloc[-1]), 4)
        for (a, b), s in moveis.items() if a < b and not s.empty}
    (SAIDA / "dados.json").write_text(
        json.dumps(dados_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Painel gerado em {SAIDA / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(executar())
