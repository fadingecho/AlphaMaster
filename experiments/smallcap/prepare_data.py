"""Freeze a current-cap universe and public daily quotes for an exploratory run.

Selection uses today's snapshot: this is deliberately NOT a point-in-time backtest.
Only public, unauthenticated market-data GET requests are made.
"""
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/raw'
END = '2026-10-23'


def fetch(url, path):
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=25) as response:
                data = json.loads(response.read())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            return data
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def history(symbol, adjustment):
    rows = {}
    for start, end in [('2022-01-01', '2023-12-31'),
                       ('2024-01-01', '2025-12-31'), ('2026-01-01', END)]:
        params = urllib.parse.urlencode({'param': ','.join([symbol, 'day', start, end, '640', adjustment])})
        data = fetch('https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?' + params,
                     RAW / (symbol + '_' + (adjustment or 'none') + '_' + start[:4] + '.json'))
        body = data['data'][symbol]
        bars = body.get(adjustment + 'day', body.get('day', []))
        for row in bars:
            if start <= row[0] <= end:
                rows[row[0]] = row[:6]
    return [rows[k] for k in sorted(rows)]


def load_stock(row):
    symbol = row['symbol']
    hfq = history(symbol, 'hfq')
    raw = history(symbol, '')
    # Require coverage, positive prices and nonzero recent volume, without choosing
    # on returns. This coverage filter itself can introduce survivorship bias.
    if len(hfq) < 1000 or hfq[0][0] > '2022-01-10' or hfq[-1][0] != END:
        return {'excluded': symbol, 'reason': 'insufficient history', 'bars': len(hfq)}
    if any(float(x[1]) <= 0 or float(x[2]) <= 0 for x in hfq):
        return {'excluded': symbol, 'reason': 'nonpositive price'}
    result = dict(row, market_cap_cny=float(row['mktcap']) * 10000,
                  history_start=hfq[0][0], history_end=hfq[-1][0], bars=len(hfq))
    (ROOT / 'data' / (symbol + '.json')).write_text(json.dumps({'metadata': result, 'hfq': hfq, 'raw': raw}, ensure_ascii=False), encoding='utf-8')
    return result


def main():
    candidates = []
    seen = set()
    pages = []
    for page in range(1, 21):
        params = urllib.parse.urlencode(dict(page=page, num=100, sort='mktcap', asc=1, node='hs_a', symbol='', _s_r_a='page'))
        rows = fetch('https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?' + params, RAW / f'caps_{page:02d}.json')
        pages.append(page)
        for r in rows:
            code, name, cap = r['code'], r['name'], float(r['mktcap']) * 10000
            if (r['symbol'] not in seen and code.startswith(('000','001','002','003','600','601','603','605'))
                    and not any(w in name.upper() for w in ['ST', '退'])
                    and 2e9 <= cap <= 1e10 and float(r['amount']) >= 2e7):
                seen.add(r['symbol'])
                candidates.append(r)
        print('cap page', page, 'eligible', len(candidates), flush=True)
        if len(candidates) >= 40:
            break
    candidates.sort(key=lambda r: (float(r['mktcap']), r['code']))
    selected, excluded = [], []
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        # Deterministic selection by cap rank, never by the order requests finish.
        for item in pool.map(load_stock, candidates[:40]):
            print(item.get('symbol', item.get('excluded')), item.get('name', item.get('reason')), item.get('bars'), flush=True)
            if 'excluded' in item:
                excluded.append(item)
            elif len(selected) < 20:
                selected.append(item)
    if len(selected) != 20:
        raise RuntimeError(f'Only {len(selected)} qualified stocks: do not silently loosen screen')
    manifest = {
        'retrieved_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'selection': 'current total market capitalization 2-10 billion CNY; main board; no current ST/delisting name; snapshot turnover >=20m CNY; first 20 by cap with sufficient history',
        'capitalization_unit': 'Sina mktcap is ten-thousand CNY; quote amount is CNY',
        'cap_snapshot_pages': pages, 'history_end': END,
        'selection_warning': 'CURRENT-cap retrospective universe with survival/coverage bias; NOT point-in-time universe, NOT an unbiased OOS strategy estimate',
        'prices': 'Tencent hfq for research total-return approximation, unadjusted for price-limit diagnostics; volume in lots',
        'selected': selected, 'excluded': excluded,
        'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').glob('s*.json'))},
    }
    (ROOT/'data/universe.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('FROZEN', len(selected), flush=True)


if __name__ == '__main__':
    main()
