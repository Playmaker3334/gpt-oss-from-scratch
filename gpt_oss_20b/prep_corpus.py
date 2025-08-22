#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, json, random
from collections import Counter, deque
from math import sqrt

WORD_RE = re.compile(r"\w+", re.UNICODE)

def iter_paragraphs(path):
    with open(path, "r", encoding="utf-8") as f:
        buf = []
        for line in f:
            s = line.strip()
            if s:
                buf.append(s)
            else:
                if buf:
                    yield " ".join(buf)
                    buf = []
        if buf:
            yield " ".join(buf)

def summarize(path, head_n=5, tail_n=5, random_n=5, sample_words_paras=50000, limit=None):
    stats = {
        "file": os.path.abspath(path),
        "size_bytes": os.path.getsize(path),
        "paras": 0,
        "chars_total": 0,
        "words_total_est": 0,
        "min_chars": None,
        "max_chars": 0,
        "mean_chars": 0.0,
        "std_chars": 0.0,
    }

    head = []
    tail = deque(maxlen=tail_n)
    reservoir = []
    counter_words = Counter()

    # one-pass streaming stats (Welford)
    mean = 0.0
    M2 = 0.0
    n = 0

    rng = random.Random(1337)
    for i, p in enumerate(iter_paragraphs(path), 1):
        if limit and i > limit: break
        L = len(p)
        w = len(WORD_RE.findall(p))

        if len(head) < head_n:
            head.append(p)
        tail.append(p)
        if len(reservoir) < random_n:
            reservoir.append(p)
        else:
            j = rng.randint(1, i)
            if j <= random_n:
                reservoir[j-1] = p

        stats["paras"] += 1
        stats["chars_total"] += L
        stats["words_total_est"] += w
        stats["min_chars"] = L if stats["min_chars"] is None else min(stats["min_chars"], L)
        stats["max_chars"] = max(stats["max_chars"], L)

        # Welford
        n += 1
        delta = L - mean
        mean += delta / n
        M2 += delta * (L - mean)

        # word freq only on a sample of paragraphs (to keep it fast)
        if i <= sample_words_paras:
            counter_words.update([t.lower() for t in WORD_RE.findall(p)])

    stats["mean_chars"] = mean
    stats["std_chars"] = sqrt(M2 / n) if n > 1 else 0.0

    top_words = counter_words.most_common(50)
    return stats, head, list(tail), reservoir, top_words

def human_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def main():
    ap = argparse.ArgumentParser(description="Analiza un corpus por párrafos (UTF-8).")
    ap.add_argument("--in", dest="inp", required=True, help="Ruta del archivo .txt")
    ap.add_argument("--head", type=int, default=5, help="Párrafos iniciales a mostrar")
    ap.add_argument("--tail", type=int, default=5, help="Párrafos finales a mostrar")
    ap.add_argument("--random", type=int, default=5, help="Párrafos aleatorios a mostrar")
    ap.add_argument("--sample-words", type=int, default=50000, help="Párrafos muestreados para top de palabras")
    ap.add_argument("--limit", type=int, default=None, help="Limitar num. de párrafos procesados")
    ap.add_argument("--report", type=str, default=None, help="Guardar métricas JSON en esta ruta")
    args = ap.parse_args()

    stats, head, tail, rnd, topw = summarize(
        args.inp, head_n=args.head, tail_n=args.tail, random_n=args.random,
        sample_words_paras=args.sample_words, limit=args.limit
    )

    print("\n========== RESUMEN ==========")
    print(f"Archivo: {stats['file']}")
    print(f"Tamaño: {human_bytes(stats['size_bytes'])}")
    print(f"Párrafos: {stats['paras']:,}")
    print(f"Caracteres totales: {stats['chars_total']:,}")
    print(f"Palabras (estimado): {stats['words_total_est']:,}")
    print(f"Longitud (chars)  min={stats['min_chars']}  max={stats['max_chars']}")
    print(f"Longitud (chars)  media={stats['mean_chars']:.2f}  std={stats['std_chars']:.2f}")

    def show(title, items):
        print(f"\n--- {title} ({len(items)}) ---")
        for i, p in enumerate(items, 1):
            snippet = (p[:240] + "…") if len(p) > 240 else p
            print(f"[{i}] {snippet}")

    show("HEAD", head)
    show("TAIL", tail)
    show("RANDOM", rnd)

    print("\n--- TOP 50 PALABRAS (muestra) ---")
    for w, c in topw:
        print(f"{w}\t{c}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nMétricas guardadas en: {os.path.abspath(args.report)}")

if __name__ == "__main__":
    main()
