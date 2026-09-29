from __future__ import annotations

from ashare_quant.data.synthetic import generate_synthetic_bars

if __name__ == "__main__":
    output = "data/demo_bars.csv"
    frame = generate_synthetic_bars()
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame):,} synthetic rows to {output}")
