from pathlib import Path
import numpy as np
import pandas as pd


REPO = Path.cwd()

SOURCE = (
    REPO / 'assets/pinn_supersonic/csv/computational_cost/cost500/table_cost500_coordinates.csv'
)

HYBRID_COST1 = (
    REPO / 'assets/pinn_supersonic/csv/computational_cost/cost1/table_cost1_input.csv'
)

OUT = (
    REPO
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM/N76/"
      "final_benchmarks/classical_from_scratch"
)

CHUNKS = OUT / "cost500_chunks"


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHUNKS.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(SOURCE)

    assert len(df) == 500

    if "cost_id" not in df.columns:
        df = df.copy()
        df["cost_id"] = np.arange(
            500,
            dtype=int,
        )

    df = (
        df.sort_values("cost_id")
        .reset_index(drop=True)
    )

    assert df["cost_id"].tolist() == list(
        range(500)
    )

    assert (
        df[["Mach", "alpha"]]
        .drop_duplicates()
        .shape[0]
        == 500
    )

    # Ensure COST1 is exactly the same fixed query
    # as in the hybrid benchmark.
    old = pd.read_csv(
        HYBRID_COST1
    )

    assert len(old) == 1

    assert np.isclose(
        float(df.loc[0, "Mach"]),
        float(old.loc[0, "Mach"]),
        atol=1e-12,
    )

    assert np.isclose(
        float(df.loc[0, "alpha"]),
        float(old.loc[0, "alpha"]),
        atol=1e-12,
    )

    df.iloc[[0]].to_csv(
        OUT
        / "classical_COST1_input.csv",
        index=False,
    )

    # 100 chunks x 5 points:
    # intentionally conservative for the
    # 20-hour Jean Zay job limit.
    chunks = np.array_split(
        df,
        100,
    )

    assert len(chunks) == 100

    total = 0

    for i, chunk in enumerate(chunks):

        assert len(chunk) == 5

        chunk.to_csv(
            CHUNKS
            / f"classical_COST500_chunk_{i:03d}.csv",
            index=False,
        )

        total += len(chunk)

    assert total == 500

    print(
        "COST1:",
        df.loc[
            0,
            ["Mach", "alpha"]
        ].to_dict(),
    )

    print(
        "COST500 points:",
        len(df),
    )

    print(
        "chunks:",
        len(chunks),
    )

    print("PREPARATION: PASS")


if __name__ == "__main__":
    main()
