import matplotlib.pyplot as plt


def plot_results(
    cm, importances, top=20, out_path="confusion_matrice_et_importances.png", show=True
):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 2]}
    )

    # Confusion Matrix
    cm_values = cm.values
    im = ax1.imshow(cm_values, cmap="Blues", vmin=0)
    ax1.set_xticks([0, 1], labels=["Humain", "Machine"])
    ax1.set_yticks([0, 1], labels=["Humain", "Machine"])
    ax1.set_xlabel("predicted class", fontsize=12)
    ax1.set_ylabel("real class", fontsize=12)
    ax1.set_title("Confusion Matrix", fontsize=13)
    for i in range(2):
        for j in range(2):
            n = cm_values[i, j]
            pct = n / cm_values[i].sum() * 100
            ax1.text(
                j,
                i,
                f"{n}\n({pct:.1f} %)",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if n > cm_values.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax1, shrink=0.8, label="number of conversations")

    # feature importance
    top_imp = importances.sort_values(ascending=False).head(top)
    ax2.bar(range(top), top_imp.values, color="steelblue")
    ax2.set_xticks(
        range(top), labels=top_imp.index, rotation=60, ha="right", fontsize=8
    )
    ax2.set_xlabel("SEANCE indices", fontsize=12)
    ax2.set_ylabel("Importance", fontsize=12)
    ax2.set_title(f"Top {top} of feature importance ", fontsize=13)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved : {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig
