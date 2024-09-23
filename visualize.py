import plotly.express as px
from sklearn.datasets import load_digits
from umap import UMAP

import os

print("Current working directory:", os.getcwd())


def main():
    digits = load_digits()
    umap_2d = UMAP()
    umap_2d.fit(digits.data)

    projections = umap_2d.transform(digits.data)

    fig = px.scatter(
        projections,
        x=0,
        y=1,
        color=digits.target.astype(str),
        labels={"color": "digit"},
    )

    print("Generating index.html...")
    fig.write_html("index.html")


# Call the main function
if __name__ == "__main__":
    main()
