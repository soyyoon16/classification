import plotly.express as px
from sklearn.datasets import load_iris
from umap import UMAP
import os

print("Current working directory:", os.getcwd())


def main():
    iris = load_iris()
    umap_2d = UMAP()
    umap_2d.fit(iris.data)

    projections = umap_2d.transform(iris.data)

    fig = px.scatter(
        projections,
        x=0,
        y=1,
        color=iris.target.astype(str),
        labels={"color": "iris"},
    )

    print("Generating index.html...")
    fig.write_html("index.html")


# Call the main function
if __name__ == "__main__":
    main()
