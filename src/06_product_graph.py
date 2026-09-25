from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_FILE = PROJECT_ROOT / "database" / "online_retail.db"
TABLE_NAME = "transactions"

NON_MERCHANDISE_CODES = {
    "POST",
    "DOT",
    "M",
    "C2",
    "23444",
    "AMAZONFEE",
    "S",
    "B",
}


def load_sale_baskets() -> pd.DataFrame:
    """Load unique merchandise appearances in sale invoices."""

    query = f"""
        SELECT
            InvoiceNo,
            StockCode,
            Description
        FROM {TABLE_NAME}
        WHERE TransactionType = 'Sale'
    """

    with sqlite3.connect(DATABASE_FILE) as connection:
        df = pd.read_sql_query(
            query,
            connection,
        )

    df["StockCode"] = (
        df["StockCode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Description"] = (
        df["Description"]
        .astype("string")
        .str.strip()
    )

    df = df[
        ~df["StockCode"].isin(
            NON_MERCHANDISE_CODES
        )
    ].copy()

    df = (
        df.drop_duplicates(
            subset=["InvoiceNo", "StockCode"]
        )
        .reset_index(drop=True)
    )

    return df

def build_incidence_matrix(
    df: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build invoice-product incidence matrix."""

    invoice_codes, invoices = pd.factorize(
        df["InvoiceNo"],
        sort=True,
    )

    product_codes, products = pd.factorize(
        df["StockCode"],
        sort=True,
    )

    values = np.ones(
        len(df),
        dtype=np.int32,
    )

    incidence = sparse.csr_matrix(
        (
            values,
            (
                invoice_codes,
                product_codes,
            ),
        ),
        shape=(
            len(invoices),
            len(products),
        ),
        dtype=np.int32,
    )

    return incidence, invoices, products


def build_cooccurrence(
    incidence: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Compute weighted product co-occurrence matrix."""

    cooccurrence = (
        incidence.T
        @ incidence
    ).tocsr()

    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()

    return cooccurrence


def validate_matrices(
    incidence: sparse.csr_matrix,
    cooccurrence: sparse.csr_matrix,
    invoices: np.ndarray,
    products: np.ndarray,
) -> None:
    """Validate cleaned incidence and co-occurrence matrices."""

    assert incidence.shape == (
        len(invoices),
        len(products),
    )

    assert 0 < len(invoices) <= 19_960
    assert 0 < len(products) <= 3_922

    assert incidence.nnz > 0

    assert incidence.data.min() == 1
    assert incidence.data.max() == 1

    assert cooccurrence.shape == (
        len(products),
        len(products),
    )

    assert cooccurrence.nnz > 0

    assert cooccurrence.diagonal().sum() == 0

    asymmetry = (
        cooccurrence
        - cooccurrence.T
    )

    assert asymmetry.nnz == 0

    assert np.all(
        cooccurrence.data > 0
    )


def threshold_diagnostics(
    cooccurrence: sparse.csr_matrix,
    products: np.ndarray,
    thresholds=(50, 75, 100, 150, 200, 300),
) -> pd.DataFrame:
    """Evaluate graph size at several co-purchase thresholds."""

    upper = sparse.triu(
        cooccurrence,
        k=1,
        format="coo",
    )

    rows = []

    for threshold in thresholds:
        mask = upper.data >= threshold

        edge_rows = upper.row[mask]
        edge_cols = upper.col[mask]
        weights = upper.data[mask]

        graph = nx.Graph()

        graph.add_nodes_from(
            range(len(products))
        )

        graph.add_weighted_edges_from(
            zip(
                edge_rows.tolist(),
                edge_cols.tolist(),
                weights.tolist(),
            )
        )

        active_nodes = [
            node
            for node, degree
            in graph.degree()
            if degree > 0
        ]

        active_graph = graph.subgraph(
            active_nodes
        )

        if active_graph.number_of_nodes() > 0:
            components = list(
                nx.connected_components(
                    active_graph
                )
            )

            largest_component = max(
                len(component)
                for component
                in components
            )

            n_components = len(components)

        else:
            largest_component = 0
            n_components = 0

        rows.append(
            {
                "Threshold": threshold,
                "Nodes": len(active_nodes),
                "Edges": graph.number_of_edges(),
                "Components": n_components,
                "LargestComponent": largest_component,
            }
        )

    return pd.DataFrame(rows)


def strongest_pairs(
    cooccurrence: sparse.csr_matrix,
    products: np.ndarray,
    df: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """Return the strongest weighted product pairs."""

    upper = sparse.triu(
        cooccurrence,
        k=1,
        format="coo",
    )

    order = np.argsort(
        upper.data
    )[::-1][:top_n]

    descriptions = (
        df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(
            lambda values: values.mode().iloc[0]
            if not values.mode().empty
            else values.iloc[0]
        )
        .to_dict()
    )

    rows = []

    for index in order:
        product_a = str(
            products[upper.row[index]]
        )

        product_b = str(
            products[upper.col[index]]
        )

        rows.append(
            {
                "ProductA": product_a,
                "DescriptionA": descriptions.get(
                    product_a,
                    "",
                ),
                "ProductB": product_b,
                "DescriptionB": descriptions.get(
                    product_b,
                    "",
                ),
                "Weight": int(
                    upper.data[index]
                ),
            }
        )

    return pd.DataFrame(rows)



def build_product_graph(
    cooccurrence: sparse.csr_matrix,
    products: np.ndarray,
    df: pd.DataFrame,
    threshold: int = 100,
) -> nx.Graph:
    """Build the weighted product co-purchase graph."""

    descriptions = (
        df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(
            lambda values: values.mode().iloc[0]
            if not values.mode().empty
            else values.iloc[0]
        )
        .to_dict()
    )

    upper = sparse.triu(
        cooccurrence,
        k=1,
        format="coo",
    )

    mask = upper.data >= threshold

    graph = nx.Graph()

    active_indices = set(
        upper.row[mask].tolist()
        + upper.col[mask].tolist()
    )

    for index in active_indices:
        stock_code = str(products[index])

        graph.add_node(
            index,
            StockCode=stock_code,
            Description=descriptions.get(
                stock_code,
                "",
            ),
        )

    for row, col, weight in zip(
        upper.row[mask],
        upper.col[mask],
        upper.data[mask],
    ):
        graph.add_edge(
            int(row),
            int(col),
            weight=int(weight),
        )

    return graph


def graph_summary(
    graph: nx.Graph,
) -> tuple[pd.DataFrame, dict]:
    """Calculate graph-level statistics and product centralities."""

    components = sorted(
        nx.connected_components(graph),
        key=len,
        reverse=True,
    )

    giant = graph.subgraph(
        components[0]
    ).copy()

    weighted_degree = dict(
        giant.degree(weight="weight")
    )

    degree = dict(
        giant.degree()
    )

    pagerank = nx.pagerank(
        giant,
        weight="weight",
    )

    betweenness = nx.betweenness_centrality(
        giant,
        weight=None,
        normalized=True,
    )

    rows = []

    for node in giant.nodes:
        rows.append(
            {
                "StockCode": giant.nodes[node]["StockCode"],
                "Description": giant.nodes[node]["Description"],
                "Degree": degree[node],
                "WeightedDegree": weighted_degree[node],
                "PageRank": pagerank[node],
                "Betweenness": betweenness[node],
            }
        )

    centrality = (
        pd.DataFrame(rows)
        .sort_values(
            "PageRank",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary = {
        "Nodes": graph.number_of_nodes(),
        "Edges": graph.number_of_edges(),
        "Components": len(components),
        "LargestComponent": giant.number_of_nodes(),
        "LargestComponentEdges": giant.number_of_edges(),
        "Density": nx.density(giant),
        "AverageClustering": nx.average_clustering(
            giant,
            weight=None,
        ),
    }

    return centrality, summary



def community_spectral_analysis(
    graph: nx.Graph,
) -> tuple[pd.DataFrame, dict]:
    """Analyze weighted communities and graph spectrum."""

    components = sorted(
        nx.connected_components(graph),
        key=len,
        reverse=True,
    )

    giant = graph.subgraph(
        components[0]
    ).copy()

    communities = (
        nx.community.louvain_communities(
            giant,
            weight="weight",
            seed=42,
            resolution=1.0,
        )
    )

    communities = sorted(
        communities,
        key=len,
        reverse=True,
    )

    modularity = nx.community.modularity(
        giant,
        communities,
        weight="weight",
    )

    pagerank = nx.pagerank(
        giant,
        weight="weight",
    )

    rows = []

    for community_id, nodes in enumerate(
        communities,
        start=1,
    ):
        subgraph = giant.subgraph(nodes)

        top_nodes = sorted(
            nodes,
            key=lambda node: pagerank[node],
            reverse=True,
        )[:5]

        top_products = " | ".join(
            (
                f"{giant.nodes[node]['StockCode']}: "
                f"{giant.nodes[node]['Description']}"
            )
            for node in top_nodes
        )

        internal_weight = sum(
            data["weight"]
            for _, _, data
            in subgraph.edges(data=True)
        )

        rows.append(
            {
                "Community": community_id,
                "Nodes": subgraph.number_of_nodes(),
                "InternalEdges": subgraph.number_of_edges(),
                "InternalWeight": internal_weight,
                "TopProducts": top_products,
            }
        )

    community_table = pd.DataFrame(rows)

    laplacian = (
        nx.normalized_laplacian_matrix(
            giant,
            weight="weight",
        )
        .astype(float)
    )

    eigenvalues = eigsh(
        laplacian,
        k=6,
        which="SM",
        return_eigenvectors=False,
        tol=1e-9,
    )

    eigenvalues = np.sort(
        np.real(eigenvalues)
    )

    eigenvalues[
        np.abs(eigenvalues) < 1e-10
    ] = 0.0

    spectral = {
        "CommunityCount": len(communities),
        "Modularity": modularity,
        "Eigenvalues": eigenvalues,
        "Lambda2": float(eigenvalues[1]),
        "Communities": communities,
    }

    return community_table, spectral




def save_product_graph_outputs(
    graph: nx.Graph,
    centrality: pd.DataFrame,
    community_table: pd.DataFrame,
    spectral: dict,
    top_pairs: pd.DataFrame,
    threshold: int = 100,
) -> tuple[Path, Path]:
    """Save the community visualization and Spanish report."""

    figures_dir = PROJECT_ROOT / "figures"
    reports_dir = PROJECT_ROOT / "reports"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    components = sorted(
        nx.connected_components(graph),
        key=len,
        reverse=True,
    )

    giant = graph.subgraph(
        components[0]
    ).copy()

    communities = spectral[
        "Communities"
    ]

    membership = {}

    for community_id, nodes in enumerate(
        communities,
        start=1,
    ):
        for node in nodes:
            membership[node] = community_id

    community_graph = nx.Graph()

    for community_id, nodes in enumerate(
        communities,
        start=1,
    ):
        community_graph.add_node(
            community_id,
            size=len(nodes),
        )

    for node_a, node_b, data in giant.edges(
        data=True
    ):
        community_a = membership[node_a]
        community_b = membership[node_b]

        if community_a == community_b:
            continue

        weight = float(
            data.get(
                "weight",
                1.0,
            )
        )

        if community_graph.has_edge(
            community_a,
            community_b,
        ):
            community_graph[
                community_a
            ][
                community_b
            ]["weight"] += weight

        else:
            community_graph.add_edge(
                community_a,
                community_b,
                weight=weight,
            )

    position = nx.spring_layout(
        community_graph,
        weight="weight",
        seed=42,
    )

    node_sizes = [
        300
        + 30
        * community_graph.nodes[node][
            "size"
        ]
        for node in community_graph.nodes
    ]

    edge_weights = [
        data["weight"]
        for _, _, data
        in community_graph.edges(
            data=True
        )
    ]

    max_edge_weight = max(
        edge_weights,
        default=1.0,
    )

    edge_widths = [
        0.5
        + 4.0
        * weight
        / max_edge_weight
        for weight in edge_weights
    ]

    labels = {
        node: (
            f"C{node}\n"
            f"n={community_graph.nodes[node]['size']}"
        )
        for node in community_graph.nodes
    }

    plt.figure(
        figsize=(12, 9)
    )

    nx.draw_networkx_nodes(
        community_graph,
        position,
        node_size=node_sizes,
        node_color=list(
            community_graph.nodes
        ),
        cmap=plt.cm.tab20,
        alpha=0.85,
    )

    nx.draw_networkx_edges(
        community_graph,
        position,
        width=edge_widths,
        alpha=0.35,
    )

    nx.draw_networkx_labels(
        community_graph,
        position,
        labels=labels,
        font_size=9,
    )

    plt.title(
        "Red agregada de comunidades "
        "de co-compra (Louvain)"
    )

    plt.axis("off")
    plt.tight_layout()

    figure_path = (
        figures_dir
        / "05_product_communities.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    eigenvalues = spectral[
        "Eigenvalues"
    ]

    report_lines = [
        "ANÁLISIS AVANZADO DE LA RED DE CO-COMPRA",
        "=" * 48,
        "",
        "1. Construcción de la red",
        "",
        (
            "Los nodos representan productos y una arista "
            "indica que dos productos aparecieron juntos "
            "en facturas de venta."
        ),
        (
            "El peso de una arista corresponde al número "
            "de facturas en las que el par fue observado."
        ),
        (
            f"Se conservan aristas con al menos "
            f"{threshold} co-compras."
        ),
        (
            "Los códigos de servicios o ajustes "
            "administrativos fueron excluidos únicamente "
            "para este análisis de red."
        ),
        (
            "Los códigos de producto fueron normalizados "
            "a mayúsculas para evitar duplicados debidos "
            "a diferencias de capitalización."
        ),
        "",
        "2. Estructura global",
        "",
        f"Nodos: {graph.number_of_nodes():,}",
        f"Aristas: {graph.number_of_edges():,}",
        f"Componentes: {len(components):,}",
        (
            f"Nodos en la componente gigante: "
            f"{giant.number_of_nodes():,}"
        ),
        (
            f"Aristas en la componente gigante: "
            f"{giant.number_of_edges():,}"
        ),
        (
            f"Densidad de la componente gigante: "
            f"{nx.density(giant):.6f}"
        ),
        (
            f"Clustering medio: "
            f"{nx.average_clustering(giant):.6f}"
        ),
        "",
        "3. Productos centrales",
        "",
        centrality.head(15).to_string(
            index=False,
            formatters={
                "PageRank": lambda x: f"{x:.6f}",
                "Betweenness": lambda x: f"{x:.6f}",
            },
        ),
        "",
        "4. Pares de productos con mayor co-compra",
        "",
        top_pairs.to_string(
            index=False
        ),
        "",
        "5. Comunidades de productos",
        "",
        (
            f"Número de comunidades Louvain: "
            f"{spectral['CommunityCount']}"
        ),
        (
            f"Modularidad ponderada: "
            f"{spectral['Modularity']:.6f}"
        ),
        "",
        community_table.to_string(
            index=False
        ),
        "",
        "6. Análisis espectral",
        "",
    ]

    for index, eigenvalue in enumerate(
        eigenvalues,
        start=1,
    ):
        report_lines.append(
            f"lambda_{index}: "
            f"{eigenvalue:.8f}"
        )

    report_lines.extend(
        [
            "",
            (
                "Segundo autovalor del Laplaciano "
                "normalizado: "
                f"{spectral['Lambda2']:.8f}"
            ),
            "",
            "7. Interpretación",
            "",
            (
                "La red presenta alta cohesión local "
                "y una componente gigante dominante, "
                "pero también una organización interna "
                "en grupos de productos."
            ),
            (
                "Las comunidades detectadas muestran "
                "coherencia comercial, agrupando "
                "familias como bolsas, artículos "
                "navideños, vajilla y otros productos "
                "relacionados."
            ),
            (
                "La modularidad positiva indica una "
                "estructura comunitaria moderada; no "
                "implica una separación absoluta entre "
                "los grupos."
            ),
            (
                "El segundo autovalor positivo del "
                "Laplaciano normalizado confirma que la "
                "componente analizada es conexa. Su "
                "magnitud relativamente pequeña es "
                "compatible con la presencia de "
                "subestructuras y cuellos de botella "
                "dentro de la red."
            ),
            (
                "En conjunto, el análisis muestra que "
                "las compras no forman una colección "
                "de productos independientes, sino una "
                "red con asociaciones y estructura "
                "comunitaria observables."
            ),
        ]
    )

    report_path = (
        reports_dir
        / "product_graph_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    return figure_path, report_path


def main() -> None:
    df = load_sale_baskets()

    incidence, invoices, products = (
        build_incidence_matrix(df)
    )

    cooccurrence = build_cooccurrence(
        incidence
    )

    validate_matrices(
        incidence,
        cooccurrence,
        invoices,
        products,
    )

    diagnostics = threshold_diagnostics(
        cooccurrence,
        products,
    )

    top_pairs = strongest_pairs(
        cooccurrence,
        products,
        df,
    )

    graph = build_product_graph(
        cooccurrence,
        products,
        df,
        threshold=100,
    )

    centrality, summary = graph_summary(
        graph
    )

    community_table, spectral = (
        community_spectral_analysis(
            graph
        )
    )

    figure_path, report_path = (
        save_product_graph_outputs(
            graph,
            centrality,
            community_table,
            spectral,
            top_pairs,
            threshold=100,
        )
    )

    print("=== PRODUCT GRAPH DATA ===")
    print(
        f"Unique invoice-product pairs: "
        f"{len(df):,}"
    )
    print(
        f"Sale invoices: "
        f"{len(invoices):,}"
    )
    print(
        f"Products: "
        f"{len(products):,}"
    )
    print(
        f"Incidence matrix shape: "
        f"{incidence.shape}"
    )
    print(
        f"Non-zero co-occurrences: "
        f"{cooccurrence.nnz:,}"
    )

    print()
    print("Graph matrix validation: PASSED")

    print()
    print("=== THRESHOLD DIAGNOSTICS ===")
    print(
        diagnostics.to_string(
            index=False
        )
    )

    print()
    print("=== STRONGEST PRODUCT PAIRS ===")
    print(
        top_pairs.to_string(
            index=False
        )
    )

    print()
    print("=== GRAPH SUMMARY (THRESHOLD = 100) ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value:,}")

    print()
    print("=== TOP PRODUCTS BY PAGERANK ===")
    print(
        centrality.head(15).to_string(
            index=False,
            formatters={
                "PageRank": lambda x: f"{x:.6f}",
                "Betweenness": lambda x: f"{x:.6f}",
            },
        )
    )

    print()
    print("=== COMMUNITY ANALYSIS ===")
    print(
        f"Louvain communities: "
        f"{spectral['CommunityCount']}"
    )
    print(
        f"Weighted modularity: "
        f"{spectral['Modularity']:.6f}"
    )
    print()
    print(
        community_table.to_string(
            index=False
        )
    )

    print()
    print(
        "=== SPECTRAL ANALYSIS "
        "(GIANT COMPONENT) ==="
    )

    for index, value in enumerate(
        spectral["Eigenvalues"],
        start=1,
    ):
        print(
            f"lambda_{index}: "
            f"{value:.8f}"
        )

    print(
        f"Second normalized Laplacian "
        f"eigenvalue: "
        f"{spectral['Lambda2']:.8f}"
    )


    print()
    print("=== OUTPUT FILES ===")
    print(f"Figure: {figure_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()




