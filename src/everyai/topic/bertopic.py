import logging
from pathlib import Path

from bertopic import BERTopic
from plotly.graph_objs._figure import Figure


def create_topic(
    docs: list[str],
    output_folder: str | Path,
    embedding_model=None,
    topic_config: dict = None,
) -> BERTopic:
    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=5)
    docs = [doc for doc in docs if isinstance(doc, str)]
    topic_model.fit_transform(docs)
    # Save the results in the new directory
    logging.info(f"Saving results in {output_folder}")
    if isinstance(output_folder, str):
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    topic_model.get_topic_freq().to_csv(output_folder / "topic_freq.csv")
    bar_fig: Figure = topic_model.visualize_barchart(
        **topic_config["bar_config"]
    )
    scatter_fig = topic_model.visualize_topics(
        **topic_config["scatter_config"]
    )
    heatmap_fig = topic_model.visualize_heatmap(
        **topic_config["heatmap_config"]
    )
    hierarchy_fig = topic_model.visualize_hierarchy(
        **topic_config["hierarchy_config"]
    )
    termrank_fig = topic_model.visualize_term_rank(
        **topic_config["termrank_config"]
    )
    fig_dict = {
        bar_fig: "bar",
        scatter_fig: "scatter",
        heatmap_fig: "heatmap",
        hierarchy_fig: "hierarchy",
        termrank_fig: "termrank",
    }
    for fig in fig_dict.items():
        if isinstance(fig, Figure):
            fig.update_layout(**topic_config["layout_config"])
            match topic_config["suffix"]:
                case "html":
                    fig.write_html(
                        output_folder
                        / f"{fig_dict[fig]}.{topic_config['suffix']}"
                    )
                case ["png", "jpeg", "jpg", "webp"]:
                    fig.write_image(
                        output_folder
                        / f"{fig_dict[fig]}.{topic_config['suffix']}"
                    )
        else:
            logging.error(f"Error in saving {fig[0]} figure")
    return topic_model
