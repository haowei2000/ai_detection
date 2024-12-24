import logging
from pathlib import Path
from typing import Union

from bertopic import BERTopic
from plotly.graph_objs._figure import Figure


def create_topic(
    docs: list[str],
    output_folder: Union[str, Path],
    embedding_model=None,
    topic_config: dict = {},
) -> BERTopic:
    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=5)

    topic_model.fit_transform(docs)
    # Save the results in the new directory
    logging.info(f"Saving results in {output_folder}")
    if isinstance(output_folder, str):
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    topic_model.get_topic_freq().to_csv(output_folder / "topic_freq.csv")
    if topic_config is None:
        topic_config = {}
    bar_fig: Figure = topic_model.visualize_barchart(
        **topic_config.get("bar_config", {})
    )
    scatter_fig = topic_model.visualize_topics(**topic_config.get("scatter_config", {}))
    heatmap_fig = topic_model.visualize_heatmap(
        **topic_config.get("heatmap_config", {})
    )
    hierarchy_fig = topic_model.visualize_hierarchy(
        **topic_config.get("hierarchy_config", {})
    )
    termrank_fig = topic_model.visualize_term_rank(
        **topic_config.get("termrank_config", {})
    )
    fig_dict = {
        "bar": bar_fig,
        "scatter": scatter_fig,
        "heatmap": heatmap_fig,
        "hierarchy": hierarchy_fig,
        "termrank": termrank_fig,
    }
    for fig_name, fig in fig_dict.items():
        if isinstance(fig, Figure):
            match topic_config["suffix"]:
                case "html":
                    fig.write_html(
                        output_folder / f"{fig_name}.{topic_config['suffix']}"
                    )
                case ["png", "jpeg", "jpg", "webp"]:
                    fig.write_image(
                        output_folder / f"{fig_name}.{topic_config['suffix']}"
                    )
                case _:
                    logging.error(f"Unsupported file format: {topic_config['suffix']}")
        else:
            logging.error(f"Error in saving {fig_name} figure")
    return topic_model
