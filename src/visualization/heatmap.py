import numpy as np
import matplotlib.pyplot as plt


def heatmap(outputs, image_file: str = None):
    """Generates and saves the visualization

    Args:
        outputs: The outputs of the LM-EMD model.
            outputs[0]: The document scores.
            outputs[1]: The cost matrix.
            outputs[2]: The transportation matrix.

        examples: The query and documents text tokenizer.
            example.q_input_ids: The tensor containing the query token IDs.
            example.d_input_ids: The tensor containing the document token IDs.

        image_file: The path where the image should be stored.
    """
    # get the output values
    scores = outputs[0].cpu()
    cm = outputs[1].cpu()
    tm = outputs[2].cpu()

    # get the max sizes (batch, query, document)
    bsize, qsize, dsize = cm.shape

    # prepare the figure size based on the input

    # initialize the figure
    fig, big_axes = plt.subplots(nrows=1, ncols=1, figsize=(8.2, 4.8))
    if bsize == 1:
        big_axes = [big_axes]

    for idx, big_ax in enumerate(big_axes):
        # Turn off axis lines and ticks of the big subplot
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(
            labelcolor=(1.0, 1.0, 1.0, 0.0),
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        # removes the white frame
        big_ax._frameon = False

    # iterate through the examples
    for i in range(bsize):
        ax_distance = fig.add_subplot(bsize, 2, 2 * i + 1)
        ax_transport = fig.add_subplot(bsize, 2, 2 * i + 2)

        # the cosine distance matrix
        ax_distance.set_title("distance matrix", fontsize="large")
        cmim = ax_distance.imshow(cm[i].numpy(), cmap="PuBu", vmin=0)
        cbar = fig.colorbar(cmim, ax=ax_distance, shrink=0.9)
        cbar.ax.set_ylabel("cosine distance", rotation=-90, va="bottom")

        # the EMD transport matrix
        ax_transport.set_title("transportation matrix", fontsize="large")
        tmim = ax_transport.imshow(tm[i] / tm[i].max(), cmap="Greens", vmin=0, vmax=1)
        cbar = fig.colorbar(tmim, ax=ax_transport, shrink=0.9)
        cbar.ax.set_ylabel("mass transport (match)", rotation=-90, va="bottom")

        plots = [ax_distance, ax_transport]

        # assign the document score (lower scores -> greater rank)
        d_score = round(scores[i].item(), 3)
        plots[0].set_ylabel(
            f"Document score: {d_score}",
            rotation=-90,
            va="bottom",
            labelpad=30,
            fontsize=12,
        )

    # make the layout more tight
    plt.tight_layout()

    # save the plot in a file
    if image_file:
        plt.savefig(image_file, dpi=300, transparent=True, bbox_inches="tight")
