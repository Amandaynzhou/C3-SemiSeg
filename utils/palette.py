import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
def get_voc_palette(num_classes):
    n = num_classes
    palette = [0]*(n*3)
    for j in range(0,n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while (lab > 0):
                    palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return palette

ADE20K_palette = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,
                    3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,
                    5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,
                    255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,
                    6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,
                    92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,
                    10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,
                    0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,
                    163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224
                    ,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,
                    200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,
                    163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,
                    255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,
                    255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,
                    255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,
                    122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,
                    255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,
                    255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,
                    0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,
                    0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,
                    20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,
                    255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,
                    255,214,0,25,194,194,102,255,0,92,0,255]

CityScpates_palette = [128,64,128,244,35,232,70,70,70,102,102,156,190,153,153,153,153,153,
                        250,170,30,220,220,0,107,142,35,152,251,152,70,130,180,220,20,60,255,0,0,0,0,142,
                        0,0,70,0,60,100,0,80,100,0,0,230,119,11,32,128,192,0,0,64,128,128,64,128,0,192,
                        128,128,192,128,64,64,0,192,64,0,64,192,0,192,192,0,64,64,128,192,64,128,64,192,
                        128,192,192,128,0,0,64,128,0,64,0,128,64,128,128,64,0,0,192,128,0,192,0,128,192,
                        128,128,192,64,0,64,192,0,64,64,128,64,192,128,64,64,0,192,192,0,192,64,128,192,
                        192,128,192,0,64,64,128,64,64,0,192,64,128,192,64,0,64,192,128,64,192,0,192,192,
                        128,192,192,64,64,64,192,64,64,64,192,64,192,192,64,64,64,192,192,64,192,64,192,
                        192,192,192,192,32,0,0,160,0,0,32,128,0,160,128,0,32,0,128,160,0,128,32,128,128,
                        160,128,128,96,0,0,224,0,0,96,128,0,224,128,0,96,0,128,224,0,128,96,128,128,224,
                        128,128,32,64,0,160,64,0,32,192,0,160,192,0,32,64,128,160,64,128,32,192,128,160,
                        192,128,96,64,0,224,64,0,96,192,0,224,192,0,96,64,128,224,64,128,96,192,128,224,
                        192,128,32,0,64,160,0,64,32,128,64,160,128,64,32,0,192,160,0,192,32,128,192,160,
                        128,192,96,0,64,224,0,64,96,128,64,224,128,64,96,0,192,224,0,192,96,128,192,224,
                        128,192,32,64,64,160,64,64,32,192,64,160,192,64,32,64,192,160,64,192,32,192,192,
                        160,192,192,96,64,64,224,64,64,96,192,64,224,192,64,96,64,192,224,64,192,96,192,
                        192,224,192,192,0,32,0,128,32,0,0,160,0,128,160,0,0,32,128,128,32,128,0,160,128,
                        128,160,128,64,32,0,192,32,0,64,160,0,192,160,0,64,32,128,192,32,128,64,160,128,
                        192,160,128,0,96,0,128,96,0,0,224,0,128,224,0,0,96,128,128,96,128,0,224,128,128,
                        224,128,64,96,0,192,96,0,64,224,0,192,224,0,64,96,128,192,96,128,64,224,128,192,
                        224,128,0,32,64,128,32,64,0,160,64,128,160,64,0,32,192,128,32,192,0,160,192,128,
                        160,192,64,32,64,192,32,64,64,160,64,192,160,64,64,32,192,192,32,192,64,160,192,
                        192,160,192,0,96,64,128,96,64,0,224,64,128,224,64,0,96,192,128,96,192,0,224,192,
                        128,224,192,64,96,64,192,96,64,64,224,64,192,224,64,64,96,192,192,96,192,64,224,
                        192,192,224,192,32,32,0,160,32,0,32,160,0,160,160,0,32,32,128,160,32,128,32,160,
                        128,160,160,128,96,32,0,224,32,0,96,160,0,224,160,0,96,32,128,224,32,128,96,160,
                        128,224,160,128,32,96,0,160,96,0,32,224,0,160,224,0,32,96,128,160,96,128,32,224,
                        128,160,224,128,96,96,0,224,96,0,96,224,0,224,224,0,96,96,128,224,96,128,96,224,
                        128,224,224,128,32,32,64,160,32,64,32,160,64,160,160,64,32,32,192,160,32,192,32,
                        160,192,160,160,192,96,32,64,224,32,64,96,160,64,224,160,64,96,32,192,224,32,192,
                        96,160,192,224,160,192,32,96,64,160,96,64,32,224,64,160,224,64,32,96,192,160,96,
                        192,32,224,192,160,224,192,96,96,64,224,96,64,96,224,64,224,224,64,96,96,192,224,
                        96,192,96,224,192,0,0,0]


COCO_palette = [31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227,
    119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44,
    214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207,
    31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75,
    227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44,
    214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189,
    34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75,
    227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127,
    14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189,
    34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103,
    189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127,
    14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127
    , 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103,
    189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14,
    44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127,
    127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189,
    140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44,
    160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190,
    207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194,
    127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148,
    103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127,
    14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34,
    23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227,
    119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39,
    40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119,
    180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127,
    127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14]


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def save_confusion_matrix(array):
    fig, ax = plt.subplots()
    name =['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign',
           'vegetation','terrain','sky', 'person','rider','car','truck','bus','train',
           'motorcycle','bicycle']
    im, cbar = heatmap(array,name, name, ax=ax,
                       cmap="YlGn", cbarlabel="harvest [t/year]")
    texts = annotate_heatmap(im, valfmt="{x:.3f} t")

    fig.tight_layout()
    plt.savefig('cm.png', dpi=100)