from random import shuffle


def rotate(l, n=1):
    """Returns a list rotated forward by n steps."""
    return l[n:] + l[:n]


def make_slides(cats=["injection", "negative", "neutral", "positive"],
                trial_length=12,
                imgformat="bmp"):
    """Returns a list of lists of filenames, one per slide.

    Filenames are given in the format {category}-{number}.{imgformat}.
    Categories are counterbalanced to appear equally in each area of
    the slide, and each image appears exactly once per trial.
    """

    # templates is a list of lists of categories, showing how the image types
    # will be arranged on each slide
    templates = []
    # for twelve trials, we'll generate three groups of four slides. If the
    # trial length isn't divisible by the number of categories, we'll get an
    # error. Could fix by rounding up instead of down, but for now an error is
    # probably better.
    for m in range(int(trial_length/len(cats))):
        # get four counterbalanced slides by adding all rotations of the
        # current (random) order of categories
        shuffle(cats)
        templates.extend([rotate(cats, n) for n in range(len(cats))])

    # indices is a dictionary of lists, containing the order of images for each
    # category
    indices = {}
    for cat in cats:
        indices[cat] = list(range(1, 13))
        shuffle(indices[cat])

    # slides is a list of lists of filenames as strings, in the order in which
    # they'll appear on the sketchpad.
    slides = []
    # Each image is determined by getting the next index value for the
    # appropriate template.
    for s in range(trial_length):
        # a list of filenames for the current slide
        filenames = []
        for i in templates[s]:
            filenames.append("{0}-{1:=02d}.bmp".format(i, indices[i][s]))
        slides.append(filenames)

    return slides

if __name__ == "__main__":
    print(make_slides())
