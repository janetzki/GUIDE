vrefs = None


def load_vrefs():
    global vrefs
    if vrefs is not None:
        return vrefs
    with open('data/vref.txt', 'r') as f:
        vrefs = f.readlines()
    vrefs = [vref.strip() for vref in vrefs]
    return vrefs


def convert_verse_id_to_bible_reference(verse_id):
    # e.g., 0 -> Gen 1:1, 23213 -> Mat 1:1
    vrefs = load_vrefs()
    vref = vrefs[verse_id]
    vref = vref[0] + vref[1].lower() + vref[2].lower() + vref[3:]
    return vref


def convert_bible_reference_to_verse_id(book, chapter, verse):
    # e.g., GEN 1:1 -> 0, MAT 1:1 -> 23213
    vrefs = load_vrefs()
    vref = f'{book} {chapter}:{verse}'
    return vrefs.index(vref)
