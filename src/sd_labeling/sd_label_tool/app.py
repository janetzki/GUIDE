import random
from flask import Flask, jsonify, render_template, request
from src.dictionary_creator.tfidf_dictionary_creator import TfidfDictionaryCreator
from src.semantic_domain_identifier import SemanticDomainIdentifier

app = Flask(__name__)

# load all verses from the bid-eng-DBY bible
dc = TfidfDictionaryCreator(['bid-eng-web', 'bid-deu'], score_threshold=0.2, state_files_path='../../../data/0_state')
sdi = SemanticDomainIdentifier(dc)
verses_eng = sdi.dc.wtxts_by_verse_by_bid['bid-eng-web']
verses_deu = sdi.dc.wtxts_by_verse_by_bid['bid-deu']


def generate_sentence():
    idx = random.randrange(0, len(verses_eng))
    return idx, verses_eng[idx], verses_deu[idx]


def generate_checkboxes():
    idx, words_eng, words_deu = generate_sentence()
    sentence = f'{idx}<br>' + ' '.join(words_eng) + '<br>'
    sentence += ' '.join([wtxt.split('_')[0] for wtxt in words_eng]) + '<br>' + ' '.join(words_deu)

    sdi.qid_by_wtxt = sdi.gt_qid_by_wtxt
    identified_qids = sdi.identify_semantic_domains([words_eng])
    identified_qids = sorted(identified_qids, key=lambda t: (t[0], t[2]))

    identified_qids = [(idx, f'"{wtxt.split("_")[0].upper()}"', qid, sd_name, question, words)
                       for (idx, wtxt, qid, sd_name, question, words) in identified_qids]
    checkboxes = [{'checked': '',
                   'word': f'{question.replace("#", wtxt)}'}
                  for (idx, wtxt, qid, sd_name, question, words) in identified_qids]

    # for each start_token_idx that occurs only once, mark it as checked (because it is the only option)
    for outer_idx, (idx, wtxt, qid, sd_name, question, words) in enumerate(identified_qids):
        if len([t for t in identified_qids if t[0] == idx]) == 1:
            checkboxes[outer_idx]['checked'] = 'checked'

    return sentence, checkboxes


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        selected_words = request.form.getlist('word')
        sentence = request.form['sentence']
        print(sentence, selected_words)
    sentence, checkboxes = generate_checkboxes()
    return render_template('index.html', sentence=sentence, checkboxes=checkboxes)


@app.route('/generate_checkboxes')
def fetch_checkboxes():
    sentence, checkboxes = generate_checkboxes()
    return jsonify({'sentence': sentence, 'checkboxes': checkboxes})


if __name__ == '__main__':
    app.run(debug=True)
