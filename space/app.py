"""Demo: a retrieval assistant on Tiny Aya that withholds answers it is likely to get wrong."""
import html
import os
from pathlib import Path

import gradio as gr
from pipeline import Pipeline

if os.environ.get("SPACES_ZERO_GPU"):
    import spaces
    gpu, device = spaces.GPU, "cuda"
else:
    gpu, device = (lambda function: function), None
assets = os.environ.get("AYA_ASSETS_DIR", Path(__file__).resolve().parent)
pipeline = Pipeline(folder=assets, device=device)

LANGUAGES = {"English": "en", "తెలుగు (Telugu)": "te", "বাংলা (Bengali)": "bn", "Kiswahili (Swahili)": "sw"}
RULES = {"TyDiQA target: 10% wrong out of all questions": "joint",
         "TyDiQA target: 20% wrong among delivered answers": "selective"}
EXAMPLES = [
    ["What causes malaria?", "English"],
    ["Who won the 1998 football world cup?", "English"],
    ["క్షయ వ్యాధికి కారణం ఏమిటి?", "తెలుగు (Telugu)"],
    ["Malaria husababishwa na nini?", "Kiswahili (Swahili)"],
]
ABOUT = """# Knows When It Doesn't Know
A small multilingual model answers from public-health passages. Before you see the answer, a linear probe on its
layer-24 activation, retrieval similarity and log-probability produce a risk score. One threshold per rule
applies across all four languages. **Research demo, not medical advice.** The targets refer to TyDiQA.
The original thresholds missed both targets in Telugu, Bengali and Swahili on the health benchmark."""


def card(result):
    passage = result["passage"]
    risk, limit = result["risk"], result["threshold"]
    answer = html.escape(result["answer"] or "(no answer produced)")
    verdict = "Answer delivered" if result["delivered"] else "Abstained: not confident enough to answer"
    colour = "#1a7f4b" if result["delivered"] else "#b3541e"
    main = f"<p style='font-size:1.3em'>{answer}</p>" if result["delivered"] else ""
    retry_note = "<p>Checked a second passage.</p>" if result.get("retrieval_attempts") == 2 else ""
    return f"""<div style='border-left:6px solid {colour};padding:8px 16px'>
<h3 style='color:{colour};margin:0 0 8px'>{verdict}</h3>{main}
<p>Risk score {risk:.2f}, threshold {limit:.2f}. Passage match {result['similarity']:.2f}.</p>{retry_note}</div>
<h4>Passage used: {html.escape(passage['topic'])}</h4><p>{html.escape(passage['text'])}</p>
<p><a href='{html.escape(passage.get('url') or '')}' target='_blank'>Wikipedia source</a> (CC BY-SA 4.0)</p>"""


@gpu
def ask(question, language, rule, retry):
    if not question.strip():
        return "<p>Type a question first.</p>"
    return card(pipeline.respond(question.strip(), LANGUAGES[language], rule=RULES[rule], retry=retry))


with gr.Blocks(title="Knows When It Doesn't Know") as demo:
    gr.Markdown(ABOUT)
    language = gr.Dropdown(list(LANGUAGES), value="English", label="Language")
    question = gr.Textbox(label="Question", lines=2)
    rule = gr.Radio(list(RULES), value=list(RULES)[0], label="Abstention rule")
    retry = gr.Checkbox(label="Try a second passage after abstaining (experimental)", value=False)
    button = gr.Button("Ask", variant="primary")
    output = gr.HTML()
    button.click(ask, [question, language, rule, retry], output)
    question.submit(ask, [question, language, rule, retry], output)
    gr.Examples(EXAMPLES, [question, language])

if __name__ == "__main__":
    demo.launch()
