"""Demo: a retrieval assistant on Tiny Aya that withholds answers it is likely to get wrong."""
import html
import os

import gradio as gr
from pipeline import Pipeline

if os.environ.get("SPACES_ZERO_GPU"):
    import spaces
    gpu, device = spaces.GPU, "cuda"
else:
    gpu, device = (lambda function: function), None
pipeline = Pipeline(device=device)

LANGUAGES = {"English": "en", "తెలుగు (Telugu)": "te", "বাংলা (Bengali)": "bn", "Kiswahili (Swahili)": "sw"}
RULES = {"Wrong and delivered at most 10% (answers more)": "joint", "Wrong among answered at most 20% (stricter)": "selective"}
EXAMPLES = [
    ["What causes malaria?", "English"],
    ["Who won the 1998 football world cup?", "English"],
    ["క్షయ వ్యాధికి కారణం ఏమిటి?", "తెలుగు (Telugu)"],
    ["Malaria husababishwa na nini?", "Kiswahili (Swahili)"],
]
ABOUT = """# Knows When It Doesn't Know
A small multilingual model answers from public-health passages. Before you see the answer, a linear probe on its
layer-24 activation, combined with retrieval similarity and log-probability, estimates the chance it is wrong.
One threshold, set on held-out data, decides for all four languages. **Research demo, not medical advice.**
The guard was calibrated on TyDiQA; on health questions the wrong-and-delivered rate rose to 6.6-12.4%."""


def card(result):
    passage = result["passage"]
    risk, limit = result["risk"], result["threshold"]
    answer = html.escape(result["answer"] or "(no answer produced)")
    verdict = "Answer delivered" if result["delivered"] else "Abstained: not confident enough to answer"
    colour = "#1a7f4b" if result["delivered"] else "#b3541e"
    main = f"<p style='font-size:1.3em'>{answer}</p>" if result["delivered"] else (
        f"<details><summary>Show the withheld answer</summary><p>{answer}</p></details>"
    )
    return f"""<div style='border-left:6px solid {colour};padding:8px 16px'>
<h3 style='color:{colour};margin:0 0 8px'>{verdict}</h3>{main}
<p>Risk {risk:.2f}, limit {limit:.2f}. Passage match {result['similarity']:.2f}.</p></div>
<h4>Passage used: {html.escape(passage['topic'])}</h4><p>{html.escape(passage['text'])}</p>
<p><a href='{html.escape(passage.get('url') or '')}' target='_blank'>Wikipedia source</a> (CC BY-SA 4.0)</p>"""


@gpu
def ask(question, language, rule):
    if not question.strip():
        return "<p>Type a question first.</p>"
    return card(pipeline.respond(question.strip(), LANGUAGES[language], rule=RULES[rule]))


with gr.Blocks(title="Knows When It Doesn't Know") as demo:
    gr.Markdown(ABOUT)
    language = gr.Dropdown(list(LANGUAGES), value="English", label="Language")
    question = gr.Textbox(label="Question", lines=2)
    rule = gr.Radio(list(RULES), value=list(RULES)[0], label="Risk target")
    button = gr.Button("Ask", variant="primary")
    output = gr.HTML()
    button.click(ask, [question, language, rule], output)
    question.submit(ask, [question, language, rule], output)
    gr.Examples(EXAMPLES, [question, language])

if __name__ == "__main__":
    demo.launch()
