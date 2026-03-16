from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Replace with your actual HF username/model-name
model_path = "your-hf-username/my-t5-nl-to-sql"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def get_sql(query):
    input_text = "translate English to SQL: %s </s>" % query
    features = tokenizer([input_text], return_tensors='pt')
    output = model.generate(
        input_ids=features['input_ids'],
        attention_mask=features['attention_mask']
    )
    return tokenizer.decode(output[0])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        generated_sql_query = get_sql(question)
        cleaned_sentence = (generated_sql_query
            .replace("<pad>", "")
            .replace("</s>", " ;")
            .replace("<unk>", "=")
            .replace("table ", "Your_table_Name ")
            .replace("not ", "!  "))
        return render_template('index.html', processed_text=cleaned_sentence, question=question)
    return render_template('index.html', processed_text='', question='')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)