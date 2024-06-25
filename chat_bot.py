import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import sentencepiece  # Ensure sentencepiece is imported

# Load the T5 model and tokenizer
model_name = "t5-small"  # You can use a larger model like "t5-base" or "t5-large" if needed
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to generate an answer using T5
def generate_answer(context, question):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Streamlit app
st.title("QA System with T5")

st.write("Upload a text file containing unstructured data, and then ask questions based on that data.")

# File uploader
uploaded_file = st.file_uploader("Choose a text file", type="txt")
if uploaded_file is not None:
    # Read the file
    text = uploaded_file.read().decode("utf-8")
    st.write("File content loaded successfully!")

    # User input for questions
    question = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                answer = generate_answer(text, question)
                st.write("Answer:", answer)
        else:
            st.write("Please enter a question.")