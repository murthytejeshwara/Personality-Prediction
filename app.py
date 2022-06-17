import streamlit as st
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import numpy as np



class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word)>2]


vectorizer  = pkl.load(open('tfidf.pkl', 'rb'))
model = pkl.load(open('model_linear_svc.pickle', 'rb'))


st.title("Predict your Personality")
st.write("The Myers Briggs Type Indicator (or MBTI for short) is a \
        personality type system that divides everyone into 16 distinct \
        personality types across 4 axis:  Introversion(I) – Extroversion (E) Intuition (N) – Sensing (S) Thinking (T) – Feeling (F)\
        Judging (J) – Perceiving (P).")
st.write("So for example, someone who prefers introversion, intuition, \
        thinking and perceiving would be labelled an INTP in the MBTI system, and there are lots of personality \
        based components that would model or describe this person’s preferences or behaviour based on the label. \
        It is one of, if not the, the most popular personality test in the world. It is used in businesses, online, \
        for fun, for research and lots more. A simple google search reveals all of the different ways the test has been\
         used over time. It’s safe to say that this test is still very relevant in the world in terms of its use.")

st.subheader("Please Answer Few Questions Below: ")
questions=['What do you consider your greatest achievement in life?','What word would people close to you use to describe you?','What is a question people ask you often?','How do you deal with failure?','What are you most proud of?','What do you do with your free time when you have no plans?','What is your favorite place you\'ve ever visited?']

for i in range(len(questions)):
    st.write("{}. {}".format(i+1,questions[i]))

answers = st.text_area("Answer each Question in seperate paragraphs.")





if st.button("Submit Answers"):
    if len(answers)>10:
        with st.spinner("Please Wait Predicting your Personality"):
            X=vectorizer.transform([" ".join(answers)]).toarray()
            decoder = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP','INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP']
            personality = decoder[model.predict(X)[0]]


        st.success("Your Personality is "+str(personality))
        st.write("Introversion(I) – Extroversion (E) Intuition (N) – Sensing (S) Thinking (T) – Feeling (F) Judging (J) – Perceiving (P).")
    else:
        st.error("Please Enter Answers")
