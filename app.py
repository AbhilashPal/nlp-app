import streamlit as st 

#nlp pkgs
import spacy 
from textblob import TextBlob 
from gensim.summarization import summarize

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

# Function to Analyse Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = spacy.load("en_core_web_sm")
	docx = nlp(my_text)
	tokens = [token.text for token in docx]
	allData = [('"Tokens" :{},\n"Lemma":{}'.format(token.text,token.lemma_)) for token in docx]
	return allData 

# Function For Extracting Entities
@st.cache
def entity_analyzer(my_text):
	nlp = spacy.load("en_core_web_sm")
	docx = nlp(my_text)
	tokens = [token.text for token in docx]
	ents = [(entity.text,entity.label_) for entity in docx.ents]
	allData = [('"Tokens" :{},\n"Entities":{}'.format(tokens,ents))]	
	return allData
#pkgs

def main():
	"""NLP APP WITH STREAMLIT"""
	st.title("NLPify with Streamlit")
	st.subheader("Natural Language Processing on the Go")

	#tokenization
	if st.checkbox("Show Tokens and Lemma"):
		st.subheader("Tokenize Your Text")
		message = st.text_area("Enter your text","Type Here")
		if st.button("Analyze"):
			nlp_result = text_analyzer(message)
			st.json(nlp_result)


	#NER
	if st.checkbox("Show Named Entities"):
		st.subheader("Extract Entities from your text")
		message = st.text_area("Enter your text","Type Here")
		if st.button("Extract"):
			nlp_result = entity_analyzer(message)
			st.json(nlp_result)


	# Sentiment Analysis
	if st.checkbox("Show Sentiment Analysis"):
		st.subheader("Analyse Your Text")

		message = st.text_area("Enter Text","Type Here ..")
		if st.button("Analyze"):
			blob = TextBlob(message)
			result_sentiment = blob.sentiment
			st.success(result_sentiment)

	# Summarization
	if st.checkbox("Show Text Summarization"):
		st.subheader("Summarize Your Text")

		message = st.text_area("Enter Text","Type Here ..")
		summary_options = st.selectbox("Choose Summarizer",['sumy','gensim'])
		if st.button("Summarize"):
			if summary_options == 'sumy':
				st.text("Using Sumy Summarizer ..")
				summary_result = sumy_summarizer(message)
			elif summary_options == 'gensim':
				st.text("Using Gensim Summarizer ..")
				summary_result = summarize(message)
			else:
				st.warning("Using Default Summarizer")
				st.text("Using Gensim Summarizer ..")
				summary_result = summarize(message)

		
			st.success(summary_result)



	st.sidebar.subheader("About App")
	st.sidebar.text("NLPiffy App with Streamlit")
	st.sidebar.info("Cudos to the Streamlit Team")
	

	st.sidebar.subheader("By")
	st.sidebar.text("Jesse E.Agbe(JCharis)")
	st.sidebar.text("Jesus saves@JCharisTech")




if __name__ == '__main__':
	main()

