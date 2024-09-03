# Imports and necessary installations
# !pip install langchain
# !pip install faiss-cpu
# !pip install openai

import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

# API key
os.environ["OPENAI_API_KEY"] = ""

# List of URLs
urls = [
    'https://www.openavenuesfoundation.org/',
    'https://www.openavenuesfoundation.org/universities',
    'https://www.openavenuesfoundation.org/companies',
    'https://www.openavenuesfoundation.org/global-talent-fellows',
    'https://www.openavenuesfoundation.org/donate',
    'https://www.openavenuesfoundation.org/annual-report',
    'https://www.openavenuesfoundation.org/about-us',
    'https://www.openavenuesfoundation.org/contact',
    'https://www.openavenuesfoundation.org/resources',
    'https://www.openavenuesfoundation.org/in-the-news',
    'https://www.openavenuesfoundation.org/global-talent',
    'https://www.openavenuesfoundation.org/micro-internships',
    'https://www.openavenuesfoundation.org/career-pathways-program',
    'https://www.openavenuesfoundation.org/nominate',
    'https://www.openavenuesfoundation.org/company-partners',
    'https://www.openavenuesfoundation.org/careers',
    'https://www.openavenuesfoundation.org/global-talent-overview',
    'https://www.openavenuesfoundation.org/justice',
    'https://www.openavenuesfoundation.org/leadership/angie-adkins',
    'https://www.openavenuesfoundation.org/leadership/brendan-lind',
    'https://www.openavenuesfoundation.org/leadership/caroline-hoogland',
    'https://www.openavenuesfoundation.org/leadership/danielle-goldman',
    'https://www.openavenuesfoundation.org/leadership/danila-blanco-travnicek',
    'https://www.openavenuesfoundation.org/leadership/elena-semeyko',
    'https://www.openavenuesfoundation.org/leadership/erin-carlini',
    'https://www.openavenuesfoundation.org/leadership/jeff-goldman',
    'https://www.openavenuesfoundation.org/leadership/miriam-cohen',
    'https://www.openavenuesfoundation.org/leadership/rua-hamid',
    'https://www.openavenuesfoundation.org/leadership/sol-bender',
    'https://www.openavenuesfoundation.org/leadership/tanashya-batra',
    'https://www.openavenuesfoundation.org/directors/amy-stephens',
    'https://www.openavenuesfoundation.org/directors/dr-marvin-loiseau',
    'https://www.openavenuesfoundation.org/directors/juan-carlos-lopez',
    'https://www.openavenuesfoundation.org/directors/leia-ruseva',
    'https://www.openavenuesfoundation.org/directors/nicholas-sykes',
    'https://www.openavenuesfoundation.org/global-talent-fellow/ada-del-cid',
    'https://www.openavenuesfoundation.org/global-talent-fellow/adi-soloducho',
    'https://www.openavenuesfoundation.org/global-talent-fellow/aditya-goel',
    'https://www.openavenuesfoundation.org/global-talent-fellow/ali-al-abdullatif',
    'https://www.openavenuesfoundation.org/global-talent-fellow/alice-ballard-rossiter',
    'https://www.openavenuesfoundation.org/global-talent-fellow/anant-jain',
    'https://www.openavenuesfoundation.org/global-talent-fellow/apurva-bafana',
    'https://www.openavenuesfoundation.org/global-talent-fellow/ashok-bagadiya',
    'https://www.openavenuesfoundation.org/global-talent-fellow/bhupendra-singh-thakur',
    'https://www.openavenuesfoundation.org/global-talent-fellow/boyun-wang',
    'https://www.openavenuesfoundation.org/global-talent-fellow/carlos-martinez',
    'https://www.openavenuesfoundation.org/global-talent-fellow/cem-sengel',
    'https://www.openavenuesfoundation.org/global-talent-fellow/dana-rakhimzhanova',
    'https://www.openavenuesfoundation.org/global-talent-fellow/ece-bicak',
    'https://www.openavenuesfoundation.org/global-talent-fellow/ekin-keserer',
    'https://www.openavenuesfoundation.org/global-talent-fellow/filip-kos',
    'https://www.openavenuesfoundation.org/global-talent-fellow/foram-joshi',
    'https://www.openavenuesfoundation.org/global-talent-fellow/fred-hamlin',
    'https://www.openavenuesfoundation.org/global-talent-fellow/gabrielle-coseteng',
    'https://www.openavenuesfoundation.org/global-talent-fellow/hanseul-nam',
    'https://www.openavenuesfoundation.org/global-talent-fellow/harini-sivagurunatha-krishnan',
    'https://www.openavenuesfoundation.org/global-talent-fellow/harry-clarke',
    'https://www.openavenuesfoundation.org/global-talent-fellow/himanshu-raghuvanshi',
    'https://www.openavenuesfoundation.org/global-talent-fellow/ignacio-garcia-leon',
    'https://www.openavenuesfoundation.org/global-talent-fellow/ignacio-ojanguren',
    'https://www.openavenuesfoundation.org/global-talent-fellow/jainik-majumdar',
    'https://www.openavenuesfoundation.org/global-talent-fellow/jasper-ng',
    'https://www.openavenuesfoundation.org/global-talent-fellow/jay-jha',
    'https://www.openavenuesfoundation.org/global-talent-fellow/juan-cristo',
    'https://www.openavenuesfoundation.org/global-talent-fellow/kai-biegun',
    'https://www.openavenuesfoundation.org/global-talent-fellow/karan-kwatra',
    'https://www.openavenuesfoundation.org/global-talent-fellow/kirill-noskov',
    'https://www.openavenuesfoundation.org/global-talent-fellow/krishang-nadgauda',
    'https://www.openavenuesfoundation.org/global-talent-fellow/kshitij-chopra',
    'https://www.openavenuesfoundation.org/global-talent-fellow/leon-staubach',
    'https://www.openavenuesfoundation.org/global-talent-fellow/luis-sarmiento',
    'https://www.openavenuesfoundation.org/global-talent-fellow/luisa-rios',
    'https://www.openavenuesfoundation.org/global-talent-fellow/maria-rocha',
    'https://www.openavenuesfoundation.org/global-talent-fellow/marie-goulard',
    'https://www.openavenuesfoundation.org/global-talent-fellow/marja-pimentel',
    'https://www.openavenuesfoundation.org/global-talent-fellow/myriam-belghiti',
    'https://www.openavenuesfoundation.org/global-talent-fellow/nayana-nagaraj',
    'https://www.openavenuesfoundation.org/global-talent-fellow/nika-diomidovskaia',
    'https://www.openavenuesfoundation.org/global-talent-fellow/noe-fontana',
    'https://www.openavenuesfoundation.org/global-talent-fellow/peace-kim',
    'https://www.openavenuesfoundation.org/global-talent-fellow/rishabh-gupta',
    'https://www.openavenuesfoundation.org/global-talent-fellow/sai-krishna-bashetty',
    'https://www.openavenuesfoundation.org/global-talent-fellow/sal-topal',
    'https://www.openavenuesfoundation.org/global-talent-fellow/sanjeev-vijayaraj',
    'https://www.openavenuesfoundation.org/global-talent-fellow/shritesh-bhattarai',
    'https://www.openavenuesfoundation.org/global-talent-fellow/srutartha-bose',
    'https://www.openavenuesfoundation.org/global-talent-fellow/tim-nguyen',
    'https://www.openavenuesfoundation.org/global-talent-fellow/toby-hong',
    'https://www.openavenuesfoundation.org/global-talent-fellow/vaclav-hasenohrl',
    'https://www.openavenuesfoundation.org/global-talent-fellow/vera-schroeder',
    'https://www.openavenuesfoundation.org/global-talent-fellow/vivek-krishnan',
    'https://www.openavenuesfoundation.org/global-talent-fellow/yanling-wang',
    'https://www.openavenuesfoundation.org/global-talent-fellow/yavuz-sahin',
    'https://www.openavenuesfoundation.org/global-talent-fellow/yuya-fujimoto',
    'https://www.openavenuesfoundation.org/alumni/alex-batchelor',
    'https://www.openavenuesfoundation.org/alumni/antoine-gargot',
    'https://www.openavenuesfoundation.org/alumni/carole-sioufi',
    'https://www.openavenuesfoundation.org/alumni/ketaki-adhikari'
]

# Loading data from the provided URLs
loaders = UnstructuredURLLoader(urls = urls)
data = loaders.load()

# Splitting the loaded documents into chunks
text_splitter = CharacterTextSplitter(separator = '\n', chunk_size = 2000, chunk_overlap = 200)
docs = text_splitter.split_documents(data)

# Loading embeddings and a pre-trained vector store
embeddings = OpenAIEmbeddings()

with open("faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)

# Creating a ChatOpenAI model for conversation
llm = ChatOpenAI(temperature = 0.0, model_name='gpt-3.5-turbo')
memory = ConversationBufferWindowMemory(k = 10)
conversation = ConversationChain(
    llm = llm, 
    memory = memory,
    verbose = True
)

# Creating the question-answering chain
chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = VectorStore.as_retriever())

# Interactive loop for user to ask questions
print("How may I help you today?")
while True:
    # Prompt the user for a question
    user_question = input()
    
    # Perform question-answering
    conversation.predict({'input': user_question})
    output = chain({"question": user_question}, return_only_outputs = True)
    answer = output.get('answer')

    print(answer)
    
    if user_question.lower() == 'bye':
        break