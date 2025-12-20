!pip install --upgrade "langchain>=0.3.29" "langchain-core>=1.0.0" langchain-openai langchain_community

#We need to downgrade protobuf to a version compatible with TensorFlow & other Google packages.
# 1. Fix protobuf
!pip install protobuf==3.20.3 --upgrade --force-reinstall

# 2. Optional: fix requests (to avid warnings)
#!pip install requests==2.32.4 --upgrade --force-reinstall

#Restart runtime to clear old imports

import langchain
import langchain_core
import langchain_openai
import langchain_community
from importlib.metadata import version

print(version("langchain"))
print(version("langchain-core"))
print(version("langchain-openai"))
print(version("langchain-community"))

#OR
#Since, all Python packages donot expose a __version__ attribute at the top level.

print(langchain.__version__)
print(langchain_core.__version__)
print(langchain_community.__version__)

# Chains
#from langchain.chains import LLMChain, SimpleSequentialChain --for older versions
from langchain_core.runnables import RunnableSequence

# Models
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings

# Documents
from langchain_core.documents import Document

# Prompts
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)

# Messages
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)

# Vector stores
from langchain_community.vectorstores import Chroma

# Text splitters
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

