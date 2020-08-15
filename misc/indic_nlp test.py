# Importing libraries for handling text data in Devnagri
import sys
from indicnlp import common


INDIC_NLP_LIB_HOME = r"C:/Users/Aditya Singh/Desktop/Deep Learning/7. Language Modelling/English to Hindi/Translation Model/indic_nlp_library"
INDIC_NLP_RESOURCES = r"C:/Users/Aditya Singh/Desktop/Deep Learning/7. Language Modelling/English to Hindi/Translation Model/indic_nlp_resources"

# Add library to Python path
sys.path.append(r'{}/src'.format(INDIC_NLP_LIB_HOME))
# Set environment variable for resources folder
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp.tokenize import sentence_tokenize

indic_string = """तो क्या विश्व कप 2019 में मैच का बॉस टॉस है? यानी मैच में हार-जीत में 
टॉस की भूमिका अहम है? आप ऐसा सोच सकते हैं। विश्वकप के अपने-अपने पहले मैच में बुरी तरह हारने वाली एशिया की दो टीमों
पाकिस्तान और श्रीलंका के कप्तान ने हालांकि अपने हार के पीछे टॉस की दलील तो नहीं दी, लेकिन यह जरूर कहा था कि वह एक अहम टॉस हार गए थे।
"""

# Split the sentence, language code "hi" is passed for hindi
sentences = sentence_tokenize.sentence_split(indic_string, lang='hi')
for t in sentences:
    print(t)


# Tokenizing Hindi sentence
from indicnlp.tokenize import indic_tokenize
hindi_text = "किताब रखी जाएगी तो अपराधियों को देखोंगे कि जो कुछ उसमें होगा उससे डर रहे है और कह रहे है ,  हाय , हमारा दुर्भाग्य ! यह कैसी किताब है कि यह न कोई छोटी बात छोड़ती है न बड़ी , बल्कि सभी को इसने अपने अन्दर समाहित कर रखा है । जो कुछ उन्होंने किया होगा सब मौजूद पाएँगे । तुम्हारा रब किसी पर ज़ुल्म न करेगा"
_tokens = indic_tokenize.trivial_tokenize(hindi_text)


