from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import twokenize, re, json
from twokenize import Protected2, happy_emote, sad_emote, other_emote, deco_emote, heart_emote, url_regex, email_regex 

#tokenise using nltk.tokenize word_tokenize
def nltk_tokenise(tweet):
    tokens = word_tokenize(tweet)
    tokenised = ''
    for token in tokens:
        tokenised += str(token.encode('utf-8')+' ')
    return tokenised.strip()


#replace URLs with "URLLINK"
def replaceURLs(tweet):
    return re.sub(url_regex, "URLLINK", tweet)

def replaceemail(tweet):
    return re.sub(email_regex, "EMAIL", tweet)

# replace emotes
def replacehappy(tweet):
    return re.sub(happy_emote, "{HAPPYEMOTE}", tweet)

def replacesad(tweet):
    return re.sub(sad_emote, "{SADEMOTE}", tweet)

def replaceother(tweet):
    return re.sub(other_emote, "OTHEREMOTE", tweet)

def replacedeco(tweet):
    return re.sub(deco_emote, "DECOEMOTE", tweet)

def replaceheart(tweet):
    return re.sub(heart_emote, "HEARTEMOTE", tweet)

def replaceemotes(tweet):
    return re.sub(Protected2, "EMOTE", tweet)

#replace user mentions with "USERMENTION"
def replaceUserMentions(tweet):
    return re.sub("(@[A-Za-z0-9_]+)", "USERMENTION", tweet)


#replace all non-alphanumeric
def replaceRest(tweet):
    result = re.sub("[^a-zA-Z0-9]", " ", tweet)
    return re.sub(' +',' ', result)

# replace numbers
def replacenumbers(tweet):
    return re.sub(r"\b[0-9]+\b", " ", tweet)

# replace non-alphanumeric
def replacenonalpha(tweet):
    return re.sub(r"([^\s\w]|_)+", " ", tweet)

# remove words with less than 4 chars
def replaceshort(tweet):
    return re.sub(r"\b\w{1,3}\b", " ", tweet)

# remove extra whitespace
def replacespace(tweet):
    return re.sub(' +',' ', tweet)


def stop_words():
    stop_words = set(stopwords.words('english'))
    stop_words = {replacenonalpha(w) for w in stop_words}
    stop_words = {replaceshort(w) for w in stop_words}
    stop_words = {replacespace(w) for w in stop_words}
    stop_words = {w.strip() for w in stop_words}
    return {w for w in stop_words if len(w)>3}


# remove stopwords
def removestop(tweet, stop_words):
    return [w for w in tweet if not w in stop_words] 


def testit():
    with open("mytweets.json", 'r') as f:
        for line in f:
            text = json.loads(line)['text']
            newtext = text.lower()
#            newtext = nltk_tokenise(text).lower()
            newtext = replaceURLs(newtext)
            newtext = replaceUserMentions(newtext)
            newtext = replaceRest(newtext)
            print(text + '\n' + newtext + '\n')
    f.close()


clean_stop_words = stop_words()

# Full preprocessing
def fullproc(text):
    newtext = text.lower()
    newtext = replaceemail(newtext)
    newtext = replaceURLs(newtext)
    newtext = replaceemotes(newtext)
#    newtext = replacehappy(newtext)
#    newtext = replacesad(newtext)
#    newtext = replaceother(newtext)
#    newtext = replaceheart(newtext)
#    newtext = replacedeco(newtext)
    newtext = replaceUserMentions(newtext)
    newtext = replacenonalpha(newtext)
    newtext = replacenumbers(newtext)
    newtext = replaceshort(newtext)
    newtext = replacespace(newtext)
    newtext = newtext.strip()
    newtext = twokenize.tokenize(newtext)
    newtext = removestop(newtext, clean_stop_words)
    newtext = ' '.join(newtext)
    
    return newtext

def fullproctoken(text):
    newtext = text.lower()
    newtext = replaceemail(newtext)
    newtext = replaceURLs(newtext)
    newtext = replaceemotes(newtext)
    newtext = replaceUserMentions(newtext)
    newtext = replacenonalpha(newtext)
    newtext = replacenumbers(newtext)
    newtext = replaceshort(newtext)
    newtext = replacespace(newtext)
    newtext = newtext.strip()
    newtext = twokenize.tokenize(newtext)
    newtext = removestop(newtext, clean_stop_words)
    
    return newtext

def pipeproc(text):
    newtext = text.lower()
    newtext = replaceemail(newtext)
    newtext = replaceUserMentions(newtext)
    newtext = replaceURLs(newtext)
    newtext = replacenumbers(newtext)

    return newtext

# without stop words
def test(text):
    newtext = text.lower()
    newtext = replaceURLs(newtext)
    newtext = replaceUserMentions(newtext)
    newtext = replacenonalpha(newtext)
    newtext = replacenumbers(newtext)
    newtext = replaceshort(newtext)
    newtext = replacespace(newtext)
    newtext = newtext.strip()
    newtext = twokenize.tokenize(newtext)
    
    return newtext


def preproc(text):
    newtext = text.lower()
    newtext = replaceURLs(newtext)
    newtext = replaceUserMentions(newtext)
    newtext = replacenonalpha(newtext)
    newtext = replacenumbers(newtext)
    newtext = replaceshort(newtext)
    newtext = replacespace(newtext)
    newtext = newtext.strip()
    newtext = twokenize.tokenize(newtext)
    newtext = removestop(newtext, clean_stop_words)
    newtext = ' '.join(newtext)
    
    return newtext

#a = "sdfsdf @test :) dfsdfgd :( :) sdfds@google.com dsfsd bb ;) fngdsfgsd.com rtreter :))):):)):)) dfsdf.co.uk dsfds dfff.xxx :D dfds <3 dfd"
#
##b =re.sub(Protected, " TEST ", a)
#
#print(a)
#print(fullproc(a))
