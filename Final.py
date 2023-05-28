import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
from collections import Counter
from nltk.corpus import stopwords
import os
nltk.download('cmudict')
nltk.download('stopwords')
nltk.download('punkt')


'''-------------------DATA EXTRACTION--------------------------------------'''

# Read the input file
df = pd.read_excel('/home/likhith/Text_Analysis/Data/input.xlsx')

# Function to count the number of syllables in a word
def count_syllables(word):
    phones = cmudict.dict().get(word.lower())
    if phones is None:
        return 1  # Approximate the number of syllables as 1 for unknown words
    else:
        return max([len([y for y in x if y[-1].isdigit()]) for x in phones])
    
# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the article title
        title = soup.title.string.strip()

        # Find the main article content
        article_content = soup.find('article')

        # Extract the article text
        article_text = ''
        for paragraph in article_content.find_all('p'):
            article_text += paragraph.get_text() + '\n'

        # Save the extracted article as a text file
        with open(f'{url_id}.txt', 'w', encoding='utf-8') as file:
            file.write(f'{title}\n\n{article_text}')

        print(f'Saved {url_id}.txt')
    except Exception as e:
        print(f'Error processing {url_id}: {str(e)}')

print('Extraction completed.')

'''-------------------DATA EXTRACTION--------------------------------------'''

'''-------------------COMBINED STOP WORDS--------------------------------------'''

import glob

# File pattern to match the text files
file_pattern = '*.txt'

# Output file name
output_file = 'stopwords.txt'

# Get a list of all text file paths that match the pattern
file_paths = glob.glob(file_pattern)

# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Iterate over each text file
    for file_path in file_paths:
        # Open each text file in read mode
        with open(file_path, 'r', encoding='latin-1') as infile:
            # Read the content of the text file
            content = infile.read()

            # Write the content to the output file
            outfile.write(content)
            outfile.write('\n')

print(f"All text files combined into '{output_file}'.")

'''-------------------COMBINED STOP WORDS--------------------------------------'''

'''-------------------STOP WORDS REMOVAL--------------------------------------'''

stop_words_file = 'stopwords.txt'
# Load the stop words from the file
with open(stop_words_file, 'r', encoding='utf-8') as file:
    stop_words = file.read().splitlines()

folder_path = '/home/likhith/Text_Analysis/Data/'

# Iterate over each text file in the directory
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        url_id = os.path.splitext(file_name)[0]
        file_path = os.path.join(folder_path, file_name)

        try:
            # Read the article text from the file
            with open(file_path, 'r', encoding='utf-8') as file:
                article_text = file.read()
                #print(type(article_text))

            # Tokenize the article text
            words = nltk.word_tokenize(article_text)
            #print(type(words))

            # Remove stop words
            filtered_text = [word for word in words if word not in stop_words]

            # Join the filtered words back into a single text string
            filtered_text = ' '.join(filtered_text)

            # Save the filtered article back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(filtered_text)

            print(f'Stop words and punctuation removed from {file_name}')

        except FileNotFoundError:
            print(f"File '{file_path}' does not exist.")

        except Exception as e:
            print(f'Error processing {file_path}: {str(e)}')

print('Stop word removal completed.')

'''-------------------STOP WORDS REMOVAL--------------------------------------'''

'''-------------------METRICS CALCULATIONS--------------------------------------'''

# Path to the Excel file
excel_file = 'Output Data Structure.xlsx'

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_file)

positive_words_file = '/home/likhith/Text_Analysis/MasterDictionary/positive-words.txt'
neg_words_file = '/home/likhith/Text_Analysis/MasterDictionary/negative-words.txt'

# Load the positive words from the file
with open(positive_words_file, 'r', encoding='utf-8') as file:
    positive_words_list = file.read().splitlines()

# Load the negative words from the file
with open(neg_words_file, 'r', encoding='latin-1') as file:
    negative_words_list = file.read().splitlines()

positive_words = {word: 1 for word in positive_words_list}
negative_words = {word: -1 for word in negative_words_list}

text_files_folder = '/home/likhith/Text_Analysis/Data/'

# Function to count the number of syllables in a word
def count_syllables(word):
    phones = cmudict.dict().get(word.lower())
    if phones is None:
        return 1  # Approximate the number of syllables as 1 for unknown words
    else:
        return max([len([y for y in x if y[-1].isdigit()]) for x in phones])
    
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def get_word_count(text):
    # Remove punctuation marks
    text_without_punctuation = remove_punctuation(text)

    # Tokenize the text into words
    words = re.findall(r'\b\w+\b', text_without_punctuation)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Count the number of words
    word_count = len(filtered_words)

    return word_count

# Function to calculate the Average Number of Words per Sentence, Complex Word Count, Word Count,
# Syllables per Word, Personal Pronouns, and Average Word Length
def calculate_metrics(text):
    sentences = sent_tokenize(text)
    words = [word for sentence in sentences for word in word_tokenize(sentence)]
    total_words = len(words)
    total_sentences = len(sentences)

    pos_score = sum(positive_words.get(word, 0) for word in words)
    neg_score = sum(negative_words.get(word, 0) for word in words)

    polarity_score = (pos_score - abs(neg_score)) / ((pos_score + abs(neg_score)) + 0.000001)
    subjectivity_score = (pos_score + abs(neg_score)) / (total_words + 0.000001)

    avg_sentence_length = total_words / total_sentences
    
    complex_words = [word for word in words if count_syllables(word) >= 3]
    complex_word_count = len(complex_words)
    per_of_complex_words = (complex_word_count/total_words)*100

    fog_index = 0.4*(avg_sentence_length + per_of_complex_words)

    avg_no_of_words_per_sentence = avg_sentence_length

    personal_pronouns = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours','he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']
    personal_pronouns_count = sum(Counter(words)[pronoun] for pronoun in personal_pronouns)
    word_lengths = [len(word) for word in words]
    syllables_per_word = sum(count_syllables(word) for word in words) / total_words
    word_count = get_word_count(text)
    avg_word_length = sum(word_lengths) / total_words

    return pos_score,neg_score,polarity_score,subjectivity_score,avg_sentence_length, fog_index,avg_no_of_words_per_sentence,per_of_complex_words,complex_word_count, word_count, syllables_per_word, personal_pronouns_count, avg_word_length

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    url_id = row['URL_ID']
    file_name = str(url_id) + '.txt'
    file_path = os.path.join(text_files_folder, file_name)

    try:
        # Read the text from the file using 'latin-1' encoding
        with open(file_path, 'r', encoding='latin-1') as file:
            article_text = file.read()

        # Calculate the Average Sentence Length, Percentage of Complex Words, and Fog Index
        pos_score,neg_score,polarity_score,subjectivity_score,avg_sentence_length, fog_index,avg_no_of_words_per_sentence,per_of_complex_words,complex_word_count, word_count, syllables_per_word, personal_pronouns_count, avg_word_length = calculate_metrics(article_text)

        # Update the columns in the DataFrame
        df.loc[index, 'POSITIVE SCORE'] = pos_score
        df.loc[index, 'NEGATIVE SCORE'] = neg_score
        df.loc[index, 'POLARITY SCORE'] = polarity_score
        df.loc[index, 'SUBJECTIVITY SCORE'] = subjectivity_score
        df.loc[index, 'AVERAGE SENTENCE LENGTH'] = avg_sentence_length
        df.loc[index, 'PERCENTAGE OF COMPLEX WORDS'] = per_of_complex_words
        df.loc[index, 'FOG INDEX'] = fog_index
        df.loc[index, 'AVG NUMBER OF WORDS PER SENTENCE'] = avg_no_of_words_per_sentence
        df.loc[index, 'COMPLEX WORD COUNT'] = complex_word_count
        df.loc[index, 'WORD COUNT'] = word_count
        df.loc[index, 'SYLLABLE PER WORD'] = syllables_per_word
        df.loc[index, 'PERSONAL PRONOUNS'] = personal_pronouns_count
        df.loc[index, 'AVG WORD LENGTH'] = avg_word_length


        print(f'Additional scores calculated for {file_name}')

    except FileNotFoundError:
        print(f"File '{file_name}' does not exist.")

    except Exception as e:
        print(f'Error processing {file_name}: {str(e)}')

# Save the updated DataFrame to the Excel file
df.to_excel(excel_file, index=False)

print('Additional score calculation completed.')

'''-------------------METRICS CALCULATIONS--------------------------------------'''
