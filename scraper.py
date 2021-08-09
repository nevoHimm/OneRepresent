import pandas as pd
import requests
from bs4 import BeautifulSoup as Soup
from nltk.corpus import words
# from nltk.corpus import wordnet as wn
import wordninja
import string
import re

# import nltk
# nltk.download('words')
# nltk.download('wordnet')

DATA_PATH = 'data\\companies_list.csv'
OUT_PATH = 'data\\data_out.csv'
OUT_FAIL_PATH = 'data\\failed_data.csv'
URL = 'https://www.ycdb.co/company/'
INFO_PATH = 'https://www.ycdb.co/custom-score-creator'
OUT_INFO = 'data\\features_info.txt'
HTTP_STATUS = 200

# 1. Name
# 2. Details
# 3. Template
# 4. Words Used
# 5. Important Words
COLUMNS = ['Name', 'Description', 'Template', 'Words Used', 'Important Words']
words = words.words()
words.append("software")
words.append("technology")
words.append("aerospace")
words.append("Robotics")
words.append("co")
words.append("Inc")
words.append("I")
words.append("couture")
words.append("App")
words.append("Biotic")
words = list(set(words)-set(string.ascii_lowercase)-set(string.ascii_uppercase))


def load_data(path):
    return pd.read_csv(path)


def clean_val(value, status):
    if status == 0:
        return value.lower().encode("ascii", errors="ignore").decode().replace('\'', '').replace(', Inc.', ' inc').replace('.', '-').replace(' ', '-').replace('&', 'and')
    elif status == 1:
        return value.strip().replace('$', '').replace('m', '').replace(',', '')
    else:
        return value.strip().replace('Location: ', '')


def scraper(companies_data):
    data = pd.DataFrame(columns=COLUMNS)
    failed_data = pd.DataFrame(columns=[])
    for i, row in enumerate(companies_data.itertuples()):
        # sleep(1/100)
        company_name = clean_val(row[1], 0)
        if company_name[-1] == '-':
            company_name = company_name[:-1]

        html = requests.get(URL + company_name)
        if html.status_code == HTTP_STATUS:
            soup = Soup(html.text, features='html.parser')
            general_details = soup.find_all('div', {'class': 'container pt-4'})
            details = soup.find_all('p', {'class': 'lighter'})
            for k in range(0, len(details), 3):
                key, value = details[k].text.strip().split(':')
                if (key == 'Category') and key != 'Batch' and key != 'Founded':
                    data.at[i, key] = value.strip()

                company = general_details[0].h1.text
                cap_name = company.replace('.com', '').replace('.', " ").replace('-', ' ')
                data.at[i, 'Name'] = cap_name

                name = cap_name.lower()

                desc = general_details[0].p.text.replace("'", "'").replace("Live", "").lower()
                name_no_space = name.replace(" ", '')
                desc = desc.replace(name, "").replace(',', '')\
                    .replace('.', '').replace('-', '').replace(name_no_space, '')
                data.at[i, 'Description'] = desc
                count = 1

                # Company's name is a real english word.
                if name in words or name.lower() in words or \
                        fix_suffix(name) in words or \
                        fix_suffix(name).lower() in words:
                    data.at[i, 'Template'] = str(count)
                    data.at[i, 'Words Used'] = name.lower()
                    importants([name], desc, i , data)

                elif " " in name:
                    split = cap_name.split(" ")
                    if len(split) > 2:
                        print("erase me>?!")
                    #     exit loop

                    split_suf = [fix_suffix(split[0]), fix_suffix(split[1])]

                    # Company's name is 2 real english words, separated by blank space
                    if (split[0] in words or
                        split[0].lower() in words or
                        split_suf[0] in words or
                        split_suf[0].lower() in words) and \
                            (split[1] in words or
                             split[1].lower() in words or
                             split_suf[1] in words or
                             split_suf[1].lower() in words):
                        count = 2
                        data.at[i, 'Words Used'] = split[0].lower() + ',' + split[1].lower()
                        importants(split, desc, i, data)

                    # Company's name is 1 real word, 1 made-up, separated by blank space
                    # 3.1 = first word real, 3.2 = second word is real
                    elif split[0] in words or \
                            split[0].lower() in words or \
                            split_suf[0] in words or \
                            split_suf[0].lower() in words:
                        count = 3.1
                        data.at[i, 'Words Used'] = split[0].lower()
                    elif split[1] in words or \
                            split[1].lower() in words or \
                            split_suf[1] in words or \
                            split_suf[1].lower() in words:
                        count = 3.2
                        data.at[i, 'Words Used'] = split[1].lower()
                    else:
                        # Company's name is 2 made-up words, separated by blank space
                        # WEIRD SITUATION = DELETE THIS DATA
                        data.at[i, 'Template'] = '10'
                    importants(split, desc, i, data)
                    data.at[i, 'Template'] = str(count)

                elif wordninja.split(cap_name):
                    # count = 4: 2 words made-up
                    # count = 5: Portmanteau of 1 real 1 and one fake/partial (5.1: first real, 5.2: second real)
                    # count = 6: Portmanteau 2 real words
                    sub_names = wordninja.split(cap_name)
                    count = 4
                    word = ''

                    if sub_names[0] in words or \
                            sub_names[0].lower() in words or \
                            fix_suffix(sub_names[0]) in words or \
                            fix_suffix(sub_names[0]).lower() in words:
                        count += 1.1
                        word += sub_names[0].lower()

                    if sub_names[-1] in words or \
                            sub_names[-1].lower() in words or \
                            fix_suffix(sub_names[-1]) in words or \
                            fix_suffix(sub_names[-1]).lower() in words:
                        count += 1.2
                        if len(word) > 0:
                            word += ',' + sub_names[-1].lower()
                        else:
                            word += sub_names[-1].lower()

                    important = ''

                    for sub_name in sub_names:
                        sub_low = sub_name.lower()
                        if sub_low in desc and len(sub_name) > 2:
                            suffix = desc.split(sub_low)[1].split(' ')[0]
                            prefix = desc.split(sub_low)[0].split(' ')[-1]

                            cur_imp = ''
                            if len(suffix) > 0:
                                cur_imp = sub_low + suffix
                            if len(prefix) > 0:
                                cur_imp = prefix + cur_imp

                            important = cur_imp
                            cur_word = ''

                            if len(cur_imp) > 0:
                                for l in range(sub_names.index(sub_name)-1, -1, -1):
                                    if (sub_names[l] + sub_name).lower() in cur_imp.lower():
                                        cur_word = sub_names[l] + sub_low
                                    else:
                                        break

                                for p in range(sub_names.index(sub_name)+1, len(sub_names)):
                                    if (sub_name + sub_names[p]).lower() in cur_imp.lower():
                                        cur_word = cur_word + sub_names[p]
                                    else:
                                        break
                                word += ',' + cur_word.lower()

                    if count == 6.3:
                        count = 6
                    data.at[i, 'Words Used'] = word
                    data.at[i, 'Important Words'] = important
                    data.at[i, 'Template'] = str(count)

        else:
            failed_data = failed_data.append({'Company': row[1]}, ignore_index=True)
        print(str(i) + ": " + row[1])

    data.to_csv(OUT_PATH, index=False)
    failed_data.to_csv(OUT_FAIL_PATH, index=False)


def importants(sub_names, desc, i, data):
    if len(sub_names) == 1:
        sub_names = wordninja.split(sub_names[0])

    important = ''
    for sub_name in sub_names:
        sub_low = sub_name.lower()
        sub_suf = fix_suffix(sub_low)
        if sub_low in desc and len(sub_name) > 2:
            suffix = desc.split(sub_low)[1].split(' ')[0]
            prefix = desc.split(sub_low)[0].split(' ')[-1]

            cur_imp = ''
            if len(suffix) > 0:
                cur_imp = sub_low + suffix
            if len(prefix) > 0:
                cur_imp = prefix + cur_imp

            important += ',' + cur_imp

        if (sub_low in desc) and (sub_low in words or sub_suf in words)\
                and (sub_low not in important):
            if len(desc) > desc.index(sub_low)+len(sub_low) and desc.index(sub_low) > 0:
                if desc[desc.index(sub_low)-1] == ' ' \
                        and desc[desc.index(sub_low)+len(sub_low)] == ' ':
                    important += ',' + sub_low
            else:
                important += ',' + sub_low
        elif (sub_suf in desc) and (sub_suf in words) \
                and (sub_suf not in important):
            if len(desc) > desc.index(sub_suf)+len(sub_suf) and desc.index(sub_suf) > 0:
                if desc[desc.index(sub_suf)-1] == ' ' \
                        and desc[desc.index(sub_suf)+len(sub_suf)] == ' ':
                    important += ',' + sub_suf
            elif desc.index(sub_suf) == 0 and len(desc) > desc.index(sub_suf)+len(sub_suf):
                if desc[desc.index(sub_suf)+len(sub_suf)] == ' ':
                    important += ',' + sub_suf
            elif len(desc) <= desc.index(sub_suf)+len(sub_suf) and desc.index(sub_suf) > 0:
                if desc[desc.index(sub_suf)-1] == ' ':
                    important += ',' + sub_suf

    data.at[i, 'Important Words'] = important


def fix_suffix(word):
    """
    fixes a word with undetectable vocabulary suffix
    :param word: word to fix
    :return: a detectable word for vocabulary
    """
    if len(word) < 1:
        return word
    if word[-2:] == "'s":
        new_word = word[:-2]
    elif len(word) > 3 and word[-3:] == 'ies':
        new_word = word.replace('ies', 'y')
    elif len(word) > 3 and word[-3:] == 'ses':
        new_word = word.replace('ses', 's')
    elif len(word) > 3 and word[-3:] == 'ves':
        new_word = word.replace('ves', 'f')
    elif len(word) > 3 and word[-3:] == 'oes':
        new_word = word.replace('oes', 'o')
    elif len(word) > 3 and word[-2:] == 'es':
        new_word = word.replace('es', 'e')
    elif len(word) > 3 and word[-2:] == 'ity':
        new_word = word.replace('ity', '')
    elif len(word) > 3 and word[-2:] == 'ing':
        new_word = word.replace('ing', '')
    elif word[-1] == 's':
        new_word = word[:-1]
    else:
        return word
    return new_word


def scrape_info():
    with open(OUT_INFO, 'w+') as out_file:
        html = requests.get(INFO_PATH)
        soup = Soup(html.text, features='html.parser')
        features = soup.find_all('div', {'class': 'col-3'})
        for i in range(len(features)):
            if i:
                feature = features[i].text.strip()
                description = features[i].span['title']
                out_file.write(feature + ': ' + description + '\n')


scrape_info()
scraper(load_data(DATA_PATH))

# print("drone" in words)
# print(wordninja.split("AskMyClass"))

# sub_low = "edit"
# desc = " is a modern credit card company in africa that allows consumers make " \
#        "purchases and pay in monthly installments across online and offline merchants. "

# fashion = wn.synset('fashion.n.01')
# clothes = wn.synset('clothes.n.01')
# blah = {fashion, clothes}
#
# print(couture.wup_similarity(blah))
