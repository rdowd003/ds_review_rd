## Instructions and set-up we're incorrect, here's best solution

#First test case was:
# title_case(''),'') --> incorrect syntax, and instructions required arguments 

def title_case(title, minor_words=''):
    title = title.capitalize().split()
    minor_words = minor_words.lower().split()
    return ' '.join([word if word in minor_words else word.capitalize() for word in title])