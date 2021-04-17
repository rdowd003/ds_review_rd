import nltk # Imports the library

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))

for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')